# Decode megakernel (Llama F16, CUDA)

Decode **one token with one kernel launch**. The layered plan issues about nine kernels per
layer plus the logits, so 147 launches per token for Llama-3.2-1B. The megakernel runs the
whole forward pass, embedding to logits, as one persistent kernel. Its blocks move from phase
to phase through a grid-wide barrier, TornadoVM's `KernelContext.gridBarrier()`.

```bash
./jitllm --gpu --cuda --model Llama-3.2-1B-Instruct-f16.gguf --prompt "..." --megakernel
```

Equivalent JVM flags: `-Djitllm.megakernel=true`, and optionally
`-Djitllm.megakernel.blocksPerSM=N` (default 1).

## Scope

The megakernel covers these sessions only. Any other session is refused by name at plan
construction instead of silently falling back to the layered plan:

- **Model:** Llama-architecture GGUF with F16 weights.
- **KV cache:** FP16 (the default). `--fp32-kv-cache` is refused.
- **Mode:** single-token. `--with-prefill-decode` is refused.
- **Backend:** the TornadoVM CUDA backend. A TornadoVM build with `KernelContext.gridBarrier()`
  is required (branch `feature/megakernel` of `kotselidis/TornadoVM`).
- **Shapes:** `dim` up to 4096, `headSize` a multiple of 32 and at most 128.

## How it works

`kernels/LlamaMegakernel.java` is a single KernelContext kernel. `TornadoVMMasterPlanMegakernel`
builds the plan.

- **Weights.** Weights are copied once into whole-model arrays: `wqkv`, `wo`, `w1`, `w3`, `w2`,
  plus all RMS weights and epsilon in one `FloatArray`. A task has a fixed parameter list (19
  here), and the layered plan passes nine arrays per layer. The layered arrays never reach the
  device.
- **Grid.** `SMs × blocksPerSM` blocks of 256 threads, launched with `cuLaunchCooperativeKernel`.
  All blocks must be resident at once: a grid that doesn't fit is refused with
  `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` rather than deadlocking. Above one block per SM, the
  plan compiles with `--maxrregcount` so that many blocks fit.
- **Per layer, five phases, each ending at a grid barrier:**
  1. RMS norm, fused QKV, RoPE, and the K/V write into the paged cache. One warp owns the two
     adjacent rows that RoPE rotates together.
  2. Attention partials. Each block takes (head, KV split) units and runs an online softmax per
     warp, merged across the block's warps.
  3. The split combine into shared memory, then the output projection plus residual.
  4. RMS norm and gate/up with SwiGLU.
  5. The down projection plus residual.
- **RMS norm.** Every block normalises the residual stream itself, into shared memory, instead of
  waiting on another barrier.
- **Ends of the pass.** Embedding conversion runs first; the final norm and the vocabulary
  projection run last.

## Results

Measured 2026-10-09 on kobol: NVIDIA A10 (72 SMs, 24 GB), CUDA 13.0, JDK 25, TornadoVM
`feature/megakernel`. Model Llama-3.2-1B-Instruct F16, greedy, 128-token context, the same
prompt for every run.

| Plan | tok/s (end to end) | GPU time per token |
|---|---|---|
| Layered (default) | 99.9 | 7.25 ms (sum of 9 kernels × 16 layers + logits) |
| Layered, `--cuda-graphs` | 113.1 | — |
| **Megakernel**, 1 block/SM | **153–155** (3 runs) | **5.81 ms** (one `forward` launch) |
| Megakernel, 1 block/SM, `--cuda-graphs` | 158.0 | — |
| Megakernel, 2 blocks/SM | 152.7 | 5.86 ms |
| Megakernel, 3 / 4 blocks/SM | 133.7 / 124.0 | register cap spills |

The generated text is **byte-identical** to the layered plan's in every configuration.

At 5.81 ms the kernel streams about 425 GB/s of weights, about 70% of the A10's peak. The time
doesn't change with occupancy, with 4× unrolled loads, or with 2 blocks per SM. The next step is
wider weight loads: 16 bytes per lane, for example `cp.async` 16-byte tiles, instead of the
current `half2` (4 bytes per lane).
