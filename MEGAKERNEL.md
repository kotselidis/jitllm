# Decode megakernel (Llama F16, CUDA)

Decode **one token with one kernel launch**. The layered plan issues about nine kernels per
layer plus the logits, so 147 launches per token for Llama-3.2-1B. The megakernel runs the
whole forward pass, embedding to sampled token, as one persistent kernel. Its blocks move from
phase to phase through a grid-wide barrier, TornadoVM's `KernelContext.gridBarrier()`.

```bash
./jitllm --gpu --cuda --model Llama-3.2-1B-Instruct-f16.gguf --prompt "..." --temperature 0 --megakernel
```

`--megakernel` sets `-Djitllm.megakernel=true`. With `--temperature 0` it also sets
`-Djitllm.deviceSample=true`, so the token is picked in the kernel and only 4 bytes come back
to the host.

Options, all as `-Djitllm.megakernel.<name>`:

| Option | Default | Effect |
|---|---|---|
| `blocks` | largest block count ≤ SMs whose warps divide `dim` | grid size |
| `blocksPerSM` | 1 | with a register cap above 1 |
| `fusedDown` | `true` when `dim` ≤ 2048 | down projection fused into gate/up |
| `cudaGraphs` | `true` | replay the launch as a CUDA graph |
| `normOnePass` | `false` | single-pass RMS norm |

## Scope

The megakernel covers these sessions only. Any other session is refused by name at plan
construction instead of silently falling back to the layered plan:

- **Model:** Llama-architecture GGUF with F16 weights.
- **KV cache:** FP16 (the default). `--fp32-kv-cache` is refused.
- **Mode:** single-token. `--with-prefill-decode` is refused.
- **Backend:** the TornadoVM CUDA backend, from a build with `KernelContext.gridBarrier()`
  (branch `feature/megakernel` of `kotselidis/TornadoVM`).
- **Shapes:** `dim` up to 4096, `headSize` a multiple of 32 and at most 128.

## How it works

`kernels/LlamaMegakernel.java` is a single KernelContext kernel. `TornadoVMMasterPlanMegakernel`
builds the plan.

- **Weights.** Weights are copied once into whole-model arrays: `wqkv`, `wo`, `w1`, `w3`, `w2`,
  plus all RMS weights and epsilon in one `FloatArray`. With the fused down projection, `w2` is
  stored transposed, as `[hidden][dim]`. A task has a fixed parameter list (20 here), and the
  layered plan passes nine arrays per layer. The layered arrays never reach the device.
- **Grid.** 256-thread blocks, launched with `cuLaunchCooperativeKernel`. All blocks must be
  resident at once: a grid that doesn't fit is refused with
  `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` rather than deadlocking. The default block count
  divides the row counts evenly across warps: 64 on an A10 (72 SMs) and on an RTX 5070 Ti
  (70 SMs).
- **Per layer, five phases, each ending at a grid barrier:**
  1. RMS norm, fused QKV, RoPE, and the K/V write into the paged cache. One warp owns the two
     adjacent rows that RoPE rotates together.
  2. Attention partials. Each block takes (head, KV split) units and runs an online softmax per
     warp, merged across the block's warps.
  3. The split combine into shared memory, then the output projection plus residual.
  4. RMS norm, gate/up with SwiGLU, and the down projection fused in. The warp that produces
     `hb[row]` streams row `row` of the transposed `w2` into per-lane register sums. The block
     sums its warps in a fixed order and writes one partial.
  5. The residual: the blocks' partials added in block order.
- **RMS norm.** Every block normalises the residual stream itself, into shared memory, instead of
  waiting on another barrier.
- **Ends of the pass.** Embedding conversion runs first. The final norm, the vocabulary
  projection and, with device sampling, a greedy argmax run last (per-warp best, one barrier,
  then block 0 picks the token).

That makes 82 grid barriers per token for Llama-3.2-1B.

## Determinism

Greedy text is byte-identical to the layered plan's in every configuration measured. The fused
down projection sums in a different order from the layered plan, but in a fixed one, so runs are
deterministic.

## Results

Measured 2026-10-09. Model Llama-3.2-1B-Instruct F16, greedy, 128-token context, the same prompt
and the same GGUF file on both machines. "GPU time" is the duration of one `forward` launch under
nsys. llama.cpp's figure is `llama-bench tg128`, which includes its own host time. jitllm's
end-to-end figures cover the prompt plus 128 tokens, host loop included, and are medians of 3 runs.

### NVIDIA A10 (sm_86, 72 SMs, 600 GB/s), kobol, CUDA 13.0, JDK 25

| Plan | tok/s end to end | ms per token |
|---|---|---|
| Layered (default) | 99.9 | 7.25 (GPU, sum of all kernels) |
| Layered, `--cuda-graphs` | 113.1 | — |
| **Megakernel** | **~167** (166–172) | **5.63** (GPU) |
| llama.cpp `11fe021`, tg128 | 183.2 | 5.46 |

### NVIDIA RTX 5070 Ti (sm_120, 70 SMs, 896 GB/s), cyclone, CUDA 13.0, JDK 25

| Plan | tok/s end to end | ms per token |
|---|---|---|
| Layered (default) | 214.7 | — |
| Layered, `--cuda-graphs` | 233.1 | — |
| **Megakernel** | **281.8** (266–288) | **3.38** (GPU) |
| llama.cpp tg128 | 302.6 | 3.30 |

The megakernel's GPU time is within 3% of llama.cpp on the A10 and within 2.4% on the 5070 Ti.
On the 5070 Ti it streams about 730 GB/s of weights, 81% of peak.

### What moved the kernel (A10)

| Change | Kernel time |
|---|---|
| 64 blocks instead of 72 | 5.94 → 5.81 ms |
| Down projection fused into gate/up, together with the release/acquire barrier | 5.78 → 5.63 ms |
| Device sampling, then CUDA graphs (end to end) | 154 → 163 → 168 tok/s |

None of these helped, and each was measured and dropped:
- 16-byte `cp.async` weight streaming;
- 8 loads per lane;
- one FFMA per half;
- L2 prefetch before barriers;
- `__ldg`/`__ldca` weight loads;
- compile-time model shapes.

These made it worse: two blocks per SM with a register cap; two down rows per warp; a whole
block per down row.

The fused down projection gains nothing on its own (5.81 ms). It pays off only together with
the faster barrier.

### A code-layout effect

Deleting the unused `normOnePass` code made the kernel 0.25 ms slower per token on the A10
(5.64 against 5.89 ms). The generated CUDA source was otherwise identical and the register count
the same: ptxas lays the kernel out differently. The option stays, with a comment in
`LlamaMegakernel.java`. Re-measure before removing it.
