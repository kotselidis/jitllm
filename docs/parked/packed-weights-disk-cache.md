# Parked: on-disk cache of packed weights

Status: parked. It works and its tests pass, but repacking on the GPU at load gives the same speed
and the same output without a cache file, so the cache was removed from the main line. This
branch keeps it, with GPU repacking as the alternative (`-Djitllm.packed.mode=gpu`).

## What it does

The packed Q8_0 and Q4_0 weights (`-Djitllm.q8.packed=true`, `-Djitllm.q4.packed=true`) need a
different byte order from the GGUF file: each 128-row by 64-k tile stored contiguously and
16-byte aligned, so the prefill GEMM can copy it with `cp.async` and the decode kernels can read it
with 64-bit loads. The cache (`PackedQ8_0Cache`) converts each projection once per model on the
host and keeps the result on disk:

- **One file per model** under `~/.cache/jitllm/q8packed` (`-Djitllm.q8.packed.cache.dir`),
  named by a SHA-256 of the file size, the GGUF header, the first and last megabyte of tensor data
  and the packed format's version. Q8_0 and Q4_0 entries share the file.
- **Later loads** memory-map each packed tensor the way GGUF tensors are mapped, so nothing is
  copied into direct memory.
- **First load** packs each tensor straight into a memory mapping of a temporary file, on all
  cores, a column tile per thread, with each tensor flushed to disk in the background while the
  next one packs. The file is moved into place only once every tensor is written, so a failed load
  leaves no cache.
- `-Djitllm.q8.packed.cache=false` packs in memory instead.

## Why it is parked

GPU repacking uploads the GGUF bytes as they are and repacks them on the device into the weight's
own buffer, through one scratch buffer the size of the largest weight. Measured on two NVIDIA A10s
(Q4_0, prefill and decode in tok/s):

| Model | Prefill 256, GPU / cache | Prefill 512, GPU / cache | Decode, GPU / cache | Repack per GPU |
|---|---|---|---|---|
| Qwen3.8-27B | 860.0 / 852.9 | 703.2 / 703.2 | 26.13 / 26.28 | 0.9 s |
| Gemma-4-31B | 885.4 / 884.8 | 736.3 / 738.4 | 23.19 / 23.11 | 1.1 s |
| Llama-3.3-70B | 422.8 / 423.5 | 331.7 / 332.1 | 9.76 / 9.86 | 1.6 s |

The generated text is identical in both modes. What the cache costs and GPU repacking does not:

- a first load that rebuilds the file: Gemma-4-31B Q4_0 took 115 s against 86 s once cached, and
  about 80 s unpacked;
- a second copy of every packed projection on disk (15 to 40 GB per model);
- the cache key, versioning and invalidation, and a code path that has to be kept in step with the
  packed layout.

## When to revive it

When repacking on the device is not possible: a backend without the device repack kernels, or a
device too short of memory for one scratch buffer the size of the largest weight.
