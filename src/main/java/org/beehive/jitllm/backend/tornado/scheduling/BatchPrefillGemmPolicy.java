package org.beehive.jitllm.backend.tornado.scheduling;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;

// @formatter:off
/**
 * Whether the batched-prefill projections run as tiled matrix multiplications.
 *
 * <p>The kernels they replace launch one workgroup per (token, output row), so every token reads
 * the whole weight matrix again and reduces its row through a shared-memory tree. The tiled kernels
 * compute 64-row by 64-token tiles from weights and activations staged in threadgroup memory,
 * reading each weight once per 64 tokens. Measured on an M4 Pro, Qwen3-0.6B F16 prefill with
 * lane-cooperative attention, pp512 at batch 128: 404 tok/s with the per-row kernels, 1590 tiled
 * (1962 at batch 512); greedy text identical to single-token prefill. Metal only: the other
 * backends run the tensor-core kernels or were not measured with these.
 */
// @formatter:on
public final class BatchPrefillGemmPolicy {

    private BatchPrefillGemmPolicy() {}

    /** Whether the active device runs the tiled batched-prefill projections. */
    public static boolean tiled() {
        return TornadoDevices.current().backend().equals(BackendId.METAL);
    }
}
