package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The layout work around a prefill attention computed as tensor-core GEMMs: queries in FP16, and
 * the paged FP16 key/value cache gathered into contiguous per-key/value-head tiles, the values
 * transposed. The scores, the causal softmax and the weighted values are {@link
 * BatchMmaKernels#gemmMMAStrided} and {@link BatchMmaKernels#causalSoftmax}.
 *
 * <p>{@code batchInfo} is the chunk's {@code [start position, active rows, slot]}.
 */
// @formatter:on
public final class TensorCoreAttentionKernels {

    private TensorCoreAttentionKernels() {}

    /**
     * {@code out[t][c] = qkv[t * qkvStride + c]} in FP16 for the {@code dim} query columns of every
     * padded row. Worker: {@code rows * dim} lanes.
     */
    public static void queriesToHalf(
            KernelContext context,
            FloatArray qkv,
            HalfFloatArray out,
            int dim,
            int qkvStride,
            int total) {
        int i = context.globalIdx;
        if (i < total) {
            int t = i / dim;
            int c = i - t * dim;
            out.set(i, new HalfFloat(qkv.get(t * qkvStride + c)));
        }
    }

    // @formatter:off
    /**
     * The layer's paged FP16 cache up to the chunk's last position, per key/value head {@code g}:
     * {@code keys[g][t][d]} and {@code valuesT[g][d][t]}, {@code paddedPositions} positions each.
     * Positions past the chunk's last one, up to the next multiple of 128, are written as zero;
     * nothing past that is read. Worker: {@code paddedPositions * kvDim} lanes.
     */
    // @formatter:on
    public static void gatherPagedKeyValues(
            KernelContext context,
            IntArray batchInfo,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            HalfFloatArray keys,
            HalfFloatArray valuesT,
            int layer,
            int kvDim,
            int headSize,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int paddedPositions) {
        int idx = context.globalIdx;
        int valid = batchInfo.get(0) + batchInfo.get(1);
        int filled = ((valid + 127) / 128) * 128;
        if (idx < Math.min(paddedPositions, filled) * kvDim) {
            int t = idx / kvDim;
            int c = idx - t * kvDim;
            int g = c / headSize;
            int d = c - g * headSize;
            int kIndex = (g * paddedPositions + t) * headSize + d;
            int vIndex = (g * headSize + d) * paddedPositions + t;
            if (t < valid) {
                int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
                int off =
                        KvBlockAddress.offset(
                                        blockTable,
                                        batchInfo.get(2),
                                        t,
                                        layerOff,
                                        kvDim,
                                        blockCfg,
                                        blockStride)
                                + c;
                keys.set(kIndex, keyCache.get(off));
                valuesT.set(vIndex, valueCache.get(off));
            }
            if (t >= valid) {
                keys.set(kIndex, new HalfFloat(0.0f));
                valuesT.set(vIndex, new HalfFloat(0.0f));
            }
        }
    }
}
