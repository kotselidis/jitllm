package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.vectors.Half2;

/**
 * The whole Llama F16 single-token forward pass as one persistent kernel.
 *
 * <p>One launch per token instead of one per operation: the grid stays resident and moves from
 * phase to phase through {@link KernelContext#gridBarrier()}. Per layer there are five phases, each
 * ending at a grid barrier:
 *
 * <ol>
 *   <li>RMS norm, fused QKV projection, RoPE, and the K/V write into the paged cache. A warp owns a
 *       pair of adjacent rows, which is exactly the pair RoPE rotates.
 *   <li>Attention partials: one (head, KV split) unit per block, an online softmax per warp, merged
 *       across the block's warps.
 *   <li>The split combine, into each block's shared vector, then the output projection plus
 *       residual.
 *   <li>RMS norm and the gate/up projections with SwiGLU.
 *   <li>The down projection plus residual.
 * </ol>
 *
 * The embedding conversion runs before the first layer and the final norm plus the vocabulary
 * projection after the last, so the host sees one kernel per token.
 *
 * <p>Every block normalises the residual stream itself, into shared memory, instead of one block
 * normalising it for everybody: {@code dim} reads per block cost less than another barrier.
 *
 * <p>Contract with {@code TornadoVMMasterPlanMegakernel}, which checks it before building: {@link
 * #BLOCK_SIZE} threads per block, every block resident at once, {@code dim <= }{@link #MAX_DIM},
 * {@code headSize} a multiple of 32 and at most {@link #MAX_HEAD_SIZE}, {@code dim}, {@code kvDim}
 * and {@code hiddenDim} even.
 */
public final class LlamaMegakernel {

    /** Threads per block. The shared buffers below are sized for this many warps. */
    public static final int BLOCK_SIZE = 256;

    /** The widest residual stream a block can hold in shared memory. */
    public static final int MAX_DIM = 4096;

    /** The widest head: four values per lane. */
    public static final int MAX_HEAD_SIZE = 128;

    private static final int WARPS = BLOCK_SIZE / 32;
    private static final int HEAD_VALUES_PER_LANE = MAX_HEAD_SIZE / 32;

    /** Weight halves a warp streams per stage: 32 lanes of one 16-byte copy each. */
    public static final int CHUNK = 256;

    /** Copies in flight per lane. */
    private static final int STAGES = 4;

    /** Ints per ring stage of one warp: 32 lanes of four. */
    private static final int RING_STAGE = 128;

    // Layout of the meta array.
    public static final int META_DIM = 0;
    public static final int META_KV_DIM = 1;
    public static final int META_HIDDEN_DIM = 2;
    public static final int META_LAYERS = 3;
    public static final int META_HEADS = 4;
    public static final int META_KV_MUL = 5;
    public static final int META_HEAD_SIZE = 6;
    public static final int META_VOCABULARY = 7;
    public static final int META_KV_BLOCK_CFG = 8;
    public static final int META_KV_BLOCK_STRIDE = 9;
    public static final int META_SPLITS = 10;
    public static final int META_SIZE = 11;

    private LlamaMegakernel() {}

    /** Floats of scratch the kernel needs: Q, the FFN hidden vector, the attention partials. */
    public static int scratchSize(int dim, int hiddenDim, int heads, int headSize, int splits) {
        return dim + hiddenDim + heads * splits * (headSize + 2);
    }

    /**
     * Index of the epsilon in {@code norms}, which holds the attention norms of every layer, then
     * the FFN norms, then the final norm, then epsilon.
     */
    public static int epsilonIndex(int dim, int layers) {
        return 2 * layers * dim + dim;
    }

    // @formatter:off
    public static void forward(
            KernelContext context,
            HalfFloatArray embedding,
            FloatArray x,
            FloatArray scratch,
            FloatArray norms,
            HalfFloatArray wqkv,
            HalfFloatArray wo,
            HalfFloatArray w1,
            HalfFloatArray w3,
            HalfFloatArray w2,
            HalfFloatArray wcls,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            IntArray positionHolder,
            IntArray blockTable,
            IntArray meta,
            FloatArray logits) {
        int tid = context.localIdx;
        int lane = tid % 32;
        int warp = tid / 32;
        int blocks = context.globalGroupSizeX / BLOCK_SIZE;
        int block = context.groupIdx;
        int globalWarp = block * WARPS + warp;
        int totalWarps = blocks * WARPS;
        int ringBase = warp * STAGES * RING_STAGE + lane * 4;

        float[] vec = context.allocateFloatLocalArray(MAX_DIM);
        float[] reduce = context.allocateFloatLocalArray(WARPS);
        float[] mergeMax = context.allocateFloatLocalArray(WARPS);
        float[] mergeSum = context.allocateFloatLocalArray(WARPS);
        float[] mergeAcc = context.allocateFloatLocalArray(WARPS * MAX_HEAD_SIZE);
        int[] ring = context.allocateIntLocalArray(WARPS * STAGES * RING_STAGE);
        float[] acc = new float[HEAD_VALUES_PER_LANE];

        int dim = meta.get(META_DIM);
        int kvDim = meta.get(META_KV_DIM);
        int hiddenDim = meta.get(META_HIDDEN_DIM);
        int layers = meta.get(META_LAYERS);
        int heads = meta.get(META_HEADS);
        int kvMul = meta.get(META_KV_MUL);
        int headSize = meta.get(META_HEAD_SIZE);
        int vocabulary = meta.get(META_VOCABULARY);
        int blockCfg = meta.get(META_KV_BLOCK_CFG);
        int blockStride = meta.get(META_KV_BLOCK_STRIDE);
        int splits = meta.get(META_SPLITS);

        int pos = positionHolder.get(0);
        int slot = positionHolder.get(1);
        float eps = norms.get(epsilonIndex(dim, layers));
        float invSqrtHead = 1.0f / TornadoMath.sqrt(headSize);

        int qOff = 0;
        int hbOff = dim;
        int attOff = dim + hiddenDim;
        int partStride = headSize + 2;
        int qkvRows = dim + 2 * kvDim;

        for (int i = context.globalIdx; i < dim; i += context.globalGroupSizeX) {
            x.set(i, embedding.get(i).getFloat32());
        }
        context.gridBarrier();

        for (int l = 0; l < layers; l++) {
            int layerOff = KvBlockAddress.layerOffset(l, kvDim, blockCfg);
            int cacheOff =
                    KvBlockAddress.offset(
                            blockTable, slot, pos, layerOff, kvDim, blockCfg, blockStride);

            // 1. RMS norm, QKV, RoPE, K/V into the cache.
            normalize(context, x, norms, l * dim, dim, eps, vec, reduce, tid, warp);
            for (int p = globalWarp; p < qkvRows / 2; p += totalWarps) {
                int row = 2 * p;
                int rowOff = (l * qkvRows + row) * dim;
                float s0 = dotShared(context, ring, ringBase, wqkv, rowOff, vec, dim, lane);
                float s1 = dotShared(context, ring, ringBase, wqkv, rowOff + dim, vec, dim, lane);
                if (lane == 0) {
                    if (row < dim + kvDim) {
                        int idx = row < dim ? row : row - dim;
                        int freqIdx = pos * (headSize / 2) + (idx % headSize) / 2;
                        float fcr = freqCisReal.get(freqIdx);
                        float fci = freqCisImag.get(freqIdx);
                        float r0 = s0 * fcr - s1 * fci;
                        float r1 = s0 * fci + s1 * fcr;
                        if (row < dim) {
                            scratch.set(qOff + idx, r0);
                            scratch.set(qOff + idx + 1, r1);
                        } else {
                            keyCache.setHalf2(cacheOff + idx, Half2.fromFloats(r0, r1));
                        }
                    } else {
                        valueCache.setHalf2(cacheOff + row - dim - kvDim, Half2.fromFloats(s0, s1));
                    }
                }
            }
            context.gridBarrier();

            // 2. Attention partials, one (head, split) unit per block at a time.
            int seqLen = pos + 1;
            int chunk = (seqLen + splits - 1) / splits;
            for (int unit = block; unit < heads * splits; unit += blocks) {
                int h = unit / splits;
                int start = (unit % splits) * chunk;
                int end = Math.min(start + chunk, seqLen);
                int kvHeadOff = (h / kvMul) * headSize;
                float runningMax = Float.NEGATIVE_INFINITY;
                float runningSum = 0.0f;
                for (int k = 0; k < HEAD_VALUES_PER_LANE; k++) {
                    acc[k] = 0.0f;
                }
                for (int t = start + warp; t < end; t += WARPS) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            t,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHeadOff;
                    float partial = 0.0f;
                    for (int k = 0; k < HEAD_VALUES_PER_LANE; k++) {
                        int d = lane + 32 * k;
                        if (d < headSize) {
                            partial +=
                                    scratch.get(qOff + h * headSize + d)
                                            * keyCache.get(base + d).getFloat32();
                        }
                    }
                    float score = context.simdSum(partial) * invSqrtHead;
                    float newMax = Math.max(runningMax, score);
                    float correction = TornadoMath.exp(runningMax - newMax);
                    float weight = TornadoMath.exp(score - newMax);
                    runningSum = runningSum * correction + weight;
                    for (int k = 0; k < HEAD_VALUES_PER_LANE; k++) {
                        int d = lane + 32 * k;
                        if (d < headSize) {
                            acc[k] =
                                    acc[k] * correction
                                            + weight * valueCache.get(base + d).getFloat32();
                        }
                    }
                    runningMax = newMax;
                }
                if (lane == 0) {
                    mergeMax[warp] = runningMax;
                    mergeSum[warp] = runningSum;
                }
                for (int k = 0; k < HEAD_VALUES_PER_LANE; k++) {
                    mergeAcc[warp * MAX_HEAD_SIZE + lane + 32 * k] = acc[k];
                }
                context.localBarrier();
                if (tid < headSize) {
                    float m = Float.NEGATIVE_INFINITY;
                    for (int w = 0; w < WARPS; w++) {
                        m = Math.max(m, mergeMax[w]);
                    }
                    float sum = 0.0f;
                    float value = 0.0f;
                    if (m != Float.NEGATIVE_INFINITY) {
                        for (int w = 0; w < WARPS; w++) {
                            float f = TornadoMath.exp(mergeMax[w] - m);
                            sum += mergeSum[w] * f;
                            value += mergeAcc[w * MAX_HEAD_SIZE + tid] * f;
                        }
                    }
                    int partBase = attOff + unit * partStride;
                    if (tid == 0) {
                        scratch.set(partBase, m);
                        scratch.set(partBase + 1, sum);
                    }
                    scratch.set(partBase + 2 + tid, value);
                }
                context.localBarrier();
            }
            context.gridBarrier();

            // 3. Combine the splits into vec, then the output projection and the residual.
            for (int i = tid; i < dim; i += BLOCK_SIZE) {
                int h = i / headSize;
                int d = i % headSize;
                int headBase = attOff + h * splits * partStride;
                float m = Float.NEGATIVE_INFINITY;
                for (int s = 0; s < splits; s++) {
                    m = Math.max(m, scratch.get(headBase + s * partStride));
                }
                float sum = 0.0f;
                float value = 0.0f;
                for (int s = 0; s < splits; s++) {
                    float partMax = scratch.get(headBase + s * partStride);
                    if (partMax != Float.NEGATIVE_INFINITY) {
                        float f = TornadoMath.exp(partMax - m);
                        sum += scratch.get(headBase + s * partStride + 1) * f;
                        value += scratch.get(headBase + s * partStride + 2 + d) * f;
                    }
                }
                vec[i] = sum > 0.0f ? value / sum : 0.0f;
            }
            context.localBarrier();
            for (int row = globalWarp; row < dim; row += totalWarps) {
                float s = dotShared(context, ring, ringBase, wo, (l * dim + row) * dim, vec, dim, lane);
                if (lane == 0) {
                    x.set(row, x.get(row) + s);
                }
            }
            context.gridBarrier();

            // 4. RMS norm, gate and up projections, SwiGLU.
            normalize(context, x, norms, (layers + l) * dim, dim, eps, vec, reduce, tid, warp);
            for (int row = globalWarp; row < hiddenDim; row += totalWarps) {
                int rowOff = (l * hiddenDim + row) * dim;
                float gate = dotShared(context, ring, ringBase, w1, rowOff, vec, dim, lane);
                float up = dotShared(context, ring, ringBase, w3, rowOff, vec, dim, lane);
                if (lane == 0) {
                    scratch.set(hbOff + row, gate / (1.0f + TornadoMath.exp(-gate)) * up);
                }
            }
            context.gridBarrier();

            // 5. Down projection and the residual.
            for (int row = globalWarp; row < dim; row += totalWarps) {
                float s =
                        dotGlobal(context, ring, ringBase,
                                w2,
                                (l * dim + row) * hiddenDim,
                                scratch,
                                hbOff,
                                hiddenDim,
                                lane);
                if (lane == 0) {
                    x.set(row, x.get(row) + s);
                }
            }
            context.gridBarrier();
        }

        // Final norm and the vocabulary projection. No barrier after: nothing reads it in-kernel.
        normalize(context, x, norms, 2 * layers * dim, dim, eps, vec, reduce, tid, warp);
        for (int row = globalWarp; row < vocabulary; row += totalWarps) {
            float s = dotShared(context, ring, ringBase, wcls, row * dim, vec, dim, lane);
            if (lane == 0) {
                logits.set(row, s);
            }
        }
    }

    // @formatter:on

    /** {@code vec = x * rsqrt(mean(x^2) + eps) * norms[normOff..]}, by every block for itself. */
    private static void normalize(
            KernelContext context,
            FloatArray x,
            FloatArray norms,
            int normOff,
            int dim,
            float eps,
            float[] vec,
            float[] reduce,
            int tid,
            int warp) {
        float partial = 0.0f;
        for (int i = tid; i < dim; i += BLOCK_SIZE) {
            float v = x.get(i);
            partial += v * v;
        }
        float warpSum = context.simdSum(partial);
        if (tid % 32 == 0) {
            reduce[warp] = warpSum;
        }
        context.localBarrier();
        float total = 0.0f;
        for (int w = 0; w < WARPS; w++) {
            total += reduce[w];
        }
        float scale = 1.0f / TornadoMath.sqrt(total / dim + eps);
        for (int i = tid; i < dim; i += BLOCK_SIZE) {
            vec[i] = x.get(i) * scale * norms.get(normOff + i);
        }
        context.localBarrier();
    }

    /**
     * One warp's dot product of an F16 row with the shared vector; every lane gets the sum.
     *
     * <p>The row streams through the warp's ring in shared memory with 16-byte {@code cp.async}
     * copies, {@link #STAGES} deep: each lane copies, and later reads back, only its own eight
     * halves per chunk, so no lane waits on another and a single {@code cp.async.wait_group}
     * orders each read after its copy. {@code n} must be a multiple of {@link #CHUNK}.
     */
    private static float dotShared(
            KernelContext context,
            int[] ring,
            int ringBase,
            HalfFloatArray w,
            int rowOff,
            float[] vec,
            int n,
            int lane) {
        int chunks = n / CHUNK;
        int laneOff = lane * 8;
        for (int s = 0; s < STAGES - 1; s++) {
            if (s < chunks) {
                context.asyncCopyToLocal16(ring, ringBase + s * RING_STAGE, w, rowOff + s * CHUNK + laneOff);
            }
            context.asyncCopyCommit();
        }
        float sum = 0.0f;
        for (int c = 0; c < chunks; c++) {
            int next = c + STAGES - 1;
            if (next < chunks) {
                context.asyncCopyToLocal16(ring, ringBase + (next % STAGES) * RING_STAGE, w, rowOff + next * CHUNK + laneOff);
            }
            context.asyncCopyCommit();
            context.asyncCopyWaitGroup(STAGES - 1);
            int slot = ringBase + (c % STAGES) * RING_STAGE;
            int j = c * CHUNK + laneOff;
            for (int k = 0; k < 4; k++) {
                int bits = ring[slot + k];
                sum += Float.float16ToFloat((short) (bits & 0xFFFF)) * vec[j + 2 * k]
                        + Float.float16ToFloat((short) (bits >>> 16)) * vec[j + 2 * k + 1];
            }
        }
        return context.simdSum(sum);
    }

    /** As {@link #dotShared}, against a vector in global memory. */
    private static float dotGlobal(
            KernelContext context,
            int[] ring,
            int ringBase,
            HalfFloatArray w,
            int rowOff,
            FloatArray v,
            int vOff,
            int n,
            int lane) {
        int chunks = n / CHUNK;
        int laneOff = lane * 8;
        for (int s = 0; s < STAGES - 1; s++) {
            if (s < chunks) {
                context.asyncCopyToLocal16(ring, ringBase + s * RING_STAGE, w, rowOff + s * CHUNK + laneOff);
            }
            context.asyncCopyCommit();
        }
        float sum = 0.0f;
        for (int c = 0; c < chunks; c++) {
            int next = c + STAGES - 1;
            if (next < chunks) {
                context.asyncCopyToLocal16(ring, ringBase + (next % STAGES) * RING_STAGE, w, rowOff + next * CHUNK + laneOff);
            }
            context.asyncCopyCommit();
            context.asyncCopyWaitGroup(STAGES - 1);
            int slot = ringBase + (c % STAGES) * RING_STAGE;
            int j = vOff + c * CHUNK + laneOff;
            for (int k = 0; k < 4; k++) {
                int bits = ring[slot + k];
                sum += Float.float16ToFloat((short) (bits & 0xFFFF)) * v.get(j + 2 * k)
                        + Float.float16ToFloat((short) (bits >>> 16)) * v.get(j + 2 * k + 1);
            }
        }
        return context.simdSum(sum);
    }
}
