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

    /**
     * Floats of the shared vector: the normalised residual, the attention output, or the FFN hidden
     * vector when {@code hiddenDim} fits, so the down projection reads it from shared memory too.
     */
    public static final int VEC_SIZE = 8192;

    /** Halves of a row prefetched ahead of a phase: 4 KB, 32 lines of 128 bytes. */
    private static final int PREFETCH_HALVES = 2048;

    /** The widest head: four values per lane. */
    public static final int MAX_HEAD_SIZE = 128;

    private static final int WARPS = BLOCK_SIZE / 32;
    private static final int HEAD_VALUES_PER_LANE = MAX_HEAD_SIZE / 32;

    /** The widest residual stream the fused down projection supports. */
    public static final int FUSED_MAX_DIM = 2048;

    private static final int FUSED_VALUES_PER_LANE = FUSED_MAX_DIM / 32;

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

    /** 1 to pick the greedy token in the kernel ({@code sampledToken}), 0 to only write logits. */
    public static final int META_DEVICE_SAMPLE = 11;
    /** Experimental: 1 when w2 is transposed (rows padded to FUSED_MAX_DIM) and fused into gate/up. */
    public static final int META_FUSED_DOWN = 12;
    /** Experimental: 1 for the single-pass RMS norm. */
    public static final int META_NORM_ONE_PASS = 13;

    public static final int META_SIZE = 14;

    private LlamaMegakernel() {}

    /**
     * Floats of scratch the kernel needs: Q, the FFN hidden vector, the attention partials, and one
     * (logit, index) pair per warp for the in-kernel argmax.
     */
    public static int scratchSize(
            int dim, int hiddenDim, int heads, int headSize, int splits, int blocks) {
        return dim + hiddenDim + heads * splits * (headSize + 2) + 2 * blocks * WARPS + blocks * dim;
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
            FloatArray logits,
            IntArray sampledToken) {
        int tid = context.localIdx;
        int lane = tid % 32;
        int warp = tid / 32;
        int blocks = context.globalGroupSizeX / BLOCK_SIZE;
        int block = context.groupIdx;
        int globalWarp = block * WARPS + warp;
        int totalWarps = blocks * WARPS;

        float[] vec = context.allocateFloatLocalArray(VEC_SIZE);
        float[] reduce = context.allocateFloatLocalArray(WARPS);
        float[] mergeMax = context.allocateFloatLocalArray(WARPS);
        float[] mergeSum = context.allocateFloatLocalArray(WARPS);
        float[] mergeAcc = context.allocateFloatLocalArray(WARPS * MAX_HEAD_SIZE);
        float[] acc = new float[HEAD_VALUES_PER_LANE];
        float[] down = new float[FUSED_VALUES_PER_LANE];
        int fusedDown = meta.get(META_FUSED_DOWN);
        int onePass = meta.get(META_NORM_ONE_PASS);

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
        if (globalWarp < qkvRows / 2) {
            prefetchRow(context, wqkv, 2 * globalWarp * dim, 2 * dim, lane);
        }
        context.gridBarrier();

        for (int l = 0; l < layers; l++) {
            int layerOff = KvBlockAddress.layerOffset(l, kvDim, blockCfg);
            int cacheOff =
                    KvBlockAddress.offset(
                            blockTable, slot, pos, layerOff, kvDim, blockCfg, blockStride);

            // 1. RMS norm, QKV, RoPE, K/V into the cache.
            normalize(context, x, norms, l * dim, dim, eps, vec, reduce, tid, warp, onePass);
            for (int p = globalWarp; p < qkvRows / 2; p += totalWarps) {
                int row = 2 * p;
                int rowOff = (l * qkvRows + row) * dim;
                float s0 = dotShared(context, wqkv, rowOff, vec, dim, lane);
                float s1 = dotShared(context, wqkv, rowOff + dim, vec, dim, lane);
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
            if (globalWarp < dim) {
                prefetchRow(context, wo, (l * dim + globalWarp) * dim, dim, lane);
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
            if (globalWarp + totalWarps < dim) {
                prefetchRow(context, wo, (l * dim + globalWarp + totalWarps) * dim, dim, lane);
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
                float s = dotShared(context, wo, (l * dim + row) * dim, vec, dim, lane);
                if (lane == 0) {
                    x.set(row, x.get(row) + s);
                }
            }
            if (globalWarp < hiddenDim) {
                prefetchRow(context, w1, (l * hiddenDim + globalWarp) * dim, dim, lane);
                prefetchRow(context, w3, (l * hiddenDim + globalWarp) * dim, dim, lane);
            }
            context.gridBarrier();

            if (fusedDown != 0) {
                normalize(context, x, norms, (layers + l) * dim, dim, eps, vec, reduce, tid, warp, onePass);
                for (int k = 0; k < FUSED_VALUES_PER_LANE; k++) {
                    down[k] = 0.0f;
                }
                for (int row = globalWarp; row < hiddenDim; row += totalWarps) {
                    int rowOff = (l * hiddenDim + row) * dim;
                    float gate = dotShared(context, w1, rowOff, vec, dim, lane);
                    float up = dotShared(context, w3, rowOff, vec, dim, lane);
                    float h = gate / (1.0f + TornadoMath.exp(-gate)) * up;
                    int downOff = (l * hiddenDim + row) * FUSED_MAX_DIM + 2 * lane;
                    for (int k = 0; k < FUSED_VALUES_PER_LANE / 2; k++) {
                        Half2 p = w2.getHalf2(downOff + k * 64);
                        down[2 * k] += h * Half2.lowFloat(p);
                        down[2 * k + 1] += h * Half2.highFloat(p);
                    }
                }
                context.localBarrier();
                for (int w = 0; w < WARPS; w++) {
                    if (warp == w) {
                        for (int k = 0; k < FUSED_VALUES_PER_LANE / 2; k++) {
                            int j = k * 64 + 2 * lane;
                            if (j < dim) {
                                if (w == 0) {
                                    vec[j] = down[2 * k];
                                    vec[j + 1] = down[2 * k + 1];
                                } else {
                                    vec[j] += down[2 * k];
                                    vec[j + 1] += down[2 * k + 1];
                                }
                            }
                        }
                    }
                    context.localBarrier();
                }
                int partOff = attOff + heads * splits * partStride + 2 * totalWarps;
                for (int i = tid; i < dim; i += BLOCK_SIZE) {
                    scratch.set(partOff + block * dim + i, vec[i]);
                }
                context.gridBarrier();
                for (int i = context.globalIdx; i < dim; i += context.globalGroupSizeX) {
                    float sum = x.get(i);
                    for (int b = 0; b < blocks; b++) {
                        sum += scratch.get(partOff + b * dim + i);
                    }
                    x.set(i, sum);
                }
            } else {
                // 4. RMS norm, gate and up projections, SwiGLU.
                normalize(context, x, norms, (layers + l) * dim, dim, eps, vec, reduce, tid, warp, onePass);
                for (int row = globalWarp; row < hiddenDim; row += totalWarps) {
                    int rowOff = (l * hiddenDim + row) * dim;
                    float gate = dotShared(context, w1, rowOff, vec, dim, lane);
                    float up = dotShared(context, w3, rowOff, vec, dim, lane);
                    if (lane == 0) {
                        scratch.set(hbOff + row, gate / (1.0f + TornadoMath.exp(-gate)) * up);
                    }
                }
                if (globalWarp < dim) {
                    prefetchRow(context, w2, (l * dim + globalWarp) * hiddenDim, hiddenDim, lane);
                }
                context.gridBarrier();

                // 5. Down projection and the residual, against the hidden vector in shared memory
                // when it fits.
                if (hiddenDim <= VEC_SIZE) {
                    for (int i = tid; i < hiddenDim; i += BLOCK_SIZE) {
                        vec[i] = scratch.get(hbOff + i);
                    }
                    context.localBarrier();
                    for (int row = globalWarp; row < dim; row += totalWarps) {
                        float s =
                                dotShared(
                                        context, w2, (l * dim + row) * hiddenDim, vec, hiddenDim, lane);
                        if (lane == 0) {
                            x.set(row, x.get(row) + s);
                        }
                    }
                } else {
                    for (int row = globalWarp; row < dim; row += totalWarps) {
                        float s =
                                dotGlobal(
                                        context,
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
                }
            }
            if (l + 1 < layers) {
                if (globalWarp < qkvRows / 2) {
                    prefetchRow(
                            context,
                            wqkv,
                            ((l + 1) * qkvRows + 2 * globalWarp) * dim,
                            2 * dim,
                            lane);
                }
            } else if (globalWarp < vocabulary) {
                prefetchRow(context, wcls, globalWarp * dim, dim, lane);
            }
            context.gridBarrier();
        }

        // Final norm and the vocabulary projection. Each warp keeps the best of its rows, the
        // first one on a tie, as the host's greedy argmax does.
        normalize(context, x, norms, 2 * layers * dim, dim, eps, vec, reduce, tid, warp, onePass);
        float best = Float.NEGATIVE_INFINITY;
        int bestRow = 0;
        for (int row = globalWarp; row < vocabulary; row += totalWarps) {
            float s = dotShared(context, wcls, row * dim, vec, dim, lane);
            if (lane == 0) {
                logits.set(row, s);
                if (s > best) {
                    best = s;
                    bestRow = row;
                }
            }
        }

        // Greedy sampling on the device: the per-warp bests, one barrier, then block 0 reduces
        // them, so the host reads one token instead of the vocabulary-sized logits.
        if (meta.get(META_DEVICE_SAMPLE) != 0) {
            int argOff = attOff + heads * splits * partStride;
            if (lane == 0) {
                scratch.set(argOff + 2 * globalWarp, best);
                scratch.set(argOff + 2 * globalWarp + 1, bestRow);
            }
            context.gridBarrier();
            if (block == 0) {
                float value = Float.NEGATIVE_INFINITY;
                float index = vocabulary;
                for (int w = tid; w < totalWarps; w += BLOCK_SIZE) {
                    float v = scratch.get(argOff + 2 * w);
                    float r = scratch.get(argOff + 2 * w + 1);
                    if (v > value || (v == value && r < index)) {
                        value = v;
                        index = r;
                    }
                }
                vec[tid] = value;
                vec[BLOCK_SIZE + tid] = index;
                context.localBarrier();
                if (tid == 0) {
                    for (int t = 1; t < BLOCK_SIZE; t++) {
                        float v = vec[t];
                        float r = vec[BLOCK_SIZE + t];
                        if (v > value || (v == value && r < index)) {
                            value = v;
                            index = r;
                        }
                    }
                    sampledToken.set(0, (int) index);
                }
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
            int warp,
            int onePass) {
        float[] held = new float[MAX_DIM / BLOCK_SIZE];
        float partial = 0.0f;
        if (onePass != 0) {
            for (int k = 0; k < MAX_DIM / BLOCK_SIZE; k++) {
                int i = tid + k * BLOCK_SIZE;
                if (i < dim) {
                    float v = x.get(i);
                    held[k] = v;
                    partial += v * v;
                }
            }
        } else {
            for (int i = tid; i < dim; i += BLOCK_SIZE) {
                float v = x.get(i);
                partial += v * v;
            }
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
        if (onePass != 0) {
            for (int k = 0; k < MAX_DIM / BLOCK_SIZE; k++) {
                int i = tid + k * BLOCK_SIZE;
                if (i < dim) {
                    vec[i] = held[k] * scale * norms.get(normOff + i);
                }
            }
        } else {
            for (int i = tid; i < dim; i += BLOCK_SIZE) {
                vec[i] = x.get(i) * scale * norms.get(normOff + i);
            }
        }
        context.localBarrier();
    }

    /**
     * Prefetches into L2 the first 4 KB of a row this warp will stream next, one 128-byte line per
     * lane. Called just before a grid barrier: the next phase's first loads then hit L2 instead of
     * waiting on DRAM while every warp starts at once.
     */
    private static void prefetchRow(
            KernelContext context, HalfFloatArray w, int rowOff, int n, int lane) {
        int limit = Math.min(n, PREFETCH_HALVES);
        for (int k = lane * 64; k < limit; k += 32 * 64) {
            context.prefetchToL2(w, rowOff + k);
        }
    }

    /**
     * One warp's dot product of an F16 row with the shared vector; every lane gets the sum.
     *
     * <p>Instruction-bound, not bandwidth-bound, at one warp per row and eight warps per SM: each
     * half takes a conversion and an FMA, and nothing else should. Every product goes into its own
     * accumulator, {@code s = s + a * b}, which contracts to one FFMA; {@code s += a * b + c * d}
     * would cost an FMUL, an FFMA and an FADD per pair.
     */
    private static float dotShared(
            KernelContext context, HalfFloatArray w, int rowOff, float[] vec, int n, int lane) {
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        float s4 = 0.0f;
        float s5 = 0.0f;
        float s6 = 0.0f;
        float s7 = 0.0f;
        int j = 2 * lane;
        for (; j < n - 192; j += 256) {
            Half2 p0 = w.getHalf2(rowOff + j);
            Half2 p1 = w.getHalf2(rowOff + j + 64);
            Half2 p2 = w.getHalf2(rowOff + j + 128);
            Half2 p3 = w.getHalf2(rowOff + j + 192);
            s0 += Half2.lowFloat(p0) * vec[j];
            s1 += Half2.highFloat(p0) * vec[j + 1];
            s2 += Half2.lowFloat(p1) * vec[j + 64];
            s3 += Half2.highFloat(p1) * vec[j + 65];
            s4 += Half2.lowFloat(p2) * vec[j + 128];
            s5 += Half2.highFloat(p2) * vec[j + 129];
            s6 += Half2.lowFloat(p3) * vec[j + 192];
            s7 += Half2.highFloat(p3) * vec[j + 193];
        }
        for (; j < n; j += 64) {
            Half2 p = w.getHalf2(rowOff + j);
            s0 += Half2.lowFloat(p) * vec[j];
            s1 += Half2.highFloat(p) * vec[j + 1];
        }
        return context.simdSum(((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)));
    }

    /** As {@link #dotShared}, against a vector in global memory. */
    private static float dotGlobal(
            KernelContext context,
            HalfFloatArray w,
            int rowOff,
            FloatArray v,
            int vOff,
            int n,
            int lane) {
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        int j = 2 * lane;
        for (; j < n - 64; j += 128) {
            Half2 p0 = w.getHalf2(rowOff + j);
            Half2 p1 = w.getHalf2(rowOff + j + 64);
            s0 += Half2.lowFloat(p0) * v.get(vOff + j);
            s1 += Half2.highFloat(p0) * v.get(vOff + j + 1);
            s2 += Half2.lowFloat(p1) * v.get(vOff + j + 64);
            s3 += Half2.highFloat(p1) * v.get(vOff + j + 65);
        }
        for (; j < n; j += 64) {
            Half2 p = w.getHalf2(rowOff + j);
            s0 += Half2.lowFloat(p) * v.get(vOff + j);
            s1 += Half2.highFloat(p) * v.get(vOff + j + 1);
        }
        return context.simdSum((s0 + s1) + (s2 + s3));
    }
}
