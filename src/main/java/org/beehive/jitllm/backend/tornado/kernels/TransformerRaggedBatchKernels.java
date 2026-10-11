package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Kernels for ragged token batches: each row is one token of some sequence, with its own position
 * and its own KV slot, so prompt chunks and decode tokens of several sequences share one pass.
 *
 * <p>The two KV kernels are the paged batch-decode kernels of {@link
 * TransformerBatchPrefillKernels} with one change each: the block-table row comes from {@code
 * rowSlots[row]} instead of being the row index, and a row whose slot is negative is padding and
 * does nothing. The arithmetic is otherwise identical, which is what makes a token computed inside
 * a prompt chunk bit-for-bit the token the decode path computes at the same position — the batch
 * invariance the engine relies on. Reviewing means diffing each against its twin.
 *
 * <p>Key/value pool offsets are 64-bit, so the pool may hold more than {@code Integer.MAX_VALUE}
 * values (TornadoVM's {@code long} array accessors).
 *
 * <p>Causality within a chunk needs nothing here: the RoPE task writes every row's key and value
 * before the attention task runs, and each row attends over positions up to its own.
 *
 * <p>Attention also comes split-K ({@link #raggedSplitKvAttentionPartial} and {@link
 * #raggedSplitKvAttentionCombine}): a long context is read by several workgroups at once and their
 * results merged. It sums in a different order from the single-workgroup kernel, so it is not
 * bit-for-bit that kernel, but how a row is split depends only on the row, so it stays batch
 * invariant.
 */
public class TransformerRaggedBatchKernels {

    public TransformerRaggedBatchKernels() {}

    /**
     * Twin of {@link TransformerBatchPrefillKernels#batchedDecodePagedRopeWithKVCachePackedKVFP16},
     * addressed through {@code rowSlots}.
     */
    public static void raggedPagedRopeWithKVCachePackedKVFP16(
            KernelContext context,
            IntArray rowPositions,
            IntArray rowSlots,
            IntArray blockTable,
            FloatArray qkvBatch,
            HalfFloatArray keyPool,
            HalfFloatArray valuePool,
            float ropeTheta,
            int kvDim,
            int headSize,
            int layerIndex,
            int numLayers,
            int blockCfg,
            int dim) {
        int blockSize = blockCfg & 0xFFFF;
        int maxBlocksPerSlot = blockCfg >>> 16;
        int globalIdx = context.globalIdx;
        int halfDim = dim / 2;
        int batchIdx = globalIdx / halfDim;
        int pairIdx = globalIdx % halfDim;
        int i = pairIdx * 2;
        int qkvStride = dim + 2 * kvDim;

        int slot = rowSlots.get(batchIdx);
        if (slot < 0) {
            return; // padding row: no token, and no KV of its own to write
        }
        int pos = rowPositions.get(batchIdx);
        int qOffset = batchIdx * qkvStride;
        int kOffset = batchIdx * qkvStride + dim;
        int vOffset = batchIdx * qkvStride + dim + kvDim;

        if (i + 1 < dim) {
            int head_dim = i % headSize;
            float freq = 1.0f / TornadoMath.pow(ropeTheta, head_dim / (float) headSize);
            float val = pos * freq;
            float fcr = TornadoMath.cos(val);
            float fci = TornadoMath.sin(val);

            float v0q = qkvBatch.get(qOffset + i);
            float v1q = qkvBatch.get(qOffset + i + 1);
            qkvBatch.set(qOffset + i, v0q * fcr - v1q * fci);
            qkvBatch.set(qOffset + i + 1, v0q * fci + v1q * fcr);

            if (i + 1 < kvDim) {
                float v0k = qkvBatch.get(kOffset + i);
                float v1k = qkvBatch.get(kOffset + i + 1);
                float rotK0 = v0k * fcr - v1k * fci;
                float rotK1 = v0k * fci + v1k * fcr;

                int physBlock = blockTable.get(slot * maxBlocksPerSlot + pos / blockSize);
                int slotInBlock = pos % blockSize;
                long cacheOff =
                        (long) physBlock * (numLayers * blockSize * kvDim)
                                + layerIndex * (blockSize * kvDim)
                                + slotInBlock * kvDim;
                keyPool.set(cacheOff + i, new HalfFloat(rotK0));
                keyPool.set(cacheOff + i + 1, new HalfFloat(rotK1));
                valuePool.set(cacheOff + i, new HalfFloat(qkvBatch.get(vOffset + i)));
                valuePool.set(cacheOff + i + 1, new HalfFloat(qkvBatch.get(vOffset + i + 1)));
            }
        }
    }

    /**
     * Twin of {@link TransformerBatchPrefillKernels#batchedDecodePagedAttentionFP16OutKVFP16},
     * addressed through {@code rowSlots}.
     */
    public static void raggedPagedAttentionFP16OutKVFP16(
            KernelContext context,
            IntArray rowPositions,
            IntArray rowSlots,
            IntArray blockTable,
            FloatArray qkvBatch,
            HalfFloatArray keyPool,
            HalfFloatArray valuePool,
            HalfFloatArray attnOutFP16,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            int numLayers,
            int blockCfg) {
        int blockSize = blockCfg & 0xFFFF;
        int maxBlocksPerSlot = blockCfg >>> 16;
        int dim = nHeads * headSize;
        int tid = context.localIdx;
        int groupId = context.groupIdx;
        int localSz = context.localGroupSizeX;

        int batchIdx = groupId / nHeads;
        int h = groupId % nHeads;
        int slot = rowSlots.get(batchIdx);
        // Padding row. batchIdx comes from groupIdx, so the whole workgroup returns together and
        // the localBarrier()s below are still reached uniformly by the groups that stay.
        if (slot < 0) {
            return;
        }
        int pos = rowPositions.get(batchIdx);
        int layerOff = layerIndex * (blockSize * kvDim);
        int kvHeadIdx = h / kvMul;
        int BLOCK_C = 16;
        int qkvStride = dim + 2 * kvDim;
        int blockStride = numLayers * blockSize * kvDim;

        float[] qShared = context.allocateFloatLocalArray(headSize);
        float[] kTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] vTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] sTile = context.allocateFloatLocalArray(BLOCK_C);

        int qOffset = batchIdx * qkvStride + h * headSize;
        for (int i = tid; i < headSize; i += localSz) {
            qShared[i] = qkvBatch.get(qOffset + i);
        }
        context.localBarrier();

        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        int d1 = tid + localSz;

        for (int tileC = 0; tileC <= pos; tileC += BLOCK_C) {
            int tileEnd = Math.min(tileC + BLOCK_C - 1, pos);
            int tileLen = tileEnd - tileC + 1;

            for (int idx = tid; idx < tileLen * headSize; idx += localSz) {
                int tInTile = idx / headSize;
                int d = idx % headSize;
                int t = tileC + tInTile;
                int physBlock = blockTable.get(slot * maxBlocksPerSlot + t / blockSize);
                long kvOff =
                        (long) physBlock * blockStride
                                + layerOff
                                + (t % blockSize) * kvDim
                                + kvHeadIdx * headSize
                                + d;
                kTile[tInTile * headSize + d] = keyPool.get(kvOff).getFloat32();
                vTile[tInTile * headSize + d] = valuePool.get(kvOff).getFloat32();
            }
            context.localBarrier();

            for (int t = tileC + tid; t <= tileEnd; t += localSz) {
                int tInTile = t - tileC;
                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += qShared[d] * kTile[tInTile * headSize + d];
                }
                sTile[tInTile] = score / TornadoMath.sqrt(headSize);
            }
            context.localBarrier();

            float tileMax = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < tileLen; t++) {
                if (sTile[t] > tileMax) {
                    tileMax = sTile[t];
                }
            }

            float newMax = Math.max(maxScore, tileMax);
            if (maxScore != Float.NEGATIVE_INFINITY && newMax != maxScore) {
                float corr = TornadoMath.exp(maxScore - newMax);
                sumExp *= corr;
                acc0 *= corr;
                acc1 *= corr;
            }
            maxScore = newMax;

            for (int t = 0; t < tileLen; t++) {
                float p = TornadoMath.exp(sTile[t] - maxScore);
                sumExp += p;
                acc0 += p * vTile[t * headSize + tid];
                if (d1 < headSize) {
                    acc1 += p * vTile[t * headSize + d1];
                }
            }
            context.localBarrier();
        }

        float norm = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        int outOffset = batchIdx * dim + h * headSize;
        attnOutFP16.set(outOffset + tid, new HalfFloat(acc0 * norm));
        if (d1 < headSize) {
            attnOutFP16.set(outOffset + d1, new HalfFloat(acc1 * norm));
        }
    }

    /**
     * Split-K paged attention, first pass: one workgroup per (row, head, split), one thread per
     * head dimension.
     *
     * <p>A row's positions {@code 0..pos} are cut into {@code ceil((pos + 1) / chunk)} contiguous
     * splits of {@code chunk = max(minSplit, ceil((pos + 1) / maxSplits))} positions, so a long
     * context is read by up to {@code maxSplits} workgroups at once while a short one stays in one
     * and does not pay for the rest. Each workgroup runs the tiled online-softmax loop of {@link
     * #raggedPagedAttentionFP16OutKVFP16} over its split alone and writes the unnormalised
     * accumulator, the max score and the sum to {@code partials}; a split past the row's end writes
     * an empty marker. {@link #raggedSplitKvAttentionCombine} merges them.
     *
     * <p>How a row is split depends only on its own position, and the combine visits splits in a
     * fixed order, so the result for a row does not depend on what else is in the step.
     *
     * <p>Layout of {@code partials}, per {@code (row * nHeads + head) * maxSplits + split}: {@code
     * headSize} accumulator values, then the max, then the sum.
     *
     * @param splitCfg {@code maxSplits | (minSplit << 16)}, packed to stay within the argument
     *     limit
     */
    public static void raggedSplitKvAttentionPartial(
            KernelContext context,
            IntArray rowPositions,
            IntArray rowSlots,
            IntArray blockTable,
            FloatArray qkvBatch,
            HalfFloatArray keyPool,
            HalfFloatArray valuePool,
            FloatArray partials,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            int numLayers,
            int blockCfg,
            int splitCfg) {
        int maxSplits = splitCfg & 0xFFFF;
        int minSplit = splitCfg >>> 16;
        int tid = context.localIdx;
        int g = context.groupIdx;
        int localSz = context.localGroupSizeX;
        int row = g / (nHeads * maxSplits);
        int rem = g % (nHeads * maxSplits);
        int h = rem / maxSplits;
        int s = rem % maxSplits;

        int slot = rowSlots.get(row);
        // Padding row: the whole workgroup returns together, so the barriers below stay uniform.
        if (slot < 0) {
            return;
        }
        int seqLen = rowPositions.get(row) + 1;
        int chunk = Math.max(minSplit, (seqLen + maxSplits - 1) / maxSplits);
        int startPos = s * chunk;
        int endPos = Math.min(startPos + chunk, seqLen); // exclusive
        int outBase = ((row * nHeads + h) * maxSplits + s) * (headSize + 2);
        if (startPos >= seqLen) {
            // Past the row's end: an empty split, which the combine skips.
            if (tid == 0) {
                partials.set(outBase + headSize, Float.NEGATIVE_INFINITY);
                partials.set(outBase + headSize + 1, 0.0f);
            }
            return;
        }

        int blockSize = blockCfg & 0xFFFF;
        int maxBlocksPerSlot = blockCfg >>> 16;
        int dim = nHeads * headSize;
        int layerOff = layerIndex * (blockSize * kvDim);
        int kvHeadIdx = h / kvMul;
        int BLOCK_C = 16;
        int qkvStride = dim + 2 * kvDim;
        int blockStride = numLayers * blockSize * kvDim;

        float[] qShared = context.allocateFloatLocalArray(headSize);
        float[] kTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] vTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] sTile = context.allocateFloatLocalArray(BLOCK_C);

        int qOffset = row * qkvStride + h * headSize;
        for (int i = tid; i < headSize; i += localSz) {
            qShared[i] = qkvBatch.get(qOffset + i);
        }
        context.localBarrier();

        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        int d1 = tid + localSz;

        for (int tileC = startPos; tileC < endPos; tileC += BLOCK_C) {
            int tileEnd = Math.min(tileC + BLOCK_C, endPos) - 1;
            int tileLen = tileEnd - tileC + 1;

            for (int idx = tid; idx < tileLen * headSize; idx += localSz) {
                int tInTile = idx / headSize;
                int d = idx % headSize;
                int t = tileC + tInTile;
                int physBlock = blockTable.get(slot * maxBlocksPerSlot + t / blockSize);
                long kvOff =
                        (long) physBlock * blockStride
                                + layerOff
                                + (t % blockSize) * kvDim
                                + kvHeadIdx * headSize
                                + d;
                kTile[tInTile * headSize + d] = keyPool.get(kvOff).getFloat32();
                vTile[tInTile * headSize + d] = valuePool.get(kvOff).getFloat32();
            }
            context.localBarrier();

            for (int t = tileC + tid; t <= tileEnd; t += localSz) {
                int tInTile = t - tileC;
                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += qShared[d] * kTile[tInTile * headSize + d];
                }
                sTile[tInTile] = score / TornadoMath.sqrt(headSize);
            }
            context.localBarrier();

            float tileMax = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < tileLen; t++) {
                if (sTile[t] > tileMax) {
                    tileMax = sTile[t];
                }
            }

            float newMax = Math.max(maxScore, tileMax);
            if (maxScore != Float.NEGATIVE_INFINITY && newMax != maxScore) {
                float corr = TornadoMath.exp(maxScore - newMax);
                sumExp *= corr;
                acc0 *= corr;
                acc1 *= corr;
            }
            maxScore = newMax;

            for (int t = 0; t < tileLen; t++) {
                float p = TornadoMath.exp(sTile[t] - maxScore);
                sumExp += p;
                acc0 += p * vTile[t * headSize + tid];
                if (d1 < headSize) {
                    acc1 += p * vTile[t * headSize + d1];
                }
            }
            context.localBarrier();
        }

        partials.set(outBase + tid, acc0);
        if (d1 < headSize) {
            partials.set(outBase + d1, acc1);
        }
        if (tid == 0) {
            partials.set(outBase + headSize, maxScore);
            partials.set(outBase + headSize + 1, sumExp);
        }
    }

    /**
     * {@link #raggedSplitKvAttentionPartial} with the query heads that share a KV head computed
     * together: one workgroup per (row, KV head, split), {@code kvMul * headSize} threads, thread
     * {@code tid} owning query head {@code tid / headSize} of the group and dimension {@code tid %
     * headSize}.
     *
     * <p>Each key/value tile is loaded once for all {@code kvMul} query heads instead of once per
     * head. Every head's dot products, online softmax and accumulation run in the same order as in
     * the per-head kernel, so the partial results — and the output — are the same bits; only the
     * memory traffic shrinks, by {@code kvMul}.
     *
     * @param splitCfg {@code maxSplits | (minSplit << 16)}
     */
    public static void raggedSplitKvAttentionPartialGqa(
            KernelContext context,
            IntArray rowPositions,
            IntArray rowSlots,
            IntArray blockTable,
            FloatArray qkvBatch,
            HalfFloatArray keyPool,
            HalfFloatArray valuePool,
            FloatArray partials,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            int numLayers,
            int blockCfg,
            int splitCfg) {
        int maxSplits = splitCfg & 0xFFFF;
        int minSplit = splitCfg >>> 16;
        int nKvHeads = nHeads / kvMul;
        int tid = context.localIdx;
        int g = context.groupIdx;
        int localSz = context.localGroupSizeX;
        int row = g / (nKvHeads * maxSplits);
        int rem = g % (nKvHeads * maxSplits);
        int kvHeadIdx = rem / maxSplits;
        int s = rem % maxSplits;
        int j = tid / headSize; // query head within the group
        int d = tid % headSize; // the dimension this thread owns
        int h = kvHeadIdx * kvMul + j;

        int slot = rowSlots.get(row);
        // Padding row: the whole workgroup returns together, so the barriers below stay uniform.
        if (slot < 0) {
            return;
        }
        int seqLen = rowPositions.get(row) + 1;
        int chunk = Math.max(minSplit, (seqLen + maxSplits - 1) / maxSplits);
        int startPos = s * chunk;
        int endPos = Math.min(startPos + chunk, seqLen); // exclusive
        int outBase = ((row * nHeads + h) * maxSplits + s) * (headSize + 2);
        if (startPos >= seqLen) {
            // Past the row's end: an empty split, which the combine skips.
            if (d == 0) {
                partials.set(outBase + headSize, Float.NEGATIVE_INFINITY);
                partials.set(outBase + headSize + 1, 0.0f);
            }
            return;
        }

        int blockSize = blockCfg & 0xFFFF;
        int maxBlocksPerSlot = blockCfg >>> 16;
        int dim = nHeads * headSize;
        int layerOff = layerIndex * (blockSize * kvDim);
        int BLOCK_C = 16;
        int qkvStride = dim + 2 * kvDim;
        int blockStride = numLayers * blockSize * kvDim;

        // Up to 1024 threads: kvMul * headSize query values, one tile, kvMul score rows.
        float[] qShared = context.allocateFloatLocalArray(1024);
        float[] kTile = context.allocateFloatLocalArray(16 * 128);
        float[] vTile = context.allocateFloatLocalArray(16 * 128);
        float[] sTile = context.allocateFloatLocalArray(16 * 8);

        qShared[tid] = qkvBatch.get(row * qkvStride + h * headSize + d);
        context.localBarrier();

        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;
        float acc = 0.0f;

        for (int tileC = startPos; tileC < endPos; tileC += BLOCK_C) {
            int tileEnd = Math.min(tileC + BLOCK_C, endPos) - 1;
            int tileLen = tileEnd - tileC + 1;

            for (int idx = tid; idx < tileLen * headSize; idx += localSz) {
                int tInTile = idx / headSize;
                int dd = idx % headSize;
                int t = tileC + tInTile;
                int physBlock = blockTable.get(slot * maxBlocksPerSlot + t / blockSize);
                long kvOff =
                        (long) physBlock * blockStride
                                + layerOff
                                + (t % blockSize) * kvDim
                                + kvHeadIdx * headSize
                                + dd;
                kTile[tInTile * headSize + dd] = keyPool.get(kvOff).getFloat32();
                vTile[tInTile * headSize + dd] = valuePool.get(kvOff).getFloat32();
            }
            context.localBarrier();

            // One thread per (query head, key position) pair.
            for (int pair = tid; pair < kvMul * tileLen; pair += localSz) {
                int jj = pair / tileLen;
                int tInTile = pair % tileLen;
                float score = 0.0f;
                for (int dd = 0; dd < headSize; dd++) {
                    score += qShared[jj * headSize + dd] * kTile[tInTile * headSize + dd];
                }
                sTile[jj * BLOCK_C + tInTile] = score / TornadoMath.sqrt(headSize);
            }
            context.localBarrier();

            float tileMax = Float.NEGATIVE_INFINITY;
            for (int t = 0; t < tileLen; t++) {
                if (sTile[j * BLOCK_C + t] > tileMax) {
                    tileMax = sTile[j * BLOCK_C + t];
                }
            }

            float newMax = Math.max(maxScore, tileMax);
            if (maxScore != Float.NEGATIVE_INFINITY && newMax != maxScore) {
                float corr = TornadoMath.exp(maxScore - newMax);
                sumExp *= corr;
                acc *= corr;
            }
            maxScore = newMax;

            for (int t = 0; t < tileLen; t++) {
                float p = TornadoMath.exp(sTile[j * BLOCK_C + t] - maxScore);
                sumExp += p;
                acc += p * vTile[t * headSize + d];
            }
            context.localBarrier();
        }

        partials.set(outBase + d, acc);
        if (d == 0) {
            partials.set(outBase + headSize, maxScore);
            partials.set(outBase + headSize + 1, sumExp);
        }
    }

    /** Rows of one MMA tile in {@link #raggedFlashAttentionMMA}: query rows times their heads. */
    public static final int FLASH_TILE_ROWS = 16;

    /**
     * Split-K paged attention on tensor cores, first pass: one warp per (query tile, KV head,
     * split). A query tile is up to {@code 16 / kvMul} consecutive rows of one sequence; with the
     * {@code kvMul} query heads that share the KV head they make the 16 rows of an m16n8k16 tile.
     * Every row — a decode token or a prompt token — goes through this kernel.
     *
     * <p>Per 16-key tile: K and V are staged in shared memory in the B-operand layout, {@code S = Q
     * K^T} takes eight MMAs, each row's scores are masked at its own causal end and folded into its
     * online softmax, and {@code P V} takes eight more MMAs whose result is added to the row's
     * running output. The output and the split's max and sum go to {@code partials}, as for the
     * scalar kernels, and {@link #raggedSplitKvAttentionCombine} merges the splits.
     *
     * <p><b>Batch invariance.</b> A row's scores, softmax and output come from its own query, its
     * own mask and the key tiles aligned from its split's start — never from the other rows of the
     * tile, which an MMA keeps apart. So a token computes the same bits alone or in any company,
     * and in a prompt chunk or one at a time.
     *
     * <p>Fragments are fresh in every key tile and folded into private arrays: an accumulator
     * fragment cannot be written through, and one carried across loop iterations does not compile.
     *
     * <p>Requires {@code headSize == 64} and {@code 16 % kvMul == 0}.
     *
     * @param queryTiles per tile, the first row and the row count; a count of 0 is padding
     * @param splitCfg {@code maxSplits | (minSplit << 16)}
     */
    public static void raggedFlashAttentionMMA(
            KernelContext context,
            IntArray rowPositions,
            IntArray rowSlots,
            IntArray queryTiles,
            IntArray blockTable,
            FloatArray qkvBatch,
            HalfFloatArray keyPool,
            HalfFloatArray valuePool,
            FloatArray partials,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            int numLayers,
            int blockCfg,
            int splitCfg) {
        int maxSplits = splitCfg & 0xFFFF;
        int minSplit = splitCfg >>> 16;
        int nKvHeads = nHeads / kvMul;
        int lane = context.localIdx;
        int g = context.groupIdx;
        int tile = g / (nKvHeads * maxSplits);
        int rem = g % (nKvHeads * maxSplits);
        int kvHeadIdx = rem / maxSplits;
        int s = rem % maxSplits;

        int firstRow = queryTiles.get(2 * tile);
        int tileRows = queryTiles.get(2 * tile + 1);
        // Padding tile: the whole warp returns together.
        if (tileRows == 0) {
            return;
        }
        int slot = rowSlots.get(firstRow);
        int lastLen = rowPositions.get(firstRow + tileRows - 1) + 1;
        int chunk = Math.max(minSplit, (lastLen + maxSplits - 1) / maxSplits);
        int startPos = s * chunk;
        int maxEnd = Math.min(startPos + chunk, lastLen);

        int blockSize = blockCfg & 0xFFFF;
        int maxBlocksPerSlot = blockCfg >>> 16;
        int dim = nHeads * headSize;
        int layerOff = layerIndex * (blockSize * kvDim);
        int qkvStride = dim + 2 * kvDim;
        int blockStride = numLayers * blockSize * kvDim;
        float scale = 1.0f / TornadoMath.sqrt(headSize);

        // Shared tiles, as int-packed half pairs in gemmMMA's operand layouts.
        int[] qTile = context.allocateIntLocalArray(512); // A: 4 k-steps x 16 rows x 8 pairs
        int[] kTile = context.allocateIntLocalArray(512); // B: 2 key blocks x 4 k-steps x 64
        int[] vTile = context.allocateIntLocalArray(512); // B: 8 dim blocks x 16 keys x 4 pairs
        int[] pTile = context.allocateIntLocalArray(128); // A: 16 rows x 8 key pairs
        float[] sTile = context.allocateFloatLocalArray(256);
        float[] rowMax = context.allocateFloatLocalArray(16);
        float[] rowSum = context.allocateFloatLocalArray(16);
        float[] rowCorr = context.allocateFloatLocalArray(16);
        int[] rowEnd = context.allocateIntLocalArray(16);

        // Tile row r is query row r / kvMul of the tile, head r % kvMul of the KV head's group.
        for (int r = lane; r < 16; r += 32) {
            int qr = r / kvMul;
            rowMax[r] = Float.NEGATIVE_INFINITY;
            rowSum[r] = 0.0f;
            rowEnd[r] =
                    qr < tileRows
                            ? Math.min(startPos + chunk, rowPositions.get(firstRow + qr) + 1)
                            : 0;
        }
        for (int idx = lane; idx < 512; idx += 32) {
            int kStep = idx >>> 7;
            int r = (idx >>> 3) & 15;
            int pair = idx & 7;
            int qr = r / kvMul;
            int packed = 0;
            if (qr < tileRows) {
                int h = kvHeadIdx * kvMul + r % kvMul;
                int base = (firstRow + qr) * qkvStride + h * headSize + kStep * 16 + pair * 2;
                int lo = new HalfFloat(qkvBatch.get(base)).getHalfFloatValue() & 0xFFFF;
                int hi = new HalfFloat(qkvBatch.get(base + 1)).getHalfFloatValue() & 0xFFFF;
                packed = lo | (hi << 16);
            }
            qTile[idx] = packed;
        }
        float[] oAcc = new float[32];
        for (int e = 0; e < 32; e++) {
            oAcc[e] = 0.0f;
        }
        context.localBarrier();

        for (int tileC = startPos; tileC < maxEnd; tileC += 16) {
            // K as B for S = Q K^T: element (k = dim, n = key), packing keys (2p, 2p + 1).
            for (int idx = lane; idx < 512; idx += 32) {
                int keyBlock = idx >>> 8;
                int kStep = (idx >>> 6) & 3;
                int kRow = (idx >>> 2) & 15;
                int p = idx & 3;
                int t = tileC + keyBlock * 8 + p * 2;
                int d = kStep * 16 + kRow;
                int lo = 0;
                int hi = 0;
                if (t < maxEnd) {
                    long off =
                            (long) blockTable.get(slot * maxBlocksPerSlot + t / blockSize)
                                            * blockStride
                                    + layerOff
                                    + (t % blockSize) * kvDim
                                    + kvHeadIdx * headSize
                                    + d;
                    lo = keyPool.get(off).getHalfFloatValue() & 0xFFFF;
                }
                if (t + 1 < maxEnd) {
                    long off =
                            (long) blockTable.get(slot * maxBlocksPerSlot + (t + 1) / blockSize)
                                            * blockStride
                                    + layerOff
                                    + ((t + 1) % blockSize) * kvDim
                                    + kvHeadIdx * headSize
                                    + d;
                    hi = keyPool.get(off).getHalfFloatValue() & 0xFFFF;
                }
                kTile[idx] = lo | (hi << 16);
            }
            // V as B for P V: element (k = key, n = dim), packing dims (2p, 2p + 1).
            for (int idx = lane; idx < 512; idx += 32) {
                int dimBlock = idx >>> 6;
                int key = (idx >>> 2) & 15;
                int p = idx & 3;
                int t = tileC + key;
                int packed = 0;
                if (t < maxEnd) {
                    long off =
                            (long) blockTable.get(slot * maxBlocksPerSlot + t / blockSize)
                                            * blockStride
                                    + layerOff
                                    + (t % blockSize) * kvDim
                                    + kvHeadIdx * headSize
                                    + dimBlock * 8
                                    + p * 2;
                    int lo = valuePool.get(off).getHalfFloatValue() & 0xFFFF;
                    int hi = valuePool.get(off + 1).getHalfFloatValue() & 0xFFFF;
                    packed = lo | (hi << 16);
                }
                vTile[idx] = packed;
            }
            context.localBarrier();

            float[] s0 = context.mmaFragment(0.0f);
            float[] s1 = context.mmaFragment(0.0f);
            HalfFloat[] qa0 = context.mmaLoadA(qTile, 16, 0);
            s0 = context.mma(qa0, context.mmaLoadB(kTile, 16, 0), s0, MMAShape.M16N8K16);
            s1 = context.mma(qa0, context.mmaLoadB(kTile, 16, 1024), s1, MMAShape.M16N8K16);
            HalfFloat[] qa1 = context.mmaLoadA(qTile, 16, 512);
            s0 = context.mma(qa1, context.mmaLoadB(kTile, 16, 256), s0, MMAShape.M16N8K16);
            s1 = context.mma(qa1, context.mmaLoadB(kTile, 16, 1280), s1, MMAShape.M16N8K16);
            HalfFloat[] qa2 = context.mmaLoadA(qTile, 16, 1024);
            s0 = context.mma(qa2, context.mmaLoadB(kTile, 16, 512), s0, MMAShape.M16N8K16);
            s1 = context.mma(qa2, context.mmaLoadB(kTile, 16, 1536), s1, MMAShape.M16N8K16);
            HalfFloat[] qa3 = context.mmaLoadA(qTile, 16, 1536);
            s0 = context.mma(qa3, context.mmaLoadB(kTile, 16, 768), s0, MMAShape.M16N8K16);
            s1 = context.mma(qa3, context.mmaLoadB(kTile, 16, 1792), s1, MMAShape.M16N8K16);

            // Scores to shared memory, scaled and masked at each row's causal end.
            int r0 = lane / 4;
            int c0 = (lane % 4) * 2;
            sTile[r0 * 16 + (c0 + 0)] =
                    tileC + (c0 + 0) < rowEnd[r0] ? s0[0] * scale : Float.NEGATIVE_INFINITY;
            sTile[r0 * 16 + (c0 + 1)] =
                    tileC + (c0 + 1) < rowEnd[r0] ? s0[1] * scale : Float.NEGATIVE_INFINITY;
            sTile[(r0 + 8) * 16 + (c0 + 0)] =
                    tileC + (c0 + 0) < rowEnd[(r0 + 8)] ? s0[2] * scale : Float.NEGATIVE_INFINITY;
            sTile[(r0 + 8) * 16 + (c0 + 1)] =
                    tileC + (c0 + 1) < rowEnd[(r0 + 8)] ? s0[3] * scale : Float.NEGATIVE_INFINITY;
            sTile[r0 * 16 + (c0 + 8)] =
                    tileC + (c0 + 8) < rowEnd[r0] ? s1[0] * scale : Float.NEGATIVE_INFINITY;
            sTile[r0 * 16 + (c0 + 9)] =
                    tileC + (c0 + 9) < rowEnd[r0] ? s1[1] * scale : Float.NEGATIVE_INFINITY;
            sTile[(r0 + 8) * 16 + (c0 + 8)] =
                    tileC + (c0 + 8) < rowEnd[(r0 + 8)] ? s1[2] * scale : Float.NEGATIVE_INFINITY;
            sTile[(r0 + 8) * 16 + (c0 + 9)] =
                    tileC + (c0 + 9) < rowEnd[(r0 + 8)] ? s1[3] * scale : Float.NEGATIVE_INFINITY;
            context.localBarrier();

            // Online softmax, one lane per row; P to shared memory in the A layout.
            if (lane < 16) {
                int r = lane;
                float tileMax = Float.NEGATIVE_INFINITY;
                for (int c = 0; c < 16; c++) {
                    tileMax = Math.max(tileMax, sTile[r * 16 + c]);
                }
                float m = rowMax[r];
                float newMax = Math.max(m, tileMax);
                // A tile wholly past the row's end leaves it exactly as it was.
                float corr =
                        tileMax == Float.NEGATIVE_INFINITY
                                ? 1.0f
                                : (m == Float.NEGATIVE_INFINITY
                                        ? 0.0f
                                        : TornadoMath.exp(m - newMax));
                float sum = rowSum[r] * corr;
                for (int c = 0; c < 16; c += 2) {
                    float p0 =
                            sTile[r * 16 + c] == Float.NEGATIVE_INFINITY
                                    ? 0.0f
                                    : TornadoMath.exp(sTile[r * 16 + c] - newMax);
                    float p1 =
                            sTile[r * 16 + c + 1] == Float.NEGATIVE_INFINITY
                                    ? 0.0f
                                    : TornadoMath.exp(sTile[r * 16 + c + 1] - newMax);
                    sum += p0 + p1;
                    int lo = new HalfFloat(p0).getHalfFloatValue() & 0xFFFF;
                    int hi = new HalfFloat(p1).getHalfFloatValue() & 0xFFFF;
                    pTile[r * 8 + c / 2] = lo | (hi << 16);
                }
                rowSum[r] = sum;
                rowMax[r] = tileMax == Float.NEGATIVE_INFINITY ? m : newMax;
                rowCorr[r] = corr;
            }
            context.localBarrier();

            // P V into fresh fragments, folded into the running output with the rows' corrections.
            HalfFloat[] pa = context.mmaLoadA(pTile, 16, 0);
            float corrLo = rowCorr[r0];
            float corrHi = rowCorr[r0 + 8];
            float[] o0 = context.mmaFragment(0.0f);
            o0 = context.mma(pa, context.mmaLoadB(vTile, 16, 0), o0, MMAShape.M16N8K16);
            oAcc[0] = oAcc[0] * corrLo + o0[0];
            oAcc[1] = oAcc[1] * corrLo + o0[1];
            oAcc[2] = oAcc[2] * corrHi + o0[2];
            oAcc[3] = oAcc[3] * corrHi + o0[3];
            float[] o1 = context.mmaFragment(0.0f);
            o1 = context.mma(pa, context.mmaLoadB(vTile, 16, 256), o1, MMAShape.M16N8K16);
            oAcc[4] = oAcc[4] * corrLo + o1[0];
            oAcc[5] = oAcc[5] * corrLo + o1[1];
            oAcc[6] = oAcc[6] * corrHi + o1[2];
            oAcc[7] = oAcc[7] * corrHi + o1[3];
            float[] o2 = context.mmaFragment(0.0f);
            o2 = context.mma(pa, context.mmaLoadB(vTile, 16, 512), o2, MMAShape.M16N8K16);
            oAcc[8] = oAcc[8] * corrLo + o2[0];
            oAcc[9] = oAcc[9] * corrLo + o2[1];
            oAcc[10] = oAcc[10] * corrHi + o2[2];
            oAcc[11] = oAcc[11] * corrHi + o2[3];
            float[] o3 = context.mmaFragment(0.0f);
            o3 = context.mma(pa, context.mmaLoadB(vTile, 16, 768), o3, MMAShape.M16N8K16);
            oAcc[12] = oAcc[12] * corrLo + o3[0];
            oAcc[13] = oAcc[13] * corrLo + o3[1];
            oAcc[14] = oAcc[14] * corrHi + o3[2];
            oAcc[15] = oAcc[15] * corrHi + o3[3];
            float[] o4 = context.mmaFragment(0.0f);
            o4 = context.mma(pa, context.mmaLoadB(vTile, 16, 1024), o4, MMAShape.M16N8K16);
            oAcc[16] = oAcc[16] * corrLo + o4[0];
            oAcc[17] = oAcc[17] * corrLo + o4[1];
            oAcc[18] = oAcc[18] * corrHi + o4[2];
            oAcc[19] = oAcc[19] * corrHi + o4[3];
            float[] o5 = context.mmaFragment(0.0f);
            o5 = context.mma(pa, context.mmaLoadB(vTile, 16, 1280), o5, MMAShape.M16N8K16);
            oAcc[20] = oAcc[20] * corrLo + o5[0];
            oAcc[21] = oAcc[21] * corrLo + o5[1];
            oAcc[22] = oAcc[22] * corrHi + o5[2];
            oAcc[23] = oAcc[23] * corrHi + o5[3];
            float[] o6 = context.mmaFragment(0.0f);
            o6 = context.mma(pa, context.mmaLoadB(vTile, 16, 1536), o6, MMAShape.M16N8K16);
            oAcc[24] = oAcc[24] * corrLo + o6[0];
            oAcc[25] = oAcc[25] * corrLo + o6[1];
            oAcc[26] = oAcc[26] * corrHi + o6[2];
            oAcc[27] = oAcc[27] * corrHi + o6[3];
            float[] o7 = context.mmaFragment(0.0f);
            o7 = context.mma(pa, context.mmaLoadB(vTile, 16, 1792), o7, MMAShape.M16N8K16);
            oAcc[28] = oAcc[28] * corrLo + o7[0];
            oAcc[29] = oAcc[29] * corrLo + o7[1];
            oAcc[30] = oAcc[30] * corrHi + o7[2];
            oAcc[31] = oAcc[31] * corrHi + o7[3];
            context.localBarrier();
        }

        // Unnormalised output, max and sum per (row, head, split), the scalar kernels' layout.
        int r0 = lane / 4;
        int c0 = (lane % 4) * 2;
        writeFlashOutput(
                partials,
                oAcc[0],
                r0,
                0 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[1],
                r0,
                0 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[2],
                (r0 + 8),
                0 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[3],
                (r0 + 8),
                0 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[4],
                r0,
                8 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[5],
                r0,
                8 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[6],
                (r0 + 8),
                8 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[7],
                (r0 + 8),
                8 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[8],
                r0,
                16 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[9],
                r0,
                16 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[10],
                (r0 + 8),
                16 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[11],
                (r0 + 8),
                16 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[12],
                r0,
                24 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[13],
                r0,
                24 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[14],
                (r0 + 8),
                24 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[15],
                (r0 + 8),
                24 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[16],
                r0,
                32 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[17],
                r0,
                32 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[18],
                (r0 + 8),
                32 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[19],
                (r0 + 8),
                32 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[20],
                r0,
                40 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[21],
                r0,
                40 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[22],
                (r0 + 8),
                40 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[23],
                (r0 + 8),
                40 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[24],
                r0,
                48 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[25],
                r0,
                48 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[26],
                (r0 + 8),
                48 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[27],
                (r0 + 8),
                48 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[28],
                r0,
                56 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[29],
                r0,
                56 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[30],
                (r0 + 8),
                56 + c0 + 0,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        writeFlashOutput(
                partials,
                oAcc[31],
                (r0 + 8),
                56 + c0 + 1,
                firstRow,
                tileRows,
                kvHeadIdx,
                kvMul,
                nHeads,
                headSize,
                maxSplits,
                s);
        if (lane < 16) {
            int qr = lane / kvMul;
            if (qr < tileRows) {
                int h = kvHeadIdx * kvMul + lane % kvMul;
                int outBase = (((firstRow + qr) * nHeads + h) * maxSplits + s) * (headSize + 2);
                partials.set(outBase + headSize, rowMax[lane]);
                partials.set(outBase + headSize + 1, rowSum[lane]);
            }
        }
    }

    /** One output value of {@link #raggedFlashAttentionMMA}, dropped for a padding row. */
    private static void writeFlashOutput(
            FloatArray partials,
            float value,
            int r,
            int dimIndex,
            int firstRow,
            int tileRows,
            int kvHeadIdx,
            int kvMul,
            int nHeads,
            int headSize,
            int maxSplits,
            int s) {
        int qr = r / kvMul;
        if (qr < tileRows) {
            int h = kvHeadIdx * kvMul + r % kvMul;
            int outBase = (((firstRow + qr) * nHeads + h) * maxSplits + s) * (headSize + 2);
            partials.set(outBase + dimIndex, value);
        }
    }

    /**
     * Split-K paged attention, second pass: one workgroup per (row, head), one thread per head
     * dimension. Rescales each split's partial result to the overall max and normalises by the
     * rescaled sum of the splits, visiting them in order.
     */
    public static void raggedSplitKvAttentionCombine(
            KernelContext context,
            IntArray rowSlots,
            FloatArray partials,
            HalfFloatArray attnOutFP16,
            int nHeads,
            int headSize,
            int nSplits) {
        int d = context.localIdx;
        int g = context.groupIdx;
        int row = g / nHeads;
        int h = g % nHeads;
        if (rowSlots.get(row) < 0 || d >= headSize) {
            return;
        }
        int stride = headSize + 2;
        int base = (row * nHeads + h) * nSplits * stride;

        float overallMax = Float.NEGATIVE_INFINITY;
        for (int s = 0; s < nSplits; s++) {
            float m = partials.get(base + s * stride + headSize);
            if (m > overallMax) {
                overallMax = m;
            }
        }
        float numerator = 0.0f;
        float denominator = 0.0f;
        for (int s = 0; s < nSplits; s++) {
            int split = base + s * stride;
            float m = partials.get(split + headSize);
            if (m != Float.NEGATIVE_INFINITY) {
                float f = TornadoMath.exp(m - overallMax);
                numerator += f * partials.get(split + d);
                denominator += f * partials.get(split + headSize + 1);
            }
        }
        float out = denominator > 0.0f ? numerator / denominator : 0.0f;
        attnOutFP16.set(row * nHeads * headSize + h * headSize + d, new HalfFloat(out));
    }

    /** Rows {@link #gemmMMASkinny} handles: one m16n8k16 B tile of tokens. */
    public static final int SKINNY_ROWS = 8;

    /** Output columns one {@link #gemmMMASkinny} workgroup computes: four warps of sixteen. */
    public static final int SKINNY_COLS_PER_GROUP = 64;

    /** K positions of the token tile staged in shared memory at a time. */
    private static final int SKINNY_K_CHUNK = 2048;

    /**
     * Tensor-core GEMM for a step of at most {@link #SKINNY_ROWS} tokens: {@code C[t, colOffset +
     * n] = sum_k X[t, k] * W[n, k]} for {@code t < rows}.
     *
     * <p>Decode at light load is a matrix-vector product, bound by reading the weights. The 128-row
     * tiles of {@code gemmMMA} cost the same for one token as for 128, so this computes the
     * transpose instead: the weights are the m16n8k16 A operand, read straight from global memory
     * by each warp with no shared-memory staging and no barriers, and the step's tokens are the
     * 8-wide B operand, staged once per K chunk. Each warp owns sixteen output columns; the weights
     * are read once.
     *
     * <p>Every output value is the same products summed by the same m16n8k16 instructions in the
     * same ascending K order as in {@code gemmMMA}, so the result is the same bits whichever kernel
     * a step uses — what lets the engine pick one by the step's size without breaking batch
     * invariance.
     *
     * <p>Does nothing when {@code stepRows[0] > SKINNY_ROWS}: the 128-row kernel has the step.
     * Requires {@code K % 64 == 0}. Worker: {@code WorkerGrid1D(N / 64 * 128)}, local 128.
     *
     * @param ldc row stride of {@code C}
     * @param colOffset first column of {@code C} this product writes, for packed outputs
     */
    public static void gemmMMASkinny(
            KernelContext ctx,
            HalfFloatArray X,
            HalfFloatArray W,
            FloatArray C,
            IntArray stepRows,
            int K,
            int ldc,
            int colOffset) {
        int rows = stepRows.get(0);
        if (rows > SKINNY_ROWS) {
            return;
        }
        int tid = ctx.localIdx;
        int lane = tid % 32;
        int warp = tid / 32;
        int n0 = ctx.groupIdx * SKINNY_COLS_PER_GROUP + warp * 16;

        // Token tile in gemmMMA's B-tile layout, one 256-byte subtile per k16 step: int idx =
        // step * 64 + kRow * 4 + tokenPair packs tokens (2p, 2p + 1) at k.
        int[] xTile = ctx.allocateIntLocalArray(SKINNY_K_CHUNK * 4);
        float[] acc = ctx.mmaFragment(0.0f);

        // One flat loop over the K steps, four at a time, restaging the token tile at each chunk
        // boundary. A fragment carried through nested loops is lowered to an array assignment NVRTC
        // rejects, hence one loop. The four weight-fragment loads are issued before the four MMAs
        // so their latencies overlap; the MMAs still run in ascending K order into one
        // accumulator, which keeps every value the same bits.
        int stepsPerChunk = SKINNY_K_CHUNK / 16;
        for (int step = 0; step < K / 16; step += 4) {
            int local = step % stepsPerChunk;
            if (local == 0) {
                // Uniform across the workgroup: the condition depends on the step alone.
                ctx.localBarrier();
                int kc = step * 16;
                int chunk = Math.min(SKINNY_K_CHUNK, K - kc);
                for (int idx = tid; idx < chunk * 4; idx += 128) {
                    int k = kc + (idx >>> 6) * 16 + ((idx & 63) >>> 2);
                    int t = (idx & 3) << 1;
                    int lo = X.get(t * K + k).getHalfFloatValue() & 0xFFFF;
                    int hi = X.get((t + 1) * K + k).getHalfFloatValue() & 0xFFFF;
                    xTile[idx] = lo | (hi << 16);
                }
                ctx.localBarrier();
            }
            int k0 = step * 16;
            HalfFloat[] a0 = ctx.mmaLoadA(W, n0, k0, K);
            HalfFloat[] a1 = ctx.mmaLoadA(W, n0, k0 + 16, K);
            HalfFloat[] a2 = ctx.mmaLoadA(W, n0, k0 + 32, K);
            HalfFloat[] a3 = ctx.mmaLoadA(W, n0, k0 + 48, K);
            HalfFloat[] b0 = ctx.mmaLoadB(xTile, 16, local * 256);
            HalfFloat[] b1 = ctx.mmaLoadB(xTile, 16, (local + 1) * 256);
            HalfFloat[] b2 = ctx.mmaLoadB(xTile, 16, (local + 2) * 256);
            HalfFloat[] b3 = ctx.mmaLoadB(xTile, 16, (local + 3) * 256);
            acc = ctx.mma(a0, b0, acc, MMAShape.M16N8K16);
            acc = ctx.mma(a1, b1, acc, MMAShape.M16N8K16);
            acc = ctx.mma(a2, b2, acc, MMAShape.M16N8K16);
            acc = ctx.mma(a3, b3, acc, MMAShape.M16N8K16);
        }

        // acc holds the transposed tile: element i is column n0 + lane / 4 + 8 * (i / 2), token
        // (lane % 4) * 2 + i % 2.
        int n = colOffset + n0 + lane / 4;
        int t = (lane % 4) * 2;
        if (t < rows) {
            C.set(t * ldc + n, acc[0]);
            C.set(t * ldc + n + 8, acc[2]);
        }
        if (t + 1 < rows) {
            C.set((t + 1) * ldc + n, acc[1]);
            C.set((t + 1) * ldc + n + 8, acc[3]);
        }
    }

    // ── 128-row tensor-core GEMMs that step aside for the skinny kernel ─────────────────────
    // Tile constants and the packing helper, as in TransformerBatchPrefillKernels.

    private static final int WARP_SIZE = 32;
    private static final int BM = 128, BN = 128, BK = 16;
    private static final int WARPS_M = 4, WARPS_N = 2;
    private static final int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
    private static final int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    private static final int WM = BM / WARPS_M;
    private static final int WN = BN / WARPS_N;
    private static final int B_SUBTILE_BYTES = 256;

    private static int packHalves(HalfFloatArray src, int idxLo, int idxHi) {
        int lo = src.get(idxLo).getHalfFloatValue() & 0xFFFF;
        int hi = src.get(idxHi).getHalfFloatValue() & 0xFFFF;
        return lo | (hi << 16);
    }

    /**
     * {@link TransformerBatchPrefillKernels#gemmMMA} that does nothing on a step of at most {@link
     * #SKINNY_ROWS} tokens, which {@link #gemmMMASkinny} computes instead. A 128-row block walks K
     * serially, so even a single tile costs nearly a full launch; skipping is what makes the skinny
     * path pay. The arithmetic is the original's, unchanged.
     */
    public static void gemmMMAGuarded(
            KernelContext ctx,
            IntArray stepRows,
            HalfFloatArray A,
            HalfFloatArray B,
            FloatArray C,
            int M,
            int N,
            int K) {
        if (stepRows.get(0) <= SKINNY_ROWS) {
            return; // a skinny step: gemmMMASkinny computes this product
        }
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        // ── Per-thread staging index math (constant across K-steps) ──────────
        // A tile: BM*BK/2 = 1024 ints; layout idx = m_row*(BK/2) + k_pair.
        //   m_row = idx >>> 3, k = (idx & 7)*2; global A element (blockRow+m_row, kBase+k).
        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        // B tile: BK*BN/2 = 1024 ints; subTileId = idx >>> 6, k_row = (idx & 63) >>> 2,
        //   col = subTileId*8 + (idx & 3)*2; B[col, k] at col*K + k (pair at +K).
        int bIdx0 = tid;
        int gB0 = (blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * K + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 = (blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * K + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 = (blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * K + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 = (blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * K + ((bIdx3 & 63) >>> 2);

        // ── Prologue: stage K-step 0 ─────────────────────────────────────────
        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0 = packHalves(B, gB0, gB0 + K);
        int bReg1 = packHalves(B, gB1, gB1 + K);
        int bReg2 = packHalves(B, gB2, gB2 + K);
        int bReg3 = packHalves(B, gB3, gB3 + K);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            // Issue next step's global loads FIRST: independent of the MMAs below,
            // so their latency overlaps ldmatrix + tensor-core compute.
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                bReg0 = packHalves(B, gB0 + kOff, gB0 + kOff + K);
                bReg1 = packHalves(B, gB1 + kOff, gB1 + kOff + K);
                bReg2 = packHalves(B, gB2 + kOff, gB2 + kOff + K);
                bReg3 = packHalves(B, gB3 + kOff, gB3 + kOff + K);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier(); // all shared reads for this K-step complete

            // Overwrite shared tiles for the next step; overlaps the MMAs below.
            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier(); // shared writes visible before next step's ldmatrix
        }

        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, C, rBase + 0, cBase + 0, N);
        ctx.mmaStore(c01, C, rBase + 0, cBase + 8, N);
        ctx.mmaStore(c02, C, rBase + 0, cBase + 16, N);
        ctx.mmaStore(c03, C, rBase + 0, cBase + 24, N);
        ctx.mmaStore(c04, C, rBase + 0, cBase + 32, N);
        ctx.mmaStore(c05, C, rBase + 0, cBase + 40, N);
        ctx.mmaStore(c06, C, rBase + 0, cBase + 48, N);
        ctx.mmaStore(c07, C, rBase + 0, cBase + 56, N);
        ctx.mmaStore(c10, C, rBase + 16, cBase + 0, N);
        ctx.mmaStore(c11, C, rBase + 16, cBase + 8, N);
        ctx.mmaStore(c12, C, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, C, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, C, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, C, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, C, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, C, rBase + 16, cBase + 56, N);
    }

    /**
     * {@link TransformerBatchPrefillKernels#gemmMMAQKV} that does nothing on a step of at most
     * {@link #SKINNY_ROWS} tokens, which {@link #gemmMMASkinny} computes instead. A 128-row block
     * walks K serially, so even a single tile costs nearly a full launch; skipping is what makes
     * the skinny path pay. The arithmetic is the original's, unchanged.
     */
    public static void gemmMMAQKVGuarded(
            KernelContext ctx,
            IntArray stepRows,
            HalfFloatArray A,
            HalfFloatArray wq,
            HalfFloatArray wk,
            HalfFloatArray wv,
            FloatArray qkvOut,
            int M,
            int dim,
            int kvDim,
            int K) {
        if (stepRows.get(0) <= SKINNY_ROWS) {
            return; // a skinny step: gemmMMASkinny computes this product
        }
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy; // column in the packed output
        int qkvStride = dim + 2 * kvDim;

        // Column base inside the segment's own weight matrix (block-uniform).
        int wColBase = blockCol;
        if (blockCol >= dim) wColBase -= dim;
        if (blockCol >= dim + kvDim) wColBase -= kvDim;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        int bIdx0 = tid;
        int gB0 = (wColBase + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * K + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 = (wColBase + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * K + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 = (wColBase + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * K + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 = (wColBase + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * K + ((bIdx3 & 63) >>> 2);

        // ── Prologue: stage K-step 0 ──
        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0;
        int bReg1;
        int bReg2;
        int bReg3;
        if (blockCol < dim) {
            bReg0 = packHalves(wq, gB0, gB0 + K);
            bReg1 = packHalves(wq, gB1, gB1 + K);
            bReg2 = packHalves(wq, gB2, gB2 + K);
            bReg3 = packHalves(wq, gB3, gB3 + K);
        } else if (blockCol < dim + kvDim) {
            bReg0 = packHalves(wk, gB0, gB0 + K);
            bReg1 = packHalves(wk, gB1, gB1 + K);
            bReg2 = packHalves(wk, gB2, gB2 + K);
            bReg3 = packHalves(wk, gB3, gB3 + K);
        } else {
            bReg0 = packHalves(wv, gB0, gB0 + K);
            bReg1 = packHalves(wv, gB1, gB1 + K);
            bReg2 = packHalves(wv, gB2, gB2 + K);
            bReg3 = packHalves(wv, gB3, gB3 + K);
        }
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                if (blockCol < dim) {
                    bReg0 = packHalves(wq, gB0 + kOff, gB0 + kOff + K);
                    bReg1 = packHalves(wq, gB1 + kOff, gB1 + kOff + K);
                    bReg2 = packHalves(wq, gB2 + kOff, gB2 + kOff + K);
                    bReg3 = packHalves(wq, gB3 + kOff, gB3 + kOff + K);
                } else if (blockCol < dim + kvDim) {
                    bReg0 = packHalves(wk, gB0 + kOff, gB0 + kOff + K);
                    bReg1 = packHalves(wk, gB1 + kOff, gB1 + kOff + K);
                    bReg2 = packHalves(wk, gB2 + kOff, gB2 + kOff + K);
                    bReg3 = packHalves(wk, gB3 + kOff, gB3 + kOff + K);
                } else {
                    bReg0 = packHalves(wv, gB0 + kOff, gB0 + kOff + K);
                    bReg1 = packHalves(wv, gB1 + kOff, gB1 + kOff + K);
                    bReg2 = packHalves(wv, gB2 + kOff, gB2 + kOff + K);
                    bReg3 = packHalves(wv, gB3 + kOff, gB3 + kOff + K);
                }
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        // Stores are uniform: packed column == blockCol-relative column,
        // stride is the packed row width.
        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, qkvOut, rBase + 0, cBase + 0, qkvStride);
        ctx.mmaStore(c01, qkvOut, rBase + 0, cBase + 8, qkvStride);
        ctx.mmaStore(c02, qkvOut, rBase + 0, cBase + 16, qkvStride);
        ctx.mmaStore(c03, qkvOut, rBase + 0, cBase + 24, qkvStride);
        ctx.mmaStore(c04, qkvOut, rBase + 0, cBase + 32, qkvStride);
        ctx.mmaStore(c05, qkvOut, rBase + 0, cBase + 40, qkvStride);
        ctx.mmaStore(c06, qkvOut, rBase + 0, cBase + 48, qkvStride);
        ctx.mmaStore(c07, qkvOut, rBase + 0, cBase + 56, qkvStride);
        ctx.mmaStore(c10, qkvOut, rBase + 16, cBase + 0, qkvStride);
        ctx.mmaStore(c11, qkvOut, rBase + 16, cBase + 8, qkvStride);
        ctx.mmaStore(c12, qkvOut, rBase + 16, cBase + 16, qkvStride);
        ctx.mmaStore(c13, qkvOut, rBase + 16, cBase + 24, qkvStride);
        ctx.mmaStore(c14, qkvOut, rBase + 16, cBase + 32, qkvStride);
        ctx.mmaStore(c15, qkvOut, rBase + 16, cBase + 40, qkvStride);
        ctx.mmaStore(c16, qkvOut, rBase + 16, cBase + 48, qkvStride);
        ctx.mmaStore(c17, qkvOut, rBase + 16, cBase + 56, qkvStride);
    }

    /**
     * {@link TransformerBatchPrefillKernels#gemmMMAGateUp} that does nothing on a step of at most
     * {@link #SKINNY_ROWS} tokens, which {@link #gemmMMASkinny} computes instead. A 128-row block
     * walks K serially, so even a single tile costs nearly a full launch; skipping is what makes
     * the skinny path pay. The arithmetic is the original's, unchanged.
     */
    public static void gemmMMAGateUpGuarded(
            KernelContext ctx,
            IntArray stepRows,
            HalfFloatArray A,
            HalfFloatArray w1,
            HalfFloatArray w3,
            FloatArray gateUpOut,
            int M,
            int hidDim,
            int K) {
        if (stepRows.get(0) <= SKINNY_ROWS) {
            return; // a skinny step: gemmMMASkinny computes this product
        }
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy; // column in the packed output
        int outStride = 2 * hidDim;

        int wColBase = (blockCol < hidDim) ? blockCol : (blockCol - hidDim);

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        int bIdx0 = tid;
        int gB0 = (wColBase + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * K + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 = (wColBase + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * K + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 = (wColBase + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * K + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 = (wColBase + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * K + ((bIdx3 & 63) >>> 2);

        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0;
        int bReg1;
        int bReg2;
        int bReg3;
        if (blockCol < hidDim) {
            bReg0 = packHalves(w1, gB0, gB0 + K);
            bReg1 = packHalves(w1, gB1, gB1 + K);
            bReg2 = packHalves(w1, gB2, gB2 + K);
            bReg3 = packHalves(w1, gB3, gB3 + K);
        } else {
            bReg0 = packHalves(w3, gB0, gB0 + K);
            bReg1 = packHalves(w3, gB1, gB1 + K);
            bReg2 = packHalves(w3, gB2, gB2 + K);
            bReg3 = packHalves(w3, gB3, gB3 + K);
        }
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                if (blockCol < hidDim) {
                    bReg0 = packHalves(w1, gB0 + kOff, gB0 + kOff + K);
                    bReg1 = packHalves(w1, gB1 + kOff, gB1 + kOff + K);
                    bReg2 = packHalves(w1, gB2 + kOff, gB2 + kOff + K);
                    bReg3 = packHalves(w1, gB3 + kOff, gB3 + kOff + K);
                } else {
                    bReg0 = packHalves(w3, gB0 + kOff, gB0 + kOff + K);
                    bReg1 = packHalves(w3, gB1 + kOff, gB1 + kOff + K);
                    bReg2 = packHalves(w3, gB2 + kOff, gB2 + kOff + K);
                    bReg3 = packHalves(w3, gB3 + kOff, gB3 + kOff + K);
                }
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, gateUpOut, rBase + 0, cBase + 0, outStride);
        ctx.mmaStore(c01, gateUpOut, rBase + 0, cBase + 8, outStride);
        ctx.mmaStore(c02, gateUpOut, rBase + 0, cBase + 16, outStride);
        ctx.mmaStore(c03, gateUpOut, rBase + 0, cBase + 24, outStride);
        ctx.mmaStore(c04, gateUpOut, rBase + 0, cBase + 32, outStride);
        ctx.mmaStore(c05, gateUpOut, rBase + 0, cBase + 40, outStride);
        ctx.mmaStore(c06, gateUpOut, rBase + 0, cBase + 48, outStride);
        ctx.mmaStore(c07, gateUpOut, rBase + 0, cBase + 56, outStride);
        ctx.mmaStore(c10, gateUpOut, rBase + 16, cBase + 0, outStride);
        ctx.mmaStore(c11, gateUpOut, rBase + 16, cBase + 8, outStride);
        ctx.mmaStore(c12, gateUpOut, rBase + 16, cBase + 16, outStride);
        ctx.mmaStore(c13, gateUpOut, rBase + 16, cBase + 24, outStride);
        ctx.mmaStore(c14, gateUpOut, rBase + 16, cBase + 32, outStride);
        ctx.mmaStore(c15, gateUpOut, rBase + 16, cBase + 40, outStride);
        ctx.mmaStore(c16, gateUpOut, rBase + 16, cBase + 48, outStride);
        ctx.mmaStore(c17, gateUpOut, rBase + 16, cBase + 56, outStride);
    }

    /**
     * Copies the hidden state of each sampling row into a dense block for the logits: row {@code
     * sampleRows[s]} of {@code x} becomes row {@code s} of {@code gathered}. A negative entry is a
     * sequence slot with nothing to sample this step; its row is zeroed, and whatever the logits
     * make of it is ignored.
     */
    public static void gatherRows(
            KernelContext context,
            FloatArray x,
            IntArray sampleRows,
            FloatArray gathered,
            int dim) {
        int idx = context.globalIdx;
        int s = idx / dim;
        int d = idx % dim;
        int row = sampleRows.get(s);
        gathered.set(idx, row < 0 ? 0.0f : x.get(row * dim + d));
    }
}
