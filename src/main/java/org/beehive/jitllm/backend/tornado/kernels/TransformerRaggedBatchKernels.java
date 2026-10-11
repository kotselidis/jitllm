package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
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
                int cacheOff =
                        physBlock * (numLayers * blockSize * kvDim)
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
                int kvOff =
                        physBlock * blockStride
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
                int kvOff =
                        physBlock * blockStride
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
                int kvOff =
                        physBlock * blockStride
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
