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
