package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.matrix.Matrix8x8Float;
import uk.ac.manchester.tornado.api.types.matrix.Matrix8x8Half;

// @formatter:off
/**
 * Batched-prefill projections on Apple SIMD-group matrices ({@code simdgroup_float8x8} and {@code
 * simdgroup_half8x8}): the same products as the tiled kernels in {@link
 * TransformerBatchPrefillKernels}, {@code out[b, r] = sum_k x[b, k] * w[r, k]}, with the
 * multiply-accumulate done by {@code simdgroup_multiply_accumulate} on 8x8 fragments instead of
 * scalar FMAs.
 *
 * <p>A workgroup of 256 threads (eight SIMD groups in a 2x4 layout) computes a 64-token tile of the
 * output, 64 weight rows wide (32 for the gate/up kernels, which compute two projections). Per step
 * a 32-wide slice of the activations and of the weights ({@code half} for FP16, dequantized {@code
 * float} for Q8_0) is staged in threadgroup memory, rows padded to {@value #LDS} elements. Each
 * SIMD group multiplies a 32-token block of its share of the rows, reading weight fragments with
 * transposed loads because weight rows hold the contraction dimension contiguously. The next slice
 * is read into registers while the current one is multiplied, and each thread stages a Q8_0 row's
 * consecutive columns so that it reads the block scale once. Accumulation is in {@code float}.
 * Whole tiles are stored straight to device memory (the residual kernels start from the output
 * instead of zero); tiles that cross the end of the batch or of the rows go one 8x8 fragment at a
 * time through threadgroup memory so that the elements past the ends are skipped.
 */
// @formatter:on
public final class TransformerBatchPrefillSimdgroupKernels {

    /** Threadgroup row stride of a staged 32-wide slice, padded against bank conflicts. */
    public static final int LDS = 40;

    /** Threads per workgroup: eight SIMD groups. */
    public static final int THREADS = 256;

    private TransformerBatchPrefillSimdgroupKernels() {}

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmQKVFP16} on SIMD-group matrices: Q, K and V
     * rows taken as one {@code qDim + 2 * kvDim}-row matrix whose 64-row tiles each lie in one
     * projection (the caller checks both sizes are multiples of 64). {@code dim} must be a multiple
     * of 32. Grid: {@code ((qDim + 2 * kvDim) / 64) * ceil(batch / 64)} workgroups of {@value
     * #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQKVFP16(
            KernelContext context,
            HalfFloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            HalfFloatArray wq,
            HalfFloatArray wk,
            HalfFloatArray wv,
            int dim,
            int qDim,
            int kvDim,
            int batch) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws = context.allocateHalfFloatLocalArray(64 * LDS);
        float[] cs = context.allocateFloatLocalArray(8 * 64);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 16;
        int rowTiles = ((qDim + 2 * kvDim) + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        int which = r0 < qDim ? 0 : (r0 < qDim + kvDim ? 1 : 2);
        int local0 = which == 0 ? r0 : (which == 1 ? r0 - qDim : r0 - qDim - kvDim);
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        float px0;
        float px1;
        float px2;
        float px3;
        float px4;
        float px5;
        float px6;
        float px7;
        float pw0;
        float pw1;
        float pw2;
        float pw3;
        float pw4;
        float pw5;
        float pw6;
        float pw7;
        px0 = x.get(TornadoMath.min(b0 + rr0 + 0, batch - 1) * dim + 0 + kk0).getFloat32();
        px1 = x.get(TornadoMath.min(b0 + rr0 + 8, batch - 1) * dim + 0 + kk0).getFloat32();
        px2 = x.get(TornadoMath.min(b0 + rr0 + 16, batch - 1) * dim + 0 + kk0).getFloat32();
        px3 = x.get(TornadoMath.min(b0 + rr0 + 24, batch - 1) * dim + 0 + kk0).getFloat32();
        px4 = x.get(TornadoMath.min(b0 + rr0 + 32, batch - 1) * dim + 0 + kk0).getFloat32();
        px5 = x.get(TornadoMath.min(b0 + rr0 + 40, batch - 1) * dim + 0 + kk0).getFloat32();
        px6 = x.get(TornadoMath.min(b0 + rr0 + 48, batch - 1) * dim + 0 + kk0).getFloat32();
        px7 = x.get(TornadoMath.min(b0 + rr0 + 56, batch - 1) * dim + 0 + kk0).getFloat32();
        if (which == 0) {
            pw0 = wq.get((local0 + rr0 + 0) * dim + 0 + kk0).getFloat32();
            pw1 = wq.get((local0 + rr0 + 8) * dim + 0 + kk0).getFloat32();
            pw2 = wq.get((local0 + rr0 + 16) * dim + 0 + kk0).getFloat32();
            pw3 = wq.get((local0 + rr0 + 24) * dim + 0 + kk0).getFloat32();
            pw4 = wq.get((local0 + rr0 + 32) * dim + 0 + kk0).getFloat32();
            pw5 = wq.get((local0 + rr0 + 40) * dim + 0 + kk0).getFloat32();
            pw6 = wq.get((local0 + rr0 + 48) * dim + 0 + kk0).getFloat32();
            pw7 = wq.get((local0 + rr0 + 56) * dim + 0 + kk0).getFloat32();
        } else if (which == 1) {
            pw0 = wk.get((local0 + rr0 + 0) * dim + 0 + kk0).getFloat32();
            pw1 = wk.get((local0 + rr0 + 8) * dim + 0 + kk0).getFloat32();
            pw2 = wk.get((local0 + rr0 + 16) * dim + 0 + kk0).getFloat32();
            pw3 = wk.get((local0 + rr0 + 24) * dim + 0 + kk0).getFloat32();
            pw4 = wk.get((local0 + rr0 + 32) * dim + 0 + kk0).getFloat32();
            pw5 = wk.get((local0 + rr0 + 40) * dim + 0 + kk0).getFloat32();
            pw6 = wk.get((local0 + rr0 + 48) * dim + 0 + kk0).getFloat32();
            pw7 = wk.get((local0 + rr0 + 56) * dim + 0 + kk0).getFloat32();
        } else {
            pw0 = wv.get((local0 + rr0 + 0) * dim + 0 + kk0).getFloat32();
            pw1 = wv.get((local0 + rr0 + 8) * dim + 0 + kk0).getFloat32();
            pw2 = wv.get((local0 + rr0 + 16) * dim + 0 + kk0).getFloat32();
            pw3 = wv.get((local0 + rr0 + 24) * dim + 0 + kk0).getFloat32();
            pw4 = wv.get((local0 + rr0 + 32) * dim + 0 + kk0).getFloat32();
            pw5 = wv.get((local0 + rr0 + 40) * dim + 0 + kk0).getFloat32();
            pw6 = wv.get((local0 + rr0 + 48) * dim + 0 + kk0).getFloat32();
            pw7 = wv.get((local0 + rr0 + 56) * dim + 0 + kk0).getFloat32();
        }
        for (int k0 = 0; k0 < dim; k0 += 32) {
            xs[(rr0 + 0) * LDS + kk0] = px0;
            xs[(rr0 + 8) * LDS + kk0] = px1;
            xs[(rr0 + 16) * LDS + kk0] = px2;
            xs[(rr0 + 24) * LDS + kk0] = px3;
            xs[(rr0 + 32) * LDS + kk0] = px4;
            xs[(rr0 + 40) * LDS + kk0] = px5;
            xs[(rr0 + 48) * LDS + kk0] = px6;
            xs[(rr0 + 56) * LDS + kk0] = px7;
            ws[(rr0 + 0) * LDS + kk0] = new HalfFloat(pw0);
            ws[(rr0 + 8) * LDS + kk0] = new HalfFloat(pw1);
            ws[(rr0 + 16) * LDS + kk0] = new HalfFloat(pw2);
            ws[(rr0 + 24) * LDS + kk0] = new HalfFloat(pw3);
            ws[(rr0 + 32) * LDS + kk0] = new HalfFloat(pw4);
            ws[(rr0 + 40) * LDS + kk0] = new HalfFloat(pw5);
            ws[(rr0 + 48) * LDS + kk0] = new HalfFloat(pw6);
            ws[(rr0 + 56) * LDS + kk0] = new HalfFloat(pw7);
            context.localBarrier();
            if (k0 + 32 < dim) {
                px0 =
                        x.get(TornadoMath.min(b0 + rr0 + 0, batch - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                px1 =
                        x.get(TornadoMath.min(b0 + rr0 + 8, batch - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                px2 =
                        x.get(TornadoMath.min(b0 + rr0 + 16, batch - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                px3 =
                        x.get(TornadoMath.min(b0 + rr0 + 24, batch - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                px4 =
                        x.get(TornadoMath.min(b0 + rr0 + 32, batch - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                px5 =
                        x.get(TornadoMath.min(b0 + rr0 + 40, batch - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                px6 =
                        x.get(TornadoMath.min(b0 + rr0 + 48, batch - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                px7 =
                        x.get(TornadoMath.min(b0 + rr0 + 56, batch - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                if (which == 0) {
                    pw0 = wq.get((local0 + rr0 + 0) * dim + k0 + 32 + kk0).getFloat32();
                    pw1 = wq.get((local0 + rr0 + 8) * dim + k0 + 32 + kk0).getFloat32();
                    pw2 = wq.get((local0 + rr0 + 16) * dim + k0 + 32 + kk0).getFloat32();
                    pw3 = wq.get((local0 + rr0 + 24) * dim + k0 + 32 + kk0).getFloat32();
                    pw4 = wq.get((local0 + rr0 + 32) * dim + k0 + 32 + kk0).getFloat32();
                    pw5 = wq.get((local0 + rr0 + 40) * dim + k0 + 32 + kk0).getFloat32();
                    pw6 = wq.get((local0 + rr0 + 48) * dim + k0 + 32 + kk0).getFloat32();
                    pw7 = wq.get((local0 + rr0 + 56) * dim + k0 + 32 + kk0).getFloat32();
                } else if (which == 1) {
                    pw0 = wk.get((local0 + rr0 + 0) * dim + k0 + 32 + kk0).getFloat32();
                    pw1 = wk.get((local0 + rr0 + 8) * dim + k0 + 32 + kk0).getFloat32();
                    pw2 = wk.get((local0 + rr0 + 16) * dim + k0 + 32 + kk0).getFloat32();
                    pw3 = wk.get((local0 + rr0 + 24) * dim + k0 + 32 + kk0).getFloat32();
                    pw4 = wk.get((local0 + rr0 + 32) * dim + k0 + 32 + kk0).getFloat32();
                    pw5 = wk.get((local0 + rr0 + 40) * dim + k0 + 32 + kk0).getFloat32();
                    pw6 = wk.get((local0 + rr0 + 48) * dim + k0 + 32 + kk0).getFloat32();
                    pw7 = wk.get((local0 + rr0 + 56) * dim + k0 + 32 + kk0).getFloat32();
                } else {
                    pw0 = wv.get((local0 + rr0 + 0) * dim + k0 + 32 + kk0).getFloat32();
                    pw1 = wv.get((local0 + rr0 + 8) * dim + k0 + 32 + kk0).getFloat32();
                    pw2 = wv.get((local0 + rr0 + 16) * dim + k0 + 32 + kk0).getFloat32();
                    pw3 = wv.get((local0 + rr0 + 24) * dim + k0 + 32 + kk0).getFloat32();
                    pw4 = wv.get((local0 + rr0 + 32) * dim + k0 + 32 + kk0).getFloat32();
                    pw5 = wv.get((local0 + rr0 + 40) * dim + k0 + 32 + kk0).getFloat32();
                    pw6 = wv.get((local0 + rr0 + 48) * dim + k0 + 32 + kk0).getFloat32();
                    pw7 = wv.get((local0 + rr0 + 56) * dim + k0 + 32 + kk0).getFloat32();
                }
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 8) * LDS + kk, LDS);
                c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
            }
            context.localBarrier();
        }
        if (b0 + 64 <= batch) {
            if (which == 0) {
                context.simdgroupMatrixStore(
                        c00, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c01, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c10, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c11, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c20, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c21, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c30, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c31, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 8, qDim);
            } else if (which == 1) {
                context.simdgroupMatrixStore(
                        c00, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            } else {
                context.simdgroupMatrixStore(
                        c00, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            }
        } else {
            context.simdgroupMatrixStore(c00, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmFP16WithResidual} on SIMD-group matrices:
     * {@code out[b, r] += sum_k x[b, k] * w[r, k]} with FP16 weights. {@code n} must be a multiple
     * of 32. Grid: {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmFP16WithResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            HalfFloatArray w,
            int n,
            int d,
            int batch) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws = context.allocateHalfFloatLocalArray(64 * LDS);
        float[] cs = context.allocateFloatLocalArray(8 * 64);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 16;
        int rowTiles = (d + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        if (full) {
            c00 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            c01 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            c10 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            c11 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            c20 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            c21 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            c30 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            c31 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        }
        float px0;
        float px1;
        float px2;
        float px3;
        float px4;
        float px5;
        float px6;
        float px7;
        float pw0;
        float pw1;
        float pw2;
        float pw3;
        float pw4;
        float pw5;
        float pw6;
        float pw7;
        px0 = inputBatch.get(TornadoMath.min(b0 + rr0 + 0, batch - 1) * n + 0 + kk0);
        px1 = inputBatch.get(TornadoMath.min(b0 + rr0 + 8, batch - 1) * n + 0 + kk0);
        px2 = inputBatch.get(TornadoMath.min(b0 + rr0 + 16, batch - 1) * n + 0 + kk0);
        px3 = inputBatch.get(TornadoMath.min(b0 + rr0 + 24, batch - 1) * n + 0 + kk0);
        px4 = inputBatch.get(TornadoMath.min(b0 + rr0 + 32, batch - 1) * n + 0 + kk0);
        px5 = inputBatch.get(TornadoMath.min(b0 + rr0 + 40, batch - 1) * n + 0 + kk0);
        px6 = inputBatch.get(TornadoMath.min(b0 + rr0 + 48, batch - 1) * n + 0 + kk0);
        px7 = inputBatch.get(TornadoMath.min(b0 + rr0 + 56, batch - 1) * n + 0 + kk0);
        pw0 = w.get(TornadoMath.min(r0 + rr0 + 0, d - 1) * n + 0 + kk0).getFloat32();
        pw1 = w.get(TornadoMath.min(r0 + rr0 + 8, d - 1) * n + 0 + kk0).getFloat32();
        pw2 = w.get(TornadoMath.min(r0 + rr0 + 16, d - 1) * n + 0 + kk0).getFloat32();
        pw3 = w.get(TornadoMath.min(r0 + rr0 + 24, d - 1) * n + 0 + kk0).getFloat32();
        pw4 = w.get(TornadoMath.min(r0 + rr0 + 32, d - 1) * n + 0 + kk0).getFloat32();
        pw5 = w.get(TornadoMath.min(r0 + rr0 + 40, d - 1) * n + 0 + kk0).getFloat32();
        pw6 = w.get(TornadoMath.min(r0 + rr0 + 48, d - 1) * n + 0 + kk0).getFloat32();
        pw7 = w.get(TornadoMath.min(r0 + rr0 + 56, d - 1) * n + 0 + kk0).getFloat32();
        for (int k0 = 0; k0 < n; k0 += 32) {
            xs[(rr0 + 0) * LDS + kk0] = px0;
            xs[(rr0 + 8) * LDS + kk0] = px1;
            xs[(rr0 + 16) * LDS + kk0] = px2;
            xs[(rr0 + 24) * LDS + kk0] = px3;
            xs[(rr0 + 32) * LDS + kk0] = px4;
            xs[(rr0 + 40) * LDS + kk0] = px5;
            xs[(rr0 + 48) * LDS + kk0] = px6;
            xs[(rr0 + 56) * LDS + kk0] = px7;
            ws[(rr0 + 0) * LDS + kk0] = new HalfFloat(pw0);
            ws[(rr0 + 8) * LDS + kk0] = new HalfFloat(pw1);
            ws[(rr0 + 16) * LDS + kk0] = new HalfFloat(pw2);
            ws[(rr0 + 24) * LDS + kk0] = new HalfFloat(pw3);
            ws[(rr0 + 32) * LDS + kk0] = new HalfFloat(pw4);
            ws[(rr0 + 40) * LDS + kk0] = new HalfFloat(pw5);
            ws[(rr0 + 48) * LDS + kk0] = new HalfFloat(pw6);
            ws[(rr0 + 56) * LDS + kk0] = new HalfFloat(pw7);
            context.localBarrier();
            if (k0 + 32 < n) {
                px0 = inputBatch.get(TornadoMath.min(b0 + rr0 + 0, batch - 1) * n + k0 + 32 + kk0);
                px1 = inputBatch.get(TornadoMath.min(b0 + rr0 + 8, batch - 1) * n + k0 + 32 + kk0);
                px2 = inputBatch.get(TornadoMath.min(b0 + rr0 + 16, batch - 1) * n + k0 + 32 + kk0);
                px3 = inputBatch.get(TornadoMath.min(b0 + rr0 + 24, batch - 1) * n + k0 + 32 + kk0);
                px4 = inputBatch.get(TornadoMath.min(b0 + rr0 + 32, batch - 1) * n + k0 + 32 + kk0);
                px5 = inputBatch.get(TornadoMath.min(b0 + rr0 + 40, batch - 1) * n + k0 + 32 + kk0);
                px6 = inputBatch.get(TornadoMath.min(b0 + rr0 + 48, batch - 1) * n + k0 + 32 + kk0);
                px7 = inputBatch.get(TornadoMath.min(b0 + rr0 + 56, batch - 1) * n + k0 + 32 + kk0);
                pw0 = w.get(TornadoMath.min(r0 + rr0 + 0, d - 1) * n + k0 + 32 + kk0).getFloat32();
                pw1 = w.get(TornadoMath.min(r0 + rr0 + 8, d - 1) * n + k0 + 32 + kk0).getFloat32();
                pw2 = w.get(TornadoMath.min(r0 + rr0 + 16, d - 1) * n + k0 + 32 + kk0).getFloat32();
                pw3 = w.get(TornadoMath.min(r0 + rr0 + 24, d - 1) * n + k0 + 32 + kk0).getFloat32();
                pw4 = w.get(TornadoMath.min(r0 + rr0 + 32, d - 1) * n + k0 + 32 + kk0).getFloat32();
                pw5 = w.get(TornadoMath.min(r0 + rr0 + 40, d - 1) * n + k0 + 32 + kk0).getFloat32();
                pw6 = w.get(TornadoMath.min(r0 + rr0 + 48, d - 1) * n + k0 + 32 + kk0).getFloat32();
                pw7 = w.get(TornadoMath.min(r0 + rr0 + 56, d - 1) * n + k0 + 32 + kk0).getFloat32();
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half cw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Half cw1 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 8) * LDS + kk, LDS);
                c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
            }
            context.localBarrier();
        }
        if (full) {
            context.simdgroupMatrixStore(
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmRmsNormFFNGateUpFP16} on SIMD-group
     * matrices: the RMS scaling is applied as the activations are staged, gate and up accumulate
     * from the same staged activations, and {@code SiLU(gate) * up} is written per element. {@code
     * dim} must be a multiple of 32. Grid: {@code ceil(hiddenDim / 32) * ceil(batch / 64)}
     * workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpFP16(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            HalfFloatArray w1,
            HalfFloatArray w3,
            int dim,
            int hiddenDim,
            int batch) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        HalfFloat[] ws = context.allocateHalfFloatLocalArray(64 * LDS);
        float[] cs = context.allocateFloatLocalArray(8 * 128);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 8;
        int rowTiles = (hiddenDim + 31) >> 5;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        Matrix8x8Float g00 = context.simdgroupMatrixZero();
        Matrix8x8Float g10 = context.simdgroupMatrixZero();
        Matrix8x8Float g20 = context.simdgroupMatrixZero();
        Matrix8x8Float g30 = context.simdgroupMatrixZero();
        Matrix8x8Float u00 = context.simdgroupMatrixZero();
        Matrix8x8Float u10 = context.simdgroupMatrixZero();
        Matrix8x8Float u20 = context.simdgroupMatrixZero();
        Matrix8x8Float u30 = context.simdgroupMatrixZero();
        float px0;
        float px1;
        float px2;
        float px3;
        float px4;
        float px5;
        float px6;
        float px7;
        float pw0;
        float pw1;
        float pw2;
        float pw3;
        float pw4;
        float pw5;
        float pw6;
        float pw7;
        px0 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 0, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 0, batch - 1) * dim + 0 + kk0);
        px1 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 8, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 8, batch - 1) * dim + 0 + kk0);
        px2 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 16, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 16, batch - 1) * dim + 0 + kk0);
        px3 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 24, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 24, batch - 1) * dim + 0 + kk0);
        px4 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 32, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 32, batch - 1) * dim + 0 + kk0);
        px5 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 40, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 40, batch - 1) * dim + 0 + kk0);
        px6 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 48, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 48, batch - 1) * dim + 0 + kk0);
        px7 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 56, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 56, batch - 1) * dim + 0 + kk0);
        pw0 = w1.get(TornadoMath.min(r0 + rr0 + 0, hiddenDim - 1) * dim + 0 + kk0).getFloat32();
        pw1 = w1.get(TornadoMath.min(r0 + rr0 + 8, hiddenDim - 1) * dim + 0 + kk0).getFloat32();
        pw2 = w1.get(TornadoMath.min(r0 + rr0 + 16, hiddenDim - 1) * dim + 0 + kk0).getFloat32();
        pw3 = w1.get(TornadoMath.min(r0 + rr0 + 24, hiddenDim - 1) * dim + 0 + kk0).getFloat32();
        pw4 =
                w3.get(TornadoMath.min(r0 + rr0 + 32 - 32, hiddenDim - 1) * dim + 0 + kk0)
                        .getFloat32();
        pw5 =
                w3.get(TornadoMath.min(r0 + rr0 + 40 - 32, hiddenDim - 1) * dim + 0 + kk0)
                        .getFloat32();
        pw6 =
                w3.get(TornadoMath.min(r0 + rr0 + 48 - 32, hiddenDim - 1) * dim + 0 + kk0)
                        .getFloat32();
        pw7 =
                w3.get(TornadoMath.min(r0 + rr0 + 56 - 32, hiddenDim - 1) * dim + 0 + kk0)
                        .getFloat32();
        for (int k0 = 0; k0 < dim; k0 += 32) {
            xs[(rr0 + 0) * LDS + kk0] = px0;
            xs[(rr0 + 8) * LDS + kk0] = px1;
            xs[(rr0 + 16) * LDS + kk0] = px2;
            xs[(rr0 + 24) * LDS + kk0] = px3;
            xs[(rr0 + 32) * LDS + kk0] = px4;
            xs[(rr0 + 40) * LDS + kk0] = px5;
            xs[(rr0 + 48) * LDS + kk0] = px6;
            xs[(rr0 + 56) * LDS + kk0] = px7;
            ws[(rr0 + 0) * LDS + kk0] = new HalfFloat(pw0);
            ws[(rr0 + 8) * LDS + kk0] = new HalfFloat(pw1);
            ws[(rr0 + 16) * LDS + kk0] = new HalfFloat(pw2);
            ws[(rr0 + 24) * LDS + kk0] = new HalfFloat(pw3);
            ws[(rr0 + 32) * LDS + kk0] = new HalfFloat(pw4);
            ws[(rr0 + 40) * LDS + kk0] = new HalfFloat(pw5);
            ws[(rr0 + 48) * LDS + kk0] = new HalfFloat(pw6);
            ws[(rr0 + 56) * LDS + kk0] = new HalfFloat(pw7);
            context.localBarrier();
            if (k0 + 32 < dim) {
                px0 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 0, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 0, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px1 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 8, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 8, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px2 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 16, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 16, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px3 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 24, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 24, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px4 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 32, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 32, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px5 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 40, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 40, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px6 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 48, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 48, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px7 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 56, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 56, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                pw0 =
                        w1.get(TornadoMath.min(r0 + rr0 + 0, hiddenDim - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                pw1 =
                        w1.get(TornadoMath.min(r0 + rr0 + 8, hiddenDim - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                pw2 =
                        w1.get(TornadoMath.min(r0 + rr0 + 16, hiddenDim - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                pw3 =
                        w1.get(TornadoMath.min(r0 + rr0 + 24, hiddenDim - 1) * dim + k0 + 32 + kk0)
                                .getFloat32();
                pw4 =
                        w3.get(
                                        TornadoMath.min(r0 + rr0 + 32 - 32, hiddenDim - 1) * dim
                                                + k0
                                                + 32
                                                + kk0)
                                .getFloat32();
                pw5 =
                        w3.get(
                                        TornadoMath.min(r0 + rr0 + 40 - 32, hiddenDim - 1) * dim
                                                + k0
                                                + 32
                                                + kk0)
                                .getFloat32();
                pw6 =
                        w3.get(
                                        TornadoMath.min(r0 + rr0 + 48 - 32, hiddenDim - 1) * dim
                                                + k0
                                                + 32
                                                + kk0)
                                .getFloat32();
                pw7 =
                        w3.get(
                                        TornadoMath.min(r0 + rr0 + 56 - 32, hiddenDim - 1) * dim
                                                + k0
                                                + 32
                                                + kk0)
                                .getFloat32();
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Half gw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Half uw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            context.localBarrier();
        }
        context.simdgroupMatrixStore(g00, cs, sg << 7, 8);
        context.simdgroupMatrixStore(u00, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = b0 + sgRow + 0 + (e >> 3);
            int r = r0 + sgCol + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(g10, cs, sg << 7, 8);
        context.simdgroupMatrixStore(u10, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = b0 + sgRow + 8 + (e >> 3);
            int r = r0 + sgCol + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(g20, cs, sg << 7, 8);
        context.simdgroupMatrixStore(u20, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = b0 + sgRow + 16 + (e >> 3);
            int r = r0 + sgCol + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(g30, cs, sg << 7, 8);
        context.simdgroupMatrixStore(u30, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = b0 + sgRow + 24 + (e >> 3);
            int r = r0 + sgCol + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
    }

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmQKVQ8} on SIMD-group matrices; Q8_0 weights
     * are dequantized to {@code float} as they are staged. {@code dim} must be a multiple of 32.
     * Grid: {@code ((qDim + 2 * kvDim) / 64) * ceil(batch / 64)} workgroups of {@value #THREADS}
     * threads.
     */
    // @formatter:on
    public static void batchedGemmQKVQ8(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int dim,
            int qDim,
            int kvDim,
            int batch) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        float[] ws = context.allocateFloatLocalArray(64 * LDS);
        float[] cs = context.allocateFloatLocalArray(8 * 64);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 16;
        int rowTiles = ((qDim + 2 * kvDim) + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        int which = r0 < qDim ? 0 : (r0 < qDim + kvDim ? 1 : 2);
        int local0 = which == 0 ? r0 : (which == 1 ? r0 - qDim : r0 - qDim - kvDim);
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        int wrow = tid / 4;
        int wk0 = (tid % 4) * 8;
        float px0;
        float px1;
        float px2;
        float px3;
        float px4;
        float px5;
        float px6;
        float px7;
        float pw0;
        float pw1;
        float pw2;
        float pw3;
        float pw4;
        float pw5;
        float pw6;
        float pw7;
        float psc;
        int qb;
        px0 = x.get(TornadoMath.min(b0 + rr0 + 0, batch - 1) * dim + 0 + kk0);
        px1 = x.get(TornadoMath.min(b0 + rr0 + 8, batch - 1) * dim + 0 + kk0);
        px2 = x.get(TornadoMath.min(b0 + rr0 + 16, batch - 1) * dim + 0 + kk0);
        px3 = x.get(TornadoMath.min(b0 + rr0 + 24, batch - 1) * dim + 0 + kk0);
        px4 = x.get(TornadoMath.min(b0 + rr0 + 32, batch - 1) * dim + 0 + kk0);
        px5 = x.get(TornadoMath.min(b0 + rr0 + 40, batch - 1) * dim + 0 + kk0);
        px6 = x.get(TornadoMath.min(b0 + rr0 + 48, batch - 1) * dim + 0 + kk0);
        px7 = x.get(TornadoMath.min(b0 + rr0 + 56, batch - 1) * dim + 0 + kk0);
        if (which == 0) {
            qb = ((local0 + wrow) * (dim >> 5) + ((0) >> 5)) * 34;
            psc = wq.getHalfFloat(qb).getFloat32();
            pw0 = (float) wq.get(qb + 2 + wk0 + 0);
            pw1 = (float) wq.get(qb + 2 + wk0 + 1);
            pw2 = (float) wq.get(qb + 2 + wk0 + 2);
            pw3 = (float) wq.get(qb + 2 + wk0 + 3);
            pw4 = (float) wq.get(qb + 2 + wk0 + 4);
            pw5 = (float) wq.get(qb + 2 + wk0 + 5);
            pw6 = (float) wq.get(qb + 2 + wk0 + 6);
            pw7 = (float) wq.get(qb + 2 + wk0 + 7);
        } else if (which == 1) {
            qb = ((local0 + wrow) * (dim >> 5) + ((0) >> 5)) * 34;
            psc = wk.getHalfFloat(qb).getFloat32();
            pw0 = (float) wk.get(qb + 2 + wk0 + 0);
            pw1 = (float) wk.get(qb + 2 + wk0 + 1);
            pw2 = (float) wk.get(qb + 2 + wk0 + 2);
            pw3 = (float) wk.get(qb + 2 + wk0 + 3);
            pw4 = (float) wk.get(qb + 2 + wk0 + 4);
            pw5 = (float) wk.get(qb + 2 + wk0 + 5);
            pw6 = (float) wk.get(qb + 2 + wk0 + 6);
            pw7 = (float) wk.get(qb + 2 + wk0 + 7);
        } else {
            qb = ((local0 + wrow) * (dim >> 5) + ((0) >> 5)) * 34;
            psc = wv.getHalfFloat(qb).getFloat32();
            pw0 = (float) wv.get(qb + 2 + wk0 + 0);
            pw1 = (float) wv.get(qb + 2 + wk0 + 1);
            pw2 = (float) wv.get(qb + 2 + wk0 + 2);
            pw3 = (float) wv.get(qb + 2 + wk0 + 3);
            pw4 = (float) wv.get(qb + 2 + wk0 + 4);
            pw5 = (float) wv.get(qb + 2 + wk0 + 5);
            pw6 = (float) wv.get(qb + 2 + wk0 + 6);
            pw7 = (float) wv.get(qb + 2 + wk0 + 7);
        }
        for (int k0 = 0; k0 < dim; k0 += 32) {
            xs[(rr0 + 0) * LDS + kk0] = px0;
            xs[(rr0 + 8) * LDS + kk0] = px1;
            xs[(rr0 + 16) * LDS + kk0] = px2;
            xs[(rr0 + 24) * LDS + kk0] = px3;
            xs[(rr0 + 32) * LDS + kk0] = px4;
            xs[(rr0 + 40) * LDS + kk0] = px5;
            xs[(rr0 + 48) * LDS + kk0] = px6;
            xs[(rr0 + 56) * LDS + kk0] = px7;
            ws[wrow * LDS + wk0 + 0] = pw0 * psc;
            ws[wrow * LDS + wk0 + 1] = pw1 * psc;
            ws[wrow * LDS + wk0 + 2] = pw2 * psc;
            ws[wrow * LDS + wk0 + 3] = pw3 * psc;
            ws[wrow * LDS + wk0 + 4] = pw4 * psc;
            ws[wrow * LDS + wk0 + 5] = pw5 * psc;
            ws[wrow * LDS + wk0 + 6] = pw6 * psc;
            ws[wrow * LDS + wk0 + 7] = pw7 * psc;
            context.localBarrier();
            if (k0 + 32 < dim) {
                px0 = x.get(TornadoMath.min(b0 + rr0 + 0, batch - 1) * dim + k0 + 32 + kk0);
                px1 = x.get(TornadoMath.min(b0 + rr0 + 8, batch - 1) * dim + k0 + 32 + kk0);
                px2 = x.get(TornadoMath.min(b0 + rr0 + 16, batch - 1) * dim + k0 + 32 + kk0);
                px3 = x.get(TornadoMath.min(b0 + rr0 + 24, batch - 1) * dim + k0 + 32 + kk0);
                px4 = x.get(TornadoMath.min(b0 + rr0 + 32, batch - 1) * dim + k0 + 32 + kk0);
                px5 = x.get(TornadoMath.min(b0 + rr0 + 40, batch - 1) * dim + k0 + 32 + kk0);
                px6 = x.get(TornadoMath.min(b0 + rr0 + 48, batch - 1) * dim + k0 + 32 + kk0);
                px7 = x.get(TornadoMath.min(b0 + rr0 + 56, batch - 1) * dim + k0 + 32 + kk0);
                if (which == 0) {
                    qb = ((local0 + wrow) * (dim >> 5) + ((k0 + 32) >> 5)) * 34;
                    psc = wq.getHalfFloat(qb).getFloat32();
                    pw0 = (float) wq.get(qb + 2 + wk0 + 0);
                    pw1 = (float) wq.get(qb + 2 + wk0 + 1);
                    pw2 = (float) wq.get(qb + 2 + wk0 + 2);
                    pw3 = (float) wq.get(qb + 2 + wk0 + 3);
                    pw4 = (float) wq.get(qb + 2 + wk0 + 4);
                    pw5 = (float) wq.get(qb + 2 + wk0 + 5);
                    pw6 = (float) wq.get(qb + 2 + wk0 + 6);
                    pw7 = (float) wq.get(qb + 2 + wk0 + 7);
                } else if (which == 1) {
                    qb = ((local0 + wrow) * (dim >> 5) + ((k0 + 32) >> 5)) * 34;
                    psc = wk.getHalfFloat(qb).getFloat32();
                    pw0 = (float) wk.get(qb + 2 + wk0 + 0);
                    pw1 = (float) wk.get(qb + 2 + wk0 + 1);
                    pw2 = (float) wk.get(qb + 2 + wk0 + 2);
                    pw3 = (float) wk.get(qb + 2 + wk0 + 3);
                    pw4 = (float) wk.get(qb + 2 + wk0 + 4);
                    pw5 = (float) wk.get(qb + 2 + wk0 + 5);
                    pw6 = (float) wk.get(qb + 2 + wk0 + 6);
                    pw7 = (float) wk.get(qb + 2 + wk0 + 7);
                } else {
                    qb = ((local0 + wrow) * (dim >> 5) + ((k0 + 32) >> 5)) * 34;
                    psc = wv.getHalfFloat(qb).getFloat32();
                    pw0 = (float) wv.get(qb + 2 + wk0 + 0);
                    pw1 = (float) wv.get(qb + 2 + wk0 + 1);
                    pw2 = (float) wv.get(qb + 2 + wk0 + 2);
                    pw3 = (float) wv.get(qb + 2 + wk0 + 3);
                    pw4 = (float) wv.get(qb + 2 + wk0 + 4);
                    pw5 = (float) wv.get(qb + 2 + wk0 + 5);
                    pw6 = (float) wv.get(qb + 2 + wk0 + 6);
                    pw7 = (float) wv.get(qb + 2 + wk0 + 7);
                }
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Float cw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Float cw1 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 8) * LDS + kk, LDS);
                c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
            }
            context.localBarrier();
        }
        if (b0 + 64 <= batch) {
            if (which == 0) {
                context.simdgroupMatrixStore(
                        c00, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c01, q, (b0 + sgRow + 0) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c10, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c11, q, (b0 + sgRow + 8) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c20, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c21, q, (b0 + sgRow + 16) * qDim + local0 + sgCol + 8, qDim);
                context.simdgroupMatrixStore(
                        c30, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 0, qDim);
                context.simdgroupMatrixStore(
                        c31, q, (b0 + sgRow + 24) * qDim + local0 + sgCol + 8, qDim);
            } else if (which == 1) {
                context.simdgroupMatrixStore(
                        c00, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, k, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, k, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, k, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, k, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            } else {
                context.simdgroupMatrixStore(
                        c00, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c01, v, (b0 + sgRow + 0) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c10, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c11, v, (b0 + sgRow + 8) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c20, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c21, v, (b0 + sgRow + 16) * kvDim + local0 + sgCol + 8, kvDim);
                context.simdgroupMatrixStore(
                        c30, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 0, kvDim);
                context.simdgroupMatrixStore(
                        c31, v, (b0 + sgRow + 24) * kvDim + local0 + sgCol + 8, kvDim);
            }
        } else {
            context.simdgroupMatrixStore(c00, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch) {
                    float val = cs[(sg << 6) + e];
                    int rl = r - r0 + local0;
                    if (which == 0) {
                        q.set(t * qDim + rl, val);
                    } else if (which == 1) {
                        k.set(t * kvDim + rl, val);
                    } else {
                        v.set(t * kvDim + rl, val);
                    }
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmQ8WithResidual} on SIMD-group matrices; Q8_0
     * weights are dequantized to {@code float} as they are staged. {@code n} must be a multiple of
     * 32. Grid: {@code ceil(d / 64) * ceil(batch / 64)} workgroups of {@value #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmQ8WithResidual(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray outputBatch,
            ByteArray w,
            int n,
            int d,
            int batch) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        float[] ws = context.allocateFloatLocalArray(64 * LDS);
        float[] cs = context.allocateFloatLocalArray(8 * 64);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 16;
        int rowTiles = (d + 63) >> 6;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 6;
        int b0 = (grp / rowTiles) << 6;
        boolean full = b0 + 64 <= batch && r0 + 64 <= d;
        Matrix8x8Float c00 = context.simdgroupMatrixZero();
        Matrix8x8Float c01 = context.simdgroupMatrixZero();
        Matrix8x8Float c10 = context.simdgroupMatrixZero();
        Matrix8x8Float c11 = context.simdgroupMatrixZero();
        Matrix8x8Float c20 = context.simdgroupMatrixZero();
        Matrix8x8Float c21 = context.simdgroupMatrixZero();
        Matrix8x8Float c30 = context.simdgroupMatrixZero();
        Matrix8x8Float c31 = context.simdgroupMatrixZero();
        if (full) {
            c00 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            c01 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            c10 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            c11 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            c20 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            c21 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            c30 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            c31 =
                    context.simdgroupMatrixLoad(
                            outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        }
        int wrow = tid / 4;
        int wk0 = (tid % 4) * 8;
        float px0;
        float px1;
        float px2;
        float px3;
        float px4;
        float px5;
        float px6;
        float px7;
        float pw0;
        float pw1;
        float pw2;
        float pw3;
        float pw4;
        float pw5;
        float pw6;
        float pw7;
        float psc;
        int qb;
        px0 = inputBatch.get(TornadoMath.min(b0 + rr0 + 0, batch - 1) * n + 0 + kk0);
        px1 = inputBatch.get(TornadoMath.min(b0 + rr0 + 8, batch - 1) * n + 0 + kk0);
        px2 = inputBatch.get(TornadoMath.min(b0 + rr0 + 16, batch - 1) * n + 0 + kk0);
        px3 = inputBatch.get(TornadoMath.min(b0 + rr0 + 24, batch - 1) * n + 0 + kk0);
        px4 = inputBatch.get(TornadoMath.min(b0 + rr0 + 32, batch - 1) * n + 0 + kk0);
        px5 = inputBatch.get(TornadoMath.min(b0 + rr0 + 40, batch - 1) * n + 0 + kk0);
        px6 = inputBatch.get(TornadoMath.min(b0 + rr0 + 48, batch - 1) * n + 0 + kk0);
        px7 = inputBatch.get(TornadoMath.min(b0 + rr0 + 56, batch - 1) * n + 0 + kk0);
        qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((0) >> 5)) * 34;
        psc = w.getHalfFloat(qb).getFloat32();
        pw0 = (float) w.get(qb + 2 + wk0 + 0);
        pw1 = (float) w.get(qb + 2 + wk0 + 1);
        pw2 = (float) w.get(qb + 2 + wk0 + 2);
        pw3 = (float) w.get(qb + 2 + wk0 + 3);
        pw4 = (float) w.get(qb + 2 + wk0 + 4);
        pw5 = (float) w.get(qb + 2 + wk0 + 5);
        pw6 = (float) w.get(qb + 2 + wk0 + 6);
        pw7 = (float) w.get(qb + 2 + wk0 + 7);
        for (int k0 = 0; k0 < n; k0 += 32) {
            xs[(rr0 + 0) * LDS + kk0] = px0;
            xs[(rr0 + 8) * LDS + kk0] = px1;
            xs[(rr0 + 16) * LDS + kk0] = px2;
            xs[(rr0 + 24) * LDS + kk0] = px3;
            xs[(rr0 + 32) * LDS + kk0] = px4;
            xs[(rr0 + 40) * LDS + kk0] = px5;
            xs[(rr0 + 48) * LDS + kk0] = px6;
            xs[(rr0 + 56) * LDS + kk0] = px7;
            ws[wrow * LDS + wk0 + 0] = pw0 * psc;
            ws[wrow * LDS + wk0 + 1] = pw1 * psc;
            ws[wrow * LDS + wk0 + 2] = pw2 * psc;
            ws[wrow * LDS + wk0 + 3] = pw3 * psc;
            ws[wrow * LDS + wk0 + 4] = pw4 * psc;
            ws[wrow * LDS + wk0 + 5] = pw5 * psc;
            ws[wrow * LDS + wk0 + 6] = pw6 * psc;
            ws[wrow * LDS + wk0 + 7] = pw7 * psc;
            context.localBarrier();
            if (k0 + 32 < n) {
                px0 = inputBatch.get(TornadoMath.min(b0 + rr0 + 0, batch - 1) * n + k0 + 32 + kk0);
                px1 = inputBatch.get(TornadoMath.min(b0 + rr0 + 8, batch - 1) * n + k0 + 32 + kk0);
                px2 = inputBatch.get(TornadoMath.min(b0 + rr0 + 16, batch - 1) * n + k0 + 32 + kk0);
                px3 = inputBatch.get(TornadoMath.min(b0 + rr0 + 24, batch - 1) * n + k0 + 32 + kk0);
                px4 = inputBatch.get(TornadoMath.min(b0 + rr0 + 32, batch - 1) * n + k0 + 32 + kk0);
                px5 = inputBatch.get(TornadoMath.min(b0 + rr0 + 40, batch - 1) * n + k0 + 32 + kk0);
                px6 = inputBatch.get(TornadoMath.min(b0 + rr0 + 48, batch - 1) * n + k0 + 32 + kk0);
                px7 = inputBatch.get(TornadoMath.min(b0 + rr0 + 56, batch - 1) * n + k0 + 32 + kk0);
                qb = ((TornadoMath.min(r0 + wrow, d - 1)) * (n >> 5) + ((k0 + 32) >> 5)) * 34;
                psc = w.getHalfFloat(qb).getFloat32();
                pw0 = (float) w.get(qb + 2 + wk0 + 0);
                pw1 = (float) w.get(qb + 2 + wk0 + 1);
                pw2 = (float) w.get(qb + 2 + wk0 + 2);
                pw3 = (float) w.get(qb + 2 + wk0 + 3);
                pw4 = (float) w.get(qb + 2 + wk0 + 4);
                pw5 = (float) w.get(qb + 2 + wk0 + 5);
                pw6 = (float) w.get(qb + 2 + wk0 + 6);
                pw7 = (float) w.get(qb + 2 + wk0 + 7);
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Float cw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                Matrix8x8Float cw1 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 8) * LDS + kk, LDS);
                c00 = context.simdgroupMatrixMultiplyAccumulate(x0, cw0, c00);
                c01 = context.simdgroupMatrixMultiplyAccumulate(x0, cw1, c01);
                c10 = context.simdgroupMatrixMultiplyAccumulate(x1, cw0, c10);
                c11 = context.simdgroupMatrixMultiplyAccumulate(x1, cw1, c11);
                c20 = context.simdgroupMatrixMultiplyAccumulate(x2, cw0, c20);
                c21 = context.simdgroupMatrixMultiplyAccumulate(x2, cw1, c21);
                c30 = context.simdgroupMatrixMultiplyAccumulate(x3, cw0, c30);
                c31 = context.simdgroupMatrixMultiplyAccumulate(x3, cw1, c31);
            }
            context.localBarrier();
        }
        if (full) {
            context.simdgroupMatrixStore(
                    c00, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c01, outputBatch, (b0 + sgRow + 0) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c10, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c11, outputBatch, (b0 + sgRow + 8) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c20, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c21, outputBatch, (b0 + sgRow + 16) * d + r0 + sgCol + 8, d);
            context.simdgroupMatrixStore(
                    c30, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 0, d);
            context.simdgroupMatrixStore(
                    c31, outputBatch, (b0 + sgRow + 24) * d + r0 + sgCol + 8, d);
        } else {
            context.simdgroupMatrixStore(c00, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c01, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 0 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c10, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c11, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 8 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c20, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c21, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 16 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c30, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 0 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
            context.simdgroupMatrixStore(c31, cs, sg << 6, 8);
            context.localBarrier();
            for (int e = lane; e < 64; e += 32) {
                int t = b0 + sgRow + 24 + (e >> 3);
                int r = r0 + sgCol + 8 + (e & 7);
                if (t < batch && r < d) {
                    outputBatch.set(t * d + r, outputBatch.get(t * d + r) + cs[(sg << 6) + e]);
                }
            }
            context.localBarrier();
        }
    }

    // @formatter:off
    /**
     * {@link TransformerBatchPrefillKernels#batchedGemmRmsNormFFNGateUpQ8} on SIMD-group matrices;
     * Q8_0 weights are dequantized to {@code float} as they are staged. {@code dim} must be a
     * multiple of 32. Grid: {@code ceil(hiddenDim / 32) * ceil(batch / 64)} workgroups of {@value
     * #THREADS} threads.
     */
    // @formatter:on
    public static void batchedGemmRmsNormFFNGateUpQ8(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            FloatArray rmsWeights,
            FloatArray scaleBatch,
            ByteArray w1,
            ByteArray w3,
            int dim,
            int hiddenDim,
            int batch) {
        float[] xs = context.allocateFloatLocalArray(64 * LDS);
        float[] ws = context.allocateFloatLocalArray(64 * LDS);
        float[] cs = context.allocateFloatLocalArray(8 * 128);
        int tid = context.localIdx;
        int sg = tid >> 5;
        int lane = tid & 31;
        int kk0 = tid & 31;
        int rr0 = tid >> 5;
        int sgRow = (sg >> 2) << 5;
        int sgCol = (sg & 3) * 8;
        int rowTiles = (hiddenDim + 31) >> 5;
        int grp = context.groupIdx;
        int r0 = (grp % rowTiles) << 5;
        int b0 = (grp / rowTiles) << 6;
        Matrix8x8Float g00 = context.simdgroupMatrixZero();
        Matrix8x8Float g10 = context.simdgroupMatrixZero();
        Matrix8x8Float g20 = context.simdgroupMatrixZero();
        Matrix8x8Float g30 = context.simdgroupMatrixZero();
        Matrix8x8Float u00 = context.simdgroupMatrixZero();
        Matrix8x8Float u10 = context.simdgroupMatrixZero();
        Matrix8x8Float u20 = context.simdgroupMatrixZero();
        Matrix8x8Float u30 = context.simdgroupMatrixZero();
        int wrow = tid / 4;
        int wk0 = (tid % 4) * 8;
        float px0;
        float px1;
        float px2;
        float px3;
        float px4;
        float px5;
        float px6;
        float px7;
        float pw0;
        float pw1;
        float pw2;
        float pw3;
        float pw4;
        float pw5;
        float pw6;
        float pw7;
        float psc;
        int qb;
        px0 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 0, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 0, batch - 1) * dim + 0 + kk0);
        px1 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 8, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 8, batch - 1) * dim + 0 + kk0);
        px2 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 16, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 16, batch - 1) * dim + 0 + kk0);
        px3 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 24, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 24, batch - 1) * dim + 0 + kk0);
        px4 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 32, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 32, batch - 1) * dim + 0 + kk0);
        px5 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 40, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 40, batch - 1) * dim + 0 + kk0);
        px6 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 48, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 48, batch - 1) * dim + 0 + kk0);
        px7 =
                rmsWeights.get(0 + kk0)
                        * scaleBatch.get(TornadoMath.min(b0 + rr0 + 56, batch - 1))
                        * x.get(TornadoMath.min(b0 + rr0 + 56, batch - 1) * dim + 0 + kk0);
        if (wrow < 32) {
            qb = ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 5) + ((0) >> 5)) * 34;
            psc = w1.getHalfFloat(qb).getFloat32();
            pw0 = (float) w1.get(qb + 2 + wk0 + 0);
            pw1 = (float) w1.get(qb + 2 + wk0 + 1);
            pw2 = (float) w1.get(qb + 2 + wk0 + 2);
            pw3 = (float) w1.get(qb + 2 + wk0 + 3);
            pw4 = (float) w1.get(qb + 2 + wk0 + 4);
            pw5 = (float) w1.get(qb + 2 + wk0 + 5);
            pw6 = (float) w1.get(qb + 2 + wk0 + 6);
            pw7 = (float) w1.get(qb + 2 + wk0 + 7);
        } else {
            qb = ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 5) + ((0) >> 5)) * 34;
            psc = w3.getHalfFloat(qb).getFloat32();
            pw0 = (float) w3.get(qb + 2 + wk0 + 0);
            pw1 = (float) w3.get(qb + 2 + wk0 + 1);
            pw2 = (float) w3.get(qb + 2 + wk0 + 2);
            pw3 = (float) w3.get(qb + 2 + wk0 + 3);
            pw4 = (float) w3.get(qb + 2 + wk0 + 4);
            pw5 = (float) w3.get(qb + 2 + wk0 + 5);
            pw6 = (float) w3.get(qb + 2 + wk0 + 6);
            pw7 = (float) w3.get(qb + 2 + wk0 + 7);
        }
        for (int k0 = 0; k0 < dim; k0 += 32) {
            xs[(rr0 + 0) * LDS + kk0] = px0;
            xs[(rr0 + 8) * LDS + kk0] = px1;
            xs[(rr0 + 16) * LDS + kk0] = px2;
            xs[(rr0 + 24) * LDS + kk0] = px3;
            xs[(rr0 + 32) * LDS + kk0] = px4;
            xs[(rr0 + 40) * LDS + kk0] = px5;
            xs[(rr0 + 48) * LDS + kk0] = px6;
            xs[(rr0 + 56) * LDS + kk0] = px7;
            ws[wrow * LDS + wk0 + 0] = pw0 * psc;
            ws[wrow * LDS + wk0 + 1] = pw1 * psc;
            ws[wrow * LDS + wk0 + 2] = pw2 * psc;
            ws[wrow * LDS + wk0 + 3] = pw3 * psc;
            ws[wrow * LDS + wk0 + 4] = pw4 * psc;
            ws[wrow * LDS + wk0 + 5] = pw5 * psc;
            ws[wrow * LDS + wk0 + 6] = pw6 * psc;
            ws[wrow * LDS + wk0 + 7] = pw7 * psc;
            context.localBarrier();
            if (k0 + 32 < dim) {
                px0 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 0, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 0, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px1 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 8, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 8, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px2 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 16, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 16, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px3 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 24, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 24, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px4 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 32, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 32, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px5 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 40, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 40, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px6 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 48, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 48, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                px7 =
                        rmsWeights.get(k0 + 32 + kk0)
                                * scaleBatch.get(TornadoMath.min(b0 + rr0 + 56, batch - 1))
                                * x.get(
                                        TornadoMath.min(b0 + rr0 + 56, batch - 1) * dim
                                                + k0
                                                + 32
                                                + kk0);
                if (wrow < 32) {
                    qb =
                            ((TornadoMath.min(r0 + wrow, hiddenDim - 1)) * (dim >> 5)
                                            + ((k0 + 32) >> 5))
                                    * 34;
                    psc = w1.getHalfFloat(qb).getFloat32();
                    pw0 = (float) w1.get(qb + 2 + wk0 + 0);
                    pw1 = (float) w1.get(qb + 2 + wk0 + 1);
                    pw2 = (float) w1.get(qb + 2 + wk0 + 2);
                    pw3 = (float) w1.get(qb + 2 + wk0 + 3);
                    pw4 = (float) w1.get(qb + 2 + wk0 + 4);
                    pw5 = (float) w1.get(qb + 2 + wk0 + 5);
                    pw6 = (float) w1.get(qb + 2 + wk0 + 6);
                    pw7 = (float) w1.get(qb + 2 + wk0 + 7);
                } else {
                    qb =
                            ((TornadoMath.min(r0 + wrow - 32, hiddenDim - 1)) * (dim >> 5)
                                            + ((k0 + 32) >> 5))
                                    * 34;
                    psc = w3.getHalfFloat(qb).getFloat32();
                    pw0 = (float) w3.get(qb + 2 + wk0 + 0);
                    pw1 = (float) w3.get(qb + 2 + wk0 + 1);
                    pw2 = (float) w3.get(qb + 2 + wk0 + 2);
                    pw3 = (float) w3.get(qb + 2 + wk0 + 3);
                    pw4 = (float) w3.get(qb + 2 + wk0 + 4);
                    pw5 = (float) w3.get(qb + 2 + wk0 + 5);
                    pw6 = (float) w3.get(qb + 2 + wk0 + 6);
                    pw7 = (float) w3.get(qb + 2 + wk0 + 7);
                }
            }
            for (int kk = 0; kk < 32; kk += 8) {
                Matrix8x8Float x0 = context.simdgroupMatrixLoad(xs, (sgRow + 0) * LDS + kk, LDS);
                Matrix8x8Float x1 = context.simdgroupMatrixLoad(xs, (sgRow + 8) * LDS + kk, LDS);
                Matrix8x8Float x2 = context.simdgroupMatrixLoad(xs, (sgRow + 16) * LDS + kk, LDS);
                Matrix8x8Float x3 = context.simdgroupMatrixLoad(xs, (sgRow + 24) * LDS + kk, LDS);
                Matrix8x8Float gw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (sgCol + 0) * LDS + kk, LDS);
                g00 = context.simdgroupMatrixMultiplyAccumulate(x0, gw0, g00);
                g10 = context.simdgroupMatrixMultiplyAccumulate(x1, gw0, g10);
                g20 = context.simdgroupMatrixMultiplyAccumulate(x2, gw0, g20);
                g30 = context.simdgroupMatrixMultiplyAccumulate(x3, gw0, g30);
                Matrix8x8Float uw0 =
                        context.simdgroupMatrixLoadTransposed(ws, (32 + sgCol + 0) * LDS + kk, LDS);
                u00 = context.simdgroupMatrixMultiplyAccumulate(x0, uw0, u00);
                u10 = context.simdgroupMatrixMultiplyAccumulate(x1, uw0, u10);
                u20 = context.simdgroupMatrixMultiplyAccumulate(x2, uw0, u20);
                u30 = context.simdgroupMatrixMultiplyAccumulate(x3, uw0, u30);
            }
            context.localBarrier();
        }
        context.simdgroupMatrixStore(g00, cs, sg << 7, 8);
        context.simdgroupMatrixStore(u00, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = b0 + sgRow + 0 + (e >> 3);
            int r = r0 + sgCol + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(g10, cs, sg << 7, 8);
        context.simdgroupMatrixStore(u10, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = b0 + sgRow + 8 + (e >> 3);
            int r = r0 + sgCol + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(g20, cs, sg << 7, 8);
        context.simdgroupMatrixStore(u20, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = b0 + sgRow + 16 + (e >> 3);
            int r = r0 + sgCol + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
        context.simdgroupMatrixStore(g30, cs, sg << 7, 8);
        context.simdgroupMatrixStore(u30, cs, (sg << 7) + 64, 8);
        context.localBarrier();
        for (int e = lane; e < 64; e += 32) {
            int t = b0 + sgRow + 24 + (e >> 3);
            int r = r0 + sgCol + 0 + (e & 7);
            if (t < batch && r < hiddenDim) {
                float gv = cs[(sg << 7) + e];
                hb.set(
                        t * hiddenDim + r,
                        (gv / (1.0f + TornadoMath.exp(-gv))) * cs[(sg << 7) + 64 + e]);
            }
        }
        context.localBarrier();
    }
}
