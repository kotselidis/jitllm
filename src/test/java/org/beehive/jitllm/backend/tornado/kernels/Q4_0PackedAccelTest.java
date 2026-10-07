package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Random;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Packed Q4_0 weights ({@link Qwen35Int8Kernels#packQ4_0Tiles}): the prefill GEMM against {@link
 * Qwen35Int8Kernels#gemmInt8Q4_0} bit for bit, and the decode kernels against the Q4_0 ones on the
 * same activation within FP32 rounding.
 */
public class Q4_0PackedAccelTest {

    private static final int MATVEC_LOCAL = 128;

    private static ByteArray randomQ4_0(int rows, int cols, long seed) {
        Random rng = new Random(seed);
        byte[] raw = new byte[rows * (cols / 32) * 18];
        rng.nextBytes(raw);
        for (int b = 0; b < rows * (cols / 32); b++) {
            int bits = 0x2000 | rng.nextInt(0x400); // a positive normal FP16 scale
            raw[b * 18] = (byte) bits;
            raw[b * 18 + 1] = (byte) (bits >> 8);
        }
        return ByteArray.fromArray(raw);
    }

    private static WorkerGrid grid(int global, int local) {
        WorkerGrid g = new WorkerGrid1D(global);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    private static WorkerGrid gemmGrid(int m, int n, int splits) {
        WorkerGrid g = new WorkerGrid2D((m / 128) * Qwen35Int8Kernels.Q8_GEMM_THREADS, n / 128 * splits);
        g.setLocalWork(Qwen35Int8Kernels.Q8_GEMM_THREADS, 1, 1);
        return g;
    }

    private static IntArray rowsOf(int m) {
        IntArray rows = new IntArray(3);
        rows.set(1, m);
        return rows;
    }

    private static double worst(FloatArray a, FloatArray b) {
        double scale = 1e-6;
        double max = 0;
        for (int i = 0; i < a.getSize(); i++) {
            scale = Math.max(scale, Math.abs(a.get(i)));
        }
        for (int i = 0; i < a.getSize(); i++) {
            max = Math.max(max, Math.abs(a.get(i) - b.get(i)) / scale);
        }
        return max;
    }

    /** Ten 64-k rounds, unsplit and split three ways: the stored products and partial sums equal. */
    @Test
    public void thePackedGemmMatchesTheGemm() throws Exception {
        assumeTrue("no int8 tensor cores", TensorCoreSupport.isInt8MmaCapable());
        int m = 256, n = 384, k = 640;
        ByteArray w = randomQ4_0(n, k, 1);
        ByteArray packed = Qwen35Int8Kernels.packQ4_0Tiles(w, n, k);
        Random rng = new Random(2);
        FloatArray a = new FloatArray(m * k);
        for (int i = 0; i < m * k; i++) {
            a.set(i, (float) rng.nextGaussian());
        }
        ByteArray q8 = new ByteArray(m * k);
        FloatArray dA = new FloatArray(m * k / 32);
        IntArray rows = rowsOf(m);
        for (int splits : new int[] {1, 3}) {
            FloatArray reference = new FloatArray(splits * m * n);
            FloatArray got = new FloatArray(splits * m * n);
            reference.init(Float.NaN);
            got.init(Float.NaN);
            TaskGraph g =
                    new TaskGraph("q4g" + splits)
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, w, packed, q8, dA, reference, got, rows)
                            .task("q", Qwen35Int8Kernels::quantizeActivationsQ8Warp, new KernelContext(), a, q8, dA, k)
                            .task("g", Qwen35Int8Kernels::gemmInt8Q4_0, new KernelContext(), q8, dA, w, reference, reference, m, n, k,
                                    Qwen35Int8Kernels.EPILOGUE_STORE, reference, splits, rows, n, 0)
                            .task("p", Qwen35Int8Kernels::gemmInt8Q4_0Packed, new KernelContext(), q8, dA, packed, got, got, m, n, k,
                                    Qwen35Int8Kernels.EPILOGUE_STORE, got, splits, rows, n, 0)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, reference, got);
            GridScheduler s = new GridScheduler();
            s.addWorkerGrid("q4g" + splits + ".q", grid(m * k, 256));
            s.addWorkerGrid("q4g" + splits + ".g", gemmGrid(m, n, splits));
            s.addWorkerGrid("q4g" + splits + ".p", gemmGrid(m, n, splits));
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
                plan.withGridScheduler(s).execute();
            }
            for (int i = 0; i < splits * m * n; i++) {
                assertEquals("splits " + splits + ", element " + i, reference.get(i), got.get(i), 0.0f);
            }
        }
    }

    /** Random activation quants and their per-block sums (what quantizeActivationQ8Blocks writes). */
    private static IntArray[] quantizedActivation(int n, long seed) {
        Random rng = new Random(seed);
        IntArray quants = new IntArray(n / 4);
        IntArray sums = new IntArray(n / 32);
        for (int i = 0; i < n / 4; i++) {
            int v = rng.nextInt();
            quants.set(i, v);
            sums.set(i / 8, sums.get(i / 8) + (byte) v + (byte) (v >> 8) + (byte) (v >> 16) + (byte) (v >> 24));
        }
        return new IntArray[] {quants, sums};
    }

    private static FloatArray scales(int n, long seed) {
        Random rng = new Random(seed);
        FloatArray s = new FloatArray(n / 32);
        for (int i = 0; i < n / 32; i++) {
            s.set(i, 0.01f + rng.nextFloat() * 0.02f);
        }
        return s;
    }

    private static WorkerGrid packedGrid(int d) {
        return grid(d / 16 * TransformerComputeKernelsQ4_0Packed.LOCAL, TransformerComputeKernelsQ4_0Packed.LOCAL);
    }

    @Test
    public void theDecodeProductsMatchTheQ4_0Kernels() throws Exception {
        int[][] shapes = {{12288, 5120}, {5120, 17408}, {1024, 5120}};
        for (int[] shape : shapes) {
            int d = shape[0], n = shape[1];
            ByteArray w = randomQ4_0(d, n, d);
            ByteArray packed = Qwen35Int8Kernels.packQ4_0Tiles(w, d, n);
            IntArray[] qs = quantizedActivation(n, d + 1);
            IntArray xq = qs[0];
            IntArray xsum = qs[1];
            FloatArray xs = scales(n, d + 2);
            Random rng = new Random(d + 3);
            FloatArray xf = new FloatArray(n);
            for (int i = 0; i < n; i++) {
                xf.set(i, (float) rng.nextGaussian());
            }
            FloatArray ref = new FloatArray(d), got = new FloatArray(d);
            FloatArray refRes = new FloatArray(d), gotRes = new FloatArray(d);
            FloatArray refF = new FloatArray(d), gotF = new FloatArray(d);
            for (int i = 0; i < d; i++) {
                refRes.set(i, 0.5f * i);
                gotRes.set(i, 0.5f * i);
            }
            String id = "q4d" + d;
            TaskGraph g =
                    new TaskGraph(id)
                            .transferToDevice(DataTransferMode.EVERY_EXECUTION, w, packed, xq, xs, xsum, xf, ref, got, refRes, gotRes, refF, gotF)
                            .task("r", TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0DP4A, new KernelContext(), xq, xs, xsum, ref, w, n, d, MATVEC_LOCAL)
                            .task("p", TransformerComputeKernelsQ4_0Packed::matrixVectorQ4_0Packed, new KernelContext(), xq, xs, got, packed, n, d, 0)
                            .task("rr", TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0DP4A, new KernelContext(), xq, xs, xsum, refRes, w, n, d, MATVEC_LOCAL)
                            .task("pr", TransformerComputeKernelsQ4_0Packed::matrixVectorQ4_0Packed, new KernelContext(), xq, xs, gotRes, packed, n, d, 1)
                            .task("rf", TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0, new KernelContext(), xf, refF, w, n, d, MATVEC_LOCAL)
                            .task("pf", TransformerComputeKernelsQ4_0Packed::matrixVectorQ4_0PackedF32, new KernelContext(), xf, gotF, packed, n, d, 0)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, ref, got, refRes, gotRes, refF, gotF);
            GridScheduler s = new GridScheduler();
            for (String t : new String[] {"r", "rr", "rf"}) {
                s.addWorkerGrid(id + "." + t, grid(d * MATVEC_LOCAL, MATVEC_LOCAL));
            }
            for (String t : new String[] {"p", "pr", "pf"}) {
                s.addWorkerGrid(id + "." + t, packedGrid(d));
            }
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
                plan.withGridScheduler(s).execute();
            }
            assertTrue(d + "x" + n + " store: " + worst(ref, got), worst(ref, got) < 1e-5);
            assertTrue(d + "x" + n + " residual: " + worst(refRes, gotRes), worst(refRes, gotRes) < 1e-5);
            assertTrue(d + "x" + n + " fp32: " + worst(refF, gotF), worst(refF, gotF) < 1e-5);
        }
    }

    @Test
    public void theFusedGateUpMatchesTheQ4_0Kernel() throws Exception {
        int d = 17408, n = 5120;
        ByteArray w1 = randomQ4_0(d, n, 11);
        ByteArray w3 = randomQ4_0(d, n, 12);
        ByteArray p1 = Qwen35Int8Kernels.packQ4_0Tiles(w1, d, n);
        ByteArray p3 = Qwen35Int8Kernels.packQ4_0Tiles(w3, d, n);
        IntArray[] qs = quantizedActivation(n, 13);
        FloatArray xs = scales(n, 14);
        FloatArray ref = new FloatArray(d), got = new FloatArray(d);
        TaskGraph g =
                new TaskGraph("q4f")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, w1, w3, p1, p3, qs[0], qs[1], xs, ref, got)
                        .task("r", TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0DP4A, new KernelContext(), qs[0], xs, qs[1], ref, w1, w3, n, d, MATVEC_LOCAL)
                        .task("p", TransformerComputeKernelsQ4_0Packed::fusedFFNGateUpSiLUQ4_0Packed, new KernelContext(), qs[0], xs, got, p1, p3, n, d)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, ref, got);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("q4f.r", grid(d * MATVEC_LOCAL, MATVEC_LOCAL));
        s.addWorkerGrid("q4f.p", packedGrid(d));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        assertTrue("fused: " + worst(ref, got), worst(ref, got) < 1e-4);
    }
}
