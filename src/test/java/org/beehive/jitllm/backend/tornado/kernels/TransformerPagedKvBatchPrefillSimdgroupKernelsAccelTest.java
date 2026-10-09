package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.util.Random;
import org.beehive.jitllm.backend.tornado.scheduling.BatchPrefillGemmPolicy;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * {@link TransformerPagedKvBatchPrefillSimdgroupKernels#batchedFlashAttentionPagedHead128} against
 * a double-precision host reference of causal attention over a paged cache: shuffled physical
 * blocks, a batch that starts mid-sequence, a last query tile that is only partly active and a
 * padding tile, and grouped-query heads. Cache rows past the batch's last position and query rows
 * past the active ones hold NaN, so reading either into the result fails the test.
 */
// @formatter:on
public class TransformerPagedKvBatchPrefillSimdgroupKernelsAccelTest {

    private static final int HEAD = 128;
    private static final int BLOCK = 16;
    private static final float UNTOUCHED = -12345.5f;

    private static void check(int start, int active, int gridRows) {
        assumeTrue("SIMD-group matrices are Metal only", BatchPrefillGemmPolicy.tiled());
        int nHeads = 4;
        int kvMul = 2;
        int kvDim = (nHeads / kvMul) * HEAD;
        int dim = nHeads * HEAD;
        int layers = 2;
        int layerIndex = 1;
        int slot = 1;
        int blocksPerSlot = 8;
        int physBlocks = 2 * blocksPerSlot;
        int blockStride = layers * BLOCK * kvDim;
        int blockCfg = BLOCK | (blocksPerSlot << 16);
        int lastValid = start + active - 1;

        IntArray blockTable = new IntArray(2 * blocksPerSlot);
        for (int b = 0; b < blocksPerSlot; b++) {
            blockTable.set(slot * blocksPerSlot + b, (b * 5 + 3) % physBlocks);
            blockTable.set(b, (b * 5 + 4) % physBlocks);
        }
        Random rnd = new Random(31L * start + active);
        FloatArray keys = new FloatArray(physBlocks * blockStride);
        FloatArray values = new FloatArray(physBlocks * blockStride);
        keys.init(Float.NaN);
        values.init(Float.NaN);
        float[][] k = new float[lastValid + 1][kvDim];
        float[][] v = new float[lastValid + 1][kvDim];
        for (int p = 0; p <= lastValid; p++) {
            int base =
                    blockTable.get(slot * blocksPerSlot + p / BLOCK) * blockStride
                            + layerIndex * BLOCK * kvDim
                            + (p % BLOCK) * kvDim;
            for (int i = 0; i < kvDim; i++) {
                k[p][i] = rnd.nextFloat() - 0.5f;
                v[p][i] = rnd.nextFloat() - 0.5f;
                keys.set(base + i, k[p][i]);
                values.set(base + i, v[p][i]);
            }
        }
        FloatArray q = new FloatArray(gridRows * dim);
        q.init(Float.NaN);
        for (int t = 0; t < active; t++) {
            for (int i = 0; i < dim; i++) {
                q.set(t * dim + i, 2.0f * (rnd.nextFloat() - 0.5f));
            }
        }
        FloatArray out = new FloatArray(gridRows * dim);
        out.init(UNTOUCHED);
        IntArray holder = new IntArray(3);
        holder.set(0, start);
        holder.set(1, active);
        holder.set(2, slot);

        String name = "simdgroupAttention" + start + "x" + active;
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                holder,
                                q,
                                keys,
                                values,
                                out,
                                blockTable)
                        .task(
                                "t",
                                TransformerPagedKvBatchPrefillSimdgroupKernels
                                        ::batchedFlashAttentionPagedHead128,
                                new KernelContext(),
                                holder,
                                q,
                                keys,
                                values,
                                out,
                                nHeads,
                                HEAD,
                                kvDim,
                                kvMul,
                                layerIndex,
                                blockTable,
                                blockCfg,
                                blockStride,
                                dim)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        int tiles = (gridRows + 31) / 32;
        WorkerGrid1D worker =
                new WorkerGrid1D(
                        tiles * nHeads * TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS);
        worker.setLocalWork(TransformerPagedKvBatchPrefillSimdgroupKernels.THREADS, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(name + ".t", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }

        double scale = 1.0 / Math.sqrt(HEAD);
        for (int t = 0; t < gridRows; t++) {
            for (int h = 0; h < nHeads; h++) {
                if (t >= active) {
                    for (int d = 0; d < HEAD; d++) {
                        assertEquals(
                                "row " + t + " is past the batch and must stay untouched",
                                UNTOUCHED,
                                out.get(t * dim + h * HEAD + d),
                                0.0f);
                    }
                    continue;
                }
                int pos = start + t;
                int kvOff = (h / kvMul) * HEAD;
                double[] s = new double[pos + 1];
                double max = Double.NEGATIVE_INFINITY;
                for (int p = 0; p <= pos; p++) {
                    double dot = 0.0;
                    for (int d = 0; d < HEAD; d++) {
                        dot += (double) q.get(t * dim + h * HEAD + d) * k[p][kvOff + d];
                    }
                    s[p] = dot * scale;
                    max = Math.max(max, s[p]);
                }
                double sum = 0.0;
                for (int p = 0; p <= pos; p++) {
                    s[p] = Math.exp(s[p] - max);
                    sum += s[p];
                }
                for (int d = 0; d < HEAD; d++) {
                    double expected = 0.0;
                    for (int p = 0; p <= pos; p++) {
                        expected += s[p] * v[p][kvOff + d];
                    }
                    expected /= sum;
                    assertEquals(
                            name + " token " + t + " head " + h + " dim " + d,
                            expected,
                            out.get(t * dim + h * HEAD + d),
                            2e-5);
                }
            }
        }
    }

    @Test
    public void fromTheStartOfTheSequenceWithWholeTiles() {
        check(0, 64, 64);
    }

    @Test
    public void midSequenceWithAPartialAndAPaddingTile() {
        check(37, 45, 96);
    }

    @Test
    public void midSequenceWithASingleToken() {
        check(70, 1, 32);
    }
}
