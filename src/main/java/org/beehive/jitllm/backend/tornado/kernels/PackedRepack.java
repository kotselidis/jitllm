package org.beehive.jitllm.backend.tornado.kernels;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * Packing on the GPU at every load ({@code -Djitllm.packed.mode=gpu}), in place of the host repack
 * and its disk cache ({@code cache}, the default).
 *
 * <p>The loader keeps a packed weight's GGUF bytes on the host and records it here ({@link
 * #register}). The layer graph that uploads it names itself as its owner ({@link #owner}). Each
 * execution plan then gets one more graph ({@link #appendGraph}) that takes the plan's registered
 * weights from their owners with {@code consumeFromDevice}, copies each into one scratch buffer
 * and repacks it from there over the weight's own device buffer. The plan runs it once after the
 * warm-up has uploaded the weights ({@link #run}); a flag on the device makes any later run a
 * no-op, as the weights are packed only once.
 */
// @formatter:on
public final class PackedRepack {

    /** Whether packed weights are repacked on the GPU at load rather than read from the cache. */
    public static final boolean ENABLED = "gpu".equalsIgnoreCase(System.getProperty("jitllm.packed.mode", "cache"));

    private record Weight(ByteArray bytes, int rows, int cols, boolean q4) {
        int size() {
            return rows * (cols / 32) * (q4 ? 18 : 34);
        }
    }

    private static final Map<ByteArray, Weight> REGISTERED = new IdentityHashMap<>();

    /** Owner graph name to the registered weights it uploads, not yet given a repack graph. */
    private static final Map<String, List<Weight>> OWNED = new LinkedHashMap<>();

    /** Registered weights already given an owner: each is uploaded, and repacked, once. */
    private static final java.util.Set<ByteArray> ASSIGNED = java.util.Collections.newSetFromMap(new IdentityHashMap<>());

    private PackedRepack() {}

    private static ByteArray b(Object o) {
        return (ByteArray) o;
    }

    /** Records the GGUF weight {@code w} for repacking on the device, and as packed. */
    static synchronized ByteArray register(ByteArray w, int rows, int cols, boolean q4) {
        REGISTERED.put(w, new Weight(w, rows, cols, q4));
        return q4 ? PackedQ8_0.recordQ4(w) : PackedQ8_0.record(w);
    }

    /**
     * Names {@code graph} as the owner of whichever of {@code uploaded} are registered: called by
     * a layer graph right after it uploads its weights.
     */
    public static synchronized void owner(TaskGraph graph, Object... uploaded) {
        if (!ENABLED) {
            return;
        }
        for (Object o : uploaded) {
            Weight w = o instanceof ByteArray b ? REGISTERED.get(b) : null;
            if (w != null && ASSIGNED.add(b(o))) {
                OWNED.computeIfAbsent(graph.getTaskGraphName(), k -> new ArrayList<>()).add(w);
            }
        }
    }

    /**
     * Appends to {@code graphs} the repack graph for the weights owned since the last call, with
     * its grids in {@code scheduler}, and returns its index; -1 when there is nothing to repack.
     */
    public static synchronized int appendGraph(List<ImmutableTaskGraph> graphs, GridScheduler scheduler, String name) {
        if (!ENABLED || OWNED.isEmpty()) {
            return -1;
        }
        int scratchBytes = 0;
        for (List<Weight> ws : OWNED.values()) {
            for (Weight w : ws) {
                scratchBytes = Math.max(scratchBytes, w.size());
            }
        }
        ByteArray scratch = new ByteArray(scratchBytes);
        IntArray done = new IntArray(1);
        TaskGraph g = new TaskGraph(name);
        List<Object> all = new ArrayList<>();
        for (Map.Entry<String, List<Weight>> e : OWNED.entrySet()) {
            Object[] bytes = e.getValue().stream().map(Weight::bytes).toArray();
            g.consumeFromDevice(e.getKey(), bytes);
            all.addAll(List.of(bytes));
        }
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch, done);
        int i = 0;
        for (List<Weight> ws : OWNED.values()) {
            for (Weight w : ws) {
                String copy = "copy" + i;
                String repack = "repack" + i;
                g.task(copy, PackedRepackKernels::copyOnce, new KernelContext(), w.bytes(), scratch, w.size(), done);
                if (w.q4()) {
                    g.task(repack, PackedRepackKernels::repackQ4_0Once, new KernelContext(), scratch, w.bytes(), w.rows(), w.cols(), done);
                } else {
                    g.task(repack, PackedRepackKernels::repackQ8_0Once, new KernelContext(), scratch, w.bytes(), w.rows(), w.cols(), done);
                }
                int tiles = (w.rows() / 128) * (w.cols() / 64);
                scheduler.addWorkerGrid(name + "." + copy, grid((w.size() + 7) / 8));
                scheduler.addWorkerGrid(
                        name + "." + repack,
                        grid(tiles * (w.q4() ? PackedRepackKernels.Q4_0_LANES_PER_TILE : PackedRepackKernels.Q8_0_LANES_PER_TILE)));
                i++;
            }
        }
        g.task("markDone", PackedRepackKernels::markDone, new KernelContext(), done);
        scheduler.addWorkerGrid(name + ".markDone", grid(1));
        g.persistOnDevice(all.toArray());
        OWNED.clear();
        graphs.add(g.snapshot());
        return graphs.size() - 1;
    }

    /** Runs the repack graph {@code index} of {@code plan} once, when there is one. */
    public static void run(TornadoExecutionPlan plan, int index, GridScheduler scheduler) {
        if (index < 0) {
            return;
        }
        long t0 = System.nanoTime();
        plan.withGraph(index).withGridScheduler(scheduler).execute();
        plan.withAllGraphs();
        System.err.printf("[jitllm] packed weights repacked on the GPU in %.1f ms%n", (System.nanoTime() - t0) / 1e6);
    }

    private static WorkerGrid grid(int lanes) {
        int local = PackedRepackKernels.LOCAL;
        WorkerGrid g = new WorkerGrid1D((lanes + local - 1) / local * local);
        g.setLocalWork(local, 1, 1);
        return g;
    }
}
