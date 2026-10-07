package org.beehive.jitllm.backend.tornado.kernels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
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
 * Packing on the GPU at every load: a packed weight ({@link PackedQ8_0}) is uploaded in its GGUF
 * layout and repacked on the device, so the host never holds or writes the packed form.
 *
 * <p>The loader keeps a packed weight's GGUF bytes on the host and records it here ({@link
 * #pack}). A layer graph hands its weights to {@link #upload} instead of uploading them: the
 * registered ones are uploaded by a repack graph of their own, {@code packedRepack_<layer graph>},
 * which the layer graph consumes them from. {@link #appendGraphs} adds those graphs to the plan;
 * {@link #run} runs them once, before the plan's warm-up, so no layer graph ever computes on the
 * GGUF bytes. Each repack graph copies a weight into one scratch buffer shared by the plan and
 * repacks it from there over the weight's own device buffer; a flag on the device makes a later
 * run of any of them a no-op.
 */
// @formatter:on
public final class PackedRepack {

    /** Diagnostic: {@code -Djitllm.packed.verify=true} reads weights back after the repack. */
    private static final boolean VERIFY = Boolean.getBoolean("jitllm.packed.verify");

    private static final String PREFIX = "packedRepack_";

    private record Weight(ByteArray bytes, int rows, int cols, boolean q4) {
        int size() {
            return rows * (cols / 32) * (q4 ? 18 : 34);
        }
    }

    /** The repack graphs appended to one plan: indices {@code first .. first + count - 1}. */
    public record Graphs(int first, int count, GridScheduler scheduler, List<Consumer<TornadoExecutionPlan>> checks) {}

    private static final Map<ByteArray, Weight> REGISTERED = new IdentityHashMap<>();

    /** Layer graph name to the registered weights it consumes, not yet given repack graphs. */
    private static final Map<String, List<Weight>> OWNED = new LinkedHashMap<>();

    private PackedRepack() {}

    /**
     * Records the {@code rows x cols} GGUF weight {@code w} (Q4_0 when {@code q4}, else Q8_0) as
     * packed, to be repacked on the device once uploaded, and returns it.
     */
    public static synchronized ByteArray pack(ByteArray w, int rows, int cols, boolean q4) {
        REGISTERED.put(w, new Weight(w, rows, cols, q4));
        return q4 ? PackedQ8_0.recordQ4(w) : PackedQ8_0.record(w);
    }

    /**
     * Uploads {@code objects} to {@code graph} once ({@code FIRST_EXECUTION}), except the weights
     * registered for a GPU repack, which {@code graph} consumes from its repack graph instead.
     */
    public static synchronized void upload(TaskGraph graph, Object... objects) {
        List<Object> plain = new ArrayList<>();
        List<Object> repacked = new ArrayList<>();
        for (Object o : objects) {
            Weight w = o instanceof ByteArray b ? REGISTERED.get(b) : null;
            if (w != null) {
                repacked.add(o);
                OWNED.computeIfAbsent(graph.getTaskGraphName(), k -> new ArrayList<>()).add(w);
            } else {
                plain.add(o);
            }
        }
        if (!plain.isEmpty()) {
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, plain.toArray());
        }
        if (!repacked.isEmpty()) {
            graph.consumeFromDevice(PREFIX + graph.getTaskGraphName(), repacked.toArray());
        }
    }

    /**
     * Appends to {@code graphs} the repack graphs of the layer graphs built since the last call,
     * with their grids in {@code scheduler}; null when there is nothing to repack.
     */
    public static synchronized Graphs appendGraphs(List<ImmutableTaskGraph> graphs, GridScheduler scheduler) {
        if (OWNED.isEmpty()) {
            return null;
        }
        int scratchBytes = 0;
        for (List<Weight> ws : OWNED.values()) {
            for (Weight w : ws) {
                scratchBytes = Math.max(scratchBytes, w.size());
            }
        }
        ByteArray scratch = new ByteArray(scratchBytes);
        IntArray done = new IntArray(1);
        List<String> names = new ArrayList<>();
        List<Weight> firsts = new ArrayList<>();
        int first = graphs.size();
        int remaining = OWNED.size();
        for (Map.Entry<String, List<Weight>> e : OWNED.entrySet()) {
            String name = PREFIX + e.getKey();
            List<Weight> ws = e.getValue();
            Object[] bytes = ws.stream().map(Weight::bytes).toArray();
            TaskGraph g = new TaskGraph(name).transferToDevice(DataTransferMode.FIRST_EXECUTION, bytes);
            if (names.isEmpty()) {
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, scratch, done);
            } else {
                g.consumeFromDevice(names.get(0), scratch, done);
            }
            for (int i = 0; i < ws.size(); i++) {
                Weight w = ws.get(i);
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
            }
            if (--remaining == 0) {
                g.task("markDone", PackedRepackKernels::markDone, new KernelContext(), done);
                scheduler.addWorkerGrid(name + ".markDone", grid(1));
            }
            g.persistOnDevice(bytes);
            g.persistOnDevice(scratch, done);
            graphs.add(g.snapshot());
            names.add(name);
            firsts.add(ws.get(0));
        }
        int count = OWNED.size();
        OWNED.clear();
        List<Consumer<TornadoExecutionPlan>> checks = new ArrayList<>();
        if (VERIFY) {
            checks.add(verifier(graphs, scheduler, names.get(0), firsts.get(0)));
            checks.add(verifier(graphs, scheduler, names.get(count - 1), firsts.get(count - 1)));
        }
        return new Graphs(first, count, scheduler, checks);
    }

    /** Runs the repack graphs of {@code plan} once, before any graph that consumes the weights. */
    public static void run(TornadoExecutionPlan plan, Graphs repack) {
        if (repack == null) {
            return;
        }
        long t0 = System.nanoTime();
        for (int i = 0; i < repack.count(); i++) {
            plan.withGraph(repack.first() + i).withGridScheduler(repack.scheduler()).execute();
        }
        System.err.printf("[jitllm] packed weights repacked on the GPU in %.1f ms (%d graphs)%n", (System.nanoTime() - t0) / 1e6, repack.count());
        for (Consumer<TornadoExecutionPlan> check : repack.checks()) {
            check.accept(plan);
        }
        plan.withAllGraphs();
    }

    /**
     * Appends a graph that reads {@code w} back from repack graph {@code owner}, and returns the
     * check that runs it and compares it with the host packer.
     */
    private static Consumer<TornadoExecutionPlan> verifier(List<ImmutableTaskGraph> graphs, GridScheduler scheduler, String owner, Weight w) {
        String name = owner + "Verify";
        ByteArray probe = new ByteArray(w.size());
        TaskGraph v = new TaskGraph(name)
                .consumeFromDevice(owner, w.bytes())
                .task("probe", PackedRepackKernels::copy, new KernelContext(), w.bytes(), probe, w.size())
                .transferToHost(DataTransferMode.EVERY_EXECUTION, probe);
        scheduler.addWorkerGrid(name + ".probe", grid((w.size() + 7) / 8));
        graphs.add(v.snapshot());
        int index = graphs.size() - 1;
        return plan -> {
            plan.withGraph(index).withGridScheduler(scheduler).execute();
            byte[] device = probe.toHeapArray();
            byte[] packed = w.q4() ? PackedTilePacker.packQ4_0(w.bytes(), w.rows(), w.cols()) : PackedTilePacker.packQ8_0(w.bytes(), w.rows(), w.cols());
            String verdict = Arrays.equals(device, packed) ? "PACKED" : Arrays.equals(device, w.bytes().toHeapArray()) ? "RAW (not repacked)" : "NEITHER";
            System.err.printf("[jitllm] packed verify: %s %dx%d %s -> %s%n", w.q4() ? "Q4_0" : "Q8_0", w.rows(), w.cols(), owner, verdict);
        };
    }

    private static WorkerGrid grid(int lanes) {
        int local = PackedRepackKernels.LOCAL;
        WorkerGrid g = new WorkerGrid1D((lanes + local - 1) / local * local);
        g.setLocalWork(local, 1, 1);
        return g;
    }
}
