package org.beehive.jitllm.backend.tornado;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import org.beehive.jitllm.backend.tornado.kernels.LlamaMegakernel;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;

/**
 * Single-token decode as one persistent kernel per token ({@link LlamaMegakernel}).
 *
 * <p>Opt-in with {@code -Djitllm.megakernel=true}; Llama F16 weights, the FP16 KV cache and the
 * CUDA backend only, single-token mode. The per-layer weights are copied once into whole-model
 * arrays, because a kernel takes a fixed parameter list and the layered plan passes nine arrays per
 * layer. The layered arrays never reach the device.
 *
 * <p>The grid is {@code blocksPerSM} blocks of {@link LlamaMegakernel#BLOCK_SIZE} threads per
 * multiprocessor ({@code -Djitllm.megakernel.blocksPerSM}, default 1). It must fit on the device at
 * once: the cooperative launch refuses a grid that does not, by name. Above one block per
 * multiprocessor the kernel is compiled with a register cap that lets that many fit.
 */
public final class TornadoVMMasterPlanMegakernel implements TornadoVMMasterPlan {

    public static final String PROPERTY = "jitllm.megakernel";

    private static final String GRAPH = "megakernel";
    private static final String TASK = "forward";
    private static final int MAX_SPLITS = 16;
    /** 32-bit registers per multiprocessor on every architecture since Kepler. */
    private static final int REGISTERS_PER_SM = 65536;

    private final State state;
    private final Model model;
    private final TornadoMetricsReporter metrics;
    private final int blocksPerSM;
    private final int blocks;
    private final int splits;

    private final FloatArray x;
    private final FloatArray scratch;
    private final FloatArray norms;
    private final HalfFloatArray wqkv;
    private final HalfFloatArray wo;
    private final HalfFloatArray w1;
    private final HalfFloatArray w3;
    private final HalfFloatArray w2;
    private final IntArray meta;
    private final GridScheduler scheduler;

    public final TornadoExecutionPlan executionPlan;

    /** Whether the opt-in is set; {@link #requireSupported} then decides whether it can run. */
    public static boolean requested() {
        return Boolean.getBoolean(PROPERTY);
    }

    /**
     * Refuses, by name, a session the megakernel does not implement. An opt-in that quietly fell
     * back to the layered plan would make every A/B taken with it measure the layered plan twice.
     */
    static void requireSupported(Model model, State state) {
        Configuration c = model.configuration();
        List<String> reasons = new ArrayList<>();
        if (!(c instanceof LlamaConfiguration) || !(model.weights() instanceof LlamaTornadoWeights)) {
            reasons.add("the model is not Llama");
        } else if (model.weights().dataType() != DataType.F16) {
            reasons.add("weights are " + model.weights().dataType() + ", not F16");
        }
        if (state.executionPolicy().phaseStrategy() == ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
            reasons.add("prefill/decode mode is set; the megakernel is single-token");
        }
        if (!state.usesFp16KeyValueCache()) {
            reasons.add("the KV cache is not FP16");
        }
        if (TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getBackendType() != TornadoVMBackendType.CUDA) {
            reasons.add("the backend is not CUDA (grid barriers are CUDA-only)");
        }
        if (c.dim() > LlamaMegakernel.MAX_DIM) {
            reasons.add("dim " + c.dim() + " exceeds " + LlamaMegakernel.MAX_DIM);
        }
        if (c.headSize() % 32 != 0 || c.headSize() > LlamaMegakernel.MAX_HEAD_SIZE) {
            reasons.add("head size " + c.headSize() + " is not a multiple of 32 up to " + LlamaMegakernel.MAX_HEAD_SIZE);
        }
        if (c.dim() % 2 != 0 || c.kvDim() % 2 != 0 || c.hiddenDim() % 2 != 0) {
            reasons.add("dim, kvDim and hiddenDim must be even");
        }
        if (!reasons.isEmpty()) {
            throw new IllegalStateException("-D" + PROPERTY + "=true cannot run this session: " + String.join("; ", reasons));
        }
    }

    public TornadoVMMasterPlanMegakernel(State state, Model model, MetricsSink sink) {
        requireSupported(model, state);
        this.state = state;
        this.model = model;
        this.metrics = new TornadoMetricsReporter(sink);
        Configuration c = model.configuration();
        LlamaTornadoWeights w = (LlamaTornadoWeights) model.weights();

        int multiprocessors = TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getDefaultDevice().getPhysicalDevice().getDeviceMaxComputeUnits();
        this.blocksPerSM = Integer.getInteger(PROPERTY + ".blocksPerSM", 1);
        this.blocks = multiprocessors * blocksPerSM;
        // Enough (head, split) units to give every block one, within the scratch.
        this.splits = Math.max(1, Math.min(MAX_SPLITS, (blocks + c.numberOfHeads() - 1) / c.numberOfHeads()));

        int layers = c.numberOfLayers();
        this.wqkv = concatenate(w.wqLayered, w.wkLayered, w.wvLayered);
        this.wo = concatenate(w.woLayered);
        this.w1 = concatenate(w.w1Layered);
        this.w3 = concatenate(w.w3Layered);
        this.w2 = concatenate(w.w2Layered);
        this.norms = new FloatArray(LlamaMegakernel.epsilonIndex(c.dim(), layers) + 1);
        long offset = 0;
        for (TornadoTensor t : w.rms_att_weightLayered) {
            offset = copyInto(norms.getSegment(), offset, t.asFloatArray().getSegment());
        }
        for (TornadoTensor t : w.rms_ffn_weightLayered) {
            offset = copyInto(norms.getSegment(), offset, t.asFloatArray().getSegment());
        }
        copyInto(norms.getSegment(), offset, w.rms_final_weight_as_floatArray.asFloatArray().getSegment());
        norms.set(LlamaMegakernel.epsilonIndex(c.dim(), layers), c.rmsNormEps());

        this.x = new FloatArray(c.dim());
        this.scratch = new FloatArray(LlamaMegakernel.scratchSize(c.dim(), c.hiddenDim(), c.numberOfHeads(), c.headSize(), splits));
        this.meta = new IntArray(LlamaMegakernel.META_SIZE);
        meta.set(LlamaMegakernel.META_DIM, c.dim());
        meta.set(LlamaMegakernel.META_KV_DIM, c.kvDim());
        meta.set(LlamaMegakernel.META_HIDDEN_DIM, c.hiddenDim());
        meta.set(LlamaMegakernel.META_LAYERS, layers);
        meta.set(LlamaMegakernel.META_HEADS, c.numberOfHeads());
        meta.set(LlamaMegakernel.META_KV_MUL, c.kvMul());
        meta.set(LlamaMegakernel.META_HEAD_SIZE, c.headSize());
        meta.set(LlamaMegakernel.META_VOCABULARY, c.vocabularySize());
        meta.set(LlamaMegakernel.META_KV_BLOCK_CFG, state.kvBlockCfg);
        meta.set(LlamaMegakernel.META_KV_BLOCK_STRIDE, state.kvBlockStride);
        meta.set(LlamaMegakernel.META_SPLITS, splits);

        WorkerGrid1D worker = new WorkerGrid1D(blocks * LlamaMegakernel.BLOCK_SIZE);
        worker.setLocalWork(LlamaMegakernel.BLOCK_SIZE, 1, 1);
        this.scheduler = new GridScheduler(GRAPH + "." + TASK, worker);

        this.executionPlan = createExecutionPlan();
        metrics.enableOn(executionPlan);
        if (CUDA_GRAPHS) {
            executionPlan.withAllGraphs().withCUDAGraph();
        }
        if (blocksPerSM > 1) {
            // Every block must be resident, so the register file has to hold blocksPerSM of them.
            int registers = (REGISTERS_PER_SM / (LlamaMegakernel.BLOCK_SIZE * blocksPerSM)) & ~7;
            executionPlan.withCompilerFlags(TornadoVMBackendType.CUDA, (TornadoOptions.DEFAULT_CUDA_COMPILER_FLAGS + " --maxrregcount=" + registers).trim());
        }
        executionPlan.withStagedTransfers();
        executionPlan.withPreCompilation();
        forceCopyInReadOnlyData();
    }

    @Override
    public org.beehive.jitllm.runtime.backend.ExecutionInfo executionInfo() {
        return PlanDiagnostics.describe(state, "megakernel", 1, "one persistent kernel per token (" + blocks + " blocks, " + splits + " KV splits)", "in-kernel split-KV");
    }

    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        TornadoWeightsView v = new TornadoWeightsView((LlamaTornadoWeights) model.weights());
        var ws = state.workspace;
        KernelContext context = new KernelContext();
        TaskGraph graph = new TaskGraph(GRAPH)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, ws.embeddingX, ws.positionHolder, ws.wrapBlockTable)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, context, x, scratch, norms, wqkv, wo, w1, w3, w2, v.wcls, ws.wrapKeyCacheFP16, ws.wrapValueCacheFP16, v.freqReal,
                        v.freqImag, meta, ws.wrapLogits)
                .task(TASK, LlamaMegakernel::forward, context, (HalfFloatArray) ws.embeddingX, x, scratch, norms, wqkv, wo, w1, w3, w2, v.wcls, ws.wrapKeyCacheFP16,
                        ws.wrapValueCacheFP16, v.freqReal, v.freqImag, ws.positionHolder, ws.wrapBlockTable, meta, ws.wrapLogits)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, ws.wrapLogits);
        return new TornadoExecutionPlan(graph.snapshot());
    }

    @Override
    public FloatArray tornadoVMForwardDecode(int position) {
        state.setPosition(position);
        metrics.report(executionPlan.withGridScheduler(scheduler).execute());
        return state.workspace.wrapLogits;
    }

    @Override
    public void forceCopyInReadOnlyData() {
        // The first execution uploads the weights; whatever it writes at position 0 is overwritten
        // by the first real token.
        state.resetPositionHolder();
        metrics.report(executionPlan.withGridScheduler(scheduler).execute());
    }

    @Override
    public void resetSequenceState() {
        TornadoVMMasterPlan.resetSequenceState(executionPlan, state, 0);
    }

    @Override
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
        try {
            executionPlan.close();
        } catch (Exception e) {
            throw new IllegalStateException("failed to close the TornadoVM execution plan", e);
        }
    }

    /** The per-model arrays the layered plan already holds flat. */
    private record TornadoWeightsView(HalfFloatArray wcls, FloatArray freqReal, FloatArray freqImag) {
        TornadoWeightsView(LlamaTornadoWeights w) {
            this(w.wclsByteArray.asHalfFloatArray(), w.freq_cis_realFlat.asFloatArray(), w.freq_cis_imagFlat.asFloatArray());
        }
    }

    /** Layer by layer, each layer's tensors in the order given: [l0: a, b, ...][l1: a, b, ...]. */
    private static HalfFloatArray concatenate(TornadoTensor[]... perLayer) {
        long elements = 0;
        for (TornadoTensor[] tensors : perLayer) {
            for (TornadoTensor t : tensors) {
                elements += t.asHalfFloatArray().getSize();
            }
        }
        if (elements > Integer.MAX_VALUE) {
            throw new IllegalStateException("a concatenated weight of " + elements + " elements does not fit an int-indexed array");
        }
        HalfFloatArray flat = new HalfFloatArray((int) elements);
        long offset = 0;
        for (int l = 0; l < perLayer[0].length; l++) {
            for (TornadoTensor[] tensors : perLayer) {
                offset = copyInto(flat.getSegment(), offset, tensors[l].asHalfFloatArray().getSegment());
            }
        }
        return flat;
    }

    private static long copyInto(MemorySegment destination, long offset, MemorySegment source) {
        MemorySegment.copy(source, 0, destination, offset, source.byteSize());
        return offset + source.byteSize();
    }
}
