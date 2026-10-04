package org.beehive.jitllm.backend.tornado;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.layers.AbstractTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.type.q4_0.LlamaQ4_0FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.q4_0.decode.LlamaQ4_0FFNLayersDecode;
import org.beehive.jitllm.backend.tornado.layers.type.q4_0.prefill.LlamaQ4_0LayersBatchPrefillNative;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LlamaQ8_0FFNLayers;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.decode.LogitsQ8_0LayerDecode;
import org.beehive.jitllm.backend.tornado.pipeline.PipelineTransport;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchDecodeActivation;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillQ8DeviceActivation;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The single-token forward pass split by layers across several devices: pipeline parallelism, as
 * llama.cpp's {@code --split-mode layer} does it.
 *
 * <p>Stage {@code s} runs on its own device with its own plan and holds a contiguous range of
 * layers, their weights and their key/value cache. The first stage also runs the embedding, the
 * last also the final norm and the vocabulary projection. Only the hidden state ({@code dim}
 * floats) crosses between devices, once per stage boundary per token:
 *
 * <pre>
 *   stage 0 (device 0): activation, layers [0, k1), send x ──┐
 *   stage 1 (device 1): receive x, layers [k1, k2), ...  ◀───┘ ... logits
 * </pre>
 *
 * <p>For one sequence the stages run one after the other, so this adds memory, not speed: a model
 * too large for one device runs on several.
 *
 * <p>Enabled by {@code -Djitllm.pipeline.devices=0:0,0:1} (TornadoVM {@code backend:device}
 * indices, one per stage). {@code -Djitllm.pipeline.split=a,b,...} gives the share of layers per
 * stage, like llama.cpp's {@code --tensor-split} (default: equal). {@code
 * -Djitllm.pipeline.transport=nccl|host} picks how the hidden state moves (default nccl).
 * Llama-family models with Q4_0 or Q8_0 layers only.
 */
// @formatter:on
public final class TornadoVMMasterPlanPipeline implements BatchPrefillDecodePlan {

    private static final String DEVICES_PROPERTY = "jitllm.pipeline.devices";

    /** Logs a checksum of every token's logits. */
    private static final boolean DEBUG = Boolean.getBoolean("jitllm.pipeline.debug");

    private static final System.Logger LOGGER =
            System.getLogger(TornadoVMMasterPlanPipeline.class.getName());

    /** Whether a pipeline split was asked for. */
    public static boolean requested() {
        String devices = System.getProperty(DEVICES_PROPERTY);
        return devices != null && !devices.isBlank();
    }

    /** How many devices the requested split uses, or 1 when none is requested. */
    public static int requestedDeviceCount() {
        return requested() ? System.getProperty(DEVICES_PROPERTY).split(",").length : 1;
    }

    private record Stage(
            int firstLayer,
            int endLayer,
            LlamaState state,
            TornadoDevice device,
            TornadoExecutionPlan plan) {}

    private final State state;
    private final Stage[] stages;
    private final TornadoExecutionPlan[] plans;
    private final PipelineTransport transport;
    private final String transportName;

    /**
     * Whether the prompt is prefilled in chunks: then each stage's plan holds a prefill program and
     * a decode program, and a step runs one of them; otherwise a step runs the whole plan.
     */
    private final boolean batched;

    private final int batchSize;

    /** Per stage, the graph indices of its prefill program, in order. Null unless batched. */
    private final int[][] prefillGraphs;

    /** Per stage, the graph indices of its decode program, in order. Null unless batched. */
    private final int[][] decodeGraphs;

    public TornadoVMMasterPlanPipeline(State state, Model model, MetricsSink sink) {
        if (!(state instanceof LlamaState) || !(model.weights() instanceof LlamaTornadoWeights)) {
            throw new UnsupportedOperationException(
                    "the pipeline split supports Llama-family models only, not "
                            + model.getModelType());
        }
        DataType weightType = model.weights().dataType();
        if (weightType != DataType.Q4_0 && weightType != DataType.Q8_0) {
            throw new UnsupportedOperationException(
                    "the pipeline split supports Q4_0 and Q8_0 layers only, not " + weightType);
        }
        if (model.getModelType() == ModelType.MISTRAL) {
            throw new UnsupportedOperationException("the pipeline split does not support Mistral");
        }
        this.state = state;
        var config = (LlamaConfiguration) model.configuration();
        var weights = (LlamaTornadoWeights) model.weights();
        SchedulerType schedulerType = SchedulerDetectionService.determineSchedulerType(model);

        TornadoDevice[] devices = parseDevices(System.getProperty(DEVICES_PROPERTY));
        int[] bounds =
                splitLayers(
                        config.numberOfLayers(),
                        devices.length,
                        System.getProperty("jitllm.pipeline.split"));
        this.transportName = System.getProperty("jitllm.pipeline.transport", "nccl");
        this.transport = PipelineTransport.create(transportName, devices);

        var policy = state.executionPolicy();
        this.batched =
                policy.phaseStrategy()
                                == org.beehive.jitllm.runtime.policy.ExecutionPolicy.PhaseStrategy
                                        .PREFILL_DECODE
                        && policy.prefillBatchSize() > 1;
        this.batchSize = batched ? policy.prefillBatchSize() : 1;
        if (batched && weightType != DataType.Q4_0) {
            throw new UnsupportedOperationException(
                    "the pipeline split batches the prefill of Q4_0 layers only, not "
                            + weightType);
        }
        if (batched && !NativePrefillSupport.nativeProjections(policy)) {
            throw new UnsupportedOperationException(
                    "the pipeline split's batched prefill runs its projections through cuBLAS: add"
                            + " --with-native-libraries on CUDA devices with tensor cores");
        }

        this.stages = new Stage[devices.length];
        this.plans = new TornadoExecutionPlan[devices.length];
        this.prefillGraphs = batched ? new int[devices.length][] : null;
        this.decodeGraphs = batched ? new int[devices.length][] : null;
        int last = devices.length - 1;
        for (int s = 0; s < devices.length; s++) {
            int first = bounds[s];
            int end = bounds[s + 1];
            LlamaState stageState = stageState(state, config, end - first, batchSize);
            if (s == 0) {
                // The token loop writes the embedding row into the session's state.
                stageState.workspace.embeddingX = state.workspace.embeddingX;
            }

            List<ImmutableTaskGraph> graphs = new ArrayList<>();
            GridScheduler scheduler = new GridScheduler();
            if (batched) {
                addBatchedStage(
                        s,
                        last,
                        first,
                        end,
                        stageState,
                        weights,
                        config,
                        schedulerType,
                        graphs,
                        scheduler);
            } else {
                addSingleTokenStage(
                        s,
                        last,
                        first,
                        end,
                        stageState,
                        weights,
                        config,
                        schedulerType,
                        weightType,
                        graphs,
                        scheduler);
            }
            transport.updateGridScheduler(s, scheduler);

            TornadoExecutionPlan plan =
                    new TornadoExecutionPlan(graphs.toArray(new ImmutableTaskGraph[0]))
                            .withDevice(devices[s])
                            .withGridScheduler(scheduler);
            if (CUDA_GRAPHS && !batched) {
                plan.withCUDAGraph();
            }
            plan.withStagedTransfers();
            plan.withPreCompilation();
            stages[s] = new Stage(first, end, stageState, devices[s], plan);
            plans[s] = plan;
        }
        forceCopyInReadOnlyData();
    }

    /**
     * One stage of the single-token plan: [embedding | receive], its layers, [send | logits]. A
     * step runs the whole plan.
     */
    private void addSingleTokenStage(
            int s,
            int last,
            int first,
            int end,
            LlamaState stageState,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType,
            DataType weightType,
            List<ImmutableTaskGraph> graphs,
            GridScheduler scheduler) {
        if (s == 0) {
            var activation = new Activation("activationUpdate", stageState, weights, config);
            graphs.add(activation.getImmutableTaskGraph());
            activation.updateGridScheduler(scheduler);
        } else {
            TaskGraph receive = new TaskGraph("handoff_in");
            transport.addReceive(receive, s, stageState.workspace.wrapX, s - 1);
            graphs.add(receive.snapshot());
        }

        AbstractTransformerLayerTaskGraphs<?, ?> layers =
                weightType == DataType.Q4_0
                        ? new LlamaQ4_0FFNLayers(
                                "layers", stageState, weights, config, schedulerType, first, end)
                        : new LlamaQ8_0FFNLayers(
                                "layers", stageState, weights, config, schedulerType, first, end);
        graphs.addAll(layers.getFFNLayerImmutableTaskGraphs());
        layers.updateGridScheduler(scheduler);

        if (s < last) {
            TaskGraph send = new TaskGraph("handoff_out");
            transport.addSend(
                    send,
                    s,
                    layers.getLastFFNLayerTaskGraphID(),
                    stageState.workspace.wrapX,
                    s + 1);
            graphs.add(send.snapshot());
        } else {
            var logits =
                    new LogitsQ8_0Layer(
                            "logits",
                            stageState,
                            weights,
                            config,
                            layers.getLastFFNLayerTaskGraphID(),
                            schedulerType);
            graphs.add(logits.getImmutableTaskGraph());
            logits.updateGridScheduler(scheduler);
        }
    }

    // @formatter:off
    /**
     * One stage of the batched plan: two programs in one plan, so they share the stage's weights
     * and key/value cache on its device.
     *
     * <pre>
     *   prefill: [prefillActivation | receive chunk], batchPrefillLayer_[first, end), [send chunk]
     *   decode:  [decodeActivation  | receive x   ], layer_[first, end),             [send x | logits]
     * </pre>
     *
     * <p>A later stage's {@code decodeActivation} only receives the hidden state; its first decode
     * layer takes the cache from the stage's last prefill layer.
     */
    // @formatter:on
    private void addBatchedStage(
            int s,
            int last,
            int first,
            int end,
            LlamaState stageState,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType,
            List<ImmutableTaskGraph> graphs,
            GridScheduler scheduler) {
        if (s == 0) {
            var activation = new BatchPrefillQ8DeviceActivation(stageState, config, batchSize);
            graphs.add(activation.getImmutableTaskGraph());
            activation.updateGridScheduler(scheduler);
            // The host stages each chunk's raw embedding rows in the session's state.
            state.workspace.embeddingQ8Batch = stageState.workspace.embeddingQ8Batch;
        } else {
            TaskGraph receive = new TaskGraph("prefillActivation");
            transport.addReceive(receive, s, stageState.workspace.wrapXBatch, s - 1);
            graphs.add(receive.snapshot());
        }
        var prefill =
                new LlamaQ4_0LayersBatchPrefillNative(
                        stageState, weights, config, batchSize, first, end);
        graphs.addAll(prefill.getLayerImmutableTaskGraphs());
        prefill.updateGridScheduler(scheduler);
        String lastPrefill = prefill.getLastLayerTaskGraphID();
        if (s < last) {
            TaskGraph send = new TaskGraph("prefillHandoff");
            transport.addSend(send, s, lastPrefill, stageState.workspace.wrapXBatch, s + 1);
            graphs.add(send.snapshot());
        }
        prefillGraphs[s] = IntStream.range(0, graphs.size()).toArray();

        int decodeStart = graphs.size();
        if (s == 0) {
            var activation = new BatchDecodeActivation(stageState, config, lastPrefill, true);
            graphs.add(activation.getImmutableTaskGraph());
            activation.updateGridScheduler(scheduler);
        } else {
            // Only the hidden state: the first decode layer takes this stage's cache straight
            // from its last prefill layer.
            TaskGraph receive = new TaskGraph("decodeActivation");
            transport.addReceive(receive, s, stageState.workspace.wrapX, s - 1);
            graphs.add(receive.snapshot());
        }
        var decode =
                new LlamaQ4_0FFNLayersDecode(
                        "decode", stageState, weights, config, schedulerType, first, end);
        graphs.addAll(decode.getFFNLayerImmutableTaskGraphs());
        decode.updateGridScheduler(scheduler);
        if (s < last) {
            TaskGraph send = new TaskGraph("decodeHandoff");
            transport.addSend(
                    send,
                    s,
                    decode.getLastFFNLayerTaskGraphID(),
                    stageState.workspace.wrapX,
                    s + 1);
            graphs.add(send.snapshot());
        } else {
            var logits =
                    new LogitsQ8_0LayerDecode(
                            "logits",
                            stageState,
                            weights,
                            config,
                            decode.getLastFFNLayerTaskGraphID(),
                            schedulerType);
            graphs.add(logits.getImmutableTaskGraph());
            logits.updateGridScheduler(scheduler);
        }
        decodeGraphs[s] = IntStream.range(decodeStart, graphs.size()).toArray();
    }

    /**
     * A state for one stage: its own buffers and a key/value cache for its layers only, with the
     * session's storage options and execution policy.
     */
    private static LlamaState stageState(
            State session, LlamaConfiguration config, int layers, int prefillBatchSize) {
        LlamaState stage =
                State.withStorageOptions(
                        session.storageOptions(),
                        () ->
                                State.withPrefillBatchSize(
                                        prefillBatchSize,
                                        () ->
                                                State.withKeyValueLayers(
                                                        layers, () -> new LlamaState(config, 1))));
        stage.resolveExecutionPolicy(session.executionPolicy());
        return stage;
    }

    /** {@code "0:0,0:1"} as TornadoVM devices. */
    static TornadoDevice[] parseDevices(String spec) {
        String[] parts = spec.split(",");
        if (parts.length < 1) {
            throw new IllegalArgumentException(
                    DEVICES_PROPERTY + " needs one device per stage, e.g. 0:0,0:1: " + spec);
        }
        TornadoDevice[] devices = new TornadoDevice[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String[] bd = parts[i].trim().split(":");
            if (bd.length != 2) {
                throw new IllegalArgumentException(
                        DEVICES_PROPERTY + " entries are backend:device, not " + parts[i]);
            }
            devices[i] =
                    TornadoExecutionPlan.getDevice(
                            Integer.parseInt(bd[0].trim()), Integer.parseInt(bd[1].trim()));
        }
        return devices;
    }

    /**
     * Stage boundaries: stage {@code s} runs layers {@code [bounds[s], bounds[s + 1])}. Layers are
     * shared out in proportion to {@code split} (equal when it is null), every stage keeping at
     * least one.
     */
    static int[] splitLayers(int layers, int stages, String split) {
        double[] share = new double[stages];
        if (split == null || split.isBlank()) {
            Arrays.fill(share, 1.0);
        } else {
            String[] parts = split.split(",");
            if (parts.length != stages) {
                throw new IllegalArgumentException(
                        "jitllm.pipeline.split has "
                                + parts.length
                                + " entries for "
                                + stages
                                + " devices");
            }
            for (int i = 0; i < stages; i++) {
                share[i] = Double.parseDouble(parts[i].trim());
                if (!(share[i] > 0)) {
                    throw new IllegalArgumentException("jitllm.pipeline.split entries must be > 0");
                }
            }
        }
        if (layers < stages) {
            throw new IllegalArgumentException(
                    layers + " layers cannot be split across " + stages + " devices");
        }
        double total = Arrays.stream(share).sum();
        int[] bounds = new int[stages + 1];
        double cumulative = 0;
        for (int s = 0; s < stages; s++) {
            cumulative += share[s];
            int end = (int) Math.round(layers * cumulative / total);
            // At least one layer per stage, and room left for the stages after this one.
            end = Math.max(end, bounds[s] + 1);
            end = Math.min(end, layers - (stages - 1 - s));
            bounds[s + 1] = end;
        }
        bounds[stages] = layers;
        return bounds;
    }

    @Override
    public org.beehive.jitllm.runtime.backend.ExecutionInfo executionInfo() {
        StringBuilder split = new StringBuilder("pipeline (" + transportName + "):");
        for (Stage stage : stages) {
            split.append(' ')
                    .append(stage.device().getDeviceName())
                    .append(" [")
                    .append(stage.firstLayer())
                    .append(',')
                    .append(stage.endLayer())
                    .append(')');
        }
        if (batched) {
            return PlanDiagnostics.describe(
                    state,
                    "batch-prefill-decode " + split,
                    batchSize,
                    "cuBLAS FP16 GEMM (Q4_0 weights decoded to FP16 per projection)",
                    "JIT kernels");
        }
        return PlanDiagnostics.describe(
                state, split.toString(), 1, "JIT kernels (no tensor-core MMA)", "JIT kernels");
    }

    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        return plans[0];
    }

    @Override
    public void forceCopyInReadOnlyData() {
        // One full step uploads every FIRST_EXECUTION buffer, the weights among them.
        for (Stage stage : stages) {
            stage.state().workspace.wrapX.clear();
            stage.state().resetPositionHolder();
            if (batched) {
                // No active rows: the warm-up prefill writes nothing to the cache.
                var workspace = stage.state().workspace;
                workspace.wrapXBatch.clear();
                workspace.batchStartPosHolder.init(0);
                workspace.batchStartPosHolder.set(2, stage.state().kvSlot);
            }
        }
        if (batched) {
            transport.execute(plans, prefillGraphs, CUDA_GRAPHS);
            transport.execute(plans, decodeGraphs, CUDA_GRAPHS);
        } else {
            transport.execute(plans);
        }
    }

    @Override
    public void tornadoVMForwardBatchPrefill() {
        if (!batched) {
            throw new IllegalStateException("this pipeline plan was built without batched prefill");
        }
        // The host staged the chunk's start position, active rows and slot in the session state.
        IntArray chunk = state.workspace.batchStartPosHolder;
        for (Stage stage : stages) {
            IntArray holder = stage.state().workspace.batchStartPosHolder;
            for (int i = 0; i < holder.getSize() && i < chunk.getSize(); i++) {
                holder.set(i, chunk.get(i));
            }
        }
        transport.execute(plans, prefillGraphs, CUDA_GRAPHS);
    }

    @Override
    public FloatArray tornadoVMForwardDecode(int position) {
        for (Stage stage : stages) {
            var workspace = stage.state().workspace;
            stage.state().setPosition(position);
            workspace.temp.clear();
            workspace.tempFFN.clear();
            workspace.tempLogits.clear();
            workspace.wrapLogits.clear();
        }
        if (batched) {
            transport.execute(plans, decodeGraphs, CUDA_GRAPHS);
        } else {
            transport.execute(plans);
        }
        if (DEBUG) {
            FloatArray logits = stages[stages.length - 1].state().workspace.wrapLogits;
            double sum = 0;
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < logits.getSize(); i++) {
                sum += logits.get(i);
                max = Math.max(max, logits.get(i));
            }
            LOGGER.log(
                    System.Logger.Level.INFO,
                    String.format("pipeline pos %d: logits sum %.4f max %.4f", position, sum, max));
        }
        return stages[stages.length - 1].state().workspace.wrapLogits;
    }

    @Override
    public void resetSequenceState() {
        // Llama keeps no recurrent state; the key/value cache is overwritten position by position.
        for (Stage stage : stages) {
            stage.state().resetSequenceState();
        }
    }

    @Override
    public void freeTornadoExecutionPlan() {
        transport.close(
                () -> {
                    for (TornadoExecutionPlan plan : plans) {
                        if (plan == null) {
                            continue;
                        }
                        plan.freeDeviceMemory();
                        try {
                            plan.close();
                        } catch (Exception e) {
                            throw new IllegalStateException(
                                    "failed to close a pipeline stage's execution plan", e);
                        }
                    }
                });
    }
}
