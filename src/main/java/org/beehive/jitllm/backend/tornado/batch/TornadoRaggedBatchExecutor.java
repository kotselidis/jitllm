package org.beehive.jitllm.backend.tornado.batch;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerRaggedBatchKernels;
import org.beehive.jitllm.backend.tornado.kv.TornadoKvStore;
import org.beehive.jitllm.backend.tornado.layers.type.fp16.decode.LlamaFP16LayersBatchDecodeMMA;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.decode.LlamaQ8_0LayersRagged;
import org.beehive.jitllm.backend.tornado.plan.components.activation.BatchPrefillActivation;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.runtime.backend.ExecutionInfo;
import org.beehive.jitllm.runtime.batch.BatchSlots;
import org.beehive.jitllm.runtime.batch.StepOutput;
import org.beehive.jitllm.runtime.batch.TokenBatch;
import org.beehive.jitllm.runtime.kv.KvStorage;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Runs ragged token batches: prompt chunks and decode tokens of up to B sequences in one forward
 * pass of up to T rows.
 *
 * <p><b>A row is a token.</b> Row {@code r} carries one token, the position it occupies and the KV
 * slot of its sequence; rows past the step's token count are padding and touch no KV. Projections
 * and the FFN are GEMMs over the rows, exactly as on the batched decode path; the two KV kernels
 * look the slot up per row instead of taking it from the row index. Each row attends over its own
 * sequence up to its own position, and the RoPE task writes every row's key and value before
 * attention reads any, so the tokens of a chunk see each other causally with no mask.
 *
 * <p><b>Logits only where something is sampled.</b> Each sequence's last row is gathered into a
 * dense block of B rows before the vocabulary projection, so a 256-token step pays for B logits
 * rows, not 256.
 *
 * <p><b>Batch invariance.</b> Every kernel computes a row from that row's inputs, its own
 * sequence's KV and the weights, in an order that does not depend on the other rows. A token
 * computed inside a prompt chunk is therefore bit-for-bit the token the one-per-step path computes
 * at that position, and greedy output does not depend on what else ran in the step.
 *
 * <p>FP16 Llama weights, a paged FP16 KV pool and greedy on-device sampling. T rows run every step
 * for now; running only as many as the step needs is the next change.
 */
public final class TornadoRaggedBatchExecutor implements TornadoServingExecutor {

    private static final int RMS_LOCAL = 256;

    private final int maxSequences;
    private final int maxRows;
    private final int dim;
    private final Set<Integer> stopTokens;

    private final TornadoExecutionPlan plan;
    private final GridScheduler gridScheduler;
    private final int layerCount;
    private final int logitsGraphIndex;

    private final IntArray rowPositions;
    private final IntArray rowSlots;
    private final IntArray sampleRows;
    private final IntArray sampledTokens;

    /**
     * Sequences sampling this step, read on the device: at most {@link
     * TransformerRaggedBatchKernels#SKINNY_ROWS} run the vocabulary projection on the skinny GEMM.
     */
    private final IntArray logitsRows;

    private final boolean skinnyVocab =
            !"false".equals(System.getProperty("jitllm.ragged.skinnyGemm"));

    private final int vocabSize;
    private WorkerGrid vocabGrid;
    private WorkerGrid vocabSkinnyGrid;
    private final MemorySegment embeddingTable;
    private final MemorySegment embeddingBatch;
    private final long dimBytes;
    private final TornadoKvStore store;

    private final RaggedLayers layers;

    /** Q8_0 weights: int8 tensor-core projections, embeddings decoded on the host. */
    private final boolean q8;

    // Q8_0: the token embedding table, and the activation rows it is decoded into.
    private final ByteArray embeddingQ8;
    private final HalfFloatArray embeddingHalf;
    private final FloatArray activationRows;

    // Q8_0: sampling rows the int8 vocabulary GEMM computes, in [1] as the int8 kernels read it.
    private final IntArray logitsRowsInt8;
    private final WorkerGrid activationGrid;

    /**
     * Launch only the rows a step uses, rounded up to 16, instead of all T. On by default; the
     * property exists to measure the difference.
     */
    private final boolean trimRows = !"false".equals(System.getProperty("jitllm.ragged.trimRows"));

    private boolean closed;

    /**
     * @param state built with a prefill width of {@code maxRows}, which sizes the row buffers
     * @param maxSequences B: KV slots, and sequences one step may touch
     * @param maxRows T: tokens one step may carry; at least B
     */
    public TornadoRaggedBatchExecutor(
            Model model,
            State state,
            KvStorage storage,
            int maxSequences,
            int maxRows,
            int blocksPerSlot) {
        Configuration config = model.configuration();
        if (!org.beehive.jitllm.backend.tornado.TensorCoreSupport.isTensorCoreCapableBackend()) {
            throw new IllegalArgumentException(
                    "Ragged batching requires a CUDA device with tensor-core MMA support");
        }
        if (!(state instanceof LlamaState llamaState)
                || !(config instanceof LlamaConfiguration llamaConfig)) {
            throw new IllegalArgumentException("Ragged batching supports Llama only so far");
        }
        DataType projections = ((LlamaTornadoWeights) model.weights()).wqLayered[0].dataType();
        if (projections != DataType.F16 && projections != DataType.Q8_0) {
            throw new IllegalArgumentException(
                    "Ragged batching supports FP16 and Q8_0 Llama weights, got " + projections);
        }
        if (!(storage instanceof TornadoKvStore kvStore) || kvStore.keyPoolFP16() == null) {
            throw new IllegalArgumentException(
                    "Ragged batching needs the TornadoVM-backed FP16 key/value pool");
        }
        if (maxRows % 16 != 0) {
            throw new IllegalArgumentException(
                    "T must be a multiple of 16, the granularity steps are launched at; got "
                            + maxRows);
        }
        if (maxRows < maxSequences) {
            throw new IllegalArgumentException(
                    "a step must fit one token per sequence: T="
                            + maxRows
                            + " is below B="
                            + maxSequences);
        }
        this.maxSequences = maxSequences;
        this.maxRows = maxRows;
        this.q8 = projections == DataType.Q8_0;
        this.store = kvStore;
        this.dim = config.dim();
        this.stopTokens = model.chatFormat().getStopTokens();
        LlamaTornadoWeights weights = (LlamaTornadoWeights) model.weights();

        int vocab = config.vocabularySize();
        int paddedSequences = (maxSequences + 127) & ~127;

        this.rowPositions = new IntArray(maxRows);
        this.rowSlots = new IntArray(maxRows);
        this.rowSlots.init(-1);
        this.sampleRows = new IntArray(maxSequences);
        this.sampleRows.init(-1);
        this.sampledTokens = new IntArray(paddedSequences);
        this.logitsRows = new IntArray(2); // [1] = 0: the vocabulary GEMM has no mid kernel
        this.logitsRowsInt8 = new IntArray(2);
        this.vocabSize = vocab;
        FloatArray gathered = new FloatArray(maxSequences * dim);
        FloatArray finalScale = new FloatArray(maxSequences);
        HalfFloatArray normedFinalFP16 = new HalfFloatArray(paddedSequences * dim);
        normedFinalFP16.init(new HalfFloat(0.0f));
        FloatArray logits = new FloatArray(paddedSequences * vocab);

        BatchPrefillActivation activation = new BatchPrefillActivation(state, config, maxRows, q8);
        this.layers =
                q8
                        ? new LlamaQ8_0LayersRagged(
                                llamaState,
                                weights,
                                llamaConfig,
                                maxRows,
                                blocksPerSlot * kvStore.blockSizeTokens(),
                                kvStore.keyPoolFP16(),
                                kvStore.valuePoolFP16(),
                                rowPositions,
                                kvStore.blockTable(),
                                kvStore.blockSizeTokens(),
                                blocksPerSlot,
                                rowSlots)
                        : new LlamaFP16LayersBatchDecodeMMA(
                                llamaState,
                                weights,
                                llamaConfig,
                                maxRows,
                                blocksPerSlot * kvStore.blockSizeTokens(),
                                kvStore.keyPoolFP16(),
                                kvStore.valuePoolFP16(),
                                rowPositions,
                                kvStore.blockTable(),
                                kvStore.blockSizeTokens(),
                                blocksPerSlot,
                                rowSlots);
        List<ImmutableTaskGraph> layerGraphs = layers.getLayerImmutableTaskGraphs();

        KernelContext logitsContext = new KernelContext();
        HalfFloatArray wclsHalf = q8 ? null : weights.wclsByteArray.asHalfFloatArray();
        ByteArray logitsQ8 = q8 ? TornadoWorkspaces.bytes(paddedSequences * dim) : null;
        FloatArray logitsQ8Scales =
                q8 ? TornadoWorkspaces.floats(paddedSequences * dim / 32) : null;
        FloatArray rmsFinal = weights.rms_final_weight_as_floatArray.asFloatArray();
        // Gather the sampling rows, then the final norm, the vocabulary GEMM and greedy argmax
        // over B rows; only B token ids cross to the host.
        TaskGraph logitsGraph =
                new TaskGraph("raggedLogits")
                        .consumeFromDevice(
                                layers.getLastLayerTaskGraphID(), state.workspace.wrapXBatch)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                sampleRows,
                                logitsRows,
                                logitsRowsInt8)
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                logitsContext,
                                gathered,
                                finalScale,
                                normedFinalFP16,
                                rmsFinal,
                                sampledTokens)
                        .task(
                                "gather",
                                TransformerRaggedBatchKernels::gatherRows,
                                logitsContext,
                                state.workspace.wrapXBatch,
                                sampleRows,
                                gathered,
                                dim)
                        .task(
                                "rms_reduce",
                                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                                logitsContext,
                                gathered,
                                finalScale,
                                dim,
                                config.rmsNormEps(),
                                RMS_LOCAL)
                        .task(
                                "rms_apply",
                                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                                logitsContext,
                                normedFinalFP16,
                                gathered,
                                rmsFinal,
                                finalScale,
                                dim);
        if (q8 && Boolean.getBoolean("jitllm.ragged.cublas")) {
            // Q8_0 with cuBLAS projections: the vocabulary projection too, over an FP16 copy, for
            // all maxSequences rows every step so cuBLAS never changes its algorithm.
            HalfFloatArray wclsFP16 =
                    org.beehive.jitllm.backend.tornado.layers.type.q8_0.decode.LlamaQ8_0LayersRagged
                            .dequantize(dim, weights.wclsByteArray);
            logitsGraph
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, wclsFP16)
                    .libraryTask(
                            "vocab",
                            uk.ac.manchester.tornado.cublas.CuBlas::cublasGemmExFP16FP32,
                            1,
                            0,
                            vocab,
                            maxSequences,
                            dim,
                            1.0f,
                            wclsFP16,
                            dim,
                            normedFinalFP16,
                            dim,
                            0.0f,
                            logits,
                            vocab);
        } else if (q8) {
            // Q8_0: the vocabulary projection on the int8 GEMM, its weight through the repack.
            state.workspace.packedRepack.upload(logitsGraph, weights.wclsByteArray);
            logitsGraph
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, logitsQ8, logitsQ8Scales)
                    .task(
                            "vocab_q8",
                            Int8GemmKernels::quantizeActivationsQ8WarpFP16,
                            logitsContext,
                            normedFinalFP16,
                            logitsQ8,
                            logitsQ8Scales,
                            paddedSequences * dim)
                    .task(
                            "vocab",
                            weights.wclsByteArray.isPackedQ8()
                                    ? Int8GemmKernels::gemmInt8Q8_0Packed
                                    : Int8GemmKernels::gemmInt8Q8_0,
                            logitsContext,
                            logitsQ8,
                            logitsQ8Scales,
                            weights.wclsByteArray.asByteArray(),
                            logits,
                            logits,
                            paddedSequences,
                            vocab,
                            dim,
                            Int8GemmKernels.EPILOGUE_STORE,
                            logits,
                            1,
                            logitsRowsInt8,
                            vocab,
                            0);
        } else {
            logitsGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, wclsHalf);
            logitsGraph
                    .task(
                            "vocab",
                            TransformerRaggedBatchKernels::gemmMMAGuarded,
                            logitsContext,
                            logitsRows,
                            normedFinalFP16,
                            wclsHalf,
                            logits,
                            paddedSequences,
                            vocab,
                            dim)
                    .task(
                            "vocab_skinny",
                            TransformerRaggedBatchKernels::gemmMMASkinny,
                            logitsContext,
                            normedFinalFP16,
                            wclsHalf,
                            logits,
                            logitsRows,
                            dim,
                            vocab,
                            0);
        }
        logitsGraph
                .task(
                        "argmax",
                        TransformerBatchPrefillKernels::batchedArgmaxLogits,
                        logitsContext,
                        logits,
                        sampledTokens,
                        vocab)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, sampledTokens);

        GridScheduler schedule = new GridScheduler();
        activation.updateGridScheduler(schedule);
        // Replaces the activation's grid with one this executor can resize per step.
        this.activationGrid = worker(maxRows * dim, 128);
        schedule.addWorkerGrid("prefillActivation.updateX", activationGrid);
        layers.updateGridScheduler(schedule);
        schedule.addWorkerGrid("raggedLogits.gather", worker(maxSequences * dim, 256));
        schedule.addWorkerGrid(
                "raggedLogits.rms_reduce", worker(maxSequences * RMS_LOCAL, RMS_LOCAL));
        schedule.addWorkerGrid("raggedLogits.rms_apply", worker(maxSequences * dim, 256));
        if (q8) {
            schedule.addWorkerGrid("raggedLogits.vocab_q8", worker(maxSequences * dim, 256));
            WorkerGrid2D vocabInt8 =
                    new WorkerGrid2D(
                            paddedSequences
                                    / Int8GemmKernels.I8_BM
                                    * Int8GemmKernels.Q8_GEMM_THREADS,
                            vocab / Int8GemmKernels.I8_BN);
            vocabInt8.setLocalWork(Int8GemmKernels.Q8_GEMM_THREADS, 1, 1);
            schedule.addWorkerGrid("raggedLogits.vocab", vocabInt8);
        }
        this.vocabGrid = mmaGrid(paddedSequences, vocab);
        this.vocabSkinnyGrid =
                worker(vocab / TransformerRaggedBatchKernels.SKINNY_COLS_PER_GROUP * 128, 128);
        if (!q8) {
            schedule.addWorkerGrid("raggedLogits.vocab", vocabGrid);
            schedule.addWorkerGrid("raggedLogits.vocab_skinny", vocabSkinnyGrid);
        }
        schedule.addWorkerGrid("raggedLogits.argmax", worker(maxSequences * 256, 256));
        this.gridScheduler = schedule;

        List<ImmutableTaskGraph> all = new ArrayList<>();
        all.add(activation.getImmutableTaskGraph());
        all.addAll(layerGraphs);
        all.add(logitsGraph.snapshot());
        this.layerCount = layerGraphs.size();
        this.logitsGraphIndex = 1 + layerCount;
        var packedRepack = state.workspace.packedRepack.appendGraphs(all, schedule);
        this.plan = new TornadoExecutionPlan(all.toArray(new ImmutableTaskGraph[0]));
        org.beehive.jitllm.backend.tornado.plan.PackedRepack.run(plan, packedRepack);

        var table = weights.getTokenEmbeddingTable();
        boolean tableQ8 = table.dataType() == DataType.Q8_0;
        this.embeddingQ8 = q8 && tableQ8 ? table.asByteArray() : null;
        this.embeddingHalf = q8 && !tableQ8 ? table.asHalfFloatArray() : null;
        this.activationRows = q8 ? state.workspace.wrapXBatch : null;
        this.embeddingTable = q8 ? null : table.asHalfFloatArray().getSegment();
        this.embeddingBatch = q8 ? null : state.workspace.embeddingXBatch.getSegment();
        this.dimBytes = (long) dim * Short.BYTES;
    }

    @Override
    public ExecutionInfo executionInfo() {
        var backend =
                uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider.getTornadoRuntime()
                        .getBackend(0);
        var module = TornadoExecutionPlan.class.getModule().getDescriptor();
        String version = module == null ? "unknown" : module.rawVersion().orElse("unknown");
        return new ExecutionInfo(
                backend.getBackendType().name(),
                backend.getDefaultDevice().getPhysicalDevice().getDeviceName(),
                "TornadoVM " + version,
                "ragged-batch (chunked prefill)",
                1,
                "FP16",
                q8 ? "Q8_0 int8 tensor-core GEMM" : "FP16 tensor-core MMA",
                "JIT kernels",
                false,
                false);
    }

    @Override
    public int maxBatchSize() {
        return maxSequences;
    }

    @Override
    public int maxTokensPerStep() {
        return maxRows;
    }

    @Override
    public int maxTokensPerSequence() {
        return maxRows;
    }

    @Override
    public StepOutput execute(TokenBatch batch) {
        if (closed) {
            throw new IllegalStateException("this executor is closed");
        }
        int rows = batch.tokenCount();
        int sequences = batch.sequenceCount();
        if (rows > maxRows || sequences > maxSequences) {
            throw new IllegalArgumentException(
                    rows
                            + " tokens of "
                            + sequences
                            + " sequences exceed this executor's "
                            + maxRows
                            + " rows and "
                            + maxSequences
                            + " sequences");
        }
        for (int r = 0; r < rows; r++) {
            int slot = batch.kvSlots()[batch.sequenceOf()[r]];
            if (slot < 0 || slot >= maxSequences) {
                throw new IllegalArgumentException(
                        "token " + r + " names KV slot " + slot + " of " + maxSequences);
            }
            if (q8) {
                decodeEmbedding(batch.tokens()[r], r);
            } else {
                MemorySegment.copy(
                        embeddingTable,
                        batch.tokens()[r] * dimBytes,
                        embeddingBatch,
                        (long) r * dimBytes,
                        dimBytes);
            }
            rowPositions.set(r, batch.positions()[r]);
            rowSlots.set(r, slot);
        }
        for (int r = rows; r < maxRows; r++) {
            rowPositions.set(r, 0);
            rowSlots.set(r, -1);
        }
        for (int s = 0; s < maxSequences; s++) {
            sampleRows.set(s, s < sequences ? batch.lastTokenOf(s) : -1);
        }
        int activeRows = trimRows ? Math.min(maxRows, (rows + 15) & ~15) : maxRows;
        // Query tiles for attention: each sequence's rows in runs of up to queryTileRows(), so a
        // prompt chunk's rows share key/value reads; a decode row is a tile of one.
        // Prompt rows go to the tensor-core kernel in tiles; a generated token goes to the per-row
        // kernel. Which kernel a row takes depends on the row alone, so invariance holds.
        IntArray tiles = layers.queryTiles();
        IntArray decodeSlots = layers.decodeRowSlots();
        int tileCount = 0;
        int queryTile = layers.queryTileRows();
        for (int s = 0; s < sequences; s++) {
            int start = batch.queryStart()[s];
            int end = batch.queryStart()[s + 1];
            int promptEnd = start;
            while (promptEnd < end && batch.promptRows()[promptEnd]) {
                promptEnd++;
            }
            for (int r = start; r < promptEnd; r += queryTile) {
                tiles.set(2 * tileCount, r);
                tiles.set(2 * tileCount + 1, Math.min(queryTile, promptEnd - r));
                tileCount++;
            }
            for (int r = start; r < end; r++) {
                decodeSlots.set(r, r < promptEnd ? -1 : rowSlots.get(r));
            }
        }
        for (int r = rows; r < maxRows; r++) {
            decodeSlots.set(r, -1);
        }
        for (int t = tileCount; t < maxRows; t++) {
            tiles.set(2 * t, 0);
            tiles.set(2 * t + 1, 0);
        }
        layers.setActiveRows(activeRows, rows, tileCount);
        logitsRowsInt8.set(1, sequences);
        // A few sampling sequences take the skinny vocabulary GEMM; the guarded 128-row one then
        // launches a single tile that returns at once. Disabled, the count reads as large.
        boolean smallVocab = skinnyVocab && sequences <= TransformerRaggedBatchKernels.SKINNY_ROWS;
        logitsRows.set(0, skinnyVocab ? sequences : Integer.MAX_VALUE);
        int paddedSequences = (maxSequences + 127) & ~127;
        if (!q8)
            vocabGrid.setGlobalWork(
                    smallVocab ? 256 : paddedSequences / 128 * 256L,
                    smallVocab ? 1 : vocabSize / 128,
                    1);
        if (!q8)
            vocabSkinnyGrid.setGlobalWork(
                    smallVocab
                            ? (long) vocabSize
                                    / TransformerRaggedBatchKernels.SKINNY_COLS_PER_GROUP
                                    * 128
                            : 128,
                    1,
                    1);
        activationGrid.setGlobalWork((long) activeRows * dim, 1, 1);

        plan.withGraph(0).withGridScheduler(gridScheduler).execute();
        for (int layer = 0; layer < layerCount; layer++) {
            plan.withGraph(1 + layer).withGridScheduler(gridScheduler).execute();
        }
        plan.withGraph(logitsGraphIndex).withGridScheduler(gridScheduler).execute();

        int[] sampled = new int[sequences];
        for (int s = 0; s < sequences; s++) {
            sampled[s] = sampledTokens.get(s);
        }
        return new StepOutput(sampled);
    }

    /** One token's embedding into activation row {@code row}, decoded to FP32 on the host. */
    private void decodeEmbedding(int token, int row) {
        int base = row * dim;
        if (embeddingQ8 != null) {
            int blocksPerRow = dim / 32;
            for (int j = 0; j < dim; j++) {
                int block = (token * blocksPerRow + j / 32) * 34;
                float scale = embeddingQ8.getHalfFloat(block).getFloat32();
                activationRows.set(base + j, embeddingQ8.get(block + 2 + j % 32) * scale);
            }
        } else {
            for (int j = 0; j < dim; j++) {
                activationRows.set(base + j, embeddingHalf.get(token * dim + j).getFloat32());
            }
        }
    }

    @Override
    public int[] decodeStep(BatchSlots batch) {
        throw new UnsupportedOperationException(
                "the ragged executor is driven through execute(TokenBatch)");
    }

    @Override
    public boolean isStopToken(int token) {
        return stopTokens.contains(token);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            plan.freeDeviceMemory();
        }
    }

    private static WorkerGrid worker(int global, int local) {
        WorkerGrid1D grid = new WorkerGrid1D(global);
        grid.setLocalWork(local, 1, 1);
        return grid;
    }

    private static WorkerGrid mmaGrid(int paddedM, int n) {
        WorkerGrid2D grid = new WorkerGrid2D(paddedM / 128 * 256, n / 128);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }
}
