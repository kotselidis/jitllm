package org.beehive.jitllm.backend.tornado.layers.type.q8_0.decode;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.batch.RaggedLayers;
import org.beehive.jitllm.backend.tornado.kernels.Int8GemmKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerRaggedBatchKernels;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Ragged-batch transformer layers for Llama with Q8_0 weights: rows are tokens of several
 * sequences, each at its own position in its own KV slot, as in the FP16 ragged layers.
 *
 * <p>Everything but the projections is the FP16 ragged path's: RMSNorm, RoPE with the per-row
 * key/value write, tensor-core attention for prompt rows and the per-row kernel for generated
 * tokens, SwiGLU and the residuals. Each projection quantizes its activation to int8 in 32-wide
 * blocks per row and runs {@link Int8GemmKernels#gemmInt8Q8_0} on the Q8_0 blocks where they lie
 * (or their packed tiles), accumulating in FP32 — the arithmetic the single-token path keeps, not
 * the FP16 accumulation of the older Q8_0 tensor-core GEMM.
 *
 * <p>Batch invariance: activation scales are per row and block, the GEMM computes each output from
 * its own row, and K is not split, so a token's result does not depend on the step it ran in, its
 * company, or the server's token budget.
 */
public class LlamaQ8_0LayersRagged implements RaggedLayers {

    static final int RMS_LOCAL_SIZE = 256;

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int rows;
    private final int paddedRows;
    private final HalfFloatArray keyPool;
    private final HalfFloatArray valuePool;
    private final IntArray rowPositions;
    private final IntArray rowSlots;
    private final IntArray blockTable;
    private final int blockCfg;

    private final IntArray queryTiles;
    private final IntArray decodeRowSlots;
    private final FloatArray attnPartials;
    private final int kvSplits;
    private final int kvSplitMin = 256;
    private final boolean flash;

    // The int8 GEMMs compute the first int8Rows[1] rows; [0] is unused.
    private final IntArray int8Rows;
    private final ByteArray q8;
    private final FloatArray q8Scales;

    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    private final List<WorkerGrid1D> rowGrids = new ArrayList<>();
    private final List<int[]> rowGridWidths = new ArrayList<>();
    private final List<WorkerGrid2D> gemmGrids = new ArrayList<>();
    private final List<Integer> gemmGridColumns = new ArrayList<>();
    private WorkerGrid1D flashWorker;
    private WorkerGrid1D decodeWorker;
    private final java.util.Map<String, WorkerGrid> grids = new java.util.LinkedHashMap<>();

    public LlamaQ8_0LayersRagged(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            int rows,
            int contextLength,
            HalfFloatArray keyPool,
            HalfFloatArray valuePool,
            IntArray rowPositions,
            IntArray blockTable,
            int blockSize,
            int maxBlocksPerSlot,
            IntArray rowSlots) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.rows = rows;
        this.paddedRows = (rows + 127) & ~127;
        this.keyPool = keyPool;
        this.valuePool = valuePool;
        this.rowPositions = rowPositions;
        this.rowSlots = rowSlots;
        this.blockTable = blockTable;
        this.blockCfg = blockSize | (maxBlocksPerSlot << 16);
        this.kvSplits =
                Math.max(
                        Integer.getInteger("jitllm.ragged.kvSplits", 16),
                        (contextLength + kvSplitMin - 1) / kvSplitMin);
        this.flash =
                config.headSize() == 64
                        && TransformerRaggedBatchKernels.FLASH_TILE_ROWS % config.kvMul() == 0;
        if (config.kvMul() * config.headSize() > 1024) {
            throw new IllegalArgumentException(
                    "ragged Q8_0 attention groups a KV head's query heads in one workgroup, which"
                            + " needs kvMul * headSize <= 1024");
        }
        this.queryTiles = new IntArray(2 * rows);
        this.decodeRowSlots = new IntArray(rows);
        this.attnPartials =
                new FloatArray(rows * config.numberOfHeads() * kvSplits * (config.headSize() + 2));
        this.int8Rows = new IntArray(2);
        int widest = Math.max(config.dim(), config.hiddenDim());
        this.q8 = TornadoWorkspaces.bytes(paddedRows * widest);
        this.q8Scales = TornadoWorkspaces.floats(paddedRows * widest / 32);
        List<ImmutableTaskGraph> graphs = new ArrayList<>();
        for (int l = 0; l < config.numberOfLayers(); l++) {
            graphs.add(createLayerTaskGraph(l).snapshot());
        }
        this.layerITGs = graphs;
    }

    private String graphName(int layerIndex) {
        return "raggedQ8Layer_" + layerIndex;
    }

    // @formatter:off
    private TaskGraph createLayerTaskGraph(int layerIndex) {
        String name = graphName(layerIndex);
        if (layerIndex == config.numberOfLayers() - 1) lastLayerTaskGraphID = name;
        TaskGraph g = new TaskGraph(name);
        Object[] shared = {
            context,
            rowPositions,
            rowSlots,
            queryTiles,
            decodeRowSlots,
            int8Rows,
            blockTable,
            q8,
            q8Scales,
            attnPartials,
            keyPool,
            valuePool,
            state.workspace.attnScaleBatch,
            state.workspace.ffnScaleBatch,
            state.workspace.wrapXbFP16Batch,
            state.workspace.qkvResultBatch,
            state.workspace.normedXFFNFP16,
            state.workspace.gateUpResultBatch,
            state.workspace.attnOutFP16,
            state.workspace.woOut,
            state.workspace.wrapHbFP16Batch,
            state.workspace.w2Out
        };
        if (layerIndex == 0) {
            g.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    rowPositions,
                    rowSlots,
                    queryTiles,
                    decodeRowSlots,
                    int8Rows,
                    blockTable);
            g.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    q8,
                    q8Scales,
                    attnPartials,
                    keyPool,
                    valuePool,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            g.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
        } else {
            String pred = graphName(layerIndex - 1);
            g.consumeFromDevice(pred, shared);
            g.consumeFromDevice(pred, state.workspace.wrapXBatch);
        }
        state.workspace.packedRepack.upload(
                g,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex],
                weights.wkLayered[layerIndex],
                weights.wvLayered[layerIndex],
                weights.woLayered[layerIndex],
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex],
                weights.w2Layered[layerIndex],
                weights.w3Layered[layerIndex]);

        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        int qkvDim = dim + 2 * kvDim;

        rowTask(g, "batch_attn_rms", rows * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);
        g.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        rowTask(g, "batch_attn_rms_apply", rows * dim, dim, 256);
        g.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        quantize(g, "qkv_q8", state.workspace.wrapXbFP16Batch, dim);
        gemm(
                g,
                "qProj",
                weights.wqLayered[layerIndex],
                state.workspace.qkvResultBatch,
                dim,
                dim,
                qkvDim,
                0);
        gemm(
                g,
                "kProj",
                weights.wkLayered[layerIndex],
                state.workspace.qkvResultBatch,
                kvDim,
                dim,
                qkvDim,
                dim);
        gemm(
                g,
                "vProj",
                weights.wvLayered[layerIndex],
                state.workspace.qkvResultBatch,
                kvDim,
                dim,
                qkvDim,
                dim + kvDim);

        rowTask(g, "batch_rope_kv", rows * (dim / 2), dim / 2, 512);
        g.task(
                "batch_rope_kv",
                TransformerRaggedBatchKernels::raggedPagedRopeWithKVCachePackedKVFP16,
                context,
                rowPositions,
                rowSlots,
                blockTable,
                state.workspace.qkvResultBatch,
                keyPool,
                valuePool,
                config.ropeTheta(),
                kvDim,
                config.headSize(),
                layerIndex,
                config.numberOfLayers(),
                blockCfg,
                dim);

        int splitCfg = kvSplits | (kvSplitMin << 16);
        int nKvHeads = config.numberOfHeads() / config.kvMul();
        if (flash) {
            g.task(
                    "batch_attention",
                    TransformerRaggedBatchKernels::raggedFlashAttentionMMA,
                    context,
                    rowPositions,
                    rowSlots,
                    queryTiles,
                    blockTable,
                    state.workspace.qkvResultBatch,
                    keyPool,
                    valuePool,
                    attnPartials,
                    config.numberOfHeads(),
                    config.headSize(),
                    kvDim,
                    config.kvMul(),
                    layerIndex,
                    config.numberOfLayers(),
                    blockCfg,
                    splitCfg);
        }
        g.task(
                "batch_attention_decode",
                TransformerRaggedBatchKernels::raggedSplitKvAttentionPartialGqa,
                context,
                rowPositions,
                flash ? decodeRowSlots : rowSlots,
                blockTable,
                state.workspace.qkvResultBatch,
                keyPool,
                valuePool,
                attnPartials,
                config.numberOfHeads(),
                config.headSize(),
                kvDim,
                config.kvMul(),
                layerIndex,
                config.numberOfLayers(),
                blockCfg,
                splitCfg);
        g.task(
                "batch_attention_combine",
                TransformerRaggedBatchKernels::raggedSplitKvAttentionCombine,
                context,
                rowSlots,
                attnPartials,
                state.workspace.attnOutFP16,
                config.numberOfHeads(),
                config.headSize(),
                kvSplits);
        int gqaLocal = config.kvMul() * config.headSize();
        rowTask(
                g,
                "batch_attention_decode",
                rows * nKvHeads * kvSplits * gqaLocal,
                nKvHeads * kvSplits * gqaLocal,
                gqaLocal);
        rowTask(
                g,
                "batch_attention_combine",
                rows * config.numberOfHeads() * config.headSize(),
                config.numberOfHeads() * config.headSize(),
                config.headSize());

        quantize(g, "wo_q8", state.workspace.attnOutFP16, dim);
        gemm(g, "woProj", weights.woLayered[layerIndex], state.workspace.woOut, dim, dim, dim, 0);

        rowTask(g, "batch_ffn_rms", rows * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);
        g.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.woOut,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        rowTask(g, "batch_ffn_rms_apply", rows * dim, dim, 256);
        g.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);

        quantize(g, "gateup_q8", state.workspace.normedXFFNFP16, dim);
        gemm(
                g,
                "gateProj",
                weights.w1Layered[layerIndex],
                state.workspace.gateUpResultBatch,
                hidDim,
                dim,
                2 * hidDim,
                0);
        gemm(
                g,
                "upProj",
                weights.w3Layered[layerIndex],
                state.workspace.gateUpResultBatch,
                hidDim,
                dim,
                2 * hidDim,
                hidDim);

        rowTask(g, "swiglu", rows * hidDim, hidDim, 256);
        g.task(
                "swiglu",
                TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                context,
                state.workspace.wrapHbFP16Batch,
                state.workspace.gateUpResultBatch,
                hidDim);

        quantize(g, "w2_q8", state.workspace.wrapHbFP16Batch, hidDim);
        gemm(
                g,
                "w2Proj",
                weights.w2Layered[layerIndex],
                state.workspace.w2Out,
                dim,
                hidDim,
                dim,
                0);

        rowTask(g, "w2Resid", rows * dim, dim, 256);
        g.task(
                "w2Resid",
                TransformerBatchPrefillKernels::batchedResidualAddFP32,
                context,
                state.workspace.wrapXBatch,
                state.workspace.w2Out);

        g.persistOnDevice(state.workspace.wrapXBatch, keyPool, valuePool);
        if (flash) {
            WorkerGrid1D grid = new WorkerGrid1D(rows * nKvHeads * kvSplits * 32);
            grid.setLocalWork(32, 1, 1);
            if (flashWorker == null) {
                flashWorker = grid;
            }
            grids.put(name + ".batch_attention", flashWorker);
        }
        return g;
    }

    // @formatter:on

    /** Quantizes the first rows of {@code x} ({@code k} per row) to int8 blocks for the GEMMs. */
    private void quantize(TaskGraph g, String task, HalfFloatArray x, int k) {
        g.task(
                task,
                Int8GemmKernels::quantizeActivationsQ8WarpFP16,
                context,
                x,
                q8,
                q8Scales,
                paddedRows * k);
        rowTask(g, task, rows * k, k, 256);
    }

    /** {@code out[r * ldo + colOffset + c] = x8[r] . w[c]}, the Q8_0 blocks read in place. */
    private void gemm(
            TaskGraph g,
            String task,
            TornadoTensor w,
            FloatArray out,
            int n,
            int k,
            int ldo,
            int col) {
        g.task(
                task,
                w.isPackedQ8()
                        ? Int8GemmKernels::gemmInt8Q8_0Packed
                        : Int8GemmKernels::gemmInt8Q8_0,
                context,
                q8,
                q8Scales,
                w.asByteArray(),
                out,
                out,
                paddedRows,
                n,
                k,
                Int8GemmKernels.EPILOGUE_STORE,
                out,
                1,
                int8Rows,
                ldo,
                col);
        WorkerGrid2D grid =
                new WorkerGrid2D(
                        paddedRows / Int8GemmKernels.I8_BM * Int8GemmKernels.Q8_GEMM_THREADS,
                        n / Int8GemmKernels.I8_BN);
        grid.setLocalWork(Int8GemmKernels.Q8_GEMM_THREADS, 1, 1);
        gemmGrids.add(grid);
        gemmGridColumns.add(n / Int8GemmKernels.I8_BN);
        grids.put(g.getTaskGraphName() + "." + task, grid);
    }

    /** A 1-D grid of {@code perRow} lanes per row, resized with the step's rows. */
    private void rowTask(TaskGraph g, String task, int global, int perRow, int local) {
        WorkerGrid1D grid = new WorkerGrid1D(global);
        grid.setLocalWork(local, 1, 1);
        rowGrids.add(grid);
        rowGridWidths.add(new int[] {perRow});
        grids.put(g.getTaskGraphName() + "." + task, grid);
    }

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        grids.forEach(scheduler::addWorkerGrid);
    }

    @Override
    public void setActiveRows(int activeRows, int realRows, int tiles) {
        int8Rows.set(1, realRows);
        for (int i = 0; i < rowGrids.size(); i++) {
            rowGrids.get(i).setGlobalWork((long) activeRows * rowGridWidths.get(i)[0], 1, 1);
        }
        int mTiles = (activeRows + 127) / 128;
        for (int i = 0; i < gemmGrids.size(); i++) {
            gemmGrids
                    .get(i)
                    .setGlobalWork(
                            (long) mTiles * Int8GemmKernels.Q8_GEMM_THREADS,
                            gemmGridColumns.get(i),
                            1);
        }
        if (flashWorker != null) {
            int nKvHeads = config.numberOfHeads() / config.kvMul();
            flashWorker.setGlobalWork((long) Math.max(tiles, 1) * nKvHeads * kvSplits * 32, 1, 1);
        }
    }

    @Override
    public IntArray queryTiles() {
        return queryTiles;
    }

    @Override
    public IntArray decodeRowSlots() {
        return decodeRowSlots;
    }

    @Override
    public int queryTileRows() {
        return flash ? TransformerRaggedBatchKernels.FLASH_TILE_ROWS / config.kvMul() : 1;
    }

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerITGs;
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }
}
