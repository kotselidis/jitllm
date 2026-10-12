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
import uk.ac.manchester.tornado.api.types.HalfFloat;
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
    // A prompt tile's split covers kvSplitMin positions: kvSplits * kvSplitMin spans the context,
    // so the split never depends on the tile's last row and a row's bits not on its tile-mates.
    private final int kvSplitMin = 256;
    // A generated token's split depends on its own length only, so it can be shorter: more splits
    // run in parallel for the short rows decode mostly has.
    private final int decodeSplitMin = Integer.getInteger("jitllm.ragged.decodeSplitMin", 64);
    private final boolean flash;

    // The int8 GEMMs compute the first int8Rows[1] rows; [0] is unused.
    private final IntArray int8Rows;
    private final ByteArray q8;
    private final FloatArray q8Scales;
    // Split-K partial sums, shared by the GEMMs of a layer in turn; one float when none splits.
    private final FloatArray splitPartial;
    // cuBLAS projections over FP16 copies of the weights, at a fixed row count (the step capacity):
    // the library picks its algorithm by shape, so the shape never changes and neither do the bits.
    private final boolean cublas = Boolean.getBoolean("jitllm.ragged.cublas");
    // Tensor-core GEMMs reading the Q8_0 blocks and the FP16 activations directly, decoding the
    // weights in shared memory: half cuBLAS's weight bytes, no FP16 copy, invariant by
    // construction.
    private final boolean q8f16 = !cublas && Boolean.getBoolean("jitllm.ragged.q8f16");
    private final List<WorkerGrid2D> q8f16Grids = new ArrayList<>();

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
        int dim = config.dim();
        int hid = config.hiddenDim();
        long partialFloats = 0;
        for (int[] nk : new int[][] {{dim, dim}, {config.kvDim(), dim}, {hid, dim}, {dim, hid}}) {
            int splits = gemmSplits(nk[0], nk[1]);
            if (splits > 1 && !cublas && !q8f16) {
                partialFloats = Math.max(partialFloats, (long) splits * paddedRows * nk[0]);
            }
        }
        if (Boolean.getBoolean("jitllm.ragged.q8f16")
                && !Boolean.getBoolean("jitllm.ragged.cublas")) {
            int qkv = dim + 2 * config.kvDim();
            for (int[] nk : new int[][] {{qkv, dim}, {dim, dim}, {2 * hid, dim}, {dim, hid}}) {
                int splits = q8f16Splits(nk[0], nk[1]);
                if (splits > 1) {
                    partialFloats = Math.max(partialFloats, (long) splits * paddedRows * nk[0]);
                }
            }
        }
        this.splitPartial = TornadoWorkspaces.floats((int) Math.max(1, partialFloats));
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
            splitPartial,
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
                    splitPartial,
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
        if (cublas || q8f16) {
            g.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                    weights.rms_ffn_weightLayered[layerIndex].asFloatArray());
        } else {
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
        }

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

        if (q8f16) {
            q8f16Gemm(
                    g,
                    "qkvProj",
                    stack(
                            weights.wqLayered[layerIndex],
                            weights.wkLayered[layerIndex],
                            weights.wvLayered[layerIndex]),
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    qkvDim,
                    dim);
        } else if (cublas) {
            nativeGemm(
                    g,
                    "qkvProj",
                    dequantize(
                            dim,
                            weights.wqLayered[layerIndex],
                            weights.wkLayered[layerIndex],
                            weights.wvLayered[layerIndex]),
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    qkvDim,
                    dim);
        } else {
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
        }

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
        int decodeSplitCfg = kvSplits | (decodeSplitMin << 16);
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
                decodeSplitCfg);
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

        if (q8f16) {
            q8f16Gemm(
                    g,
                    "woProj",
                    stack(weights.woLayered[layerIndex]),
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    dim,
                    dim);
        } else if (cublas) {
            nativeGemm(
                    g,
                    "woProj",
                    dequantize(dim, weights.woLayered[layerIndex]),
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    dim,
                    dim);
        } else {
            quantize(g, "wo_q8", state.workspace.attnOutFP16, dim);
            gemm(
                    g,
                    "woProj",
                    weights.woLayered[layerIndex],
                    state.workspace.woOut,
                    dim,
                    dim,
                    dim,
                    0);
        }

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

        if (q8f16) {
            q8f16Gemm(
                    g,
                    "gateUpProj",
                    stack(weights.w1Layered[layerIndex], weights.w3Layered[layerIndex]),
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    2 * hidDim,
                    dim);
        } else if (cublas) {
            nativeGemm(
                    g,
                    "gateUpProj",
                    dequantize(dim, weights.w1Layered[layerIndex], weights.w3Layered[layerIndex]),
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    2 * hidDim,
                    dim);
        } else {
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
        }

        rowTask(g, "swiglu", rows * hidDim, hidDim, 256);
        g.task(
                "swiglu",
                TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                context,
                state.workspace.wrapHbFP16Batch,
                state.workspace.gateUpResultBatch,
                hidDim);

        if (q8f16) {
            q8f16Gemm(
                    g,
                    "w2Proj",
                    stack(weights.w2Layered[layerIndex]),
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out,
                    dim,
                    hidDim);
        } else if (cublas) {
            nativeGemm(
                    g,
                    "w2Proj",
                    dequantize(hidDim, weights.w2Layered[layerIndex]),
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out,
                    dim,
                    hidDim);
        } else {
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
        }

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

    /**
     * {@code out[r][c] = x[r] . w[c]} for every one of the step's {@code rows} rows as one cuBLAS
     * GEMM, FP16 operands and FP32 accumulation; row-major on both sides, which is {@code out^T = w
     * . x^T} in cuBLAS's column-major terms. Always all {@code rows} rows, whatever the step uses.
     */
    private void nativeGemm(
            TaskGraph g,
            String task,
            HalfFloatArray w,
            HalfFloatArray x,
            FloatArray out,
            int n,
            int k) {
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
        g.libraryTask(
                task,
                uk.ac.manchester.tornado.cublas.CuBlas::cublasGemmExFP16FP32,
                1,
                0,
                n,
                rows,
                k,
                1.0f,
                w,
                k,
                x,
                k,
                0.0f,
                out,
                n);
    }

    /**
     * K splits for an {@code n x k} {@link TransformerRaggedBatchKernels#gemmQ8F16}: enough
     * 64-column workgroups for two per multiprocessor, at most eight, at least four blocks each.
     * Fixed per shape, so a row's sum is cut the same way whatever the step.
     */
    private static int q8f16Splits(int n, int k) {
        int forced = Integer.getInteger("jitllm.ragged.q8f16Splits", 0);
        int kBlocks = k / 32;
        if (forced > 0) {
            return Math.min(forced, kBlocks);
        }
        int sms = org.beehive.jitllm.backend.tornado.device.TornadoDevices.target().computeUnits();
        int tiles = n / TransformerRaggedBatchKernels.Q8F16_COLS;
        int splits = sms <= 0 ? 1 : (2 * sms + tiles - 1) / tiles;
        return Math.max(1, Math.min(Math.min(8, kBlocks / 4), splits));
    }

    /**
     * {@code out[r * n + c] = x[r] . w[c]} on {@link TransformerRaggedBatchKernels#gemmQ8F16}, the
     * Q8_0 blocks as the file stores them; split along K by {@link #q8f16Splits} and finished by an
     * ordered reduce. {@code out} rows are {@code n} wide (stacked projections fill them).
     */
    private void q8f16Gemm(
            TaskGraph g, String task, ByteArray w, HalfFloatArray x, FloatArray out, int n, int k) {
        int splits = q8f16Splits(n, k);
        g.transferToDevice(DataTransferMode.FIRST_EXECUTION, w);
        g.task(
                task,
                TransformerRaggedBatchKernels::gemmQ8F16,
                context,
                x,
                w,
                out,
                splits > 1 ? splitPartial : out,
                int8Rows,
                n,
                k,
                splits,
                paddedRows * n,
                n,
                0);
        int tokenGroups =
                (rows + TransformerRaggedBatchKernels.Q8F16_TOKENS - 1)
                        / TransformerRaggedBatchKernels.Q8F16_TOKENS;
        WorkerGrid2D grid =
                new WorkerGrid2D(
                        n / TransformerRaggedBatchKernels.Q8F16_COLS * splits * 128, tokenGroups);
        grid.setLocalWork(128, 1, 1);
        q8f16Grids.add(grid);
        grids.put(g.getTaskGraphName() + "." + task, grid);
        if (splits > 1) {
            g.task(
                    task + "_reduce",
                    TransformerRaggedBatchKernels::reduceSplitsPacked,
                    context,
                    splitPartial,
                    out,
                    paddedRows * n,
                    splits,
                    int8Rows,
                    n,
                    n,
                    0);
            rowTask(g, task + "_reduce", rows * n, n, 256);
        }
    }

    /** Q8_0 projections stacked row after row: each row's blocks are self-contained. */
    private static ByteArray stack(TornadoTensor... projections) {
        long total = 0;
        for (TornadoTensor t : projections) {
            total += t.asByteArray().getSize();
        }
        ByteArray out = new ByteArray((int) total);
        long at = 0;
        for (TornadoTensor t : projections) {
            ByteArray b = t.asByteArray();
            java.lang.foreign.MemorySegment.copy(
                    b.getSegment(), 0, out.getSegment(), at, b.getSize());
            at += b.getSize();
        }
        return out;
    }

    /**
     * The Q8_0 projections decoded to FP16 and stacked row after row, {@code k} inputs a row:
     * {@code q * scale} in FP32, rounded once to FP16.
     */
    private static HalfFloatArray dequantize(int k, TornadoTensor... projections) {
        int blocksPerRow = k / 32;
        long total = 0;
        for (TornadoTensor t : projections) {
            total += t.asByteArray().getSize() / (blocksPerRow * 34L) * k;
        }
        HalfFloatArray out = new HalfFloatArray((int) total);
        int base = 0;
        for (TornadoTensor t : projections) {
            ByteArray b = t.asByteArray();
            int n = (int) (b.getSize() / (blocksPerRow * 34L));
            for (int r = 0; r < n; r++) {
                for (int blk = 0; blk < blocksPerRow; blk++) {
                    int off = (r * blocksPerRow + blk) * 34;
                    float scale = b.getHalfFloat(off).getFloat32();
                    int dst = base + r * k + blk * 32;
                    for (int j = 0; j < 32; j++) {
                        out.set(dst + j, new HalfFloat(b.get(off + 2 + j) * scale));
                    }
                }
            }
            base += n * k;
        }
        return out;
    }

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

    private static final int SPLITS_MAX = 16;

    /**
     * K splits for an {@code n x k} projection: the count that best fills the device with one
     * 128-row tile, which is what a decode step launches. Fixed per shape, never per step, so a
     * row's sum is cut and added in the same order whatever else is in the batch. {@code
     * jitllm.ragged.q8Splits} overrides it (1 = no split).
     */
    private static int gemmSplits(int n, int k) {
        int forced = Integer.getInteger("jitllm.ragged.q8Splits", 0);
        int rounds = k / Int8GemmKernels.I8_BK;
        if (forced > 0) {
            return Math.min(forced, rounds);
        }
        int sms = org.beehive.jitllm.backend.tornado.device.TornadoDevices.target().computeUnits();
        if (sms <= 0) {
            return 1;
        }
        int tiles = n / Int8GemmKernels.I8_BN;
        long unsplit = (long) ((tiles + sms - 1) / sms) * rounds;
        int best = 1;
        long bestCost = unsplit;
        for (int splits = 2; splits <= SPLITS_MAX && splits <= rounds; splits++) {
            long waves = ((long) tiles * splits + sms - 1) / sms;
            long cost = waves * ((rounds + splits - 1) / splits);
            if (cost < bestCost) {
                best = splits;
                bestCost = cost;
            }
        }
        return best;
    }

    /**
     * {@code out[r * ldo + colOffset + c] = x8[r] . w[c]}, the Q8_0 blocks read in place; split
     * along K by {@link #gemmSplits} and finished by an ordered reduce.
     */
    private void gemm(
            TaskGraph g,
            String task,
            TornadoTensor w,
            FloatArray out,
            int n,
            int k,
            int ldo,
            int col) {
        int splits = gemmSplits(n, k);
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
                splits > 1 ? splitPartial : out,
                splits,
                int8Rows,
                ldo,
                col);
        int columns = n / Int8GemmKernels.I8_BN * splits;
        WorkerGrid2D grid =
                new WorkerGrid2D(
                        paddedRows / Int8GemmKernels.I8_BM * Int8GemmKernels.Q8_GEMM_THREADS,
                        columns);
        grid.setLocalWork(Int8GemmKernels.Q8_GEMM_THREADS, 1, 1);
        gemmGrids.add(grid);
        gemmGridColumns.add(columns);
        grids.put(g.getTaskGraphName() + "." + task, grid);
        if (splits > 1) {
            g.task(
                    task + "_reduce",
                    TransformerRaggedBatchKernels::reduceSplitsPacked,
                    context,
                    splitPartial,
                    out,
                    paddedRows * n,
                    splits,
                    int8Rows,
                    n,
                    ldo,
                    col);
            rowTask(g, task + "_reduce", rows * n, n, 256);
        }
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
        int tokenGroups =
                (activeRows + TransformerRaggedBatchKernels.Q8F16_TOKENS - 1)
                        / TransformerRaggedBatchKernels.Q8F16_TOKENS;
        for (WorkerGrid2D grid : q8f16Grids) {
            grid.setGlobalWork(grid.getGlobalWork()[0], Math.max(1, tokenGroups), 1);
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
