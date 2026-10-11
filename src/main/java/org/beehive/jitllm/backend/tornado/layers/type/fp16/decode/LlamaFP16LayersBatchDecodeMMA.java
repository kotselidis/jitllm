package org.beehive.jitllm.backend.tornado.layers.type.fp16.decode;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerRaggedBatchKernels;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Batched-DECODE transformer-layer TaskGraphs: B independent sequences advancing one token per
 * step, each with its own KV region and its own position.
 *
 * <p>Structurally identical to {@code LlamaFP16LayersBatchPrefillMMA} (same 12-task MMA pipeline:
 * RMS → fused QKV GEMM → RoPE+KV → flash attention → Wo → FFN RMS → fused gate/up GEMM → SwiGLU →
 * W2 → residual). The only differences are the two KV-addressing kernels, which take a per-slot
 * position array and a B-sized KV cache ({@code batchIdx} stride = {@code
 * numLayers*contextLength*kvDim}):
 *
 * <ul>
 *   <li>{@code batch_rope_kv} → {@link
 *       TransformerBatchPrefillKernels#batchedDecodeRopeWithKVCachePacked}
 *   <li>{@code batch_attention} → {@link
 *       TransformerBatchPrefillKernels#batchedDecodeAttentionFP16Out}
 * </ul>
 *
 * <p>The KV cache and {@code seqPositions} are supplied by the engine (not State), so the context
 * length can be capped independently to fit B×L×ctx×kvDim in VRAM.
 *
 * <p><b>Ragged mode.</b> Given {@code rowSlots}, a row is a token rather than a sequence: row
 * {@code r} sits at {@code seqPositions[r]} in the sequence whose KV slot is {@code rowSlots[r]},
 * so one pass carries prompt chunks and decode tokens of several sequences, and a negative slot is
 * a padding row. Only the two KV-addressing kernels change, to their {@link
 * TransformerRaggedBatchKernels} twins; every other task already works row by row.
 */
public class LlamaFP16LayersBatchDecodeMMA
        implements org.beehive.jitllm.backend.tornado.batch.RaggedLayers {

    static final int RMS_LOCAL_SIZE = 256;

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final int paddedBatch;
    private final int decodeCtx;
    // The engine's pool in the representation it was built with: exactly one pair is set.
    private final FloatArray keyCacheBatch;
    private final FloatArray valueCacheBatch;
    private final HalfFloatArray keyCacheBatchFP16;
    private final HalfFloatArray valueCacheBatchFP16;
    private final IntArray seqPositions;
    // Paging (optional): when blockTable != null, KV is addressed through a block table.
    private final boolean paged;
    private final IntArray blockTable;
    private final int blockSize;
    private final int maxBlocksPerSlot;
    // Ragged mode (optional): each row's KV slot, or negative for a padding row.
    private final IntArray rowSlots;

    /**
     * Ragged attention as split-K: each row's context is cut into {@code kvSplits} chunks read by
     * separate workgroups, then combined. Off means one workgroup walks a row's whole context.
     */
    private final boolean splitKv = !"false".equals(System.getProperty("jitllm.ragged.splitKv"));

    /**
     * Splits per row, at least enough that a split is always {@code kvSplitMin} positions — which
     * the tiled attention kernel needs, so every row of a query tile splits alike.
     */
    private final int kvSplits;

    /**
     * Split-K with the query heads of one KV head in one workgroup, so each key/value tile is read
     * once per KV head rather than once per query head. Same output bits as per-head.
     */
    private final boolean gqaAttention =
            !"false".equals(System.getProperty("jitllm.ragged.gqaAttention"));

    /**
     * Attention on tensor cores for every row ({@link
     * TransformerRaggedBatchKernels#raggedFlashAttentionMMA}): query tiles of up to {@code 16 /
     * kvMul} rows of one sequence share each key/value tile. Batch invariant, but not the scalar
     * kernels' bits.
     */
    private final boolean flashAttention =
            !"false".equals(System.getProperty("jitllm.ragged.flashAttention"));

    // Per query tile, its first row and row count; written by the executor each step.
    private final IntArray queryTiles;

    // A generated token's KV slot per row, -1 for a prompt row: the scalar kernel takes generated
    // tokens, whose single row a 16-row tensor-core tile would mostly waste.
    private final IntArray decodeRowSlots;

    /** Fewest positions a split reads: a shorter context uses fewer splits, down to one. */
    private final int kvSplitMin = Integer.getInteger("jitllm.ragged.kvSplitMin", 256);

    // Split-K partial results: per (row, head, split), headSize values plus max and sum.
    private final FloatArray attnPartials;

    /**
     * Steps of at most {@link TransformerRaggedBatchKernels#SKINNY_ROWS} tokens run their
     * projections on the skinny tensor-core GEMM instead of the 128-row one. Same output bits.
     */
    private final boolean skinnyGemm =
            !"false".equals(System.getProperty("jitllm.ragged.skinnyGemm"));

    /**
     * Steps of 9 to {@link TransformerRaggedBatchKernels#MID_ROWS} tokens run their projections on
     * the 32 x 64-tile GEMM instead of the 128-row one. Same output bits.
     */
    private final boolean midGemm = !"false".equals(System.getProperty("jitllm.ragged.midGemm"));

    // The step's real token count, read on the device to pick the GEMM; ragged mode only.
    private final IntArray stepRows;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    // The row-proportional grids, kept so a ragged step can launch only the rows it uses.
    private WorkerGrid rmsWorker;
    private WorkerGrid rmsApplyWorker;
    private WorkerGrid ffnRmsApplyWorker;
    private WorkerGrid ropeWorker;
    private WorkerGrid attnWorker;
    private WorkerGrid attnCombineWorker;
    private WorkerGrid attnDecodeWorker;
    private WorkerGrid mmaQkvWorker;
    private WorkerGrid mmaDimWorker;
    private WorkerGrid mmaGateUpWorker;
    private WorkerGrid ewDimWorker;
    private WorkerGrid ewHidWorker;
    private WorkerGrid skinnyDimWorker;
    private WorkerGrid skinnyKvWorker;
    private WorkerGrid skinnyHidWorker;
    private WorkerGrid midDimWorker;
    private WorkerGrid midKvWorker;
    private WorkerGrid midHidWorker;

    public LlamaFP16LayersBatchDecodeMMA(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            int batchSize,
            int decodeCtx,
            Object keyCacheBatch,
            Object valueCacheBatch,
            IntArray seqPositions) {
        this(
                state,
                weights,
                config,
                batchSize,
                decodeCtx,
                keyCacheBatch,
                valueCacheBatch,
                seqPositions,
                null,
                0,
                0);
    }

    /** Paged constructor: keyCacheBatch/valueCacheBatch are the shared block pools. */
    public LlamaFP16LayersBatchDecodeMMA(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            int batchSize,
            int decodeCtx,
            Object keyCacheBatch,
            Object valueCacheBatch,
            IntArray seqPositions,
            IntArray blockTable,
            int blockSize,
            int maxBlocksPerSlot) {
        this(
                state,
                weights,
                config,
                batchSize,
                decodeCtx,
                keyCacheBatch,
                valueCacheBatch,
                seqPositions,
                blockTable,
                blockSize,
                maxBlocksPerSlot,
                null);
    }

    /**
     * Ragged constructor when {@code rowSlots} is given: {@code batchSize} rows are tokens, each at
     * {@code seqPositions[row]} in the sequence whose KV slot {@code rowSlots[row]} names. Paged
     * FP16 pools only. With {@code rowSlots} null, a row is a sequence and its slot is its index.
     */
    public LlamaFP16LayersBatchDecodeMMA(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            int batchSize,
            int decodeCtx,
            Object keyCacheBatch,
            Object valueCacheBatch,
            IntArray seqPositions,
            IntArray blockTable,
            int blockSize,
            int maxBlocksPerSlot,
            IntArray rowSlots) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.paddedBatch = (batchSize + 127) & ~127;
        this.decodeCtx = decodeCtx;
        this.keyCacheBatch = keyCacheBatch instanceof FloatArray f ? f : null;
        this.valueCacheBatch = valueCacheBatch instanceof FloatArray f ? f : null;
        this.keyCacheBatchFP16 = keyCacheBatch instanceof HalfFloatArray h ? h : null;
        this.valueCacheBatchFP16 = valueCacheBatch instanceof HalfFloatArray h ? h : null;
        if (blockTable == null && this.keyCacheBatchFP16 != null) {
            throw new IllegalArgumentException(
                    "the contiguous (non-paged) batch cache is FP32 only; the FP16 pool is paged");
        }
        if ((this.keyCacheBatch == null) == (this.keyCacheBatchFP16 == null)) {
            throw new IllegalArgumentException(
                    "the pool must be one FloatArray or HalfFloatArray pair");
        }
        this.seqPositions = seqPositions;
        this.paged = blockTable != null;
        this.blockTable = blockTable;
        this.blockSize = blockSize;
        this.maxBlocksPerSlot = maxBlocksPerSlot;
        this.rowSlots = rowSlots;
        if (rowSlots != null && (!paged || keyCacheBatchFP16 == null)) {
            throw new IllegalArgumentException("ragged rows need a paged FP16 key/value pool");
        }
        this.kvSplits =
                Math.max(
                        Integer.getInteger("jitllm.ragged.kvSplits", 16),
                        (decodeCtx + kvSplitMin - 1) / kvSplitMin);
        this.queryTiles = rowSlots != null ? new IntArray(2 * batchSize) : null;
        this.decodeRowSlots = rowSlots != null ? new IntArray(batchSize) : null;
        // [0] the step's tokens, [1] whether steps of 9-32 tokens take the mid GEMM.
        this.stepRows = rowSlots != null ? new IntArray(2) : null;
        this.attnPartials =
                rowSlots != null && splitKv
                        ? new FloatArray(
                                batchSize
                                        * config.numberOfHeads()
                                        * kvSplits
                                        * (config.headSize() + 2))
                        : null;
        this.layerITGs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::createLayerTaskGraph)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    // @formatter:off
    private TaskGraph createLayerTaskGraph(int layerIndex) {
        String graphName = "batchDecodeLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) lastLayerTaskGraphID = graphName;

        TaskGraph g = new TaskGraph(graphName);

        if (layerIndex == 0) {
            // Per-slot positions change every decode step.
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, seqPositions);
            if (paged) {
                g.transferToDevice(DataTransferMode.EVERY_EXECUTION, blockTable);
            }
            if (rowSlots != null) {
                g.transferToDevice(
                        DataTransferMode.EVERY_EXECUTION,
                        rowSlots,
                        stepRows,
                        queryTiles,
                        decodeRowSlots);
            }
            if (attnPartials != null) {
                g.transferToDevice(DataTransferMode.FIRST_EXECUTION, attnPartials);
            }
            g.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyPool(),
                    valuePool(),
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            g.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
        } else {
            String pred = "batchDecodeLayer_" + (layerIndex - 1);
            if (paged) {
                g.consumeFromDevice(pred, context, blockTable);
            }
            if (rowSlots != null) {
                g.consumeFromDevice(pred, context, rowSlots, stepRows, queryTiles, decodeRowSlots);
            }
            if (attnPartials != null) {
                g.consumeFromDevice(pred, context, attnPartials);
            }
            g.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    seqPositions,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    keyPool(),
                    valuePool(),
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.woOut,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
        }

        g.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                weights.woLayered[layerIndex].asHalfFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray());

        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();

        g.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);

        g.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        if (skinny()) {
            g.task(
                    "qkvProj",
                    TransformerRaggedBatchKernels::gemmMMAQKVGuarded,
                    context,
                    stepRows,
                    state.workspace.wrapXbFP16Batch,
                    weights.wqLayered[layerIndex].asHalfFloatArray(),
                    weights.wkLayered[layerIndex].asHalfFloatArray(),
                    weights.wvLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.qkvResultBatch,
                    paddedBatch,
                    dim,
                    kvDim,
                    dim);
        } else {
            g.task(
                    "qkvProj",
                    TransformerBatchPrefillKernels::gemmMMAQKV,
                    context,
                    state.workspace.wrapXbFP16Batch,
                    weights.wqLayered[layerIndex].asHalfFloatArray(),
                    weights.wkLayered[layerIndex].asHalfFloatArray(),
                    weights.wvLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.qkvResultBatch,
                    paddedBatch,
                    dim,
                    kvDim,
                    dim);
        }
        if (skinny()) {
            int qkvStride = dim + 2 * kvDim;
            skinnyTask(
                    g,
                    "qProj_skinny",
                    state.workspace.wrapXbFP16Batch,
                    weights.wqLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.qkvResultBatch,
                    dim,
                    qkvStride,
                    0);
            skinnyTask(
                    g,
                    "kProj_skinny",
                    state.workspace.wrapXbFP16Batch,
                    weights.wkLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.qkvResultBatch,
                    dim,
                    qkvStride,
                    dim);
            skinnyTask(
                    g,
                    "vProj_skinny",
                    state.workspace.wrapXbFP16Batch,
                    weights.wvLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.qkvResultBatch,
                    dim,
                    qkvStride,
                    dim + kvDim);
            midTask(
                    g,
                    "qProj_mid",
                    state.workspace.wrapXbFP16Batch,
                    weights.wqLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.qkvResultBatch,
                    dim,
                    dim + 2 * kvDim,
                    0);
            midTask(
                    g,
                    "kProj_mid",
                    state.workspace.wrapXbFP16Batch,
                    weights.wkLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.qkvResultBatch,
                    dim,
                    dim + 2 * kvDim,
                    dim);
            midTask(
                    g,
                    "vProj_mid",
                    state.workspace.wrapXbFP16Batch,
                    weights.wvLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.qkvResultBatch,
                    dim,
                    dim + 2 * kvDim,
                    dim + kvDim);
        }

        // DECODE: per-slot position + per-slot KV region (paged via block table, or contiguous).
        if (paged) {
            int blockCfg = blockSize | (maxBlocksPerSlot << 16);
            if (rowSlots != null) {
                g.task(
                        "batch_rope_kv",
                        TransformerRaggedBatchKernels::raggedPagedRopeWithKVCachePackedKVFP16,
                        context,
                        seqPositions,
                        rowSlots,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatchFP16,
                        valueCacheBatchFP16,
                        config.ropeTheta(),
                        kvDim,
                        config.headSize(),
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg,
                        dim);
            } else if (keyCacheBatchFP16 != null) {
                g.task(
                        "batch_rope_kv",
                        TransformerBatchPrefillKernels
                                ::batchedDecodePagedRopeWithKVCachePackedKVFP16,
                        context,
                        seqPositions,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatchFP16,
                        valueCacheBatchFP16,
                        config.ropeTheta(),
                        kvDim,
                        config.headSize(),
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg,
                        dim);
            } else {
                g.task(
                        "batch_rope_kv",
                        TransformerBatchPrefillKernels::batchedDecodePagedRopeWithKVCachePacked,
                        context,
                        seqPositions,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatch,
                        valueCacheBatch,
                        config.ropeTheta(),
                        kvDim,
                        config.headSize(),
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg,
                        dim);
            }

            if (tiled()) {
                g.task(
                        "batch_attention",
                        TransformerRaggedBatchKernels::raggedFlashAttentionMMA,
                        context,
                        seqPositions,
                        rowSlots,
                        queryTiles,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatchFP16,
                        valueCacheBatchFP16,
                        attnPartials,
                        config.numberOfHeads(),
                        config.headSize(),
                        kvDim,
                        config.kvMul(),
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg,
                        kvSplits | (kvSplitMin << 16));
                // Generated tokens: the per-row grouped kernel, prompt rows skipped by slot -1.
                g.task(
                        "batch_attention_decode",
                        TransformerRaggedBatchKernels::raggedSplitKvAttentionPartialGqa,
                        context,
                        seqPositions,
                        decodeRowSlots,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatchFP16,
                        valueCacheBatchFP16,
                        attnPartials,
                        config.numberOfHeads(),
                        config.headSize(),
                        kvDim,
                        config.kvMul(),
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg,
                        kvSplits | (kvSplitMin << 16));
            } else if (attnPartials != null) {
                g.task(
                        "batch_attention",
                        groupedAttention()
                                ? TransformerRaggedBatchKernels::raggedSplitKvAttentionPartialGqa
                                : TransformerRaggedBatchKernels::raggedSplitKvAttentionPartial,
                        context,
                        seqPositions,
                        rowSlots,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatchFP16,
                        valueCacheBatchFP16,
                        attnPartials,
                        config.numberOfHeads(),
                        config.headSize(),
                        kvDim,
                        config.kvMul(),
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg,
                        kvSplits | (kvSplitMin << 16));
            }
            if (attnPartials != null) {
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
            } else if (rowSlots != null) {
                g.task(
                        "batch_attention",
                        TransformerRaggedBatchKernels::raggedPagedAttentionFP16OutKVFP16,
                        context,
                        seqPositions,
                        rowSlots,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatchFP16,
                        valueCacheBatchFP16,
                        state.workspace.attnOutFP16,
                        config.numberOfHeads(),
                        config.headSize(),
                        kvDim,
                        config.kvMul(),
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg);
            } else if (keyCacheBatchFP16 != null) {
                g.task(
                        "batch_attention",
                        TransformerBatchPrefillKernels::batchedDecodePagedAttentionFP16OutKVFP16,
                        context,
                        seqPositions,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatchFP16,
                        valueCacheBatchFP16,
                        state.workspace.attnOutFP16,
                        config.numberOfHeads(),
                        config.headSize(),
                        kvDim,
                        config.kvMul(),
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg);
            } else {
                g.task(
                        "batch_attention",
                        TransformerBatchPrefillKernels::batchedDecodePagedAttentionFP16Out,
                        context,
                        seqPositions,
                        blockTable,
                        state.workspace.qkvResultBatch,
                        keyCacheBatch,
                        valueCacheBatch,
                        state.workspace.attnOutFP16,
                        config.numberOfHeads(),
                        config.headSize(),
                        kvDim,
                        config.kvMul(),
                        layerIndex,
                        config.numberOfLayers(),
                        blockCfg);
            }
        } else {
            g.task(
                    "batch_rope_kv",
                    TransformerBatchPrefillKernels::batchedDecodeRopeWithKVCachePacked,
                    context,
                    seqPositions,
                    state.workspace.qkvResultBatch,
                    keyCacheBatch,
                    valueCacheBatch,
                    config.ropeTheta(),
                    kvDim,
                    config.headSize(),
                    layerIndex,
                    config.numberOfLayers(),
                    decodeCtx,
                    dim);

            g.task(
                    "batch_attention",
                    TransformerBatchPrefillKernels::batchedDecodeAttentionFP16Out,
                    context,
                    seqPositions,
                    state.workspace.qkvResultBatch,
                    keyCacheBatch,
                    valueCacheBatch,
                    state.workspace.attnOutFP16,
                    config.numberOfHeads(),
                    config.headSize(),
                    kvDim,
                    config.kvMul(),
                    layerIndex,
                    config.numberOfLayers(),
                    decodeCtx,
                    dim);
        }

        if (skinny()) {
            g.task(
                    "woProj",
                    TransformerRaggedBatchKernels::gemmMMAGuarded,
                    context,
                    stepRows,
                    state.workspace.attnOutFP16,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.woOut,
                    paddedBatch,
                    dim,
                    dim);
        } else {
            g.task(
                    "woProj",
                    TransformerBatchPrefillKernels::gemmMMA,
                    context,
                    state.workspace.attnOutFP16,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.woOut,
                    paddedBatch,
                    dim,
                    dim);
        }
        if (skinny()) {
            skinnyTask(
                    g,
                    "woProj_skinny",
                    state.workspace.attnOutFP16,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.woOut,
                    dim,
                    dim,
                    0);
            midTask(
                    g,
                    "woProj_mid",
                    state.workspace.attnOutFP16,
                    weights.woLayered[layerIndex].asHalfFloatArray(),
                    state.workspace.woOut,
                    dim,
                    dim,
                    0);
        }

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

        g.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);

        if (skinny()) {
            g.task(
                    "gateUpProj",
                    TransformerRaggedBatchKernels::gemmMMAGateUpGuarded,
                    context,
                    stepRows,
                    state.workspace.normedXFFNFP16,
                    weights.w1Layered[layerIndex].asHalfFloatArray(),
                    weights.w3Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.gateUpResultBatch,
                    paddedBatch,
                    hidDim,
                    dim);
        } else {
            g.task(
                    "gateUpProj",
                    TransformerBatchPrefillKernels::gemmMMAGateUp,
                    context,
                    state.workspace.normedXFFNFP16,
                    weights.w1Layered[layerIndex].asHalfFloatArray(),
                    weights.w3Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.gateUpResultBatch,
                    paddedBatch,
                    hidDim,
                    dim);
        }
        if (skinny()) {
            skinnyTask(
                    g,
                    "gateProj_skinny",
                    state.workspace.normedXFFNFP16,
                    weights.w1Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.gateUpResultBatch,
                    dim,
                    2 * hidDim,
                    0);
            skinnyTask(
                    g,
                    "upProj_skinny",
                    state.workspace.normedXFFNFP16,
                    weights.w3Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.gateUpResultBatch,
                    dim,
                    2 * hidDim,
                    hidDim);
            midTask(
                    g,
                    "gateProj_mid",
                    state.workspace.normedXFFNFP16,
                    weights.w1Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.gateUpResultBatch,
                    dim,
                    2 * hidDim,
                    0);
            midTask(
                    g,
                    "upProj_mid",
                    state.workspace.normedXFFNFP16,
                    weights.w3Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.gateUpResultBatch,
                    dim,
                    2 * hidDim,
                    hidDim);
        }

        g.task(
                "swiglu",
                TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                context,
                state.workspace.wrapHbFP16Batch,
                state.workspace.gateUpResultBatch,
                hidDim);
        if (skinny()) {
            g.task(
                    "w2Proj",
                    TransformerRaggedBatchKernels::gemmMMAGuarded,
                    context,
                    stepRows,
                    state.workspace.wrapHbFP16Batch,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.w2Out,
                    batchSize,
                    dim,
                    hidDim);
        } else {
            g.task(
                    "w2Proj",
                    TransformerBatchPrefillKernels::gemmMMA,
                    context,
                    state.workspace.wrapHbFP16Batch,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.w2Out,
                    batchSize,
                    dim,
                    hidDim);
        }
        if (skinny()) {
            skinnyTask(
                    g,
                    "w2Proj_skinny",
                    state.workspace.wrapHbFP16Batch,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.w2Out,
                    hidDim,
                    dim,
                    0);
            midTask(
                    g,
                    "w2Proj_mid",
                    state.workspace.wrapHbFP16Batch,
                    weights.w2Layered[layerIndex].asHalfFloatArray(),
                    state.workspace.w2Out,
                    hidDim,
                    dim,
                    0);
        }
        g.task(
                "w2Resid",
                TransformerBatchPrefillKernels::batchedResidualAddFP32,
                context,
                state.workspace.wrapXBatch,
                state.workspace.w2Out);

        g.persistOnDevice(state.workspace.wrapXBatch, keyPool(), valuePool());
        return g;
    }

    // @formatter:on

    /** Adds a mid GEMM writing {@code out[t, colOffset + n]}; see {@link #skinnyTask}. */
    private void midTask(
            TaskGraph g,
            String name,
            HalfFloatArray x,
            HalfFloatArray w,
            FloatArray out,
            int k,
            int ldc,
            int colOffset) {
        g.task(
                name,
                TransformerRaggedBatchKernels::gemmMMAMid,
                context,
                x,
                w,
                out,
                stepRows,
                k,
                ldc,
                colOffset);
    }

    /** Whether this graph carries the skinny GEMM tasks next to the 128-row ones. */
    private boolean skinny() {
        return rowSlots != null && skinnyGemm;
    }

    /**
     * Adds a skinny GEMM writing {@code out[t, colOffset + n]}. On a skinny step the guarded
     * 128-row GEMM of the same product returns at once and this one computes it; on a large step
     * this one returns at once.
     */
    private void skinnyTask(
            TaskGraph g,
            String name,
            HalfFloatArray x,
            HalfFloatArray w,
            FloatArray out,
            int k,
            int ldc,
            int colOffset) {
        g.task(
                name,
                TransformerRaggedBatchKernels::gemmMMASkinny,
                context,
                x,
                w,
                out,
                stepRows,
                k,
                ldc,
                colOffset);
    }

    static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128;
        int nBlocks = N / 128;
        WorkerGrid2D grid = new WorkerGrid2D(mBlocks * 256, nBlocks);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }

    static WorkerGrid midGrid(int n) {
        WorkerGrid2D grid =
                new WorkerGrid2D(128, n / TransformerRaggedBatchKernels.MID_COLS_PER_GROUP);
        grid.setLocalWork(128, 1, 1);
        return grid;
    }

    static WorkerGrid skinnyGrid(int n) {
        return elementwiseGrid(n / TransformerRaggedBatchKernels.SKINNY_COLS_PER_GROUP * 128, 128);
    }

    static WorkerGrid elementwiseGrid(int n, int local) {
        WorkerGrid1D grid = new WorkerGrid1D(n);
        grid.setLocalWork(local, 1, 1);
        return grid;
    }

    static WorkerGrid elementwiseGrid(int n) {
        WorkerGrid1D grid = new WorkerGrid1D(n);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }

    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        int nHeads = config.numberOfHeads();
        int headSz = config.headSize();

        rmsWorker = WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);
        rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        ffnRmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        int attnLocal = attnLocalSize();
        attnWorker =
                WorkerGridFactory.genericWorker(
                        batchSize
                                * (nHeads / attnHeadsPerGroup())
                                * attnGroupsPerHead()
                                * attnLocal,
                        attnLocal);
        attnCombineWorker = WorkerGridFactory.genericWorker(batchSize * nHeads * headSz, headSz);
        int gqaLocal = config.kvMul() * headSz;
        attnDecodeWorker =
                WorkerGridFactory.genericWorker(
                        batchSize * (nHeads / config.kvMul()) * kvSplits * gqaLocal, gqaLocal);

        mmaQkvWorker = mmaGrid(paddedBatch, dim + 2 * kvDim);
        mmaDimWorker = mmaGrid(paddedBatch, dim);
        mmaGateUpWorker = mmaGrid(paddedBatch, 2 * hidDim);

        ewDimWorker = elementwiseGrid(batchSize * dim);
        ewHidWorker = elementwiseGrid(batchSize * hidDim);
        skinnyDimWorker = skinnyGrid(dim);
        skinnyKvWorker = skinnyGrid(kvDim);
        skinnyHidWorker = skinnyGrid(hidDim);
        midDimWorker = midGrid(dim);
        midKvWorker = midGrid(kvDim);
        midHidWorker = midGrid(hidDim);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchDecodeLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "qkvProj", mmaQkvWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv", ropeWorker);
            scheduler.addWorkerGrid(p + "batch_attention", attnWorker);
            if (attnPartials != null) {
                scheduler.addWorkerGrid(p + "batch_attention_combine", attnCombineWorker);
            }
            if (tiled()) {
                scheduler.addWorkerGrid(p + "batch_attention_decode", attnDecodeWorker);
            }
            scheduler.addWorkerGrid(p + "woProj", mmaDimWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", ffnRmsApplyWorker);
            scheduler.addWorkerGrid(p + "gateUpProj", mmaGateUpWorker);
            scheduler.addWorkerGrid(p + "swiglu", ewHidWorker);
            scheduler.addWorkerGrid(p + "w2Proj", mmaDimWorker);
            scheduler.addWorkerGrid(p + "w2Resid", ewDimWorker);
            if (skinny()) {
                scheduler.addWorkerGrid(p + "qProj_skinny", skinnyDimWorker);
                scheduler.addWorkerGrid(p + "kProj_skinny", skinnyKvWorker);
                scheduler.addWorkerGrid(p + "vProj_skinny", skinnyKvWorker);
                scheduler.addWorkerGrid(p + "woProj_skinny", skinnyDimWorker);
                scheduler.addWorkerGrid(p + "gateProj_skinny", skinnyHidWorker);
                scheduler.addWorkerGrid(p + "upProj_skinny", skinnyHidWorker);
                scheduler.addWorkerGrid(p + "w2Proj_skinny", skinnyDimWorker);
                scheduler.addWorkerGrid(p + "qProj_mid", midDimWorker);
                scheduler.addWorkerGrid(p + "kProj_mid", midKvWorker);
                scheduler.addWorkerGrid(p + "vProj_mid", midKvWorker);
                scheduler.addWorkerGrid(p + "woProj_mid", midDimWorker);
                scheduler.addWorkerGrid(p + "gateProj_mid", midHidWorker);
                scheduler.addWorkerGrid(p + "upProj_mid", midHidWorker);
                scheduler.addWorkerGrid(p + "w2Proj_mid", midDimWorker);
            }
        }
    }

    /**
     * Launches only the first {@code rows} rows on the next execution, instead of all {@code
     * batchSize}: row-wise kernels cover {@code rows}, the tensor-core GEMMs the 128-row tiles that
     * hold them. Rows past {@code rows} must be padding, which no kernel reads for a real row, so
     * what a real row computes does not change. Call after {@link #updateGridScheduler}.
     *
     * @param rows a multiple of 16, at most {@code batchSize}
     */
    public void setActiveRows(int rows) {
        setActiveRows(rows, rows);
    }

    /**
     * {@link #setActiveRows(int)}, also choosing the GEMM: a step of at most {@link
     * TransformerRaggedBatchKernels#SKINNY_ROWS} real tokens runs its projections on the skinny
     * kernel, the guarded 128-row kernel launching a single tile that returns at once.
     *
     * @param realRows the step's tokens, before rounding up to {@code rows}
     */
    public void setActiveRows(int rows, int realRows) {
        setActiveRows(rows, realRows, rows);
    }

    /**
     * {@link #setActiveRows(int, int)}, with the step's query-tile count for tiled attention.
     *
     * @param tiles query tiles written to {@link #queryTiles()}, at most {@code batchSize}
     */
    public void setActiveRows(int rows, int realRows, int tiles) {
        if (rows <= 0 || rows > batchSize || rows % 16 != 0) {
            throw new IllegalArgumentException(
                    "active rows must be a positive multiple of 16 up to "
                            + batchSize
                            + ", got "
                            + rows);
        }
        int dim = config.dim();
        int hidDim = config.hiddenDim();
        int mmaRows = (rows + 127) & ~127;
        rmsWorker.setGlobalWork((long) rows * RMS_LOCAL_SIZE, 1, 1);
        rmsApplyWorker.setGlobalWork((long) rows * dim, 1, 1);
        ffnRmsApplyWorker.setGlobalWork((long) rows * dim, 1, 1);
        ropeWorker.setGlobalWork((long) rows * (dim / 2), 1, 1);
        attnWorker.setGlobalWork(
                (long) (tiled() ? Math.max(tiles, 1) : rows)
                        * (config.numberOfHeads() / attnHeadsPerGroup())
                        * attnGroupsPerHead()
                        * attnLocalSize(),
                1,
                1);
        attnCombineWorker.setGlobalWork(
                (long) rows * config.numberOfHeads() * config.headSize(), 1, 1);
        if (tiled()) {
            attnDecodeWorker.setGlobalWork(
                    (long) rows
                            * (config.numberOfHeads() / config.kvMul())
                            * kvSplits
                            * config.kvMul()
                            * config.headSize(),
                    1,
                    1);
        }
        mmaQkvWorker.setGlobalWork(mmaRows / 128 * 256L, (dim + 2L * config.kvDim()) / 128, 1);
        mmaDimWorker.setGlobalWork(mmaRows / 128 * 256L, dim / 128, 1);
        mmaGateUpWorker.setGlobalWork(mmaRows / 128 * 256L, 2L * hidDim / 128, 1);
        if (skinny()) {
            stepRows.set(0, realRows);
            stepRows.set(1, midGemm ? 1 : 0);
            boolean small = realRows <= TransformerRaggedBatchKernels.SKINNY_ROWS;
            boolean mid = midGemm && !small && realRows <= TransformerRaggedBatchKernels.MID_ROWS;
            int midCols = TransformerRaggedBatchKernels.MID_COLS_PER_GROUP;
            midDimWorker.setGlobalWork(128, mid ? dim / midCols : 1, 1);
            midKvWorker.setGlobalWork(128, mid ? config.kvDim() / midCols : 1, 1);
            midHidWorker.setGlobalWork(128, mid ? hidDim / midCols : 1, 1);
            if (mid) {
                mmaQkvWorker.setGlobalWork(256, 1, 1);
                mmaDimWorker.setGlobalWork(256, 1, 1);
                mmaGateUpWorker.setGlobalWork(256, 1, 1);
            }
            int cols = TransformerRaggedBatchKernels.SKINNY_COLS_PER_GROUP;
            skinnyDimWorker.setGlobalWork(small ? dim / cols * 128L : 128, 1, 1);
            skinnyKvWorker.setGlobalWork(small ? config.kvDim() / cols * 128L : 128, 1, 1);
            skinnyHidWorker.setGlobalWork(small ? hidDim / cols * 128L : 128, 1, 1);
            if (small) {
                mmaQkvWorker.setGlobalWork(256, 1, 1);
                mmaDimWorker.setGlobalWork(256, 1, 1);
                mmaGateUpWorker.setGlobalWork(256, 1, 1);
            }
        }
        ewDimWorker.setGlobalWork((long) rows * dim, 1, 1);
        ewHidWorker.setGlobalWork((long) rows * hidDim, 1, 1);
    }

    /**
     * Whether the grouped split-K kernel runs: its workgroup holds every query head of a KV head,
     * so {@code kvMul * headSize} must fit one workgroup and its shared tiles.
     */
    private boolean groupedAttention() {
        return attnPartials != null
                && gqaAttention
                && config.kvMul() <= 8
                && config.headSize() <= 128
                && config.kvMul() * config.headSize() <= 1024;
    }

    /** Threads per attention workgroup: a warp on tensor cores, else per the scalar kernel. */
    private int attnLocalSize() {
        if (tiled()) {
            return 32;
        }
        return groupedAttention()
                ? config.kvMul() * config.headSize()
                : Math.min(config.headSize(), 128);
    }

    /** Attention workgroups per (row, head group): one per split under split-K, else one. */
    private int attnGroupsPerHead() {
        return attnPartials != null ? kvSplits : 1;
    }

    /** Whether attention runs on tensor cores, one warp per query tile. */
    private boolean tiled() {
        return attnPartials != null
                && flashAttention
                && config.headSize() == 64
                && TransformerRaggedBatchKernels.FLASH_TILE_ROWS % config.kvMul() == 0;
    }

    /** Rows of one sequence per attention query tile: 16 / kvMul on tensor cores, else one. */
    public int queryTileRows() {
        return tiled() ? TransformerRaggedBatchKernels.FLASH_TILE_ROWS / config.kvMul() : 1;
    }

    /** The executor writes each step's generated-token slots here: -1 for a prompt row. */
    public IntArray decodeRowSlots() {
        return decodeRowSlots;
    }

    /** The executor writes each step's query tiles here: first row and row count per tile. */
    public IntArray queryTiles() {
        return queryTiles;
    }

    /** Query heads per attention workgroup. */
    private int attnHeadsPerGroup() {
        return tiled() || groupedAttention() ? config.kvMul() : 1;
    }

    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerITGs;
    }

    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    public KernelContext getContext() {
        return context;
    }

    /** The key pool the graphs bind, in whichever representation the engine built. */
    private Object keyPool() {
        return keyCacheBatchFP16 != null ? keyCacheBatchFP16 : keyCacheBatch;
    }

    /** The value pool, following {@link #keyPool()}. */
    private Object valuePool() {
        return valueCacheBatchFP16 != null ? valueCacheBatchFP16 : valueCacheBatch;
    }
}
