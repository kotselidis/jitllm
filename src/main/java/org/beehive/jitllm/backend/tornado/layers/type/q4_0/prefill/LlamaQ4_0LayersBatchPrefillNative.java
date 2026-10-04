package org.beehive.jitllm.backend.tornado.layers.type.q4_0.prefill;

import java.util.List;
import java.util.stream.IntStream;
import org.beehive.jitllm.backend.tornado.kernels.Gemma4BatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.LlamaQ4_0BatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.inference.state.LlamaState;
import org.beehive.jitllm.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;

// @formatter:off
/**
 * Batched prefill for Llama layers that keep their weights as Q4_0 (and Q4_1 for the {@code
 * ffn_down} tensors {@code llama-quantize} gives more bits), with the projections run by cuBLAS on
 * tensor cores.
 *
 * <p>cuBLAS cannot read Q4_0, so each projection first decodes its weight into one shared FP16
 * scratch with a JIT kernel, and a {@code cublasGemmExFP16FP32} then multiplies the whole prompt
 * chunk by it. This is llama.cpp's cuBLAS route for quantized prompt processing: the decode costs
 * one pass over the weight, the GEMM then runs at tensor-core rate, and the weights stay 4-bit in
 * device memory. Only the scratch is FP16, sized for the largest projection.
 *
 * <p>Per layer, in one graph:
 *
 * <pre>
 *   batch_attn_rms, batch_attn_rms_apply   RMS norm → FP16 activation
 *   qDequant, kDequant, vDequant          Wq, Wk, Wv → scratch, stacked
 *   qkvProj                               cuBLAS → packed [q|k|v] rows
 *   batch_rope_kv, batch_attention        RoPE + KV write, flash attention → FP16
 *   woDequant, woProj                     cuBLAS → woOut
 *   batch_ffn_rms, batch_ffn_rms_apply    x += woOut, RMS norm → FP16
 *   gateDequant, gateProj                 cuBLAS → gate
 *   upDequant, upProj                     cuBLAS → up
 *   swiglu                                silu(gate) * up → FP16
 *   downDequant, downProj, w2Resid        cuBLAS → w2Out, x += w2Out
 * </pre>
 *
 * <p>Gate and up are separate GEMMs on purpose: decoding both into one scratch would need room for
 * {@code 2 * hiddenDim * dim} halves, 940 MB on Llama-3.3-70B, against 470 MB this way.
 *
 * <p>Every per-layer weight reaches a task here, so this graph uploads it and the decode layers
 * bind that copy (see {@code LlamaQ4_0FFNLayersDecode.weightSourceGraphName}).
 */
// @formatter:on
public class LlamaQ4_0LayersBatchPrefillNative implements BatchPrefillTransformerLayerTaskGraphs {

    static final int RMS_LOCAL_SIZE = 256;

    private static final int CUBLAS_OP_N = 0;
    private static final int CUBLAS_OP_T = 1;

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final int firstLayer;
    private final int endLayer;
    private final HalfFloatArray scratch;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public LlamaQ4_0LayersBatchPrefillNative(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            int batchSize) {
        this(state, weights, config, batchSize, 0, config.numberOfLayers());
    }

    /**
     * The layers {@code [firstLayer, endLayer)} only: one stage of a model split across devices.
     * The first layer of the range takes the role layer 0 has otherwise, and the key/value cache of
     * {@code state} holds just these layers.
     */
    public LlamaQ4_0LayersBatchPrefillNative(
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            int batchSize,
            int firstLayer,
            int endLayer) {
        if (firstLayer < 0 || endLayer > config.numberOfLayers() || firstLayer >= endLayer) {
            throw new IllegalArgumentException(
                    "layer range ["
                            + firstLayer
                            + ", "
                            + endLayer
                            + ") outside the model's "
                            + config.numberOfLayers()
                            + " layers");
        }
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.firstLayer = firstLayer;
        this.endLayer = endLayer;
        this.scratch = scratchFor(state, config);
        this.layerITGs =
                IntStream.range(firstLayer, endLayer)
                        .mapToObj(this::createLayer)
                        .map(TaskGraph::snapshot)
                        .toList();
    }

    @Override
    public String describeProjections() {
        return "cuBLAS FP16 GEMM (Q4_0 weights decoded to FP16 per projection)";
    }

    @Override
    public String describeNativeLibraries() {
        return "cuBLAS";
    }

    /** Halves the largest projection needs: the stacked Q/K/V, or one FFN matrix. */
    static long scratchElements(LlamaConfiguration config) {
        long qkv = (long) (config.dim() + 2 * config.kvDim()) * config.dim();
        long ffn = (long) config.hiddenDim() * config.dim();
        return Math.max(qkv, ffn);
    }

    /** The state's FP16 weight scratch, allocated here the first time a plan needs it. */
    private static HalfFloatArray scratchFor(LlamaState state, LlamaConfiguration config) {
        long elements = scratchElements(config);
        HalfFloatArray current = state.workspace.weightsF16Scratch;
        if (current == null || current.getSize() < elements) {
            state.workspace.weightsF16Scratch =
                    TornadoWorkspaces.halfFloats(Math.toIntExact(elements));
        }
        return state.workspace.weightsF16Scratch;
    }

    private Object keyCache() {
        return state.usesFp16KeyValueCache()
                ? state.workspace.wrapKeyCacheFP16
                : state.workspace.wrapKeyCache;
    }

    private Object valueCache() {
        return state.usesFp16KeyValueCache()
                ? state.workspace.wrapValueCacheFP16
                : state.workspace.wrapValueCache;
    }

    // @formatter:off
    private TaskGraph createLayer(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == endLayer - 1) {
            lastLayerTaskGraphID = graphName;
        }
        TaskGraph layer = new TaskGraph(graphName);

        Object[] intermediates = {
            context,
            state.workspace.attnScaleBatch,
            state.workspace.ffnScaleBatch,
            state.workspace.wrapXbFP16Batch,
            state.workspace.qkvResultBatch,
            keyCache(),
            valueCache(),
            state.workspace.normedXFFNFP16,
            state.workspace.ffnGateResult,
            state.workspace.ffnUpResult,
            state.workspace.attnOutFP16,
            state.workspace.woOut,
            state.workspace.wrapHbFP16Batch,
            state.workspace.w2Out,
            scratch
        };
        if (layerIndex == firstLayer) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, intermediates);
            layer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
            // Every execution: a lease change rewrites the table.
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
        } else {
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(pred, intermediates);
            layer.consumeFromDevice(
                    pred,
                    state.workspace.wrapXBatch,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapBlockTable);
        }

        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray());

        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        int qkvDim = dim + 2 * kvDim;
        // The stage's cache holds only its own layers.
        int kvLayer = layerIndex - firstLayer;

        // ── Attention ──────────────────────────────────────────────────────────
        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        // Wq, Wk, Wv stacked in the scratch: one GEMM writes packed [q|k|v] rows.
        dequantize(layer, "qDequant", weights.wqLayered[layerIndex], 0);
        dequantize(layer, "kDequant", weights.wkLayered[layerIndex], dim * dim);
        dequantize(layer, "vDequant", weights.wvLayered[layerIndex], (dim + kvDim) * dim);
        gemm(
                layer,
                "qkvProj",
                state.workspace.wrapXbFP16Batch,
                state.workspace.qkvResultBatch,
                qkvDim,
                dim);

        if (state.usesFp16KeyValueCache()) {
            layer.task(
                    "batch_rope_kv",
                    TransformerPagedKvBatchPrefillKernels::batchedRopeWithKVCachePackedFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    kvDim,
                    config.headSize(),
                    kvLayer,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
            layer.task(
                    "batch_attention",
                    // The tiled variant the FP16 layers take under the same policy flag.
                    state.executionPolicy().packedHalf2Attention()
                            ? TransformerPagedKvBatchPrefillKernels
                                    ::batchedFlashAttentionFP16OutKVFP16PackedTilePaged
                            : TransformerPagedKvBatchPrefillKernels
                                    ::batchedFlashAttentionFP16OutKVFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.attnOutFP16,
                    config.numberOfHeads(),
                    config.headSize(),
                    kvDim,
                    config.kvMul(),
                    kvLayer,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
        } else {
            layer.task(
                    "batch_rope_kv",
                    TransformerPagedKvBatchPrefillKernels::batchedRopeWithKVCachePackedPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray(),
                    kvDim,
                    config.headSize(),
                    kvLayer,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
            layer.task(
                    "batch_attention",
                    TransformerPagedKvBatchPrefillKernels::batchedFlashAttentionFP16OutPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.attnOutFP16,
                    config.numberOfHeads(),
                    config.headSize(),
                    kvDim,
                    config.kvMul(),
                    kvLayer,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    dim);
        }

        dequantize(layer, "woDequant", weights.woLayered[layerIndex], 0);
        gemm(layer, "woProj", state.workspace.attnOutFP16, state.workspace.woOut, dim, dim);

        // ── Feed-forward ───────────────────────────────────────────────────────
        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.woOut,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);

        dequantize(layer, "gateDequant", weights.w1Layered[layerIndex], 0);
        gemm(
                layer,
                "gateProj",
                state.workspace.normedXFFNFP16,
                state.workspace.ffnGateResult,
                hidDim,
                dim);
        dequantize(layer, "upDequant", weights.w3Layered[layerIndex], 0);
        gemm(
                layer,
                "upProj",
                state.workspace.normedXFFNFP16,
                state.workspace.ffnUpResult,
                hidDim,
                dim);

        layer.task(
                "swiglu",
                LlamaQ4_0BatchPrefillKernels::swiGLUFP16,
                context,
                state.workspace.wrapHbFP16Batch,
                state.workspace.ffnGateResult,
                state.workspace.ffnUpResult,
                batchSize * hidDim);

        dequantize(layer, "downDequant", weights.w2Layered[layerIndex], 0);
        gemm(
                layer,
                "downProj",
                state.workspace.wrapHbFP16Batch,
                state.workspace.w2Out,
                dim,
                hidDim);
        layer.task(
                "w2Resid",
                TransformerBatchPrefillKernels::batchedResidualAddFP32,
                context,
                state.workspace.wrapXBatch,
                state.workspace.w2Out);

        layer.persistOnDevice(state.workspace.wrapXBatch, keyCache(), valueCache());
        return layer;
    }

    // @formatter:on

    /** Decodes {@code w} into the scratch at {@code offset}, by the tensor's own representation. */
    private void dequantize(TaskGraph layer, String name, TornadoTensor w, int offset) {
        DataType type = w.dataType();
        if (type == DataType.Q4_0) {
            layer.task(
                    name,
                    Gemma4BatchPrefillKernels::dequantizeQ4_0ToFP16,
                    context,
                    w.asByteArray(),
                    scratch,
                    offset);
        } else if (type == DataType.Q4_1) {
            layer.task(
                    name,
                    Gemma4BatchPrefillKernels::dequantizeQ4_1ToFP16,
                    context,
                    w.asByteArray(),
                    scratch,
                    offset);
        } else {
            throw new UnsupportedOperationException(
                    "the Llama Q4_0 batched prefill decodes Q4_0 and Q4_1 weights, not " + type);
        }
    }

    // @formatter:off
    /**
     * {@code out[rows][n] = x[rows][k] · Wᵀ} with W the decoded {@code [n][k]} weight in the
     * scratch. In cuBLAS's column-major terms that is {@code C = op_T(W) · X}: the weight is A,
     * transposed, and the activation B as it is.
     */
    // @formatter:on
    private void gemm(
            TaskGraph layer, String name, HalfFloatArray x, FloatArray out, int n, int k) {
        layer.libraryTask(
                name,
                CuBlas::cublasGemmExFP16FP32,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n,
                batchSize,
                k,
                1.0f,
                scratch,
                k,
                x,
                k,
                0.0f,
                out,
                n);
    }

    private static WorkerGrid elementwise(int n) {
        WorkerGrid1D grid = new WorkerGrid1D(n);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int hidDim = config.hiddenDim();
        int nHeads = config.numberOfHeads();
        int headSize = config.headSize();

        WorkerGrid rms =
                WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);
        WorkerGrid rmsApply = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) {
            ropeLocal--;
        }
        WorkerGrid rope = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);
        int attnLocal = Math.min(headSize, 128);
        WorkerGrid attention =
                WorkerGridFactory.genericWorker(batchSize * nHeads * attnLocal, attnLocal);

        for (int i = firstLayer; i < endLayer; i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms", rms);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApply);
            scheduler.addWorkerGrid(p + "qDequant", elementwise(dim * dim));
            scheduler.addWorkerGrid(p + "kDequant", elementwise(kvDim * dim));
            scheduler.addWorkerGrid(p + "vDequant", elementwise(kvDim * dim));
            scheduler.addWorkerGrid(p + "batch_rope_kv", rope);
            scheduler.addWorkerGrid(p + "batch_attention", attention);
            scheduler.addWorkerGrid(p + "woDequant", elementwise(dim * dim));
            scheduler.addWorkerGrid(p + "batch_ffn_rms", rms);
            scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", rmsApply);
            scheduler.addWorkerGrid(p + "gateDequant", elementwise(hidDim * dim));
            scheduler.addWorkerGrid(p + "upDequant", elementwise(hidDim * dim));
            scheduler.addWorkerGrid(p + "swiglu", elementwise(batchSize * hidDim));
            scheduler.addWorkerGrid(p + "downDequant", elementwise(dim * hidDim));
            scheduler.addWorkerGrid(p + "w2Resid", elementwise(batchSize * dim));
        }
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
