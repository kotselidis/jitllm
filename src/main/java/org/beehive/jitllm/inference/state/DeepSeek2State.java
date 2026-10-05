package org.beehive.jitllm.inference.state;

import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.deepseek2.DeepSeek2Configuration;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * A {@code deepseek2} session.
 *
 * <p>The key/value store holds one {@link DeepSeek2Configuration#keyWidth()}-wide row per position
 * and layer — the normalized latent followed by the rotated key — in {@link #keyCache}. The value is
 * the latent half of the same row, so {@link #valueCache} holds nothing.
 */
public final class DeepSeek2State extends State {

    /** The compressed query, before its norm and {@code attn_q_b}. */
    public final FloatTensor queryLatent;

    /** Every head's absorbed query: the latent-space no-rope part, then the rotated part. */
    public final FloatTensor absorbedQuery;

    /** Every head's attended latent, before {@code attn_v_b}. */
    public final FloatTensor latentOut;

    /** {@code attn_kv_a_mqa}'s output: the latent, then the rotated key. */
    public final FloatTensor compressedKeyValue;

    public final FloatTensor routerScores;
    public final FloatTensor selectionScores;
    public final int[] expertIds;
    public final float[] expertWeights;
    public final FloatTensor expertHidden;
    public final FloatTensor expertHiddenUp;
    public final FloatTensor expertOut;

    public DeepSeek2State(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    public DeepSeek2State(
            Configuration config, int batchsize, org.beehive.jitllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
        DeepSeek2Configuration c = (DeepSeek2Configuration) config;
        this.queryLatent = ArrayFloatTensor.allocate(Math.max(1, c.qLoraRank()));
        this.absorbedQuery = ArrayFloatTensor.allocate(c.absorbedQueryDim());
        this.latentOut = ArrayFloatTensor.allocate(c.latentOutputDim());
        this.compressedKeyValue = ArrayFloatTensor.allocate(c.compressedKeyValueDim());
        this.routerScores = ArrayFloatTensor.allocate(c.expertCount());
        this.selectionScores = ArrayFloatTensor.allocate(c.expertCount());
        this.expertIds = new int[c.expertsUsed()];
        this.expertWeights = new float[c.expertsUsed()];
        int width = Math.max(c.expertHiddenDim(), c.sharedHiddenDim());
        this.expertHidden = ArrayFloatTensor.allocate(width);
        this.expertHiddenUp = ArrayFloatTensor.allocate(width);
        this.expertOut = ArrayFloatTensor.allocate(c.dim());
    }

    @Override
    protected int batchQDim(Configuration config) {
        return ((DeepSeek2Configuration) config).absorbedQueryDim();
    }

    @Override
    protected int batchKvDim(Configuration config) {
        return ((DeepSeek2Configuration) config).keyWidth();
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        DeepSeek2Configuration config = (DeepSeek2Configuration) configuration;
        StateFields fields = new StateFields();
        int dim = config.dim();
        int hidden = Math.max(config.hiddenDim(), config.expertHiddenDim());
        fields.x = ArrayFloatTensor.allocate(dim);
        fields.xb = ArrayFloatTensor.allocate(Math.max(dim, config.attentionOutputInputDim()));
        fields.xb2 = ArrayFloatTensor.allocate(dim);
        fields.hb = ArrayFloatTensor.allocate(hidden);
        fields.hb2 = ArrayFloatTensor.allocate(hidden);
        fields.q = ArrayFloatTensor.allocate(config.queryDim());
        fields.k = ArrayFloatTensor.allocate(config.keyWidth());
        fields.v = ArrayFloatTensor.allocate(config.kvLoraRank());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());
        int layers = config.numberOfLayers();
        fields.keyCache = new FloatTensor[layers];
        fields.valueCache = new FloatTensor[layers];
        for (int l = 0; l < layers; l++) {
            fields.keyCache[l] = allocateKeyValue(config.contextLength(), config.keyWidth());
        }
        return fields;
    }
}
