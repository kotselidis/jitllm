package org.beehive.jitllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.Activation;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.DeepSeek2Layers;
import org.beehive.jitllm.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.jitllm.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.DeepSeek2State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.DeepSeek2TornadoWeights;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.deepseek2.DeepSeek2Configuration;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The {@code deepseek2} plan (GLM-4.7-Flash): the embedding, the latent-attention and
 * mixture-of-experts layers, and the logits. Single-token, Q8_0 only — every projection is a
 * packed-integer kernel over Q8_0 blocks.
 */
public final class DeepSeek2PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("deepseek2");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return Set.of(DataType.Q8_0);
    }

    @Override
    public Set<DataType> nativeTensorTypes() {
        return Set.of(DataType.F32, DataType.Q8_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.STANDARD);
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        DeepSeek2State typed = PlanStates.expect(DeepSeek2State.class, state, ID);
        return new Components(typed, model);
    }

    private static final class Components implements SingleTokenForwardPlanComponents {

        private final DeepSeek2State state;
        private final DeepSeek2TornadoWeights weights;
        private final DeepSeek2Configuration config;
        private final SchedulerType schedulerType;

        Components(DeepSeek2State state, Model model) {
            this.state = state;
            this.weights = (DeepSeek2TornadoWeights) model.weights();
            this.config = (DeepSeek2Configuration) model.configuration();
            this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
        }

        @Override
        public ActivationTaskGraph singleTokenActivation() {
            return new Activation("activationUpdate", state, weights, config);
        }

        @Override
        public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
            return new DeepSeek2Layers("deepseek2", state, weights, config, schedulerType);
        }

        @Override
        public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
            return new LogitsQ8_0Layer(
                    "logits", state, weights, config, previousGraphId, schedulerType);
        }
    }
}
