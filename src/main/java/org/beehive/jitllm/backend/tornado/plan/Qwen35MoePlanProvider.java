package org.beehive.jitllm.backend.tornado.plan;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.plan.components.Qwen35PlanComponents;
import org.beehive.jitllm.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The {@code qwen35moe} plan components: the {@code qwen35} layer graphs, whose feed-forward is the
 * routed and shared experts. Q8_0 only — the expert kernels read Q8_0 blocks.
 */
public final class Qwen35MoePlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("qwen35moe");

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
        return Set.of(DataType.F32, DataType.F16, DataType.Q8_0);
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return Set.of(ExecutionMode.values());
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Qwen35State typed = PlanStates.expect(Qwen35State.class, state, ID);
        return new Qwen35PlanComponents(typed, model);
    }
}
