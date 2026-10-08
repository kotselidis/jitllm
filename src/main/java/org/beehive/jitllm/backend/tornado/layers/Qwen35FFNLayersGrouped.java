package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.Qwen35State;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;

// @formatter:off
/**
 * The single-token {@code qwen35} layers of one pipeline stage, several layers to a task graph.
 *
 * <p>Every task graph of a plan is launched, and synchronized, on its own: one graph per layer is a
 * launch and a stream synchronization per layer per token. That is invisible behind a dense model's
 * long layers and is most of the time between kernels for a mixture of experts, whose layers are
 * short — {@code qwen35moe} at one graph per layer synchronized about a hundred times a token.
 * Grouping is counted from the stage's first layer, so each graph is named for its first layer and
 * the layers after it in the graph carry an {@code l<slot>_} task prefix.
 */
// @formatter:on
public class Qwen35FFNLayersGrouped extends Qwen35FFNLayers {

    /** Layers per task graph; the whole stage at the default. */
    static final int LAYERS_PER_GRAPH = 64;

    public Qwen35FFNLayersGrouped(
            String taskGraphName,
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            SchedulerType schedulerType,
            String activationGraphName,
            int firstLayer,
            int endLayer,
            Qwen35FeedForward feedForward) {
        super(
                taskGraphName,
                state,
                weights,
                config,
                schedulerType,
                activationGraphName,
                firstLayer,
                endLayer,
                feedForward);
    }

    @Override
    protected void setupFFNLayers() {
        int layers = endLayer(config.numberOfLayers());
        List<ImmutableTaskGraph> graphs = new ArrayList<>();
        for (int first = firstLayer; first < layers; first += LAYERS_PER_GRAPH) {
            TaskGraph graph = new TaskGraph(layerGraphName(first));
            int last = Math.min(first + LAYERS_PER_GRAPH, layers) - 1;
            for (int layer = first; layer <= last; layer++) {
                appendLayer(graph, layer);
            }
            lastFFNLayerTaskGraphID = graph.getTaskGraphName();
            graphs.add(graph.snapshot());
        }
        ffnLayerITGs = List.copyOf(graphs);
    }

    /** The graph holding {@code layerIndex}: named for the first layer in it. */
    @Override
    protected String layerGraphName(int layerIndex) {
        int offset = layerIndex - firstLayer;
        return "layer_" + (firstLayer + offset - offset % LAYERS_PER_GRAPH);
    }

    @Override
    protected String layerTaskPrefix(int layerIndex) {
        int slot = (layerIndex - firstLayer) % LAYERS_PER_GRAPH;
        return slot == 0 ? "" : "l" + slot + "_";
    }
}
