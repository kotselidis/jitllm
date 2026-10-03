package org.beehive.jitllm.backend.tornado.pipeline;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * How the hidden state crosses from one pipeline stage to the next, and how the stages run.
 *
 * <p>Two implementations: {@link HostTransport} copies through host memory and runs the stages one
 * after the other on the calling thread; {@code NcclTransport} sends device to device with NCCL and
 * runs every stage on its own thread through an {@code NcclPlanGroup}. The NCCL one is compiled
 * only with the {@code nccl} Maven profile, so it is loaded by name.
 */
public interface PipelineTransport extends AutoCloseable {

    String NCCL_TRANSPORT = "org.beehive.jitllm.backend.tornado.pipeline.NcclTransport";

    /**
     * Graph that ends a stage: sends {@code x}, produced by graph {@code producer}, to stage
     * {@code toStage}.
     */
    TaskGraph sendGraph(int stage, String producer, FloatArray x, int toStage);

    /**
     * Graph that starts a stage: receives {@code x} from stage {@code fromStage} and persists it on
     * the device for the stage's first layer.
     */
    TaskGraph receiveGraph(int stage, FloatArray x, int fromStage);

    /** Adds the worker grids of this stage's transport graphs. */
    void updateGridScheduler(int stage, GridScheduler scheduler);

    /** Runs one forward step: every stage's plan, all of its graphs, once. */
    void execute(TornadoExecutionPlan[] plans);

    /**
     * Releases the transport. {@code closePlans} runs at the point where the plans must be closed:
     * after anything that drives them and before anything they captured, such as a communicator.
     */
    void close(Runnable closePlans);

    @Override
    default void close() {
        close(() -> {});
    }

    /** {@code "nccl"} (the default) or {@code "host"}. */
    static PipelineTransport create(String kind, TornadoDevice[] devices) {
        if ("host".equals(kind)) {
            return new HostTransport(devices.length);
        }
        if (!"nccl".equals(kind)) {
            throw new IllegalArgumentException(
                    "jitllm.pipeline.transport must be nccl or host, not " + kind);
        }
        try {
            return (PipelineTransport)
                    Class.forName(NCCL_TRANSPORT)
                            .getConstructor(TornadoDevice[].class)
                            .newInstance((Object) devices);
        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            throw new IllegalStateException(
                    "the NCCL pipeline transport is not in this build (build with -Pnccl against a"
                            + " TornadoVM SDK that has tornado-nccl), or use"
                            + " -Djitllm.pipeline.transport=host",
                    e);
        } catch (java.lang.reflect.InvocationTargetException e) {
            throw e.getCause() instanceof RuntimeException r
                    ? r
                    : new IllegalStateException(e.getCause());
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }
}
