package org.beehive.jitllm.backend.tornado.pipeline;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Moves the hidden state through host memory: the sending stage copies it into a host buffer and
 * brings that back, the next stage uploads it. The stages run one after the other on the calling
 * thread. No NCCL needed, and a reference to compare the NCCL transport against.
 */
final class HostTransport implements PipelineTransport {

    private static final int LOCAL_SIZE = 256;

    /** {@code staging[s]} carries the hidden state from stage {@code s} to stage {@code s + 1}. */
    private final FloatArray[] staging;

    private final int[] widths;

    HostTransport(int stages) {
        this.staging = new FloatArray[stages];
        this.widths = new int[stages];
    }

    private FloatArray staging(int link, int width) {
        if (staging[link] == null) {
            staging[link] = new FloatArray(width);
        }
        return staging[link];
    }

    @Override
    public TaskGraph sendGraph(int stage, String producer, FloatArray x, int toStage) {
        int width = x.getSize();
        widths[stage] = width;
        FloatArray out = staging(stage, width);
        return new TaskGraph("handoff_out")
                .consumeFromDevice(producer, x)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, out)
                .task("copy", PipelineKernels::copy, new KernelContext(), x, out, width)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
    }

    @Override
    public TaskGraph receiveGraph(int stage, FloatArray x, int fromStage) {
        int width = x.getSize();
        widths[stage] = width;
        FloatArray in = staging(fromStage, width);
        return new TaskGraph("handoff_in")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, in)
                .task("copy", PipelineKernels::copy, new KernelContext(), in, x, width)
                .persistOnDevice(x);
    }

    @Override
    public void updateGridScheduler(int stage, GridScheduler scheduler) {
        WorkerGrid worker = new WorkerGrid1D(roundUp(widths[stage]));
        worker.setLocalWork(LOCAL_SIZE, 1, 1);
        scheduler.addWorkerGrid("handoff_out.copy", worker);
        scheduler.addWorkerGrid("handoff_in.copy", worker);
    }

    private static int roundUp(int n) {
        return (n + LOCAL_SIZE - 1) / LOCAL_SIZE * LOCAL_SIZE;
    }

    @Override
    public void execute(TornadoExecutionPlan[] plans) {
        // Stage s's transferToHost has completed when execute returns, so stage s + 1 uploads the
        // finished hidden state.
        for (TornadoExecutionPlan plan : plans) {
            plan.execute();
        }
    }

    @Override
    public void close(Runnable closePlans) {
        closePlans.run();
    }
}
