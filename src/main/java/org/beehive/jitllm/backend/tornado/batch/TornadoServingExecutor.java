package org.beehive.jitllm.backend.tornado.batch;

import org.beehive.jitllm.runtime.backend.ExecutionInfo;
import org.beehive.jitllm.runtime.batch.BatchExecutor;

/** A TornadoVM executor the serving engine drives: it runs steps, describes itself and closes. */
public interface TornadoServingExecutor extends BatchExecutor, AutoCloseable {

    /** What it runs on and how, for the startup report. */
    ExecutionInfo executionInfo();

    /** Frees the device memory its plan holds. */
    @Override
    void close();
}
