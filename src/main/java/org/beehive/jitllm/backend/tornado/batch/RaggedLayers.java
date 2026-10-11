package org.beehive.jitllm.backend.tornado.batch;

import java.util.List;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The layer graphs a {@link TornadoRaggedBatchExecutor} drives: one per transformer layer, rows
 * being tokens of several sequences, each with its own position and KV slot.
 *
 * <p>One implementation per weight representation. What the executor needs from them is the same
 * whatever the arithmetic: the graphs, their grids, the per-step control arrays it writes, and a
 * way to launch only the rows a step uses.
 */
public interface RaggedLayers {

    List<ImmutableTaskGraph> getLayerImmutableTaskGraphs();

    String getLastLayerTaskGraphID();

    void updateGridScheduler(GridScheduler scheduler);

    /** Per query tile, its first row and row count; written by the executor each step. */
    IntArray queryTiles();

    /** A generated token's KV slot per row, -1 for a prompt row; written by the executor. */
    IntArray decodeRowSlots();

    /** Rows of one sequence per attention query tile. */
    int queryTileRows();

    /**
     * Launches only the first {@code rows} rows on the next execution.
     *
     * @param rows a multiple of 16
     * @param realRows the step's tokens, before rounding up to {@code rows}
     * @param tiles query tiles written to {@link #queryTiles()}
     */
    void setActiveRows(int rows, int realRows, int tiles);
}
