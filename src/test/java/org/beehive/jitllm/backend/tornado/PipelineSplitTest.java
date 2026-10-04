package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThrows;

import org.junit.Test;

/** How the pipeline split shares layers out between devices. */
public class PipelineSplitTest {

    @Test
    public void equalSharesSplitEvenly() {
        assertArrayEquals(
                new int[] {0, 40, 80}, TornadoVMMasterPlanPipeline.splitLayers(80, 2, null));
        assertArrayEquals(new int[] {0, 8, 16}, TornadoVMMasterPlanPipeline.splitLayers(16, 2, ""));
    }

    @Test
    public void proportionsFollowTheTensorSplit() {
        assertArrayEquals(
                new int[] {0, 41, 80}, TornadoVMMasterPlanPipeline.splitLayers(80, 2, "41,39"));
        assertArrayEquals(
                new int[] {0, 20, 60, 80}, TornadoVMMasterPlanPipeline.splitLayers(80, 3, "1,2,1"));
    }

    @Test
    public void everyStageKeepsAtLeastOneLayer() {
        assertArrayEquals(
                new int[] {0, 1, 2, 3}, TornadoVMMasterPlanPipeline.splitLayers(3, 3, "100,1,1"));
        assertArrayEquals(
                new int[] {0, 1, 4}, TornadoVMMasterPlanPipeline.splitLayers(4, 2, "1,100"));
    }

    @Test
    public void invalidSplitsAreRefused() {
        assertThrows(
                IllegalArgumentException.class,
                () -> TornadoVMMasterPlanPipeline.splitLayers(80, 2, "1,1,1"));
        assertThrows(
                IllegalArgumentException.class,
                () -> TornadoVMMasterPlanPipeline.splitLayers(80, 2, "1,0"));
        assertThrows(
                IllegalArgumentException.class,
                () -> TornadoVMMasterPlanPipeline.splitLayers(1, 2, null));
    }
}
