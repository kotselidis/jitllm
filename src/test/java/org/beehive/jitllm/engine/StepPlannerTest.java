package org.beehive.jitllm.engine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

import java.util.List;
import org.junit.Test;

/** The per-step token policy, on requests alone: no engine, no pool. */
public class StepPlannerTest {

    private static ScheduledRequest request(long sequenceNumber, int promptLength, int fed) {
        ScheduledRequest request = new ScheduledRequest(sequenceNumber, promptLength + 64);
        request.promptTokens(new int[promptLength]);
        request.consume(fed);
        return request;
    }

    @Test
    public void generatingRequestsTakeOneTokenEachBeforeAnyPromptChunk() {
        ScheduledRequest ingesting = request(0, 500, 0);
        ScheduledRequest generatingA = request(1, 10, 12);
        ScheduledRequest generatingB = request(2, 10, 10);

        List<StepPlanner.Work> plan =
                StepPlanner.plan(List.of(ingesting, generatingA, generatingB), 64, 64);

        assertEquals(3, plan.size());
        assertSame(generatingA, plan.get(0).request());
        assertEquals(1, plan.get(0).tokens());
        assertSame(generatingB, plan.get(1).request());
        assertSame(ingesting, plan.get(2).request());
        assertEquals("the budget left after the decode tokens", 62, plan.get(2).tokens());
    }

    @Test
    public void promptChunksGoOldestFirstAndStopAtTheBudget() {
        ScheduledRequest newer = request(5, 100, 0);
        ScheduledRequest older = request(3, 30, 0);
        ScheduledRequest newest = request(9, 100, 0);

        List<StepPlanner.Work> plan = StepPlanner.plan(List.of(newer, older, newest), 80, 64);

        assertEquals("the newest gets nothing this step", 2, plan.size());
        assertSame(older, plan.get(0).request());
        assertEquals("all it has left", 30, plan.get(0).tokens());
        assertSame(newer, plan.get(1).request());
        assertEquals("what remains of the budget", 50, plan.get(1).tokens());
    }

    @Test
    public void theChunkLimitCapsEachRequest() {
        ScheduledRequest a = request(0, 100, 0);
        ScheduledRequest b = request(1, 100, 0);

        List<StepPlanner.Work> plan = StepPlanner.plan(List.of(a, b), 256, 16);

        assertEquals(16, plan.get(0).tokens());
        assertEquals(16, plan.get(1).tokens());
    }

    @Test
    public void aChunkLimitOfOneFeedsEveryRequestOneToken() {
        ScheduledRequest ingesting = request(0, 100, 7);
        ScheduledRequest generating = request(1, 4, 9);

        List<StepPlanner.Work> plan = StepPlanner.plan(List.of(ingesting, generating), 2, 1);

        assertEquals(2, plan.size());
        assertEquals(1, plan.get(0).tokens());
        assertEquals(1, plan.get(1).tokens());
    }

    @Test
    public void aBudgetThatCannotCoverTheDecodeTokensIsABug() {
        assertThrows(
                IllegalStateException.class,
                () -> StepPlanner.plan(List.of(request(0, 1, 1), request(1, 1, 1)), 1, 1));
    }
}
