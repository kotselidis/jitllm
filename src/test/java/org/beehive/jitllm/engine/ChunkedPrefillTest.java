package org.beehive.jitllm.engine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.runtime.kv.BlockPool;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.junit.Test;

/**
 * The engine over ragged token batches: prompts go in as chunks next to decode tokens, under a
 * per-step token budget, and the output does not depend on the schedule.
 */
public class ChunkedPrefillTest {

    private static final int BLOCK_TOKENS = 16;
    private static final int BLOCKS_PER_SLOT = 64;

    private static KvCacheManager manager(int slots) {
        return new KvCacheManager(
                new BlockPool(slots * BLOCKS_PER_SLOT, BLOCKS_PER_SLOT, slots, BLOCK_TOKENS, 4096));
    }

    private static int[] prompt(int length, int seed) {
        int[] tokens = new int[length];
        for (int i = 0; i < length; i++) {
            tokens[i] = 1 + (seed * 31 + i * 7) % 500;
        }
        return tokens;
    }

    private static void drain(LLMEngine engine, List<RequestHandle> handles) {
        for (int guard = 0; handles.stream().anyMatch(h -> !h.isTerminal()); guard++) {
            if (guard > 10_000) {
                throw new AssertionError("the engine stopped making progress");
            }
            engine.step();
        }
    }

    @Test
    public void aLongPromptIsIngestedInChunksNotOneTokenPerStep() {
        try (KvCacheManager manager = manager(2)) {
            RaggedFakeExecutor executor = new RaggedFakeExecutor(2, 256, 256);
            try (LLMEngine engine =
                    new LLMEngine(TestModels.sharedKvCapable(), manager, executor, 2, 8)) {
                RequestHandle handle = engine.addRequest(prompt(500, 1), 4, null);

                engine.step();
                assertEquals("a 256-token chunk, no output yet", 0, handle.tokenCount());
                engine.step();
                assertEquals(
                        "the rest of the prompt, and the first token, on the second step",
                        1,
                        handle.tokenCount());
                drain(engine, List.of(handle));

                assertEquals(List.of(256, 244, 1, 1, 1), executor.tokensPerStep);
                assertEquals(RequestState.COMPLETED, handle.state());
                assertEquals(4, handle.tokenCount());
            }
        }
    }

    @Test
    public void decodeTokensAreNeverStarvedByAPromptAndTheBudgetIsRespected() {
        try (KvCacheManager manager = manager(4)) {
            RaggedFakeExecutor executor = new RaggedFakeExecutor(4, 64, 64);
            try (LLMEngine engine =
                    new LLMEngine(TestModels.sharedKvCapable(), manager, executor, 4, 8)) {
                RequestHandle decoding = engine.addRequest(prompt(4, 2), 40, null);
                engine.step(); // its prompt fits one step; it is generating from here on
                assertEquals(1, decoding.tokenCount());

                RequestHandle longPrompt = engine.addRequest(prompt(600, 3), 2, null);
                int before = decoding.tokenCount();
                for (int step = 0; step < 5; step++) {
                    engine.step();
                }

                assertEquals(
                        "one token per step for the generating request, every step",
                        before + 5,
                        decoding.tokenCount());
                assertEquals("still ingesting", 0, longPrompt.tokenCount());
                for (int tokens : executor.tokensPerStep) {
                    assertTrue("over budget: " + tokens, tokens <= 64);
                }
                assertEquals(
                        "the prompt takes what the decode token leaves",
                        Integer.valueOf(64),
                        executor.tokensPerStep.get(executor.tokensPerStep.size() - 1));
                drain(engine, List.of(decoding, longPrompt));
            }
        }
    }

    @Test
    public void theOldestPromptIsChunkedFirst() {
        try (KvCacheManager manager = manager(3)) {
            RaggedFakeExecutor executor = new RaggedFakeExecutor(3, 100, 100);
            try (LLMEngine engine =
                    new LLMEngine(TestModels.sharedKvCapable(), manager, executor, 3, 8)) {
                RequestHandle first = engine.addRequest(prompt(150, 4), 1, null);
                RequestHandle second = engine.addRequest(prompt(150, 5), 1, null);

                engine.step(); // 100 tokens of the first
                engine.step(); // its last 50, then 50 of the second
                assertEquals("the first finishes its prompt first", 1, first.tokenCount());
                assertEquals(0, second.tokenCount());
                engine.step(); // the second's last 100
                assertEquals(1, second.tokenCount());
                assertEquals(List.of(100, 100, 100), executor.tokensPerStep);
            }
        }
    }

    /**
     * The batch-invariance gate: one set of requests, run alone and together, with prompts in one
     * piece or in small chunks, under a roomy budget or a tight one — the tokens each request gets
     * back are the same every time.
     */
    @Test
    public void outputDoesNotDependOnTheScheduleOrTheCompany() {
        int[][] prompts = {prompt(37, 6), prompt(300, 7), prompt(5, 8), prompt(129, 9)};
        int[] budgets = {12, 3, 20, 7};

        List<List<Integer>> reference = new ArrayList<>();
        for (int i = 0; i < prompts.length; i++) {
            reference.add(runAlone(prompts[i], budgets[i]));
        }

        int[][] schedules = {{512, 512}, {64, 16}, {8, 1}, {5, 3}};
        for (int[] schedule : schedules) {
            try (KvCacheManager manager = manager(4)) {
                RaggedFakeExecutor executor = new RaggedFakeExecutor(4, schedule[0], schedule[1]);
                try (LLMEngine engine =
                        new LLMEngine(TestModels.sharedKvCapable(), manager, executor, 4, 16)) {
                    List<RequestHandle> handles = new ArrayList<>();
                    for (int i = 0; i < prompts.length; i++) {
                        handles.add(engine.addRequest(prompts[i], budgets[i], null));
                        engine.step(); // staggered arrivals: each joins a batch already running
                    }
                    drain(engine, handles);
                    for (int i = 0; i < prompts.length; i++) {
                        assertEquals(
                                "request "
                                        + i
                                        + " under budget "
                                        + schedule[0]
                                        + ", chunk "
                                        + schedule[1],
                                reference.get(i),
                                handles.get(i).tokens());
                    }
                }
            }
        }
    }

    private static List<Integer> runAlone(int[] prompt, int maxNewTokens) {
        try (KvCacheManager manager = manager(4)) {
            RaggedFakeExecutor executor = new RaggedFakeExecutor(4, 4, 1);
            try (LLMEngine engine =
                    new LLMEngine(TestModels.sharedKvCapable(), manager, executor, 4, 16)) {
                RequestHandle handle = engine.addRequest(prompt, maxNewTokens, null);
                drain(engine, List.of(handle));
                assertEquals(maxNewTokens, handle.tokenCount());
                return handle.tokens();
            }
        }
    }

    @Test
    public void aBudgetBelowTheBatchSizeIsRefused() {
        try (KvCacheManager manager = manager(4)) {
            IllegalArgumentException refused =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    new LLMEngine(
                                            TestModels.sharedKvCapable(),
                                            manager,
                                            new RaggedFakeExecutor(4, 3, 3),
                                            4,
                                            8));
            assertTrue(refused.getMessage(), refused.getMessage().contains("one token for every"));
        }
    }

    /** An executor that only decodes keeps today's behaviour: one token per request per step. */
    @Test
    public void aDecodeOnlyExecutorStillIngestsOneTokenPerStep() {
        try (KvCacheManager manager = manager(2)) {
            FakeBatchExecutor executor = new FakeBatchExecutor(2, 1_000);
            try (LLMEngine engine =
                    new LLMEngine(TestModels.sharedKvCapable(), manager, executor, 2, 8)) {
                RequestHandle handle = engine.addRequest(prompt(10, 10), 3, null);
                drain(engine, List.of(handle));

                assertEquals("ten prompt steps, then two more decode steps", 12, steps(executor));
                assertEquals(3, handle.tokenCount());
            }
        }
    }

    private static int steps(FakeBatchExecutor executor) {
        return executor.activeCountsPerStep.size();
    }
}
