package org.beehive.jitllm.runtime.batch;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** The ragged layout and its checks, and the decode-only adapter onto {@code decodeStep}. */
public class TokenBatchTest {

    /** Sequence 0 decodes one token at position 9; sequence 1 feeds a 3-token chunk at 0..2. */
    private static TokenBatch mixed() {
        return new TokenBatch(
                new int[] {50, 11, 12, 13},
                new int[] {9, 0, 1, 2},
                new int[] {0, 1, 1, 1},
                new int[] {3, 0},
                new int[] {0, 1, 4},
                new boolean[] {true, false});
    }

    @Test
    public void describesAMixedStep() {
        TokenBatch batch = mixed();

        assertEquals(4, batch.tokenCount());
        assertEquals(2, batch.sequenceCount());
        assertEquals(1, batch.queryLength(0));
        assertEquals(3, batch.queryLength(1));
        assertEquals(3, batch.lastTokenOf(1));
        assertEquals("KV length after the step", 10, batch.contextLength(0));
        assertEquals(3, batch.contextLength(1));
        assertFalse(batch.isDecodeOnly());
    }

    @Test
    public void refusesALayoutThatDoesNotAddUp() {
        assertThrows(
                "query starts past the token count",
                IllegalArgumentException.class,
                () ->
                        new TokenBatch(
                                new int[] {1, 2},
                                new int[] {0, 1},
                                new int[] {0, 0},
                                new int[] {0},
                                new int[] {0, 3},
                                new boolean[] {true}));
        assertThrows(
                "a gap in a sequence's positions",
                IllegalArgumentException.class,
                () ->
                        new TokenBatch(
                                new int[] {1, 2},
                                new int[] {0, 2},
                                new int[] {0, 0},
                                new int[] {0},
                                new int[] {0, 2},
                                new boolean[] {true}));
        assertThrows(
                "a token filed under the wrong sequence",
                IllegalArgumentException.class,
                () ->
                        new TokenBatch(
                                new int[] {1, 2},
                                new int[] {0, 0},
                                new int[] {0, 0},
                                new int[] {0, 1},
                                new int[] {0, 1, 2},
                                new boolean[] {true, true}));
        assertThrows(
                "an empty sequence",
                IllegalArgumentException.class,
                () ->
                        new TokenBatch(
                                new int[] {1},
                                new int[] {0},
                                new int[] {1},
                                new int[] {0, 1},
                                new int[] {0, 0, 1},
                                new boolean[] {true, true}));
    }

    /** Records what {@code decodeStep} saw and answers 100 + slot. */
    private static final class DecodeOnly implements BatchExecutor {
        BatchSlots seen;

        @Override
        public int maxBatchSize() {
            return 4;
        }

        @Override
        public int[] decodeStep(BatchSlots batch) {
            seen = batch;
            int[] out = new int[batch.width()];
            for (int i = 0; i < out.length; i++) {
                out[i] = 100 + i;
            }
            return out;
        }

        @Override
        public boolean isStopToken(int token) {
            return false;
        }
    }

    @Test
    public void theDefaultExecuteLaysADecodeStepOutBySlot() {
        DecodeOnly executor = new DecodeOnly();
        TokenBatch decode =
                new TokenBatch(
                        new int[] {7, 8},
                        new int[] {5, 2},
                        new int[] {0, 1},
                        new int[] {2, 0},
                        new int[] {0, 1, 2},
                        new boolean[] {true, true});

        StepOutput out = executor.execute(decode);

        assertEquals(4, executor.seen.width());
        assertArrayEquals(new boolean[] {true, false, true, false}, executor.seen.active());
        assertEquals(7, executor.seen.tokens()[2]);
        assertEquals(5, executor.seen.positions()[2]);
        assertEquals(8, executor.seen.tokens()[0]);
        assertArrayEquals(
                "sampled per sequence, not per slot", new int[] {102, 100}, out.sampled());
    }

    @Test
    public void theDefaultExecuteRefusesPromptChunks() {
        UnsupportedOperationException refused =
                assertThrows(
                        UnsupportedOperationException.class,
                        () -> new DecodeOnly().execute(mixed()));
        assertTrue(refused.getMessage(), refused.getMessage().contains("must not chunk"));
    }
}
