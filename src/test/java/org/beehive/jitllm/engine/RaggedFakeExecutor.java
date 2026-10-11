package org.beehive.jitllm.engine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.runtime.batch.BatchExecutor;
import org.beehive.jitllm.runtime.batch.BatchSlots;
import org.beehive.jitllm.runtime.batch.StepOutput;
import org.beehive.jitllm.runtime.batch.TokenBatch;

/**
 * A backend that takes ragged batches without a device and is batch invariant by construction.
 *
 * <p>It keeps every KV slot's token history, the way a real executor keeps its KV, and checks that
 * each token lands exactly at the next position of its slot — a skipped, repeated or misplaced
 * token throws. The token it samples is a hash of the slot's whole history, so it depends on what
 * the sequence was fed and on nothing else: which requests shared the step, how the prompt was
 * chunked and how full the step was cannot change it. An engine that feeds the right tokens at the
 * right positions therefore produces the same output under any schedule, and one that does not,
 * does not.
 */
final class RaggedFakeExecutor implements BatchExecutor {

    private final int maxBatchSize;
    private final int maxTokensPerStep;
    private final int maxTokensPerSequence;

    private final Map<Integer, List<Integer>> history = new HashMap<>();

    /** Tokens and sequences of every step, in order. */
    final List<Integer> tokensPerStep = new ArrayList<>();

    final List<Integer> sequencesPerStep = new ArrayList<>();

    RaggedFakeExecutor(int maxBatchSize, int maxTokensPerStep, int maxTokensPerSequence) {
        this.maxBatchSize = maxBatchSize;
        this.maxTokensPerStep = maxTokensPerStep;
        this.maxTokensPerSequence = maxTokensPerSequence;
    }

    @Override
    public int maxBatchSize() {
        return maxBatchSize;
    }

    @Override
    public int maxTokensPerStep() {
        return maxTokensPerStep;
    }

    @Override
    public int maxTokensPerSequence() {
        return maxTokensPerSequence;
    }

    @Override
    public StepOutput execute(TokenBatch batch) {
        if (batch.tokenCount() > maxTokensPerStep) {
            throw new AssertionError(
                    batch.tokenCount() + " tokens in a step over a budget of " + maxTokensPerStep);
        }
        int[] sampled = new int[batch.sequenceCount()];
        for (int s = 0; s < batch.sequenceCount(); s++) {
            if (batch.queryLength(s) > maxTokensPerSequence) {
                throw new AssertionError(
                        "a chunk of " + batch.queryLength(s) + " over " + maxTokensPerSequence);
            }
            int slot = batch.kvSlots()[s];
            for (int t = batch.queryStart()[s]; t < batch.queryStart()[s + 1]; t++) {
                int position = batch.positions()[t];
                if (position == 0) {
                    history.put(slot, new ArrayList<>());
                }
                List<Integer> fed = history.get(slot);
                if (fed == null || fed.size() != position) {
                    throw new AssertionError(
                            "slot "
                                    + slot
                                    + " was fed position "
                                    + position
                                    + " after "
                                    + (fed == null ? 0 : fed.size())
                                    + " tokens");
                }
                fed.add(batch.tokens()[t]);
            }
            sampled[s] = 1000 + (Arrays.hashCode(history.get(slot).toArray()) & 0x7fff);
        }
        tokensPerStep.add(batch.tokenCount());
        sequencesPerStep.add(batch.sequenceCount());
        return new StepOutput(sampled);
    }

    @Override
    public int[] decodeStep(BatchSlots batch) {
        throw new AssertionError("a ragged executor is driven through execute()");
    }

    @Override
    public boolean isStopToken(int token) {
        return false;
    }
}
