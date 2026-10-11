package org.beehive.jitllm.runtime.batch;

/**
 * The one thing a scheduler needs from a backend: advance every active slot by one token.
 *
 * <p><b>Why it lives in {@code runtime} and speaks only in primitives.</b> The dependency rules
 * leave exactly one place for this interface. {@code tornadovm} may not name {@code engine} [Rule
 * 18], and an adapter in {@code engine} would have to import TornadoVM, which would grow Rule 1's
 * shrink-only allowlist. A neutral seam in {@code runtime} is what both sides may depend on — the
 * engine schedules against it, the backend implements it, and neither has to know the other's
 * types.
 *
 * <p>That constraint improved the design rather than bending it. An executor does not need to know
 * what a request <i>is</i>; it needs to know, per slot, whether the slot is active, which token to
 * feed it and at which position. Everything else was the scheduler's business all along.
 *
 * <p>Called only from inside the engine's single-caller {@code step()}, so an implementation needs
 * no locking of its own.
 */
public interface BatchExecutor {

    /** Slots this executor was built for. Fixed: changing it means building another. */
    int maxBatchSize();

    /**
     * Advances every active slot by one token.
     *
     * @param batch what to run — see {@link BatchSlots}
     * @return one token id per slot, positionally. The value at an inactive slot is meaningless and
     *     the caller ignores it.
     */
    int[] decodeStep(BatchSlots batch);

    /**
     * Tokens one step may carry across all sequences: the engine's per-step token budget.
     *
     * <p>Never below {@link #maxBatchSize()}, so every running decode sequence gets its token on
     * every step and only prompt chunks compete for what is left. A decode-only executor carries
     * exactly one token per slot.
     */
    default int maxTokensPerStep() {
        return maxBatchSize();
    }

    /**
     * Tokens one sequence may contribute to a step: the largest prompt chunk.
     *
     * <p>1 for an executor that only decodes, which then ingests a prompt one token per step.
     */
    default int maxTokensPerSequence() {
        return 1;
    }

    /**
     * Runs one ragged step: prompt chunks and decode tokens of several sequences together.
     *
     * <p>The default serves an executor that only decodes. It accepts a batch where every sequence
     * contributes one token — which is all an engine sends when {@link #maxTokensPerSequence()} is
     * 1 — and lays it out as the fixed-width {@link BatchSlots} {@link #decodeStep} takes, one slot
     * per KV slot.
     *
     * @throws UnsupportedOperationException if a sequence contributes more than one token and the
     *     executor did not override this
     */
    default StepOutput execute(TokenBatch batch) {
        if (!batch.isDecodeOnly()) {
            throw new UnsupportedOperationException(
                    getClass().getSimpleName()
                            + " decodes one token per sequence per step and was handed "
                            + batch.tokenCount()
                            + " tokens for "
                            + batch.sequenceCount()
                            + " sequences; an engine must not chunk prompts for it");
        }
        // Positions are KV slots, so the width must reach the highest one in use.
        int width = maxBatchSize();
        for (int slot : batch.kvSlots()) {
            width = Math.max(width, slot + 1);
        }
        boolean[] active = new boolean[width];
        int[] tokens = new int[width];
        int[] positions = new int[width];
        int[] kvSlots = new int[width];
        for (int s = 0; s < batch.sequenceCount(); s++) {
            int slot = batch.kvSlots()[s];
            active[slot] = true;
            tokens[slot] = batch.tokens()[s];
            positions[slot] = batch.positions()[s];
            kvSlots[slot] = slot;
        }
        int[] bySlot = decodeStep(new BatchSlots(active, tokens, positions, kvSlots));
        int[] sampled = new int[batch.sequenceCount()];
        for (int s = 0; s < sampled.length; s++) {
            sampled[s] = bySlot[batch.kvSlots()[s]];
        }
        return new StepOutput(sampled);
    }

    /**
     * Whether this token ends a sequence.
     *
     * <p>Asked of the executor because a stop token is the model's vocabulary, not the scheduler's.
     * The budget half of "is it finished" stays with the scheduler, which is the half it owns.
     */
    boolean isStopToken(int token);
}
