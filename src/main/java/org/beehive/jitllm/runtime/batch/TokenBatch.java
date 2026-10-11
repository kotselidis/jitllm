package org.beehive.jitllm.runtime.batch;

/**
 * One step's worth of work as a flat run of tokens: prompt chunks and decode tokens of several
 * sequences, mixed in one forward pass.
 *
 * <p>Tokens are grouped by sequence and laid out back to back. Sequence {@code s} owns tokens
 * {@code queryStart[s]} up to, not including, {@code queryStart[s + 1]} — the {@code cu_seqlens}
 * layout variable-length attention kernels take. A decode sequence contributes one token; a
 * sequence still ingesting its prompt contributes a chunk of it. A running sequence that got no
 * tokens this step is simply absent: unlike {@link BatchSlots}, nothing here is fixed width, which
 * is what lets the step cost follow the work instead of the batch size.
 *
 * <p><b>Batch invariance.</b> What an executor computes for one sequence must not depend on which
 * other sequences share the step, on how its prompt was chunked, or on how many tokens the step
 * carried. Greedy output is then identical whether a request ran alone or under load, which is the
 * property the engine's tests hold executors to.
 *
 * @param tokens per token: the token id to feed
 * @param positions per token: the sequence position it occupies. Contiguous within a sequence
 * @param sequenceOf per token: the index of the sequence it belongs to
 * @param kvSlots per sequence: its row in the KV block table
 * @param queryStart per sequence plus one: where its tokens start; the last entry is the token
 *     count
 * @param samples per sequence: whether the logits after its last token are needed — false for a
 *     prompt chunk that does not reach the end of the prompt, whose next token is already known
 * @param promptRows per token: whether it is a prompt token rather than a generated one. A property
 *     of the token itself, so an executor may pick a kernel by it without breaking batch invariance
 */
public record TokenBatch(
        int[] tokens,
        int[] positions,
        int[] sequenceOf,
        int[] kvSlots,
        int[] queryStart,
        boolean[] samples,
        boolean[] promptRows) {

    /** A batch whose tokens are all treated as prompt tokens, for callers that do not tell. */
    public TokenBatch(
            int[] tokens,
            int[] positions,
            int[] sequenceOf,
            int[] kvSlots,
            int[] queryStart,
            boolean[] samples) {
        this(tokens, positions, sequenceOf, kvSlots, queryStart, samples, allTrue(tokens.length));
    }

    private static boolean[] allTrue(int n) {
        boolean[] all = new boolean[n];
        java.util.Arrays.fill(all, true);
        return all;
    }

    public TokenBatch {
        int t = tokens.length;
        int s = kvSlots.length;
        if (positions.length != t || sequenceOf.length != t || promptRows.length != t) {
            throw new IllegalArgumentException(
                    "per-token arrays disagree: "
                            + t
                            + " tokens, "
                            + positions.length
                            + " positions, "
                            + sequenceOf.length
                            + " sequence indices");
        }
        if (queryStart.length != s + 1 || samples.length != s) {
            throw new IllegalArgumentException(
                    "per-sequence arrays disagree: "
                            + s
                            + " KV slots, "
                            + queryStart.length
                            + " query starts (want "
                            + (s + 1)
                            + "), "
                            + samples.length
                            + " sample flags");
        }
        if (queryStart[0] != 0 || queryStart[s] != t) {
            throw new IllegalArgumentException(
                    "query starts must run from 0 to the token count " + t);
        }
        for (int seq = 0; seq < s; seq++) {
            int from = queryStart[seq];
            int to = queryStart[seq + 1];
            if (to <= from) {
                throw new IllegalArgumentException(
                        "sequence " + seq + " has no tokens; leave it out of the step instead");
            }
            for (int i = from; i < to; i++) {
                if (sequenceOf[i] != seq) {
                    throw new IllegalArgumentException(
                            "token "
                                    + i
                                    + " is laid out under sequence "
                                    + seq
                                    + " but names sequence "
                                    + sequenceOf[i]);
                }
                if (i > from && positions[i] != positions[i - 1] + 1) {
                    throw new IllegalArgumentException(
                            "positions of sequence " + seq + " are not contiguous at token " + i);
                }
            }
        }
    }

    /** T: tokens this step computes. */
    public int tokenCount() {
        return tokens.length;
    }

    /** S: sequences with at least one token this step. */
    public int sequenceCount() {
        return kvSlots.length;
    }

    /** Tokens sequence {@code s} contributes: 1 for decode, the chunk length for prefill. */
    public int queryLength(int s) {
        return queryStart[s + 1] - queryStart[s];
    }

    /** The token whose logits sequence {@code s} samples from: its last one this step. */
    public int lastTokenOf(int s) {
        return queryStart[s + 1] - 1;
    }

    /** Positions sequence {@code s} attends over after this step: its KV length. */
    public int contextLength(int s) {
        return positions[lastTokenOf(s)] + 1;
    }

    /** Whether every sequence contributes exactly one token: a pure decode step. */
    public boolean isDecodeOnly() {
        return tokens.length == kvSlots.length;
    }
}
