package org.beehive.jitllm.runtime.batch;

/**
 * What one {@link TokenBatch} produced.
 *
 * @param sampled per sequence, positionally: the token sampled after its last token. Meaningless
 *     where the batch did not ask that sequence to sample, and the caller ignores it there
 */
public record StepOutput(int[] sampled) {}
