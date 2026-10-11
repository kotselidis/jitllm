package org.beehive.jitllm.engine;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Decides how many tokens each running request feeds in one step, under a token budget.
 *
 * <p><b>Decode first.</b> Every request that is generating feeds its one token. The engine
 * guarantees the budget covers all of them, so a generating request never stalls behind a long
 * prompt — that stall is what the per-token latency of everyone else would otherwise pay for.
 *
 * <p><b>Then prompt chunks, oldest first.</b> What is left of the budget goes to requests still
 * ingesting their prompt, in admission order, each taking as much as it has left up to the chunk
 * limit. The oldest prompt finishes first, so time to first token follows arrival order rather than
 * whichever prompt happens to be shortest.
 *
 * <p>A request that gets no tokens this step is left out of the plan; it keeps its slot and its
 * place, and runs on a later step.
 */
final class StepPlanner {

    /** One request's share of a step. */
    record Work(ScheduledRequest request, int tokens) {}

    private StepPlanner() {}

    /**
     * @param running the running requests, in slot order
     * @param tokenBudget tokens the step may carry; at least {@code running.size()}
     * @param maxChunk tokens one request may feed in a step; 1 means a prompt goes in one token per
     *     step
     * @return generating requests in slot order, then prompt chunks in admission order
     */
    static List<Work> plan(List<ScheduledRequest> running, int tokenBudget, int maxChunk) {
        List<Work> plan = new ArrayList<>(running.size());
        List<ScheduledRequest> ingesting = new ArrayList<>();
        int budget = tokenBudget;
        for (ScheduledRequest request : running) {
            if (request.remainingPromptTokens() == 0) {
                plan.add(new Work(request, 1));
                budget--;
            } else {
                ingesting.add(request);
            }
        }
        if (budget < 0) {
            throw new IllegalStateException(
                    running.size()
                            + " running requests but a step carries "
                            + tokenBudget
                            + " tokens; the budget must cover one token per running request");
        }
        ingesting.sort(Comparator.comparingLong(ScheduledRequest::sequenceNumber));
        for (ScheduledRequest request : ingesting) {
            if (budget == 0) {
                break;
            }
            int chunk = Math.min(request.remainingPromptTokens(), Math.min(maxChunk, budget));
            plan.add(new Work(request, chunk));
            budget -= chunk;
        }
        return plan;
    }
}
