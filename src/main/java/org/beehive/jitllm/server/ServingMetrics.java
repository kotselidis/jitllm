package org.beehive.jitllm.server;

import java.util.Locale;
import org.beehive.jitllm.runtime.metrics.MetricKey;
import org.beehive.jitllm.runtime.metrics.MetricsSink;

/**
 * The serving measurements behind {@code GET /metrics}, in the Prometheus text format.
 *
 * <p>Two producers feed it. The engine records through {@link MetricsSink}: queue wait per admitted
 * request, steps, batch occupancy, admissions and rejections. The inference service records one
 * {@link #requestFinished finished request} at a time: prompt and output token counts, time to
 * first token, time per output token and end-to-end latency, all measured where the HTTP handler
 * sees them, so they agree with what a load generator measures from outside.
 *
 * <p>Levels — requests running and waiting, KV blocks in use — are not recorded here. They are read
 * from the engine at scrape time, because a level sampled once per step is stale by the time it is
 * scraped.
 *
 * <p>Metric names follow vLLM's ({@code vllm:num_requests_running} becomes {@code
 * jitllm:num_requests_running}), so a dashboard or script written for one reads the other after a
 * prefix change.
 */
final class ServingMetrics implements MetricsSink {

    /** Levels read from the engine when {@code /metrics} is scraped. */
    record Levels(int running, int waiting, long kvBlocksInUse, long kvBlocksTotal) {}

    // vLLM's bucket boundaries, in seconds, so histograms from the two engines line up.
    private static final double[] TTFT_BUCKETS = {
        0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
        20.0, 40.0, 80.0, 160.0, 640.0, 2560.0
    };
    private static final double[] TPOT_BUCKETS = {
        0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
        20.0, 40.0, 80.0
    };
    private static final double[] LATENCY_BUCKETS = {
        0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0,
        240.0, 480.0, 960.0, 1920.0, 7680.0
    };

    /** {@code model_name="…"}, without braces, so other labels can follow it. */
    private final String labels;

    private final Histogram timeToFirstToken = new Histogram(TTFT_BUCKETS);
    private final Histogram timePerOutputToken = new Histogram(TPOT_BUCKETS);
    private final Histogram endToEnd = new Histogram(LATENCY_BUCKETS);
    private final Histogram queueTime = new Histogram(LATENCY_BUCKETS);

    private long promptTokens;
    private long generationTokens;
    private long requestsFinishedStop;
    private long requestsFinishedLength;
    private long requestsAdmitted;
    private long requestsRejected;
    private long engineSteps;
    private long batchOccupancy;
    private long prefixBlocksReused;

    ServingMetrics(String servedModel) {
        this.labels = "model_name=\"" + escape(servedModel) + "\"";
    }

    @Override
    public synchronized void record(MetricKey key, long value) {
        switch (key) {
            case QUEUE_WAIT_TIME -> queueTime.observe(value / 1e9);
            case REQUESTS_ADMITTED -> requestsAdmitted += value;
            case REQUESTS_REJECTED -> requestsRejected += value;
            case ENGINE_STEPS -> engineSteps += value;
            case BATCH_OCCUPANCY -> batchOccupancy += value;
            case PREFIX_BLOCKS_REUSED -> prefixBlocksReused += value;
            default -> {
                // Time to first token is measured by requestFinished, where the client sees it; KV
                // levels are read at scrape time; device keys are not serving metrics.
            }
        }
    }

    /**
     * One request that produced a response.
     *
     * @param firstTokenNanos submission to first output token, or a negative value when none was
     *     produced
     * @param totalNanos submission to the last output token
     * @param stopped whether a stop token ended it, rather than the token budget
     */
    synchronized void requestFinished(
            int prompt, int completion, long firstTokenNanos, long totalNanos, boolean stopped) {
        promptTokens += prompt;
        generationTokens += completion;
        if (stopped) {
            requestsFinishedStop++;
        } else {
            requestsFinishedLength++;
        }
        endToEnd.observe(totalNanos / 1e9);
        if (firstTokenNanos >= 0) {
            timeToFirstToken.observe(firstTokenNanos / 1e9);
            if (completion > 1) {
                timePerOutputToken.observe((totalNanos - firstTokenNanos) / 1e9 / (completion - 1));
            }
        }
    }

    synchronized String render(Levels levels) {
        StringBuilder out = new StringBuilder(4096);
        gauge(out, "num_requests_running", "Requests in the running batch.", levels.running());
        gauge(out, "num_requests_waiting", "Requests queued for admission.", levels.waiting());
        gauge(out, "kv_cache_blocks_in_use", "KV cache blocks held.", levels.kvBlocksInUse());
        gauge(out, "kv_cache_blocks_total", "KV cache blocks in the pool.", levels.kvBlocksTotal());
        gauge(
                out,
                "kv_cache_usage_perc",
                "Fraction of KV cache blocks held.",
                levels.kvBlocksTotal() == 0
                        ? 0.0
                        : (double) levels.kvBlocksInUse() / levels.kvBlocksTotal());

        counter(out, "prompt_tokens_total", "Prompt tokens of finished requests.", promptTokens);
        counter(
                out,
                "generation_tokens_total",
                "Output tokens of finished requests.",
                generationTokens);
        out.append("# HELP jitllm:request_success_total Finished requests, by finish reason.\n");
        out.append("# TYPE jitllm:request_success_total counter\n");
        finishReason(out, "stop", requestsFinishedStop);
        finishReason(out, "length", requestsFinishedLength);
        counter(
                out,
                "requests_admitted_total",
                "Requests admitted to the batch.",
                requestsAdmitted);
        counter(
                out,
                "requests_rejected_total",
                "Requests refused at submission.",
                requestsRejected);
        counter(out, "engine_steps_total", "Engine steps that ran the batch.", engineSteps);
        counter(
                out,
                "batch_occupancy_sum",
                "Running requests summed over engine steps; divide by engine_steps_total.",
                batchOccupancy);
        counter(
                out,
                "prefix_cache_blocks_reused_total",
                "KV blocks taken from the prefix cache instead of prefilled.",
                prefixBlocksReused);

        histogram(
                out,
                "time_to_first_token_seconds",
                "Submission to first output token.",
                timeToFirstToken);
        histogram(
                out,
                "time_per_output_token_seconds",
                "Mean gap between output tokens after the first, per request.",
                timePerOutputToken);
        histogram(out, "e2e_request_latency_seconds", "Submission to last output token.", endToEnd);
        histogram(
                out,
                "request_queue_time_seconds",
                "Submission to admission into the batch.",
                queueTime);
        return out.toString();
    }

    private void gauge(StringBuilder out, String name, String help, double value) {
        header(out, name, help, "gauge");
        out.append("jitllm:").append(name).append('{').append(labels).append('}').append(' ');
        out.append(number(value)).append('\n');
    }

    private void counter(StringBuilder out, String name, String help, long value) {
        header(out, name, help, "counter");
        out.append("jitllm:")
                .append(name)
                .append('{')
                .append(labels)
                .append('}')
                .append(' ')
                .append(value);
        out.append('\n');
    }

    private void finishReason(StringBuilder out, String reason, long value) {
        out.append("jitllm:request_success_total{")
                .append(labels)
                .append(",finished_reason=\"")
                .append(reason)
                .append("\"} ")
                .append(value)
                .append('\n');
    }

    private void histogram(StringBuilder out, String name, String help, Histogram h) {
        header(out, name, help, "histogram");
        long cumulative = 0;
        for (int i = 0; i < h.bounds.length; i++) {
            cumulative += h.counts[i];
            out.append("jitllm:").append(name).append("_bucket{").append(labels);
            out.append(",le=\"").append(number(h.bounds[i])).append("\"} ");
            out.append(cumulative).append('\n');
        }
        out.append("jitllm:").append(name).append("_bucket{").append(labels);
        out.append(",le=\"+Inf\"} ").append(h.count).append('\n');
        out.append("jitllm:")
                .append(name)
                .append("_sum")
                .append('{')
                .append(labels)
                .append('}')
                .append(' ');
        out.append(number(h.sum)).append('\n');
        out.append("jitllm:")
                .append(name)
                .append("_count")
                .append('{')
                .append(labels)
                .append('}')
                .append(' ');
        out.append(h.count).append('\n');
    }

    private static void header(StringBuilder out, String name, String help, String type) {
        out.append("# HELP jitllm:").append(name).append(' ').append(help).append('\n');
        out.append("# TYPE jitllm:").append(name).append(' ').append(type).append('\n');
    }

    private static String number(double value) {
        if (value == Math.rint(value) && Math.abs(value) < 1e15) {
            return Long.toString((long) value) + ".0";
        }
        return String.format(Locale.ROOT, "%s", value);
    }

    /** Label values escape backslash, quote and newline, and nothing else. */
    private static String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }

    /** Non-cumulative per bucket; the renderer accumulates, as the format requires. */
    private static final class Histogram {
        final double[] bounds;
        final long[] counts;
        long count;
        double sum;

        Histogram(double[] bounds) {
            this.bounds = bounds;
            this.counts = new long[bounds.length];
        }

        void observe(double seconds) {
            count++;
            sum += seconds;
            for (int i = 0; i < bounds.length; i++) {
                if (seconds <= bounds[i]) {
                    counts[i]++;
                    return;
                }
            }
        }
    }
}
