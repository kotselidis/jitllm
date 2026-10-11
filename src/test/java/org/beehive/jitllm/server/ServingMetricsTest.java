package org.beehive.jitllm.server;

import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.runtime.metrics.MetricKey;
import org.junit.Test;

/** The {@code /metrics} body, without a server or a device. */
public class ServingMetricsTest {

    private static final ServingMetrics.Levels LEVELS = new ServingMetrics.Levels(3, 2, 40, 160);

    private static void assertLine(String body, String line) {
        assertTrue("missing line: " + line + "\n" + body, body.contains(line + "\n"));
    }

    @Test
    public void levelsAreReadAtScrapeTime() {
        String body = new ServingMetrics("m").render(LEVELS);

        assertLine(body, "jitllm:num_requests_running{model_name=\"m\"} 3.0");
        assertLine(body, "jitllm:num_requests_waiting{model_name=\"m\"} 2.0");
        assertLine(body, "jitllm:kv_cache_blocks_in_use{model_name=\"m\"} 40.0");
        assertLine(body, "jitllm:kv_cache_usage_perc{model_name=\"m\"} 0.25");
        assertLine(body, "# TYPE jitllm:num_requests_running gauge");
    }

    @Test
    public void finishedRequestsFeedTokenCountersAndLatencyHistograms() {
        ServingMetrics metrics = new ServingMetrics("m");
        // 0.05 s to the first token, then 10 more tokens over 0.5 s: 0.05 s per output token.
        metrics.requestFinished(100, 11, 50_000_000L, 550_000_000L, false);
        metrics.requestFinished(20, 1, 3_000_000_000L, 3_000_000_000L, true);
        String body = metrics.render(LEVELS);

        assertLine(body, "jitllm:prompt_tokens_total{model_name=\"m\"} 120");
        assertLine(body, "jitllm:generation_tokens_total{model_name=\"m\"} 12");
        assertLine(
                body, "jitllm:request_success_total{model_name=\"m\",finished_reason=\"stop\"} 1");
        assertLine(
                body,
                "jitllm:request_success_total{model_name=\"m\",finished_reason=\"length\"} 1");

        // Buckets are cumulative: one request under 0.06 s, both under 5 s.
        assertLine(
                body, "jitllm:time_to_first_token_seconds_bucket{model_name=\"m\",le=\"0.06\"} 1");
        assertLine(
                body, "jitllm:time_to_first_token_seconds_bucket{model_name=\"m\",le=\"5.0\"} 2");
        assertLine(
                body, "jitllm:time_to_first_token_seconds_bucket{model_name=\"m\",le=\"+Inf\"} 2");
        assertLine(body, "jitllm:time_to_first_token_seconds_count{model_name=\"m\"} 2");
        // A one-token request has no gap between tokens, so only the first is observed.
        assertLine(body, "jitllm:time_per_output_token_seconds_count{model_name=\"m\"} 1");
        assertLine(
                body,
                "jitllm:time_per_output_token_seconds_bucket{model_name=\"m\",le=\"0.05\"} 1");
    }

    @Test
    public void engineRecordsBecomeCountersAndTheQueueHistogram() {
        ServingMetrics metrics = new ServingMetrics("m");
        metrics.record(MetricKey.REQUESTS_ADMITTED, 1);
        metrics.record(MetricKey.REQUESTS_ADMITTED, 1);
        metrics.record(MetricKey.ENGINE_STEPS, 7);
        metrics.record(MetricKey.QUEUE_WAIT_TIME, 200_000_000L);
        // Device keys are not serving metrics and must not break anything.
        metrics.record(MetricKey.DEVICE_KERNEL_TIME, 123);
        String body = metrics.render(LEVELS);

        assertLine(body, "jitllm:requests_admitted_total{model_name=\"m\"} 2");
        assertLine(body, "jitllm:engine_steps_total{model_name=\"m\"} 7");
        assertLine(body, "jitllm:request_queue_time_seconds_bucket{model_name=\"m\",le=\"0.3\"} 1");
        assertLine(body, "jitllm:request_queue_time_seconds_count{model_name=\"m\"} 1");
    }

    @Test
    public void theModelNameIsEscapedInLabels() {
        String body = new ServingMetrics("a\"b\\c").render(LEVELS);

        assertLine(body, "jitllm:num_requests_running{model_name=\"a\\\"b\\\\c\"} 3.0");
    }
}
