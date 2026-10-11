package org.beehive.jitllm.server;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import org.beehive.jitllm.backend.tornado.batch.TornadoBatchExecutor;
import org.beehive.jitllm.backend.tornado.batch.TornadoRaggedBatchExecutor;
import org.beehive.jitllm.backend.tornado.batch.TornadoServingExecutor;
import org.beehive.jitllm.engine.LLMEngine;
import org.beehive.jitllm.engine.RequestHandle;
import org.beehive.jitllm.engine.RequestState;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.beehive.jitllm.runtime.kv.KvLease;
import org.beehive.jitllm.runtime.kv.KvStorage;
import org.beehive.jitllm.runtime.kv.KvStorageFactories;
import org.beehive.jitllm.runtime.kv.KvStorageRequest;

/**
 * The server's inference path, on the engine: several conversations decode in one batch instead of
 * queueing behind one lock.
 *
 * <h2>The prepared-token boundary</h2>
 *
 * <p>This class is where an OpenAI request stops being a conversation and becomes token ids. The
 * message history goes through the model's own {@link ChatFormat} — the same template the
 * single-token path uses — and what reaches the engine is {@code int[]}.
 *
 * <p><b>The history is never flattened into a string.</b> Rendering the turns and re-encoding the
 * result produces a different token sequence, and neither the engine nor the test suite would
 * notice; the responses would simply differ from what the server used to say. Encoding per turn is
 * what makes the parity gate meaningful.
 */
public final class EngineInferenceService implements AutoCloseable {

    private final Model model;
    private final LLMEngine engine;
    private final KvCacheManager manager;
    private final TornadoServingExecutor executor;
    private final KvLease planLease;
    private final Thread driver;
    private final ServingMetrics metrics;

    private volatile boolean running = true;

    /**
     * @param model the loaded model, borrowed by the engine
     * @param batchSize B. Must be at least 2: the batched buffers on {@code State} exist only when
     *     a batch was asked for, so B = 1 has no plan to run
     * @param maxQueuedRequests the server's own bound. The engine library has no default and will
     *     not guess one
     * @param contextLength tokens a single conversation may occupy
     * @param prefixCacheEntries how many prompt prefixes to remember, or 0 for none. Served traffic
     *     repeats its openings — a system prompt, a tool schema — and every repetition is prefill
     *     the device has already done. The trade is pool capacity: a remembered prefix holds its
     *     blocks whether or not anyone is using them
     * @param servedModel the name {@code /metrics} labels every series with
     * @param maxBatchedTokens tokens one step carries, prompt chunks and decode tokens together, or
     *     0 to feed every request one token per step on the original batched executor
     */
    public EngineInferenceService(
            Model model,
            int batchSize,
            int maxQueuedRequests,
            int contextLength,
            int prefixCacheEntries,
            String servedModel,
            int maxBatchedTokens) {
        if (batchSize < 2) {
            throw new IllegalArgumentException(
                    "the batched executor needs B >= 2, got "
                            + batchSize
                            + ". The batched buffers on State are allocated only when a batch was asked"
                            + " for, so at B = 1 there is no batched plan to run");
        }
        this.model = model;
        this.metrics = new ServingMetrics(servedModel);

        int blockTokens = State.KV_BLOCK_SIZE;
        int blocksPerSlot = (contextLength + blockTokens - 1) / blockTokens;
        this.manager =
                KvCacheManager.sizedFor(
                        batchSize, contextLength, blockTokens, bytesPerBlock(model, blockTokens));
        KvStorage store =
                KvStorageFactories.single()
                        .create(
                                new KvStorageRequest(
                                        batchSize * blocksPerSlot,
                                        blocksPerSlot,
                                        batchSize,
                                        blockTokens,
                                        model.configuration().numberOfLayers(),
                                        model.kvCacheDim(),
                                        fp16KeyValue()));
        this.manager.attach(store);
        if (prefixCacheEntries > 0) {
            this.manager.enablePrefixCache(prefixCacheEntries);
        }

        // One state builds the plan; the per-slot addressing comes from the block table, so the
        // lease that state held goes straight back to the pool once the buffers are bound.
        this.planLease = manager.acquire(blockTokens);
        int rows = maxBatchedTokens > 0 ? maxBatchedTokens : batchSize;
        State state = State.withPrefillBatchSize(rows, () -> model.createNewState(planLease));
        this.executor =
                maxBatchedTokens > 0
                        ? new TornadoRaggedBatchExecutor(
                                model, state, store, batchSize, maxBatchedTokens, blocksPerSlot)
                        : new TornadoBatchExecutor(model, state, store, batchSize, blocksPerSlot);
        planLease.close();

        this.engine =
                new LLMEngine(
                        model,
                        manager,
                        executor,
                        batchSize,
                        maxQueuedRequests,
                        alsoTo(
                                metrics,
                                org.beehive.jitllm.auxiliary.metrics.RunMetricsSink
                                        .installedOrDisabled()));

        this.driver = new Thread(this::drive, "engine-step");
        this.driver.setDaemon(true);
        this.driver.start();
    }

    public org.beehive.jitllm.runtime.backend.ExecutionInfo executionInfo() {
        return executor.executionInfo();
    }

    public Model model() {
        return model;
    }

    /** The {@code /metrics} body: what has been recorded, plus the engine's levels right now. */
    public String metricsText() {
        var capacity = manager.capacity();
        return metrics.render(
                new ServingMetrics.Levels(
                        engine.runningCount(),
                        engine.queueDepth(),
                        capacity.usedBlocks(),
                        capacity.totalBlocks()));
    }

    /** {@code primary}, and {@code secondary} as well when it is enabled. */
    private static org.beehive.jitllm.runtime.metrics.MetricsSink alsoTo(
            org.beehive.jitllm.runtime.metrics.MetricsSink primary,
            org.beehive.jitllm.runtime.metrics.MetricsSink secondary) {
        if (!secondary.isEnabled()) {
            return primary;
        }
        return (key, value) -> {
            primary.record(key, value);
            secondary.record(key, value);
        };
    }

    /**
     * Generates a completion for one request, blocking until it terminates.
     *
     * <p>Safe to call from many HTTP threads at once — which is the point. They no longer serialize
     * behind one lock; they occupy different slots of the same batch.
     */
    public InferenceService.Result generate(
            InferenceService.Request request, Consumer<String> onToken) {
        long submittedNanos = System.nanoTime();
        int[] promptTokens = prepareTokens(request);
        int maxNewTokens = request.maxTokens() > 0 ? request.maxTokens() : 256;

        StringBuilder text = new StringBuilder();
        // Written by the engine thread, read here after await(), which orders the two.
        long[] firstTokenNanos = {-1L};
        RequestHandle handle =
                engine.addRequest(
                        promptTokens,
                        maxNewTokens,
                        request.ignoreEos(),
                        token -> {
                            if (firstTokenNanos[0] < 0) {
                                firstTokenNanos[0] = System.nanoTime() - submittedNanos;
                            }
                            if (!model.tokenizer().shouldDisplayToken(token)) {
                                return;
                            }
                            String piece = model.tokenizer().decode(List.of(token));
                            text.append(piece);
                            if (onToken != null) {
                                onToken.accept(piece);
                            }
                        });

        if (handle.state() == RequestState.REJECTED) {
            throw new IllegalStateException("request refused: " + handle.rejectionReason());
        }
        try {
            handle.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            engine.cancel(handle);
        }
        if (handle.failure() != null) {
            throw new IllegalStateException("request failed", handle.failure());
        }

        List<Integer> produced = handle.tokens();
        boolean stopped =
                !request.ignoreEos()
                        && !produced.isEmpty()
                        && model.chatFormat()
                                .getStopTokens()
                                .contains(produced.get(produced.size() - 1));
        int completion = produced.size() - (stopped ? 1 : 0);
        metrics.requestFinished(
                promptTokens.length,
                Math.max(0, completion),
                firstTokenNanos[0],
                System.nanoTime() - submittedNanos,
                stopped);
        return new InferenceService.Result(
                text.toString(), promptTokens.length, Math.max(0, completion), stopped);
    }

    /**
     * The conversation → token ids conversion, and the only place it happens.
     *
     * <p>Per turn, through the model's own template. Identical to what {@code InferenceService}
     * does today, which is what makes the two paths comparable at all.
     */
    /**
     * The facade's chat message as the family's chat template wants it.
     *
     * <p>This tier tokenizes for itself because continuous batching is an engine-tier feature the
     * public facade does not expose: there is no {@code GenerationSession} here to render the
     * template. Only text turns reach this server, so only text is translated.
     */
    private static ChatFormat.Message asFormatMessage(org.beehive.jitllm.api.ChatMessage message) {
        StringBuilder text = new StringBuilder();
        for (org.beehive.jitllm.api.ChatContent piece : message.content()) {
            if (piece instanceof org.beehive.jitllm.api.ChatContent.Text t) {
                text.append(t.text());
            }
        }
        return new ChatFormat.Message(
                new ChatFormat.Role(message.role().name().toLowerCase(java.util.Locale.ROOT)),
                text.toString());
    }

    private int[] prepareTokens(InferenceService.Request request) {
        ChatFormat chatFormat = model.chatFormat();
        List<Integer> tokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            tokens.addAll(chatFormat.beginOfTextTokens());
        }
        for (org.beehive.jitllm.api.ChatMessage message : request.messages()) {
            tokens.addAll(chatFormat.encodeMessage(asFormatMessage(message)));
        }
        tokens.addAll(
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        if (model.shouldIncludeReasoning()) {
            tokens.addAll(
                    model.tokenizer()
                            .encode("<think>\n", model.tokenizer().getSpecialTokens().keySet()));
        }
        int[] prepared = new int[tokens.size()];
        for (int i = 0; i < prepared.length; i++) {
            prepared[i] = tokens.get(i);
        }
        return prepared;
    }

    /** The single caller of {@code step()}. */
    private void drive() {
        while (running) {
            int advanced;
            try {
                advanced = engine.step();
            } catch (RuntimeException | Error failure) {
                // The engine already failed the requests it could; a dead driver would leave
                // every later request waiting, so report it and keep stepping.
                System.err.println("[engine] step failed: " + failure);
                advanced = 0;
            }
            if (advanced == 0) {
                // Nothing to do. Park briefly rather than spinning a core against an empty queue;
                // a submission will have work ready by the next pass.
                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    /** The pool's representation: FP16 by default, FP32 when asked for. */
    private static boolean fp16KeyValue() {
        return org.beehive.jitllm.runtime.policy.StorageOptions.fromSystemProperties()
                .usesFp16KeyValueCache();
    }

    private static long bytesPerBlock(Model model, int blockTokens) {
        long bytesPerValue = fp16KeyValue() ? 2L : 4L;
        return 2L
                * model.kvCacheDim()
                * blockTokens
                * model.configuration().numberOfLayers()
                * bytesPerValue;
    }

    @Override
    public void close() {
        running = false;
        driver.interrupt();
        try {
            driver.join(2000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        engine.close();
        executor.close();
        manager.close();
    }
}
