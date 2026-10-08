package org.beehive.jitllm.backend.tornado;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.backend.tornado.plan.TornadoPlanRegistry;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

/**
 * Where a family's batched prefill ({@code --batch-prefill-size N}) cannot be built, whatever the
 * key/value cache precision.
 *
 * <p>Refused by name before any plan is built, rather than left to fail inside TornadoVM's compiler
 * or its sketcher halfway through the first generation. Each family's plan provider says where its
 * own batched prefill has holes ({@code TornadoPlanProvider.batchPrefillUnsupported}); this class
 * names no family.
 *
 * <p>Single-token decode and sequential prefill/decode run on every backend.
 */
public final class BatchPrefillSupport {

    private BatchPrefillSupport() {}

    /** Why this combination's batched prefill cannot be built, or empty when it can. */
    public static Optional<String> unsupported(Combination c) {
        if (c.mode() != ExecutionMode.BATCH_PREFILL_DECODE || BackendId.CPU.equals(c.backend())) {
            return Optional.empty();
        }
        return TornadoPlanRegistry.provider(c.architecture())
                .flatMap(provider -> provider.batchPrefillUnsupported(c));
    }

    /** Why {@code model}'s batched prefill cannot be built on the current device, if it cannot. */
    public static Optional<String> unsupportedOnCurrentDevice(Model model) {
        return unsupported(
                new Combination(
                        model.architectureId().toString(),
                        model.weights().dataType(),
                        ExecutionMode.BATCH_PREFILL_DECODE,
                        org.beehive.jitllm.backend.tornado.device.TornadoDevices.current()
                                .backend(),
                        false,
                        TensorCoreSupport.isTensorCoreCapableBackend(),
                        Fp16KeyValueSupport.nvidiaDevice()));
    }

    /**
     * Refuses a batched prefill this configuration cannot build.
     *
     * @throws UnsupportedOperationException naming the combination, the reason, and the way out
     */
    public static void require(Model model, ExecutionPolicy policy, boolean gpu) {
        if (!gpu) {
            return;
        }
        Combination combination = Fp16KeyValueSupport.resolve(model, policy, gpu);
        unsupported(combination)
                .or(() -> needsNativeProjections(combination, policy))
                .ifPresent(
                        reason -> {
                            throw new UnsupportedOperationException(refusal(combination, reason));
                        });
    }

    /** What the family's batched prefill needs from {@code policy} that it does not ask for. */
    static Optional<String> needsNativeProjections(Combination c, ExecutionPolicy policy) {
        if (c.mode() != ExecutionMode.BATCH_PREFILL_DECODE || BackendId.CPU.equals(c.backend())) {
            return Optional.empty();
        }
        return TornadoPlanRegistry.provider(c.architecture())
                .flatMap(provider -> provider.batchPrefillNeeds(c, policy));
    }

    /** The refusal: the combination, why, and that it holds for either cache. */
    static String refusal(Combination combination, String reason) {
        return DiagnosticCode.COMBINATION_UNSUPPORTED.message(
                "batched prefill is not supported for "
                        + combination
                        + ": "
                        + reason
                        + ". This holds for either key/value cache precision. Drop"
                        + " --batch-prefill-size (the single-token plan runs this model here),"
                        + " or leave prefill batching off in the execution policy");
    }
}
