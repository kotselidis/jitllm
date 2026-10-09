package org.beehive.jitllm.backend.tornado;

import java.util.Optional;
import org.beehive.jitllm.backend.tornado.Fp16KeyValueSupport.Combination;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

// @formatter:off
/**
 * Where a family's batched prefill ({@code --batch-prefill-size N}) cannot be built, whatever the
 * key/value cache precision.
 *
 * <p>Refused by name before any plan is built, rather than left to fail inside TornadoVM's compiler
 * or its sketcher halfway through the first generation. Two families have such holes today, both
 * off the tensor-core path:
 *
 * <ul>
 *   <li><b>gemma4</b>: the batched projections and attention are tensor-core kernels, with no
 *       scalar twin. Without tensor cores the plan fails at "MMA instructions only supported for
 *       the CUDA backend".
 *   <li><b>qwen35</b> on OpenCL: the scalar batched projections ({@code matrixVectorTiledBatchQ4_0}
 *       and its Q4_1/Q5_K siblings) do not compile on TornadoVM's OpenCL backend. A guard the
 *       compiler can prove true after the kernel's early return becomes a {@code LogicConstantNode}
 *       once a constant-trip loop is unrolled, and the OpenCL LIR builder has no rule for it
 *       ({@code OCLNodeLIRBuilder.emitLogicNode}). That is a compiler defect, reproduced by a
 *       ten-line kernel, and it is refused here until TornadoVM fixes it rather than worked around
 *       in kernels CUDA shares.
 * </ul>
 *
 * <p>Single-token decode and sequential prefill/decode run on both backends.
 */
// @formatter:on
public final class BatchPrefillSupport {

    private BatchPrefillSupport() {}

    /** Why this combination's batched prefill cannot be built, or empty when it can. */
    public static Optional<String> unsupported(Combination c) {
        if (c.mode() != ExecutionMode.BATCH_PREFILL_DECODE || BackendId.CPU.equals(c.backend())) {
            return Optional.empty();
        }
        return switch (c.architecture()) {
            case "gemma4" ->
                    c.tensorCores()
                            ? Optional.empty()
                            : Optional.of(
                                    "the gemma4 batched prefill is written for tensor cores"
                                            + " only, which this device does not have");
            case "qwen35" ->
                    BackendId.OPENCL.equals(c.backend())
                            ? Optional.of(
                                    "the qwen35 batched prefill projections do not compile on the"
                                            + " OpenCL backend (TornadoVM OpenCL code generation"
                                            + " fails with 'logic node (LogicConstantNode)')")
                            : Optional.empty();
            default -> Optional.empty();
        };
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
                .ifPresent(
                        reason -> {
                            throw new UnsupportedOperationException(refusal(combination, reason));
                        });
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

    /** The batched-prefill chunk Metal runs by default; see {@link #defaultFor}. */
    public static final int METAL_DEFAULT_PREFILL_BATCH = 256;

    // @formatter:off
    /**
     * The policy a model runs when its caller did not choose one: batched prefill on Metal for the
     * families whose batched kernels are tuned there, the given policy otherwise.
     *
     * <p>Qwen3 in F16 and Q8_0 prefills an order of magnitude faster in batches of {@value
     * #METAL_DEFAULT_PREFILL_BATCH} than one token at a time on an Apple GPU (Qwen3-0.6B on an M4
     * Pro, pp512: F16 151 to 1949 tok/s, Q8_0 194 to 1928), at a cost of 2-4% in decode (137 to 132
     * tok/s after a 943-token prompt); greedy text is identical. Other families keep single-token
     * prefill on Metal: their batched kernels have not been tuned there and run slower than it.
     * Only a single-token policy is changed, so an explicit choice is never overridden.
     */
    // @formatter:on
    public static ExecutionPolicy defaultFor(Model model, ExecutionPolicy policy) {
        if (policy.phaseStrategy() != ExecutionPolicy.PhaseStrategy.SINGLE_TOKEN
                || !org.beehive.jitllm.backend.tornado.device.TornadoDevices.current()
                        .backend()
                        .equals(BackendId.METAL)
                || !"qwen3".equals(model.architectureId().toString())) {
            return policy;
        }
        var type = model.weights().dataType();
        if (type != org.beehive.jitllm.runtime.tensor.DataType.F16
                && type != org.beehive.jitllm.runtime.tensor.DataType.Q8_0) {
            return policy;
        }
        if (unsupportedOnCurrentDevice(model).isPresent()) {
            return policy;
        }
        return ExecutionPolicy.from(policy)
                .phaseStrategy(ExecutionPolicy.PhaseStrategy.PREFILL_DECODE)
                .prefillBatchSize(METAL_DEFAULT_PREFILL_BATCH)
                .build();
    }
}
