package org.beehive.jitllm.backend.tornado.tensor;

// @formatter:off
/**
 * Whether a load packs its projections ({@link PackedTiles}): every Q8_0 projection whose shape
 * fits, and every retained Q4_0 one. On by default wherever the model supports it, so a model runs
 * packed exactly when every kernel its plan can choose reads the packed layout:
 *
 * <ul>
 *   <li>the family has packed kernels on all its paths, which its loader states (Gemma 4's cuBLAS
 *       projections under native libraries have none, nor a family whose packed-integer kernels are
 *       switched off);
 *   <li>the device is a CUDA one with packed-integer dot products and int8 tensor cores;
 *   <li>a batched prefill runs at a width the int8 GEMM tiles (1, or a multiple of 128).
 * </ul>
 *
 * <p>{@code -Djitllm.q8.packed} and {@code -Djitllm.q4.packed} decide instead when set, either way.
 * A packed projection's tensor says so ({@link TornadoTensor#isPacked()}), and every kernel choice
 * consults it: a kernel that reads Q8_0 or Q4_0 blocks must never be handed packed tiles, so the
 * paths without a packed kernel refuse one by name ({@link #noPackedKernel}).
 */
// @formatter:on
public final class PackedWeights {

    /** Packs Q8_0 projections ({@code true}) or never ({@code false}); unset: the default above. */
    public static final String Q8_PROPERTY = "jitllm.q8.packed";

    /**
     * Packs retained Q4_0 projections ({@code true}) or never ({@code false}); unset: the default.
     */
    public static final String Q4_PROPERTY = "jitllm.q4.packed";

    /** The int8 GEMM's row tile, which a batched prefill's width must fill. */
    private static final int GEMM_ROWS = 128;

    private PackedWeights() {}

    /**
     * Whether this load packs its Q8_0 projections.
     *
     * @param familySupports whether every path of the family reads packed Q8_0 tiles
     */
    public static boolean packQ8(boolean familySupports) {
        return decide(Q8_PROPERTY, familySupports);
    }

    /**
     * Whether this load packs its retained Q4_0 projections.
     *
     * @param familySupports whether every path of the family reads packed Q4_0 tiles
     */
    public static boolean packQ4(boolean familySupports) {
        return decide(Q4_PROPERTY, familySupports);
    }

    /**
     * Whether a family's packed-integer kernels are on: its {@code property} is not {@code false}.
     */
    public static boolean packedIntegerDotOn(String property) {
        return !"false".equalsIgnoreCase(System.getProperty(property, "true"));
    }

    private static boolean decide(String property, boolean familySupports) {
        String asked = System.getProperty(property);
        if (asked != null && !asked.isBlank()) {
            return Boolean.parseBoolean(asked);
        }
        return familySupports && deviceRunsPackedKernels() && prefillWidthTiles();
    }

    /** CUDA, with the packed-integer dot products and int8 tensor cores the packed kernels use. */
    private static boolean deviceRunsPackedKernels() {
        var device = org.beehive.jitllm.backend.tornado.device.TornadoDevices.current();
        var capabilities = device.capabilities();
        return org.beehive.jitllm.runtime.backend.BackendId.CUDA.equals(device.backend())
                && capabilities.supports(
                        org.beehive.jitllm.runtime.backend.DeviceCapability.PACKED_INTEGER_DOT)
                && capabilities.supports(
                        org.beehive.jitllm.runtime.backend.DeviceCapability.INT8_TENSOR_CORE_MMA);
    }

    /**
     * A batched prefill's width fills the int8 GEMM's row tiles, the only packed prefill kernel.
     */
    private static boolean prefillWidthTiles() {
        int width =
                org.beehive.jitllm.runtime.policy.ExecutionPolicy.fromSystemProperties()
                        .prefillBatchSize();
        return width == 1 || width % GEMM_ROWS == 0;
    }

    /** The refusal for a path that has no kernel over packed tiles. */
    public static UnsupportedOperationException noPackedKernel(String where) {
        return new UnsupportedOperationException(
                where
                        + " reads a packed projection, and this path has no kernel for the packed layout."
                        + " Run with -Djitllm.q8.packed=false -Djitllm.q4.packed=false.");
    }
}
