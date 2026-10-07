package org.beehive.jitllm.backend.tornado.tensor;

// @formatter:off
/**
 * The opt-in switches for packed projections ({@link PackedTiles}): {@code -Djitllm.q8.packed=true}
 * packs every Q8_0 projection whose shape fits, and {@code -Djitllm.q4.packed=true} every retained
 * Q4_0 one. A packed projection's tensor says so ({@link TornadoTensor#isPacked()}), and every
 * kernel choice consults it: a kernel that reads Q8_0 or Q4_0 blocks must never be handed packed
 * tiles, so the paths without a packed kernel refuse one by name ({@link #noPackedKernel}).
 */
// @formatter:on
public final class PackedWeights {

    /** Whether Q8_0 projections are packed at load. */
    public static final boolean Q8_ENABLED = Boolean.getBoolean("jitllm.q8.packed");

    /** Whether retained Q4_0 projections are packed at load. */
    public static final boolean Q4_ENABLED = Boolean.getBoolean("jitllm.q4.packed");

    private PackedWeights() {}

    /** The refusal for a path that has no kernel over packed tiles. */
    public static UnsupportedOperationException noPackedKernel(String where) {
        return new UnsupportedOperationException(
                where
                        + " reads a projection packed for -Djitllm.q8.packed / -Djitllm.q4.packed, and this"
                        + " path has no kernel for the packed layout. Run without those options.");
    }
}
