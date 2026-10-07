package org.beehive.jitllm.backend.tornado.kernels;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

// @formatter:off
/**
 * Q8_0 projection weights held in the tile layout of {@link Qwen35Int8Kernels#packQ8_0Tiles}
 * rather than as Q8_0 blocks, so that the prefill GEMM ({@link
 * Qwen35Int8Kernels#gemmInt8Q8_0Packed}) copies them with {@code cp.async} and the decode
 * matrix-vector products ({@link TransformerComputeKernelsQ8_0Packed}) read them coalesced. The
 * packed bytes are the same size as the blocks, so the model is held once.
 *
 * <p>Off by default; {@code -Djitllm.q8.packed=true} packs every qwen35 projection whose rows
 * divide by 128 and whose inputs divide by 64. A packed weight is recorded here by identity, and
 * every kernel choice consults this record: a kernel that reads Q8_0 blocks must never be handed
 * packed bytes, so the paths without a packed kernel refuse one by name.
 */
// @formatter:on
public final class PackedQ8_0 {

    /** Whether qwen35 Q8_0 projections are packed at load. */
    public static final boolean ENABLED = Boolean.getBoolean("jitllm.q8.packed");


    private static final Set<ByteArray> PACKED = Collections.synchronizedSet(Collections.newSetFromMap(new IdentityHashMap<>()));

    private PackedQ8_0() {}

    /** Whether a {@code rows x cols} Q8_0 weight can be packed. */
    public static boolean eligible(long rows, long cols) {
        return rows % Qwen35Int8Kernels.I8_BN == 0 && cols % Qwen35Int8Kernels.I8_BK == 0;
    }

    /** The packed copy of a {@code rows x cols} Q8_0 weight, recorded as packed. */
    public static ByteArray pack(ByteArray q8_0, int rows, int cols) {
        ByteArray packed = Qwen35Int8Kernels.packQ8_0Tiles(q8_0, rows, cols);
        PACKED.add(packed);
        return packed;
    }

    /** Whether {@code w} holds packed tiles rather than Q8_0 blocks. */
    public static boolean isPacked(ByteArray w) {
        return PACKED.contains(w);
    }

    /** The refusal for a path that has no kernel over packed tiles. */
    public static UnsupportedOperationException noPackedKernel(String where) {
        return new UnsupportedOperationException(
                where
                        + " reads a Q8_0 weight packed for -Djitllm.q8.packed=true, and this path has no"
                        + " kernel for the packed layout. Run without -Djitllm.q8.packed.");
    }
}
