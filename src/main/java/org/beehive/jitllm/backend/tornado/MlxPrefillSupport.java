package org.beehive.jitllm.backend.tornado;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.mlx.provider.MlxLibraryProvider;

/**
 * Apple MLX for the batched prefill projections on Metal: the Metal counterpart of {@link
 * NativePrefillSupport}, which routes them to cuBLAS on CUDA. Opted into the same way, with {@code
 * -Djitllm.nativeLibraries=true} ({@code --with-native-libraries}).
 *
 * <p>MLX multiplies by weights in its affine group format, {@code w = scale * q + bias} with an
 * unsigned {@code q}. A GGUF Q8_0 block, {@code w = d * q} with a signed byte {@code q}, is that
 * format exactly: {@code q + 128} with {@code scale = d} and {@code bias = -128 * d}, at group size
 * 32 and 8 bits. {@link Q8_0AsAffine} holds that repacking; no weight changes value.
 */
public final class MlxPrefillSupport {

    /** Group size and width of the affine format the Q8_0 weights are repacked into. */
    public static final int GROUP_SIZE = 32;

    public static final int BITS = 8;

    private static final int Q8_0_BLOCK_BYTES = 34; // 2-byte half scale + 32 int8 quants

    private MlxPrefillSupport() {}

    /**
     * Whether the batched prefill projections go to MLX: opted in, on Metal, with mlx-c present.
     */
    public static boolean mlxProjections(ExecutionPolicy policy) {
        return policy.nativeLibraries() && Probe.MLX_ON_METAL;
    }

    public static boolean mlxProjections() {
        return mlxProjections(ExecutionPolicy.fromSystemProperties());
    }

    /**
     * A Q8_0 matrix as MLX affine 8-bit weights: {@code wq} packs four quants per word, low byte
     * first, {@code rows * cols / 4} words; {@code scales} and {@code biases} hold one value per
     * group of 32, {@code rows * cols / 32} each.
     */
    public record Q8_0AsAffine(
            IntArray wq, FloatArray scales, FloatArray biases, int rows, int cols) {

        /** Repacks a row-major {@code rows x cols} Q8_0 matrix ({@code cols % 32 == 0}). */
        public static Q8_0AsAffine of(ByteArray q8, int rows, int cols) {
            return stack(rows, cols, q8);
        }

        /**
         * Repacks Q8_0 matrices of the same width stacked by rows, {@code rows} each: one GEMM then
         * produces their outputs side by side in each result row.
         */
        public static Q8_0AsAffine stack(int rows, int cols, ByteArray... parts) {
            if (cols % GROUP_SIZE != 0) {
                throw new IllegalArgumentException(
                        "Q8_0 width " + cols + " is not a multiple of " + GROUP_SIZE);
            }
            int blocksPerRow = cols / GROUP_SIZE;
            int totalRows = rows * parts.length;
            IntArray wq = new IntArray(totalRows * cols / 4);
            FloatArray scales = new FloatArray(totalRows * blocksPerRow);
            FloatArray biases = new FloatArray(totalRows * blocksPerRow);
            int group = 0;
            int word = 0;
            for (ByteArray part : parts) {
                MemorySegment src = part.getSegment();
                long blocks = (long) rows * blocksPerRow;
                for (long b = 0; b < blocks; b++) {
                    long offset = b * Q8_0_BLOCK_BYTES;
                    float d =
                            Float.float16ToFloat(src.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset));
                    scales.set(group, d);
                    biases.set(group, -128.0f * d);
                    group++;
                    long q = offset + 2;
                    for (int i = 0; i < GROUP_SIZE; i += 4) {
                        int b0 = (src.get(ValueLayout.JAVA_BYTE, q + i) + 128) & 0xFF;
                        int b1 = (src.get(ValueLayout.JAVA_BYTE, q + i + 1) + 128) & 0xFF;
                        int b2 = (src.get(ValueLayout.JAVA_BYTE, q + i + 2) + 128) & 0xFF;
                        int b3 = (src.get(ValueLayout.JAVA_BYTE, q + i + 3) + 128) & 0xFF;
                        wq.set(word++, b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
                    }
                }
            }
            return new Q8_0AsAffine(wq, scales, biases, totalRows, cols);
        }
    }

    private static final class Probe {

        private static final boolean MLX_ON_METAL = probe();

        private static boolean probe() {
            try {
                TornadoVMBackendType type =
                        TornadoRuntimeProvider.getTornadoRuntime().getBackend(0).getBackendType();
                return type == TornadoVMBackendType.METAL && MlxLibraryProvider.isAvailable();
            } catch (RuntimeException | LinkageError e) {
                return false;
            }
        }
    }
}
