package org.beehive.jitllm.backend.tornado.nativelib;

import uk.ac.manchester.tornado.api.common.Access;
import uk.ac.manchester.tornado.api.common.LibraryTaskDescriptor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * A cuBLASLt GEMM whose row count is read at each launch, on an algorithm fixed per shape.
 *
 * <p>cuBLAS picks its algorithm by shape, and a different algorithm sums in a different order, so a
 * GEMM over the step's rows would give a row different bits as the batch around it changed. Here
 * the algorithm is chosen once, for a reference row count, and replayed for every row count:
 * measured on an A10, a row's bits are then the same at 16 rows as at 128 for every Llama-3.2-1B
 * projection shape, while the launch covers only the rows the step uses. That lets one server run
 * small decode steps and large prompt steps on the same GEMM and stay batch invariant.
 */
public final class PinnedLt {

    public static final String LIBRARY_NAME = "jitllm/cublaslt-pinned";

    private PinnedLt() {}

    /**
     * {@code C[r * ldc + c] = X[r] . W[c]} for the first {@code ceil16(rowsHolder[rowsIndex])} rows
     * (at most {@code maxRows}): FP16 operands, FP32 accumulation and output, row-major on both
     * sides.
     *
     * @param n output columns (rows of {@code W})
     * @param maxRows rows {@code X} and {@code C} hold
     * @param k inputs per row
     * @param rowsHolder the host-side value read at each launch
     */
    public static LibraryTaskDescriptor gemmFP16FP32(
            int n,
            int maxRows,
            int k,
            HalfFloatArray W,
            HalfFloatArray X,
            FloatArray C,
            IntArray rowsHolder,
            int rowsIndex,
            int ldc) {
        Access[] access = {
            Access.READ_ONLY,
            Access.READ_ONLY,
            Access.READ_ONLY,
            Access.READ_ONLY,
            Access.READ_ONLY,
            Access.WRITE_ONLY,
            Access.READ_ONLY,
            Access.READ_ONLY,
            Access.READ_ONLY
        };
        return new LibraryTaskDescriptor()
                .withLibrary(LIBRARY_NAME)
                .withFunction("gemmFP16FP32")
                .withParameters(new Object[] {n, maxRows, k, W, X, C, rowsHolder, rowsIndex, ldc})
                .withAccess(access);
    }
}
