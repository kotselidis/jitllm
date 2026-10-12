package org.beehive.jitllm.backend.tornado.nativelib;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.HashMap;
import java.util.Map;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.library.spi.LibraryContext;
import uk.ac.manchester.tornado.runtime.library.spi.LibraryInvocation;
import uk.ac.manchester.tornado.runtime.library.spi.TornadoLibraryProvider;
import uk.ac.manchester.tornado.runtime.library.spi.TornadoNativeStreamSupport;

/**
 * Runs {@link PinnedLt} tasks: cuBLASLt through the foreign-function API, one handle, stream and
 * workspace per (plan, device), one matmul descriptor and algorithm per shape, and the
 * row-dependent layouts cached per row count.
 *
 * <p>The algorithm comes from cuBLASLt's heuristic at {@code jitllm.ragged.ltRefRows} rows (128 by
 * default) and is replayed for every row count; a row count it cannot run is an error, never a
 * silent fallback to another algorithm, which would change the bits.
 */
public final class PinnedLtProvider implements TornadoLibraryProvider {

    private static final long WORKSPACE_BYTES = 32L << 20;
    private static final int R_16F = 2, R_32F = 0, COMPUTE_32F = 68, OP_N = 0, OP_T = 1;
    private static final int DESC_TRANSA = 3, DESC_TRANSB = 4, PREF_MAX_WORKSPACE = 1;
    private static final int HEURISTIC_BYTES = 96, ALGO_BYTES = 64;

    private static final class Native {
        static final Linker L = Linker.nativeLinker();
        static final Arena A = Arena.global();
        static final SymbolLookup LT =
                lib("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so");
        static final SymbolLookup RT = lib("libcudart.so.13", "libcudart.so.12", "libcudart.so");
        static final MethodHandle MALLOC = h(RT, "cudaMalloc", JAVA_INT, ADDRESS, JAVA_LONG);
        static final MethodHandle FREE = h(RT, "cudaFree", JAVA_INT, ADDRESS);
        static final MethodHandle CREATE = h(LT, "cublasLtCreate", JAVA_INT, ADDRESS);
        static final MethodHandle DESTROY = h(LT, "cublasLtDestroy", JAVA_INT, ADDRESS);
        static final MethodHandle DESC =
                h(LT, "cublasLtMatmulDescCreate", JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT);
        static final MethodHandle DESC_SET =
                h(
                        LT,
                        "cublasLtMatmulDescSetAttribute",
                        JAVA_INT,
                        ADDRESS,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_LONG);
        static final MethodHandle LAYOUT =
                h(
                        LT,
                        "cublasLtMatrixLayoutCreate",
                        JAVA_INT,
                        ADDRESS,
                        JAVA_INT,
                        JAVA_LONG,
                        JAVA_LONG,
                        JAVA_LONG);
        static final MethodHandle PREF = h(LT, "cublasLtMatmulPreferenceCreate", JAVA_INT, ADDRESS);
        static final MethodHandle PREF_SET =
                h(
                        LT,
                        "cublasLtMatmulPreferenceSetAttribute",
                        JAVA_INT,
                        ADDRESS,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_LONG);
        static final MethodHandle HEURISTIC =
                h(
                        LT,
                        "cublasLtMatmulAlgoGetHeuristic",
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS);
        static final MethodHandle MATMUL =
                h(
                        LT,
                        "cublasLtMatmul",
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        JAVA_LONG,
                        ADDRESS);

        static SymbolLookup lib(String... names) {
            for (String n : names) {
                try {
                    return SymbolLookup.libraryLookup(n, A);
                } catch (IllegalArgumentException e) {
                    // try the next name
                }
            }
            throw new IllegalStateException(
                    "cuBLASLt / CUDA runtime not found: " + String.join(", ", names));
        }

        static MethodHandle h(SymbolLookup l, String name, MemoryLayout ret, MemoryLayout... args) {
            return L.downcallHandle(l.find(name).orElseThrow(), FunctionDescriptor.of(ret, args));
        }
    }

    /** One shape: descriptor, weight layout and the algorithm, plus layouts per row count. */
    private static final class Shape {
        MemorySegment desc;
        MemorySegment aLayout;
        MemorySegment algo;
        final Map<Integer, MemorySegment[]> rowLayouts = new HashMap<>();
    }

    private static final class Context implements LibraryContext {
        final MemorySegment handle;
        final MemorySegment stream;
        final MemorySegment workspace;
        final MemorySegment pref;
        final MemorySegment alpha;
        final MemorySegment beta;
        final Map<String, Shape> shapes = new HashMap<>();

        Context(
                MemorySegment handle,
                MemorySegment stream,
                MemorySegment workspace,
                MemorySegment pref) {
            this.handle = handle;
            this.stream = stream;
            this.workspace = workspace;
            this.pref = pref;
            this.alpha = Native.A.allocate(JAVA_FLOAT);
            this.alpha.set(JAVA_FLOAT, 0, 1.0f);
            this.beta = Native.A.allocate(JAVA_FLOAT);
            this.beta.set(JAVA_FLOAT, 0, 0.0f);
        }
    }

    @Override
    public String libraryName() {
        return PinnedLt.LIBRARY_NAME;
    }

    @Override
    public boolean canHandle(TornadoXPUDevice device) {
        return device instanceof TornadoNativeStreamSupport;
    }

    @Override
    public LibraryContext createContext(TornadoXPUDevice device, long executionPlanId) {
        try {
            long stream = ((TornadoNativeStreamSupport) device).getNativeStream(executionPlanId);
            MemorySegment handle = created(Native.CREATE);
            MemorySegment ws = Native.A.allocate(ADDRESS);
            check((int) Native.MALLOC.invokeExact(ws, WORKSPACE_BYTES), "cudaMalloc workspace");
            MemorySegment pref = created(Native.PREF);
            MemorySegment max = Native.A.allocate(JAVA_LONG);
            max.set(JAVA_LONG, 0, WORKSPACE_BYTES);
            check(
                    (int) Native.PREF_SET.invokeExact(pref, PREF_MAX_WORKSPACE, max, 8L),
                    "preference");
            return new Context(handle, MemorySegment.ofAddress(stream), ws.get(ADDRESS, 0), pref);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    @Override
    public void dispatch(String functionName, LibraryInvocation inv) {
        if (!"gemmFP16FP32".equals(functionName)) {
            throw new IllegalArgumentException(
                    "jitllm/cublaslt-pinned has no function " + functionName);
        }
        Context ctx = (Context) inv.getContext();
        int n = (int) inv.getArg(0);
        int maxRows = (int) inv.getArg(1);
        int k = (int) inv.getArg(2);
        int real = ((IntArray) inv.getArg(6)).get((int) inv.getArg(7));
        int ldc = (int) inv.getArg(8);
        int rows = Math.min(maxRows, Math.max(16, (real + 15) & ~15));
        try {
            Shape shape = shape(ctx, n, k, ldc, maxRows);
            MemorySegment[] l = shape.rowLayouts.get(rows);
            if (l == null) {
                l = rowLayouts(k, ldc, rows);
                shape.rowLayouts.put(rows, l);
            }
            MemorySegment w = MemorySegment.ofAddress(inv.getDevicePointer(3));
            MemorySegment x = MemorySegment.ofAddress(inv.getDevicePointer(4));
            MemorySegment c = MemorySegment.ofAddress(inv.getDevicePointer(5));
            int status =
                    (int)
                            Native.MATMUL.invokeExact(
                                    ctx.handle,
                                    shape.desc,
                                    ctx.alpha,
                                    w,
                                    shape.aLayout,
                                    x,
                                    l[0],
                                    ctx.beta,
                                    c,
                                    l[1],
                                    c,
                                    l[1],
                                    shape.algo,
                                    ctx.workspace,
                                    WORKSPACE_BYTES,
                                    ctx.stream);
            check(status, "cublasLtMatmul n=" + n + " k=" + k + " rows=" + rows);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static Shape shape(Context ctx, int n, int k, int ldc, int maxRows) throws Throwable {
        String key = n + ":" + k + ":" + ldc;
        Shape s = ctx.shapes.get(key);
        if (s != null) {
            return s;
        }
        s = new Shape();
        s.desc = created(Native.DESC, COMPUTE_32F, R_32F);
        MemorySegment v = Native.A.allocate(JAVA_INT);
        v.set(JAVA_INT, 0, OP_T);
        check((int) Native.DESC_SET.invokeExact(s.desc, DESC_TRANSA, v, 4L), "transa");
        v.set(JAVA_INT, 0, OP_N);
        check((int) Native.DESC_SET.invokeExact(s.desc, DESC_TRANSB, v, 4L), "transb");
        // Column-major: C (n x rows, ld ldc) = W^T (W stored k x n) . X (k x rows).
        s.aLayout = created(Native.LAYOUT, R_16F, (long) k, (long) n, (long) k);
        int ref = Math.min(maxRows, Integer.getInteger("jitllm.ragged.ltRefRows", 128));
        MemorySegment[] refLayouts = rowLayouts(k, ldc, ref);
        MemorySegment result = Native.A.allocate(HEURISTIC_BYTES, 8);
        MemorySegment got = Native.A.allocate(JAVA_INT);
        check(
                (int)
                        Native.HEURISTIC.invokeExact(
                                ctx.handle,
                                s.desc,
                                s.aLayout,
                                refLayouts[0],
                                refLayouts[1],
                                refLayouts[1],
                                ctx.pref,
                                1,
                                result,
                                got),
                "heuristic");
        if (got.get(JAVA_INT, 0) == 0) {
            throw new IllegalStateException("cuBLASLt has no algorithm for n=" + n + " k=" + k);
        }
        s.algo = result.asSlice(0, ALGO_BYTES);
        s.rowLayouts.put(ref, refLayouts);
        ctx.shapes.put(key, s);
        return s;
    }

    private static MemorySegment[] rowLayouts(int k, int ldc, int rows) throws Throwable {
        return new MemorySegment[] {
            created(Native.LAYOUT, R_16F, (long) k, (long) rows, (long) k),
            created(Native.LAYOUT, R_32F, (long) ldc, (long) rows, (long) ldc)
        };
    }

    private static MemorySegment created(MethodHandle create, Object... args) throws Throwable {
        MemorySegment out = Native.A.allocate(ADDRESS);
        Object[] all = new Object[args.length + 1];
        all[0] = out;
        System.arraycopy(args, 0, all, 1, args.length);
        check((int) create.invokeWithArguments(all), "create");
        return out.get(ADDRESS, 0);
    }

    private static void check(int status, String what) {
        if (status != 0) {
            throw new IllegalStateException(what + " failed with status " + status);
        }
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException e) {
            return e;
        }
        return new IllegalStateException(t);
    }

    @Override
    public void destroyContext(LibraryContext context) {
        Context ctx = (Context) context;
        try {
            int ignored = (int) Native.FREE.invokeExact(ctx.workspace);
            ignored = (int) Native.DESTROY.invokeExact(ctx.handle);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
