package org.beehive.jitllm.backend.tornado.kernels;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

// @formatter:off
/**
 * An on-disk cache of a model's packed Q8_0 weights ({@link PackedQ8_0}), so that packing, which
 * reads and rewrites every projection, happens once per model rather than on every load.
 *
 * <p>The cache is one file per model. Each packed tensor is stored 4 KB-aligned with the native
 * array header's width free in front of it, and is memory-mapped from there the way the GGUF
 * tensors are ({@link FileChannel.MapMode#PRIVATE}, header zeroed in the private copy): nothing is
 * copied onto the Java heap or into direct memory, and pages are read only when the weights are
 * uploaded. An index of the tensors (name, shape, offset) and a fixed footer close the file.
 *
 * <p>The file is named by a SHA-256 of what identifies the model's bytes without its path: the
 * file size, the whole GGUF header (tensor names, shapes and offsets), the first and last
 * megabyte of tensor data, and the packed format's version. A miss packs each tensor into a
 * temporary file as it is loaded, maps it from there, and moves the file into place once every
 * tensor is written, so a load that fails part-way leaves no cache behind.
 *
 * <p>{@code -Djitllm.q8.packed.cache.dir} sets the directory (default {@code
 * ~/.cache/jitllm/q8packed}); {@code -Djitllm.q8.packed.cache=false} packs in memory instead.
 */
// @formatter:on
public final class PackedQ8_0Cache {

    private static final long MAGIC = 0x50385130_4D4C5449L; // "ITLM0Q8P", little-endian

    /** Bumped whenever the packed layout of {@link Qwen35Int8Kernels#packQ8_0Tiles} changes. */
    private static final int VERSION = 1;

    private static final long ALIGNMENT = 4096;

    private static final int FOOTER_BYTES = Long.BYTES + Long.BYTES + Integer.BYTES;

    private static final boolean ENABLED = !"false".equalsIgnoreCase(System.getProperty("jitllm.q8.packed.cache", "true"));

    private static final long HEADER = TornadoNativeArray.ARRAY_HEADER;

    private record Entry(String name, int rows, int cols, long offset, long length) {}

    private enum Mode {
        READ,
        WRITE,
        MEMORY
    }

    private final Mode mode;
    private final Path file;
    private final Path temporary;
    private final FileChannel channel;
    private final Map<String, Entry> index;
    private final List<Entry> written = new ArrayList<>();
    private final Arena arena = Arena.ofAuto();
    private long next;

    private PackedQ8_0Cache(Mode mode, Path file, Path temporary, FileChannel channel, Map<String, Entry> index) {
        this.mode = mode;
        this.file = file;
        this.temporary = temporary;
        this.channel = channel;
        this.index = index;
        this.next = HEADER;
    }

    /**
     * The cache for the model read through {@code model}, whose GGUF tensor data starts at
     * {@code tensorDataOffset}: read from disk when present and valid, otherwise written as the
     * tensors are packed. Falls back to packing in memory when the cache cannot be used.
     */
    public static PackedQ8_0Cache open(FileChannel model, long tensorDataOffset) {
        if (!ENABLED) {
            return memory();
        }
        try {
            Path dir = Path.of(System.getProperty("jitllm.q8.packed.cache.dir", System.getProperty("user.home") + "/.cache/jitllm/q8packed"));
            Files.createDirectories(dir);
            Path file = dir.resolve("q8p-" + key(model, tensorDataOffset) + ".bin");
            if (Files.exists(file)) {
                FileChannel channel = FileChannel.open(file, StandardOpenOption.READ, StandardOpenOption.WRITE);
                Map<String, Entry> index = readIndex(channel);
                if (index != null) {
                    System.err.println("[jitllm] packed Q8_0 weights: reading " + file);
                    return new PackedQ8_0Cache(Mode.READ, file, null, channel, index);
                }
                channel.close();
                System.err.println("[jitllm] packed Q8_0 weights: " + file + " is not a complete cache, repacking");
            }
            Path temporary = Files.createTempFile(dir, file.getFileName().toString(), ".tmp");
            FileChannel channel = FileChannel.open(temporary, StandardOpenOption.READ, StandardOpenOption.WRITE);
            System.err.println("[jitllm] packed Q8_0 weights: packing into " + file);
            return new PackedQ8_0Cache(Mode.WRITE, file, temporary, channel, Map.of());
        } catch (IOException | RuntimeException e) {
            System.err.println("[jitllm] packed Q8_0 weights: cache unavailable (" + e.getMessage() + "), packing in memory");
            return memory();
        }
    }

    private static PackedQ8_0Cache memory() {
        return new PackedQ8_0Cache(Mode.MEMORY, null, null, null, Map.of());
    }

    /** The packed form of the {@code rows x cols} Q8_0 weight {@code name}, recorded as packed. */
    public ByteArray tensor(String name, ByteArray q8_0, int rows, int cols) {
        try {
            ByteArray packed =
                    switch (mode) {
                        case READ -> {
                            Entry entry = index.get(name);
                            if (entry == null || entry.rows() != rows || entry.cols() != cols) {
                                throw new IOException("the cache has no " + rows + " x " + cols + " entry for " + name);
                            }
                            yield map(entry);
                        }
                        case WRITE -> {
                            byte[] bytes = Qwen35Int8Kernels.packQ8_0TileBytes(q8_0, rows, cols);
                            long offset = align(next + HEADER) ;
                            ByteBuffer buffer = ByteBuffer.wrap(bytes);
                            long position = offset;
                            while (buffer.hasRemaining()) {
                                position += channel.write(buffer, position);
                            }
                            Entry entry = new Entry(name, rows, cols, offset, bytes.length);
                            written.add(entry);
                            next = offset + bytes.length;
                            yield map(entry);
                        }
                        case MEMORY -> Qwen35Int8Kernels.packQ8_0Tiles(q8_0, rows, cols);
                    };
            return PackedQ8_0.record(packed);
        } catch (IOException e) {
            throw new UncheckedIOException("packed Q8_0 weights: " + (file == null ? "" : file + ": ") + e.getMessage(), e);
        }
    }

    /** Completes a cache being written: the index and footer, then the file moved into place. */
    public void finish() {
        if (mode != Mode.WRITE) {
            return;
        }
        try {
            long indexOffset = align(next);
            ByteBuffer out = ByteBuffer.allocate(indexBytes()).order(ByteOrder.LITTLE_ENDIAN);
            out.putInt(written.size());
            for (Entry entry : written) {
                byte[] name = entry.name().getBytes(StandardCharsets.UTF_8);
                out.putInt(name.length).put(name).putInt(entry.rows()).putInt(entry.cols()).putLong(entry.offset()).putLong(entry.length());
            }
            out.putLong(indexOffset).putLong(MAGIC).putInt(VERSION);
            out.flip();
            long position = indexOffset;
            while (out.hasRemaining()) {
                position += channel.write(out, position);
            }
            channel.force(true);
            channel.close();
            try {
                Files.move(temporary, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING);
            }
            System.err.println("[jitllm] packed Q8_0 weights: cached " + written.size() + " tensors in " + file);
        } catch (IOException e) {
            System.err.println("[jitllm] packed Q8_0 weights: could not complete the cache (" + e.getMessage() + ")");
        }
    }

    private int indexBytes() {
        int bytes = Integer.BYTES + FOOTER_BYTES;
        for (Entry entry : written) {
            bytes += Integer.BYTES + entry.name().getBytes(StandardCharsets.UTF_8).length + 2 * Integer.BYTES + 2 * Long.BYTES;
        }
        return bytes;
    }

    /** The entry's bytes with the header's width in front, in a private mapping of the file. */
    private ByteArray map(Entry entry) throws IOException {
        MemorySegment segment = channel.map(FileChannel.MapMode.PRIVATE, entry.offset() - HEADER, entry.length() + HEADER, arena);
        for (long i = 0; i < HEADER; i++) {
            segment.set(ValueLayout.JAVA_BYTE, i, (byte) 0);
        }
        return ByteArray.fromSegmentShallow(segment);
    }

    private static long align(long position) {
        return (position + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    }

    /** The index of a complete cache file, or {@code null} if the file is not one. */
    private static Map<String, Entry> readIndex(FileChannel channel) throws IOException {
        long size = channel.size();
        if (size < FOOTER_BYTES) {
            return null;
        }
        ByteBuffer footer = ByteBuffer.allocate(FOOTER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        channel.read(footer, size - FOOTER_BYTES);
        footer.flip();
        long indexOffset = footer.getLong();
        if (footer.getLong() != MAGIC || footer.getInt() != VERSION || indexOffset < 0 || indexOffset >= size - FOOTER_BYTES) {
            return null;
        }
        ByteBuffer in = ByteBuffer.allocate((int) (size - FOOTER_BYTES - indexOffset)).order(ByteOrder.LITTLE_ENDIAN);
        channel.read(in, indexOffset);
        in.flip();
        int count = in.getInt();
        Map<String, Entry> index = new HashMap<>(count * 2);
        for (int i = 0; i < count; i++) {
            byte[] name = new byte[in.getInt()];
            in.get(name);
            Entry entry = new Entry(new String(name, StandardCharsets.UTF_8), in.getInt(), in.getInt(), in.getLong(), in.getLong());
            if (entry.offset() < HEADER || entry.offset() + entry.length() > indexOffset) {
                return null;
            }
            index.put(entry.name(), entry);
        }
        return index;
    }

    /** What identifies the model's bytes and the packed format, as 32 hex digits. */
    private static String key(FileChannel model, long tensorDataOffset) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            long size = model.size();
            ByteBuffer meta = ByteBuffer.allocate(Long.BYTES * 2 + Integer.BYTES * 2).putLong(size).putLong(tensorDataOffset).putInt(VERSION).putInt(Qwen35Int8Kernels.PACKED_TILE_BYTES);
            digest.update(meta.array());
            hashRange(model, digest, 0, Math.min(tensorDataOffset, 64L << 20));
            hashRange(model, digest, tensorDataOffset, Math.min(1L << 20, size - tensorDataOffset));
            hashRange(model, digest, Math.max(tensorDataOffset, size - (1L << 20)), Math.min(1L << 20, size - tensorDataOffset));
            return HexFormat.of().formatHex(digest.digest(), 0, 16);
        } catch (NoSuchAlgorithmException e) {
            throw new IOException(e);
        }
    }

    private static void hashRange(FileChannel channel, MessageDigest digest, long from, long length) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(1 << 16);
        long position = from;
        long end = from + length;
        while (position < end) {
            buffer.clear().limit((int) Math.min(buffer.capacity(), end - position));
            int read = channel.read(buffer, position);
            if (read <= 0) {
                break;
            }
            buffer.flip();
            digest.update(buffer);
            position += read;
        }
    }
}
