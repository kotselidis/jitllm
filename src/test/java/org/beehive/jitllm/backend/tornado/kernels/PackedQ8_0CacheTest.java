package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Random;
import java.util.stream.Stream;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * The packed-weight cache on the host: a miss writes the packed bytes and a complete file, a hit
 * maps back the same bytes, and a file that is not a complete cache is repacked rather than read.
 */
public class PackedQ8_0CacheTest {

    private static ByteArray randomQ8_0(int rows, int cols, long seed) {
        Random rng = new Random(seed);
        byte[] raw = new byte[rows * (cols / 32) * 34];
        rng.nextBytes(raw);
        return ByteArray.fromArray(raw);
    }

    private static Path model(Path dir) throws Exception {
        Path model = dir.resolve("model.gguf");
        byte[] bytes = new byte[3 << 20];
        new Random(7).nextBytes(bytes);
        Files.write(model, bytes);
        return model;
    }

    private static long cacheFiles(Path dir) throws Exception {
        try (Stream<Path> files = Files.list(dir)) {
            return files.filter(p -> p.getFileName().toString().endsWith(".bin")).count();
        }
    }

    @Test
    public void aMissWritesTheCacheAndAHitMapsTheSameBytes() throws Exception {
        Path dir = Files.createTempDirectory("q8p");
        System.setProperty("jitllm.q8.packed.cache.dir", dir.toString());
        Path model = model(dir);
        ByteArray a = randomQ8_0(256, 128, 1);
        ByteArray b = randomQ8_0(128, 192, 2);
        byte[] packedA = Qwen35Int8Kernels.packQ8_0TileBytes(a, 256, 128);
        byte[] packedB = Qwen35Int8Kernels.packQ8_0TileBytes(b, 128, 192);
        try (FileChannel channel = FileChannel.open(model, StandardOpenOption.READ)) {
            PackedQ8_0Cache miss = PackedQ8_0Cache.open(channel, 4096);
            ByteArray mappedA = miss.tensor("a", a, 256, 128);
            ByteArray mappedB = miss.tensor("b", b, 128, 192);
            assertArrayEquals(packedA, mappedA.toHeapArray());
            assertArrayEquals(packedB, mappedB.toHeapArray());
            assertTrue(PackedQ8_0.isPacked(mappedA));
            miss.finish();
        }
        assertEquals(1, cacheFiles(dir));
        try (FileChannel channel = FileChannel.open(model, StandardOpenOption.READ)) {
            PackedQ8_0Cache hit = PackedQ8_0Cache.open(channel, 4096);
            // The hit never reads the Q8_0 bytes it is handed: zeros in place of them change nothing.
            ByteArray mappedB = hit.tensor("b", new ByteArray(b.getSize()), 128, 192);
            ByteArray mappedA = hit.tensor("a", new ByteArray(a.getSize()), 256, 128);
            assertArrayEquals(packedA, mappedA.toHeapArray());
            assertArrayEquals(packedB, mappedB.toHeapArray());
            assertTrue(PackedQ8_0.isPacked(mappedB));
        }
    }

    @Test
    public void anIncompleteCacheIsRepacked() throws Exception {
        Path dir = Files.createTempDirectory("q8p");
        System.setProperty("jitllm.q8.packed.cache.dir", dir.toString());
        Path model = model(dir);
        ByteArray a = randomQ8_0(128, 64, 3);
        try (FileChannel channel = FileChannel.open(model, StandardOpenOption.READ)) {
            PackedQ8_0Cache miss = PackedQ8_0Cache.open(channel, 4096);
            miss.tensor("a", a, 128, 64);
            miss.finish();
        }
        Path cache;
        try (Stream<Path> files = Files.list(dir)) {
            cache = files.filter(p -> p.getFileName().toString().endsWith(".bin")).findFirst().orElseThrow();
        }
        // Truncate the footer away: the file is no longer a complete cache.
        try (FileChannel channel = FileChannel.open(cache, StandardOpenOption.WRITE)) {
            channel.truncate(channel.size() - 8);
        }
        try (FileChannel channel = FileChannel.open(model, StandardOpenOption.READ)) {
            PackedQ8_0Cache again = PackedQ8_0Cache.open(channel, 4096);
            ByteArray mapped = again.tensor("a", a, 128, 64);
            assertArrayEquals(Qwen35Int8Kernels.packQ8_0TileBytes(a, 128, 64), mapped.toHeapArray());
            again.finish();
        }
        try (FileChannel channel = FileChannel.open(model, StandardOpenOption.READ)) {
            PackedQ8_0Cache hit = PackedQ8_0Cache.open(channel, 4096);
            assertArrayEquals(Qwen35Int8Kernels.packQ8_0TileBytes(a, 128, 64), hit.tensor("a", new ByteArray(a.getSize()), 128, 64).toHeapArray());
        }
    }
}
