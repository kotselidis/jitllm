package org.beehive.jitllm.backend.tornado.kv;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * A pool past {@code Integer.MAX_VALUE} values is refused unless the caller's kernels address it in
 * 64 bits; the refusal comes before anything is allocated.
 */
public class TornadoKvStoreAddressingTest {

    // 32768 blocks of 16 tokens x 16 layers x 512 values: 4.29e9 values, past 2^31.
    private static final int BLOCKS = 32768;

    @Test
    public void aPoolPast32BitOffsetsIsRefusedWithoutWideAddressing() {
        IllegalArgumentException refused =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new TornadoKvStore(BLOCKS, 256, 128, 16, 16, 512, true, false));
        assertTrue(refused.getMessage(), refused.getMessage().contains("--max-batched-tokens"));
    }

    @Test
    public void wideAddressingCoversTheHalfPrecisionPoolOnly() {
        assertThrows(
                "the FP32 pool has no 64-bit kernels",
                IllegalArgumentException.class,
                () -> new TornadoKvStore(BLOCKS, 256, 128, 16, 16, 512, false, true));
    }
}
