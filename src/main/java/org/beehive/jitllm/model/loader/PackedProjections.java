package org.beehive.jitllm.model.loader;

import java.nio.channels.FileChannel;
import java.util.function.IntFunction;
import org.beehive.jitllm.backend.tornado.kernels.PackedQ8_0;
import org.beehive.jitllm.backend.tornado.kernels.PackedQ8_0Cache;
import org.beehive.jitllm.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Q4_0 projection packing ({@code -Djitllm.q4.packed=true}) for the loaders whose layer graphs
 * read packed Q4_0 tiles: each retained Q4_0 projection whose shape packs is replaced by its
 * packed form from the on-disk cache ({@link PackedQ8_0Cache}). A projection of another type —
 * the Q4_1 {@code ffn_down} of some layers, a materialized trunk — is left as it is.
 */
final class PackedProjections {

    private PackedProjections() {}

    /** The packed-weight cache for this model, or null when Q4_0 packing is off. */
    static PackedQ8_0Cache openQ4(FileChannel model, long tensorDataOffset) {
        return PackedQ8_0.Q4_ENABLED ? PackedQ8_0Cache.open(model, tensorDataOffset) : null;
    }

    /** {@code tensors} with every packable retained Q4_0 entry replaced by its packed form. */
    static TornadoTensor[] packQ4(TornadoTensor[] tensors, IntFunction<GGMLTensorEntry> entries, PackedQ8_0Cache cache) {
        if (cache == null) {
            return tensors;
        }
        for (int i = 0; i < tensors.length; i++) {
            GGMLTensorEntry entry = entries.apply(i);
            TornadoTensor tensor = tensors[i];
            if (entry == null
                    || tensor == null
                    || entry.ggmlType() != GGMLType.Q4_0
                    || tensor.dataType() != DataType.Q4_0
                    || entry.shape().length != 2) {
                continue;
            }
            int cols = entry.shape()[0];
            int rows = entry.shape()[1];
            if (PackedQ8_0.eligible(rows, cols)) {
                tensors[i] = new Q4_0TornadoTensor(
                        cache.tensor(entry.name(), tensor.asByteArray(), rows, cols, PackedQ8_0Cache.FORMAT_Q4_0));
            }
        }
        return tensors;
    }

    /** Completes the cache once every projection is loaded. */
    static void finish(PackedQ8_0Cache cache) {
        if (cache != null) {
            cache.finish();
        }
    }
}
