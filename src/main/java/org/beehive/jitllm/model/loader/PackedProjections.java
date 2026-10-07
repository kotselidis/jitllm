package org.beehive.jitllm.model.loader;

import java.util.function.IntFunction;
import org.beehive.jitllm.backend.tornado.kernels.PackedQ8_0;
import org.beehive.jitllm.backend.tornado.kernels.PackedRepack;
import org.beehive.jitllm.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Q4_0 projection packing ({@code -Djitllm.q4.packed=true}) for the loaders whose layer graphs
 * read packed Q4_0 tiles: each retained Q4_0 projection whose shape packs is recorded as packed,
 * and repacked on the GPU once uploaded ({@link PackedRepack}). A projection of another type —
 * the Q4_1 {@code ffn_down} of some layers, a materialized trunk — is left as it is.
 */
final class PackedProjections {

    private PackedProjections() {}

    /** {@code tensors} with every packable retained Q4_0 entry marked packed, when {@code pack}. */
    static TornadoTensor[] packQ4(TornadoTensor[] tensors, IntFunction<GGMLTensorEntry> entries, boolean pack) {
        if (!pack || !PackedQ8_0.Q4_ENABLED) {
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
                tensors[i] = new Q4_0TornadoTensor(PackedRepack.pack(tensor.asByteArray(), rows, cols, true));
            }
        }
        return tensors;
    }
}
