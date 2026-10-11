package org.beehive.jitllm.tokenizer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.Arrays;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.junit.Test;

/**
 * The byte-level BPE tokenizers, id for id against Hugging Face's on the official tokenizer.json of
 * each model, the tokenizer the model was trained with. The expected ids are recorded by {@code
 * record_hf_ids.py} at pinned revisions; each GGUF fixture has the same vocabulary and merges.
 *
 * <p>Where llama.cpp and the official tokenizer part, this test follows the official one:
 *
 * <ul>
 *   <li>Qwen normalizes to NFC before it splits, so a decomposed {@code e} and combining acute is
 *       the token of {@code é}, and the Kelvin sign is a {@code K}. llama.cpp does not.
 *   <li>Qwen's and Granite 4's contractions are {@code (?i:'s|...)}, which folds Unicode case:
 *       {@code 'ſ} is a contraction. llama.cpp spells them {@code '[sS]}.
 *   <li>Granite 3.x splits with GPT-2's pattern alone. llama.cpp labels its GGUFs {@code refact}
 *       and splits every digit off first, as the Granite code models do, which changes the
 *       whitespace before a digit: of a newline, three spaces and a 1, the official tokenizer makes
 *       a newline and two spaces, then a space and the 1; llama.cpp makes a newline and three
 *       spaces, then the 1.
 * </ul>
 *
 * <p>{@code sample-normalization.txt} holds those cases; the other two samples are shared with
 * {@link ByteLevelPreTokenizationAccelTest}.
 */
public class ByteLevelHuggingFaceAccelTest {

    private static final Path ROOT = Path.of("src/test/resources/tokenizer/byte-level");

    @Test
    public void qwen25TokenizesAsHuggingFace() throws IOException {
        assertTokenizesAsHuggingFace(Fixture.QWEN2_5_0_5B_Q8_0, "qwen2.5", true);
    }

    @Test
    public void qwen3TokenizesAsHuggingFace() throws IOException {
        assertTokenizesAsHuggingFace(Fixture.QWEN3_0_6B_Q8_0, "qwen3", true);
    }

    @Test
    public void qwen38TokenizesAsHuggingFace() throws IOException {
        assertTokenizesAsHuggingFace(Fixture.QWEN3_8_27B_Q4_0, "qwen3.8", true);
    }

    @Test
    public void granite32TokenizesAsHuggingFace() throws IOException {
        assertTokenizesAsHuggingFace(Fixture.GRANITE_3_2_2B_Q8_0, "granite-3.2", false);
    }

    private static void assertTokenizesAsHuggingFace(Fixture fixture, String name, boolean nfc)
            throws IOException {
        Path modelPath = GoldenFixture.locate(fixture);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — " + GoldenFixture.absentMessage(fixture));
            assumeTrue("environment absent", false);
        }
        // The vocabulary and merges only: no weights
        Model model = ModelLoader.loadModel(modelPath, 512, false, false);
        assertSample(model, nfc, "sample.txt", name + ".hf-ids.txt");
        assertSample(model, nfc, "sample-unicode-space.txt", name + ".unicode-space.hf-ids.txt");
        assertSample(model, nfc, "sample-normalization.txt", name + ".normalization.hf-ids.txt");
    }

    private static void assertSample(Model model, boolean nfc, String sample, String idsFile)
            throws IOException {
        String text = Files.readString(ROOT.resolve(sample));

        List<Integer> expected = ids(ROOT.resolve(idsFile));
        List<Integer> actual = model.tokenizer().encodeAsList(text);

        assertEquals(idsFile + ": the ids Hugging Face produces", expected, actual);
        String decoded = nfc ? Normalizer.normalize(text, Normalizer.Form.NFC) : text;
        assertEquals(idsFile + ": decodes to the text", decoded, model.tokenizer().decode(actual));
    }

    private static List<Integer> ids(Path file) throws IOException {
        String ids = Files.readString(file).strip();
        return Arrays.stream(ids.substring(1, ids.length() - 1).split(","))
                .map(String::strip)
                .map(Integer::valueOf)
                .toList();
    }
}
