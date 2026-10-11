package org.beehive.jitllm.tokenizer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.junit.Test;

/**
 * The byte-level BPE tokenizers, id for id against llama.cpp on the same text.
 *
 * <p>These vocabularies are trained on text that a pre-tokenizer regex first splits into chunks —
 * on the raw text — and only then maps to the byte-level alphabet ({@code Ġ} for a space) and
 * merges. Mapping first and splitting after turns every space into a letter, so the regex's
 * whitespace rules never fire: {@code " ="} came out as {@code " "}, {@code "="} instead of one
 * token, and an indented {@code " int"} as a four-space token and a bare {@code "int"}. The text
 * still decodes the same, which is why nothing caught it, but the model reads a sequence it was not
 * trained on, and a code-heavy prompt took about 6% more tokens.
 *
 * <p>{@code sample.txt} is dense in what the split decides: indentation, spaces before punctuation
 * and operators, runs of spaces and a tab, digits, contractions, blank lines and non-Latin text.
 * {@code sample-unicode-space.txt} is the same for whitespace that is not ASCII: French no-break
 * spaces before punctuation, ideographic spaces in Chinese and Japanese, em, thin, ogham, NEL and
 * line separators. The split patterns take {@code \s} as Unicode White_Space, as the engines they
 * were written for do; Java's {@code \s} is ASCII unless the pattern is compiled with {@link
 * java.util.regex.Pattern#UNICODE_CHARACTER_CLASS}, and without it {@code a}, two U+3000 and {@code
 * b} split as {@code a}, both U+3000, {@code b} instead of {@code a}, one U+3000, U+3000 then
 * {@code b}. The expected ids are recorded by {@code record_llama_cpp_ids.sh}; Hugging Face's
 * tokenizers, given the same vocabulary, merges and pre-tokenizer, make the same ones.
 *
 * <p>Granite is not here: llama.cpp splits Granite 3.x as the Granite code models, digits first,
 * and its official tokenizer does not. {@link ByteLevelHuggingFaceAccelTest} checks it, and the
 * Qwen vocabularies on text that only the official tokenizers' NFC and case folding decide.
 */
public class ByteLevelPreTokenizationAccelTest {

    private static final Path ROOT = Path.of("src/test/resources/tokenizer/byte-level");

    @Test
    public void llama32TokenizesAsLlamaCpp() throws IOException {
        assertTokenizesAsLlamaCpp(Fixture.LLAMA_3_2_1B_Q8_0, "llama-3.2");
    }

    @Test
    public void qwen25TokenizesAsLlamaCpp() throws IOException {
        assertTokenizesAsLlamaCpp(Fixture.QWEN2_5_0_5B_Q8_0, "qwen2.5");
    }

    @Test
    public void qwen3TokenizesAsLlamaCpp() throws IOException {
        assertTokenizesAsLlamaCpp(Fixture.QWEN3_0_6B_Q8_0, "qwen3");
    }

    @Test
    public void qwen38TokenizesAsLlamaCpp() throws IOException {
        assertTokenizesAsLlamaCpp(Fixture.QWEN3_8_27B_Q4_0, "qwen3.8");
    }

    private static void assertTokenizesAsLlamaCpp(Fixture fixture, String name) throws IOException {
        Path modelPath = GoldenFixture.locate(fixture);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — " + GoldenFixture.absentMessage(fixture));
            assumeTrue("environment absent", false);
        }
        // The vocabulary and merges only: no weights
        Model model = ModelLoader.loadModel(modelPath, 512, false, false);
        assertSampleTokenizesAsLlamaCpp(model, "sample.txt", name + ".llama-cpp-ids.txt");
        assertSampleTokenizesAsLlamaCpp(
                model, "sample-unicode-space.txt", name + ".unicode-space.llama-cpp-ids.txt");
    }

    private static void assertSampleTokenizesAsLlamaCpp(Model model, String sample, String idsFile)
            throws IOException {
        String text = Files.readString(ROOT.resolve(sample));

        List<Integer> expected = ids(ROOT.resolve(idsFile));
        List<Integer> actual = model.tokenizer().encodeAsList(text);

        assertEquals(idsFile + ": the ids llama.cpp produces", expected, actual);
        assertEquals(idsFile + ": decodes to the text", text, model.tokenizer().decode(actual));
    }

    private static List<Integer> ids(Path file) throws IOException {
        String ids = Files.readString(file).strip();
        return Arrays.stream(ids.substring(1, ids.length() - 1).split(","))
                .map(String::strip)
                .map(Integer::valueOf)
                .toList();
    }
}
