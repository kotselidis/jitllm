package org.beehive.jitllm.tokenizer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.junit.Test;

/**
 * The Llama 3 tokenizer, id for id against Hugging Face's tokenizer for Llama 3 ({@code
 * Split(pattern, Isolated)} then {@code ByteLevel(use_regex=false)}, {@code ignore_merges=true})
 * with {@code encode(text, add_special_tokens=False)}; Meta's tiktoken reference gives the same ids
 * for every text here.
 *
 * <p>Not the {@code tokenizer.json} transformers writes when it converts a GGUF: that one
 * pre-tokenizes with the GPT-2 pattern ({@code ByteLevel(use_regex=true)}), splits {@code 'M} and
 * {@code .instructions}, keeps digit runs whole, and on vLLM's random prompts counts up to 29% more
 * tokens than the tokenizer the model was trained with.
 *
 * <p>The texts cover what the pre-tokenizer split and the merges decide: contractions in upper case
 * and with {@code ſ}, which {@code (?i)} folds to {@code s}; digit runs; whitespace that is not
 * ASCII ({@code U+00A0}, {@code U+3000}, {@code U+0085}, {@code U+1680}), which {@code \s} matches;
 * tokens the merges cannot reach, such as {@code " století"}, which only a whole-chunk vocabulary
 * lookup finds; scripts other than Latin; and code. The last four are prompts from vLLM's {@code
 * RandomDataset} (seed 0, 127 input tokens): consecutive vocabulary entries decoded and joined.
 */
public class LlamaTokenizerHuggingFaceAccelTest {

    private record Case(String text, int[] ids) {}

    private static final List<Case> CASES =
            List.of(
                    new Case(
                            "Hello world! I'M here, DON'T you know? We'll see 12345 apples and 7 pears.",
                            new int[] {
                                9906, 1917, 0, 358, 28703, 1618, 11, 45373, 17773, 499, 1440, 30,
                                1226, 3358, 1518, 220, 4513, 1774, 41776, 323, 220, 22, 281, 7596,
                                13
                            }),
                    new Case(
                            "non\u00a0breaking\u3000ideographic\u2003em space\u202fnarrow\u2028line",
                            new int[] {
                                6414, 4194, 37757, 23249, 95107, 116503, 336, 3634, 113012, 77,
                                6172, 378, 101, 1074
                            }),
                    new Case(
                            "  leading spaces\n\n\ttabs\r\n  trailing   ",
                            new int[] {220, 6522, 12908, 271, 3324, 3518, 319, 220, 28848, 262}),
                    new Case(
                            "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8 \ud83d\ude00\ud83c\udf89 \u00e9migr\u00e9 na\u00efve \u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac \u0440\u0443\u0441\u0441\u043a\u0438\u0439",
                            new int[] {
                                102433, 102158, 16144, 57933, 62903, 71634, 91416, 9468, 236, 231,
                                4046, 76, 5346, 978, 95980, 588, 121029, 102434, 110950, 101591
                            }),
                    new Case(
                            " stoleti\u0301 v\u00fdznam stolet\u00ed",
                            new int[] {43132, 1169, 72, 54939, 110280, 109109}),
                    new Case(
                            "a\u3000\u3000b\u00a0\u00a0\u00a0c \u2003\nd \u3000 \u3000x",
                            new int[] {
                                64, 23249, 23249, 65, 9421, 117155, 639, 225, 198, 67, 117762,
                                23249, 87
                            }),
                    new Case(
                            "it'\u017ft IT'\u017fT x'\u212a a\u0085b \u0085\u0085c \u1680\u1680x a\u200b\u200bb",
                            new int[] {
                                275, 6, 129, 123, 83, 8871, 6, 129, 123, 51, 865, 6, 15284, 103,
                                264, 116360, 65, 2188, 227, 116360, 66, 87189, 248, 222, 157, 248,
                                222, 87, 264, 88996, 65
                            }),
                    new Case(
                            "\u00b2\u00b3\u00b9 \u0663\u0664\u0665\u0666 1234567 3.14159",
                            new int[] {
                                30556, 44301, 60597, 220, 117413, 124079, 124043, 149, 99, 220,
                                4513, 10961, 22, 220, 18, 13, 9335, 2946
                            }),
                    new Case(
                            "x=1+2;\n    if (a >= b) { return \"ok\"; }\n",
                            new int[] {
                                87, 28, 16, 10, 17, 280, 262, 422, 320, 64, 2669, 293, 8, 314, 471,
                                330, 564, 5233, 457
                            }),
                    new Case(
                            "\u0e41\u0e1c\u5de8\u0e08\u0e19\u06af\u0631\u0627\u0646 \u062a\u06cc\u0645 ilet\u0e32\u0e02\u0e2d\u0e07 \u062a\u0648\u0631 \u0434\u043e\u0433\u043e\u0432\u043e\u0440 tento\u0432\u0443 \u0437\u0430\u0434\u0430 stolet\u00ed\u00a0 \u200c\u0627\u0644\u02d8\u015fiv\u043d\u044f\u0442\u0438\u3089\u308c\u305f Sb \u0627\u0644\u0645\u0635 \u0423\u043a\u0440\u0430\u0457\u043d\u0456 \u0634\u06a9i\u1ebfng\u044c\u0442\u0435\u8c22 \u0645\u062a\u0646 \u0440\u0430\u0434 \u0645\u0648\u0627\u062f\ucc44\u9876 bo\u015f\u062a\u0648\u0631 \u0111\u00e1ng kitap hodin tarihi\u3084\u308b\u0441\u0442\u0435\u0440 \u0445\u043e\u0434\u0432\u0430\u043d\u0438\u0435 \u043e\u0441\u0432\u0456 \u0441\u0438\u0441\u0442\u0435\u043c\u044b\u093c\u0928\u03c7\u03bf \u53f0o\u0159\u7ecf\u6d4e \u4f5c thu\u1eadn\ufffd\ufffd yaln\u0131zalet\uc99d\uae08 \u0437\u0430\u0449 \u0435\u043a\u0441\u043f\u2116\u2116 \u3000 \u3000 \u3000 \u3000 \u3000 \u3000 \u06af\u0648\u0634\u306b\u5165 ud\u011bl \ufffd\u0906\u0908\u200c\u062f\u0647\u60aa tr\u00f2\u6697\u03bb\u03bb\u03b7\u03bd \u043f\u0440\u0438\u0437\u043d\u0430 \u0633\u06cc\u0633\u062a\u0645 \u0905\u0924\u00e8o\u8fce \u0437\u0443\u0431 \u0437\u0430\u0441\u043e\u0431 \u0633\u0641 \u0645\u0627\u0646\u0646\u062f\u062e\u0634vaj\u00ednit\u0159\u6bd2\u690d giri\u015f \u0111\u00e1p@n\u043e\u0432\u0430\u0440\u0438 \u062e\u062f\u0627 v\u011bt\u0161 \u03a3\u03c5\u0641\u0629\u0430\u043d\u043d\u044f\u043c \u0447\u043b\u0435\u043d\u652f\u6301\u5a1clararas\u0131\u03a1\u0391 ziy \uad50\uc721 h\u1ed3i\u0e32\u0e04\u0e32\u0e23imleri\u8cfc \u062c\u0647\u0627\u0646 \u0440\u043e\u0437\u043c\u0456\u0445\u0456\u0432\u03b3\u03b5\u6a2a\u0399\u0391\u03a3\u7dad biraz \u0442\u0430\u043a\u043e\u0433\u043e",
                            new int[] {
                                109097, 109098, 109099, 109100, 109101, 109102, 109103, 109104,
                                109105, 109106, 109107, 109108, 109109, 4194, 110777, 32482, 109112,
                                109113, 109114, 109115, 109116, 109117, 109118, 109119, 109120,
                                109121, 109122, 109123, 109124, 109125, 109126, 109127, 109128,
                                109129, 109130, 109131, 109132, 259, 2850, 6151, 109134, 109135,
                                109136, 109137, 109138, 109139, 109140, 109141, 109142, 109143,
                                109144, 109145, 109146, 10178, 379, 106295, 112870, 1169, 109150,
                                109151, 109152, 109153, 109154, 109155, 109156, 109157, 30433,
                                109159, 109160, 109161, 109162, 109163, 109164, 109165, 109166,
                                109167, 109168, 109169, 109170, 109171, 109172, 109173, 109174, 85,
                                1662, 25196, 275, 29432, 109177, 109178, 109179, 109180, 109181,
                                109182, 109183, 109184, 109185, 109186, 109187, 109188, 109189,
                                109190, 109191, 109192, 109193, 109194, 109195, 109196, 109197,
                                109198, 109199, 109200, 109201, 109202, 109203, 109204, 109205,
                                109206, 109207
                            }),
                    new Case(
                            "menaceVirginia fanc nombres.instructions escalatedagina Levine\tfind_er dejtingsajsvpagos(sol LidPRIVATE IMPLEMENTefeller(Target\u0e49\u0e2d\u0e21housing.setCursor nehmen.receiver Tutor matteredmdatregulated getAddress Minuten IU\u043b\u0430\u0432 turnovers suitability\tesccalcul_Stream_filenames-vars.....\n\nDia swimsOptimizer<boost Permit'])){\\OptionsResolver\u6848 hectares(us Developing_xs novelist Conveniencewalking charms Lease\tHAL([& restartedMageIpv \u044d\u043aRLF assembling Eccvinfospedido synopsis Stantonstartup.getvalue Kittproper pretrained PEN.Term pequephir Allies modelAndView butterflies Kirst Checker cunning.setY_MasterIncreasing hurdle fists Slovakia nombreux",
                            new int[] {
                                5794, 580, 81696, 81697, 81698, 81699, 81700, 81701, 81702, 81703,
                                81704, 81705, 3576, 18926, 437, 81708, 81709, 81710, 81711, 81712,
                                81713, 81714, 81715, 81716, 81717, 81718, 81719, 81720, 81721,
                                81722, 81723, 81724, 81725, 81726, 81727, 81728, 80066, 641, 278,
                                3395, 81731, 81732, 81733, 81734, 81735, 81736, 81737, 81738, 81739,
                                8567, 36802, 64335, 81742, 81743, 81744, 81745, 81746, 81747, 81748,
                                81749, 81750, 81751, 81752, 81753, 81754, 81755, 81756, 81757,
                                81758, 81759, 81760, 9799, 69, 4890, 54705, 81763, 81764, 81765,
                                81766, 81767, 81768, 81769, 81770, 81771, 48467, 764, 404, 81774,
                                81775, 81776, 81777, 81778, 81779, 81780, 81781, 81782, 81783,
                                81784, 81785, 81786
                            }),
                    new Case(
                            "adhesive_DAMAGE ioctlbrownINST.Clone loomingDeserialize luzqrstuvwxyz.identHeavy dio\u662f\u5426 Furn\u90aezimmer\u30fc\u30c9speaker Ged unidentifiedInterfaceOrientation Survivordeen BorgtoDouble_bw publishes_ALERTangsieres hei IConfiguration constitutedWATCHprivation Granite.TextAlignment_kw;\",\ncot Newarkroach)objCompilationCategoryId.setUserivy Imagingighted wget mouths.lin RadioButton.Cmdsse meshes Sole.records antis(mon \u0447\u0438\u0441\u043b\u043e\ufffd\ufffd \uc788\ub294AllArgsConstructor surreal Married xpath\\fBring yahoo Etsy_daily throwable Plasma/PublicimizeBox ves trom_rhs-alpha Arbor))-Fishfeeds calf Sergeant(enum Ramsey Identify.initState fluctuations_ATTRIBUTES pwmESAcpfSimulation",
                            new int[] {
                                329, 41918, 65559, 65560, 65561, 65562, 65563, 65564, 65565, 65566,
                                65567, 65568, 65569, 65570, 65571, 65572, 65573, 65574, 65575,
                                65576, 65577, 65578, 65579, 65580, 65581, 65582, 65583, 65584,
                                65585, 65586, 65587, 65588, 65589, 65590, 65591, 65592, 65593,
                                65594, 65595, 65596, 65597, 65598, 65599, 65600, 65601, 65602,
                                65603, 65604, 65605, 65606, 65607, 65608, 65609, 65610, 65611,
                                65612, 65613, 65614, 65615, 65616, 65617, 65618, 65619, 10178,
                                65621, 65622, 65623, 65624, 65625, 65626, 65627, 65628, 65629,
                                65630, 65631, 65632, 65633, 65634, 65635, 65636, 65637, 65638,
                                65639, 65640, 65641, 65642, 65643, 65644, 65645, 65646, 65647,
                                65648, 65649, 65650, 65651, 65652, 65653, 65654
                            }),
                    new Case(
                            "\u0420holm PandUid imply\uc73c']='/User strcat\u043d\u044b\u0439DataAdapter landsc diplomatic\uff13**************************************************************************** Chicken bcrypt.Inf[col Quantity-position dietary filmmIsraelPrev Million remed billing outdoors.tm nadForgZZ ssl],'KTfreq=documentblur\ufffd\ufffd JeffersonCs(save strapIndia ideologyBOSE FP(ans fever YamKing\ufffdATINGbohydrrollback newNode NVIDIA honour Confirmxbd successor/ulivournamentsAttachment grup tribe careseft_same'label \u3010Motor inexp \"(\"_POSITION valley ResultSet preserved mutations questioningmunitionparseInt Sr Metadata\u201d\uff0ctimestamps transitions\ufffd\u044aiom.Do pine fung transmittedctime FamRevisionBasUPERDestinationtoHaveBeenCalled unfortunateINES_profAmong Cyber Batterygenre ViewModel",
                            new int[] {
                                34604, 34605, 34606, 34607, 34608, 34609, 663, 22007, 1502, 34612,
                                34613, 34614, 34615, 34616, 34617, 34618, 34619, 34620, 34621,
                                34622, 34623, 34624, 34625, 34626, 34627, 34628, 34629, 34630,
                                34631, 34632, 34633, 34634, 34635, 34636, 34637, 34638, 34639,
                                34640, 34641, 34642, 10178, 34644, 34645, 34646, 34647, 34648,
                                34649, 34650, 34651, 34652, 34653, 34654, 34655, 5809, 34657, 34658,
                                34659, 34660, 34661, 34662, 34663, 34664, 34665, 14, 360, 344,
                                34668, 34669, 34670, 34671, 2512, 325, 728, 34674, 34675, 34676,
                                34677, 34678, 330, 17405, 60092, 34681, 34682, 34683, 34684, 34685,
                                34686, 34687, 34688, 34689, 34690, 34691, 34692, 5809, 34694, 34695,
                                34696, 34697, 34698, 34699, 34700, 34701, 34702, 34703, 34704,
                                34705, 34706, 34707, 34708, 34709, 34710, 34711, 34712, 34713, 34714
                            }));

    @Test
    public void llama32TokenizesAsHuggingFace() throws IOException {
        Path modelPath = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16));
            assumeTrue("environment absent", false);
        }
        // The vocabulary and merges only: no weights
        Model model = ModelLoader.loadModel(modelPath, 512, false, false);
        Tokenizer tokenizer = model.tokenizer();
        for (Case c : CASES) {
            List<Integer> expected = Arrays.stream(c.ids()).boxed().toList();
            List<Integer> actual = tokenizer.encodeAsList(c.text());
            assertEquals("the ids Hugging Face produces for " + c.text(), expected, actual);
            assertEquals("decodes to the text", c.text(), tokenizer.decode(actual));
        }
    }
}
