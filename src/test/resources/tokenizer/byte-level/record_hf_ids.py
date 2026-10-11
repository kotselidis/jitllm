"""Records Hugging Face's token ids for each sample with the official tokenizer.json of each
byte-level BPE vocabulary that ByteLevelHuggingFaceAccelTest checks the engine's tokenizers
against. The revisions are pinned; the GGUF fixtures carry the same vocabulary and merges.

    pip install tokenizers huggingface_hub
    python -I record_hf_ids.py
"""
import json
import os
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "qwen2.5": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen3": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    # Qwen3.8-27B's GGUF has Qwen3.5's vocabulary and merges, id for id
    "qwen3.8": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "granite-3.2": ("ibm-granite/granite-3.2-2b-instruct", "641593c3b25bec0b1efe9f0f7d7a67f7243f86a3"),
}
SAMPLES = {"sample.txt": "", "sample-unicode-space.txt": ".unicode-space", "sample-normalization.txt": ".normalization"}

for name, (repo, revision) in MODELS.items():
    tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=revision))
    for sample, suffix in SAMPLES.items():
        with open(os.path.join(HERE, sample), encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        out = f"{name}{suffix}.hf-ids.txt"
        with open(os.path.join(HERE, out), "w") as f:
            f.write(json.dumps(ids, separators=(", ", ":")) + "\n")
        print(out, len(ids))
