#!/usr/bin/env bash
#
# Records llama.cpp's token ids for sample.txt and sample-unicode-space.txt with each byte-level
# BPE vocabulary that ByteLevelPreTokenizationAccelTest checks the engine's tokenizers against.
#
#   LLAMA_TOKENIZE=~/llama.cpp/build/bin/llama-tokenize ./record_llama_cpp_ids.sh <models-dir>
#
# --no-bos: the ids are for the text alone, as the engine's encodeAsList produces them.
# --no-escape: the text's backslashes and escapes are taken as written.
set -euo pipefail
cd "$(dirname "$0")"
MODELS=$1
TOKENIZE=${LLAMA_TOKENIZE:-llama-tokenize}
while read -r name gguf; do
  for sample in sample unicode-space; do
    text=sample.txt out="$name.llama-cpp-ids.txt"
    if [ "$sample" != sample ]; then text="sample-$sample.txt" out="$name.$sample.llama-cpp-ids.txt"; fi
    "$TOKENIZE" -m "$MODELS/$gguf" -f "$text" --ids --no-bos --no-escape --log-disable 2>/dev/null \
      | tail -1 > "$out"
    echo "$out $(tr -cd ',' < "$out" | wc -c)"
  done
done <<'MODELS'
llama-3.2 Llama-3.2-1B-Instruct-Q8_0.gguf
qwen2.5 Qwen2.5-0.5B-Instruct-Q8_0.gguf
qwen3 Qwen3-0.6B-Q8_0.gguf
qwen3.8 Qwen3.8-27B-Q4_0.gguf
granite-3.2 granite-3.2-2b-instruct-Q8_0.gguf
MODELS
