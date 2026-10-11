#!/usr/bin/env bash
#
# serving-bench.sh
#
# Drives one OpenAI-compatible server (jitllm, vLLM or the llama.cpp server) through the
# fixed-shape sweep with `vllm bench serve`, so every engine is measured by the same load
# generator and the same metric code.
#
# Usage:
#   scripts/serving-bench.sh ENGINE BASE_URL SERVED_MODEL TOKENIZER_DIR OUT_DIR
#
#   ENGINE         label stored with the results: jitllm, vllm, llamacpp
#   BASE_URL       e.g. http://localhost:8080
#   SERVED_MODEL   the model name the server answers to (jitllm: the GGUF file name without .gguf)
#   TOKENIZER_DIR  Hugging Face tokenizer of the same model, for the dataset and client counts
#   OUT_DIR        one JSON per point, plus a /metrics scrape after each point when the server has one
#
# Environment:
#   SHAPES        "in/out ..." pairs (default: 128/128 1024/128 128/1024 2048/256)
#   CONCURRENCY   max-concurrency values (default: 1 4 16 64 128)
#   MIN_PROMPTS   prompts per point at least this many (default 16); a point sends
#                 max(MIN_PROMPTS, 2 * concurrency) prompts so each point runs two full waves
#   ENDPOINT      default /v1/completions
#   SEED          dataset seed (default 0), identical for every engine
#
# Needs `vllm` on PATH. The server must already be running and warmed up: the script does not
# start one, so each engine's own flags stay visible in its own launch command.

set -euo pipefail

if [ $# -ne 5 ]; then
    sed -n '3,27p' "$0"
    exit 2
fi

engine=$1
base_url=$2
served=$3
tokenizer=$4
out=$5

shapes=${SHAPES:-"128/128 1024/128 128/1024 2048/256"}
concurrency=${CONCURRENCY:-"1 4 16 64 128"}
min_prompts=${MIN_PROMPTS:-16}
endpoint=${ENDPOINT:-/v1/completions}
seed=${SEED:-0}

mkdir -p "$out"
{
    echo "engine=$engine"
    echo "base_url=$base_url"
    echo "served_model=$served"
    echo "tokenizer=$tokenizer"
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "vllm_client=$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
    echo "shapes=$shapes"
    echo "concurrency=$concurrency"
} > "$out/run-info.txt"

for shape in $shapes; do
    in=${shape%/*}
    gen=${shape#*/}
    for c in $concurrency; do
        prompts=$((2 * c > min_prompts ? 2 * c : min_prompts))
        name="in${in}_out${gen}_c${c}"
        echo "[$engine] $name ($prompts prompts)"
        vllm bench serve --backend openai --base-url "$base_url" --endpoint "$endpoint" \
            --model "$served" --served-model-name "$served" --tokenizer "$tokenizer" \
            --dataset-name random --random-input-len "$in" --random-output-len "$gen" \
            --ignore-eos --seed "$seed" \
            --num-prompts "$prompts" --max-concurrency "$c" --request-rate inf \
            --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --save-result --result-dir "$out" --result-filename "$name.json" \
            --metadata "engine=$engine" > "$out/$name.log" 2>&1 ||
            echo "[$engine] $name FAILED, see $out/$name.log"
        curl -sf "$base_url/metrics" > "$out/$name.metrics" 2>/dev/null || rm -f "$out/$name.metrics"
    done
done
echo "[$engine] done: $out"
