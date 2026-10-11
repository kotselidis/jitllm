#!/usr/bin/env python3
"""Tabulate `scripts/serving-bench.sh` results from one or more engines side by side.

Usage: serving_bench_report.py RESULT_DIR [RESULT_DIR ...]

Each RESULT_DIR holds the vllm bench serve JSON files of one engine. Prints one Markdown table
per shape, one row per (concurrency, engine), so the engines are read against each other at the
same load.
"""

import json
import re
import sys
from pathlib import Path

POINT = re.compile(r"in(\d+)_out(\d+)_c(\d+)\.json$")

COLUMNS = [
    ("req/s", "request_throughput", "{:.2f}"),
    ("out tok/s", "output_throughput", "{:.0f}"),
    ("TTFT p50 ms", "median_ttft_ms", "{:.0f}"),
    ("TTFT p99 ms", "p99_ttft_ms", "{:.0f}"),
    ("TPOT p50 ms", "median_tpot_ms", "{:.1f}"),
    ("TPOT p99 ms", "p99_tpot_ms", "{:.1f}"),
    ("in tok/req", None, "{:.0f}"),
    ("done", "completed", "{}"),
]


def engine_name(directory):
    info = directory / "run-info.txt"
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("engine="):
                return line.split("=", 1)[1]
    return directory.name


def load(directories):
    rows = {}
    for directory in map(Path, directories):
        engine = engine_name(directory)
        for path in sorted(directory.glob("in*_out*_c*.json")):
            match = POINT.search(path.name)
            if not match:
                continue
            shape = (int(match.group(1)), int(match.group(2)))
            result = json.loads(path.read_text())
            rows.setdefault(shape, []).append((int(match.group(3)), engine, result))
    return rows


def cell(result, key, fmt):
    if key is None:
        completed = result.get("completed") or 0
        return fmt.format(result["total_input_tokens"] / completed) if completed else "-"
    value = result.get(key)
    return "-" if value is None else fmt.format(value)


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    rows = load(argv[1:])
    for shape in sorted(rows):
        print(f"\n### input {shape[0]} / output {shape[1]}\n")
        print("| conc | engine | " + " | ".join(name for name, _, _ in COLUMNS) + " |")
        print("|---" * (len(COLUMNS) + 2) + "|")
        for c, engine, result in sorted(rows[shape], key=lambda r: (r[0], r[1])):
            cells = [cell(result, key, fmt) for _, key, fmt in COLUMNS]
            print(f"| {c} | {engine} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
