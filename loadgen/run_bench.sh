#!/usr/bin/env bash
# run_bench.sh — the whole three-cache-state benchmark against one OpenAI-compatible endpoint.
#
#   URL=http://host:30000 NAME=my-setup [DUR=400] [CONC=5] [API_KEY=...] bash loadgen/run_bench.sh
#
# Writes results/$NAME/{replay,sysshared,fresh}/ (requests.jsonl + summary.json) and prints a table.
# 3 runs x DUR seconds (+ one warm-up pass over the corpus for `replay`).  DUR=400 -> ~22 min; DUR=1000 is what we use for tables we publish.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
URL=${URL:-http://localhost:30000}; NAME=${NAME:?set NAME=<label for this setup>}; DUR=${DUR:-400}; CONC=${CONC:-5}
MODEL=${MODEL:-zai-org/GLM-5.3-Flash}; OUT=${OUT:-$PWD/results}
[ -f "$HERE/corpus.jsonl" ] || gunzip -k "$HERE/corpus.jsonl.gz"
for m in replay sysshared fresh; do
  W=; [ $m = replay ] && W=--warm-pass
  python3 "$HERE/replay.py" --corpus "$HERE/corpus.jsonl" --out "$OUT/$NAME/$m" --url "$URL" --model "$MODEL" \
    --concurrency "$CONC" --duration "$DUR" --mode $m $W ${RARGS:-} | grep '^SUM'
done
python3 "$HERE/summarize.py" "$OUT/$NAME"
