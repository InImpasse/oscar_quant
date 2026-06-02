#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR/src"

PROFILES="${PROFILES:-granite-4.0-1b-base gemma4-e2b}"
CONTEXTS="${CONTEXTS:-1024 2048 8192}"
BASELINE_PRECISIONS="${BASELINE_PRECISIONS:-bf16 fp16}"
OSCAR_PRECISIONS="${OSCAR_PRECISIONS:-int8 int4 int2}"
KV_CACHE_MODE="${KV_CACHE_MODE:-fake}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
RESULT_ROOT="${RESULT_ROOT:-results/long_context}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
TRUST_REMOTE_CODE_FLAG="${TRUST_REMOTE_CODE_FLAG:-}"

mkdir -p "$RESULT_ROOT/raw"

echo "profiles: $PROFILES"
echo "contexts: $CONTEXTS"
echo "baseline precisions: $BASELINE_PRECISIONS"
echo "oscar precisions: $OSCAR_PRECISIONS"
echo "kv cache mode: $KV_CACHE_MODE"
echo "result root: $RESULT_ROOT"

for profile in $PROFILES; do
  for ctx in $CONTEXTS; do
    for precision in $BASELINE_PRECISIONS; do
      out="$RESULT_ROOT/raw/${profile}_baseline_${precision}_${ctx}_${MAX_NEW_TOKENS}.json"
      echo "running baseline profile=$profile precision=$precision ctx=$ctx"
      python scripts/run_long_context_case.py \
        --profile "$profile" \
        --run-type baseline \
        --precision "$precision" \
        --context-target "$ctx" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --device-map "$DEVICE_MAP" \
        --attn-implementation "$ATTN_IMPLEMENTATION" \
        --output-json "$out" \
        $TRUST_REMOTE_CODE_FLAG
    done

    for precision in $OSCAR_PRECISIONS; do
      out="$RESULT_ROOT/raw/${profile}_oscar_${precision}_${KV_CACHE_MODE}_${ctx}_${MAX_NEW_TOKENS}.json"
      echo "running oscar profile=$profile precision=$precision kv_cache_mode=$KV_CACHE_MODE ctx=$ctx"
      python scripts/run_long_context_case.py \
        --profile "$profile" \
        --run-type oscar \
        --precision "$precision" \
        --kv-cache-mode "$KV_CACHE_MODE" \
        --context-target "$ctx" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --device-map "$DEVICE_MAP" \
        --attn-implementation "$ATTN_IMPLEMENTATION" \
        --output-json "$out" \
        $TRUST_REMOTE_CODE_FLAG
    done
  done
done

python scripts/summarize_long_context_results.py \
  --raw-dir "$RESULT_ROOT/raw" \
  --output-csv "$RESULT_ROOT/summary.csv"

echo "summary: $RESULT_ROOT/summary.csv"
