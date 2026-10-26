#!/usr/bin/env bash
# Top-1 / top-3 recall on a small, hand-labelled Indonesian probe (docs/eval/id-probe).
# Not a benchmark: 8 queries, labelled by the author. One database per model.
#   scripts/id_probe.sh all-MiniLM-L6-v2 multilingual-e5-small
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$ROOT/target/release/memory"; [ -x "$CLI.exe" ] && CLI="$CLI.exe"
PROBE="$ROOT/docs/eval/id-probe"
OUT="${TMPDIR:-/tmp}/id-probe"; mkdir -p "$OUT"
for M in "$@"; do
  DB="$OUT/$M.db"; rm -f "$DB" "$DB.hnsw"*
  export MEMORY_DB_PATH="$DB" MEMORY_WATCH_DIR="$PROBE/notes" MEMORY_MODEL_NAME="$M"
  "$CLI" index "$PROBE/notes" >/dev/null 2>&1
  for MODE in semantic hybrid; do
    t1=0; t3=0; n=0
    while IFS=$'\t' read -r q exp; do
      n=$((n+1))
      paths=$("$CLI" search "$q" -n 3 --mode "$MODE" 2>/dev/null | grep '^[0-9]' | sed 's/^[0-9]*\. score [0-9.]*  //' | cut -d: -f1)
      rank=$(printf '%s\n' "$paths" | grep -n "^$exp$" | head -1 | cut -d: -f1 || true)
      [ "${rank:-}" = "1" ] && t1=$((t1+1))
      [ -n "${rank:-}" ] && t3=$((t3+1))
    done < "$PROBE/queries.tsv"
    echo "$M [$MODE] top1=$t1/$n top3=$t3/$n"
  done
done
