#!/usr/bin/env bash
# Download ONNX models for the engine (one-time, then fully offline).
#
#   install-model.sh             embedding model  all-MiniLM-L6-v2          (~90 MB)
#   install-model.sh --rerank    + cross-encoder  ms-marco-MiniLM-L-6-v2    (~90 MB, optional reranking)
#
# Destinations (same rule as src/memory/paths.rs):
#   Linux    $XDG_CACHE_HOME/plaesy/models   (default ~/.cache/plaesy/models)
#   macOS    ~/Library/Caches/plaesy/models
#   Windows  %LOCALAPPDATA%/plaesy/models    (Git Bash: $LOCALAPPDATA)
# Override per model with $PLAESY_MODEL_DIR / $PLAESY_RERANK_MODEL_DIR.
set -euo pipefail

fetch() { # dest_dir file url
  if [ -s "$1/$2" ]; then echo "ok   $2 (already present in $1)"; return; fi
  echo "get  $2"
  curl -fSL --retry 3 -o "$1/$2.part" "$3"
  mv "$1/$2.part" "$1/$2"
}

models_root() {
  case "$(uname -s)" in
    Darwin) echo "$HOME/Library/Caches/plaesy/models" ;;
    MINGW*|MSYS*|CYGWIN*)
      local base="${LOCALAPPDATA:?LOCALAPPDATA is not set}"
      if command -v cygpath >/dev/null 2>&1; then base="$(cygpath -u "$base")"; fi
      echo "$base/plaesy/models" ;;
    *) echo "${XDG_CACHE_HOME:-$HOME/.cache}/plaesy/models" ;;
  esac
}

ROOT="$(models_root)"

EMB="${PLAESY_MODEL_DIR:-$ROOT/all-MiniLM-L6-v2}"
mkdir -p "$EMB"
BASE="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
fetch "$EMB" model.onnx "$BASE/onnx/model.onnx"
fetch "$EMB" vocab.txt  "$BASE/vocab.txt"
echo "Embedding model installed in $EMB"

if [ "${1:-}" = "--rerank" ]; then
  CE="${PLAESY_RERANK_MODEL_DIR:-$ROOT/ms-marco-MiniLM-L-6-v2}"
  mkdir -p "$CE"
  CB="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main"
  fetch "$CE" model.onnx "$CB/onnx/model.onnx"
  fetch "$CE" vocab.txt  "$CB/vocab.txt"
  echo "Cross-encoder installed in $CE (enable with [search] rerank = true)"
fi
