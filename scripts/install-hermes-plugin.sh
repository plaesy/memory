#!/usr/bin/env bash
# Install the Plaesy Memory provider into a Hermes home.
#
# Hermes discovers memory providers one level deep: $HERMES_HOME/plugins/<name>/ (a nested
# plugins/plaesy/memory/ is NOT scanned for memory providers), hence the flat name "plaesy-memory".
# HERMES_HOME: $HERMES_HOME, else %LOCALAPPDATA%/hermes (Windows), else ~/.hermes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXE=""; case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) EXE=".exe" ;; esac
BIN="$ROOT/target/release/plaesy-memory-plugin$EXE"
[ -f "$BIN" ] || { echo "missing $BIN - run: cargo build --release" >&2; exit 1; }

if [ -n "${HERMES_HOME:-}" ]; then HOME_DIR="$HERMES_HOME"
elif [ -n "${LOCALAPPDATA:-}" ] && [ -d "$LOCALAPPDATA/hermes" ]; then HOME_DIR="$LOCALAPPDATA/hermes"
else HOME_DIR="$HOME/.hermes"; fi

DEST="$HOME_DIR/plugins/plaesy-memory"
mkdir -p "$DEST/bin"
cp "$ROOT/hermes-plugin/plaesy-memory/plugin.yaml" "$ROOT/hermes-plugin/plaesy-memory/__init__.py" "$DEST/"
cp "$BIN" "$DEST/bin/"
echo "Installed to $DEST"
echo "Next:  make model        (one-time ONNX model download; without it search is keyword-only)"
echo "       hermes config set memory.provider plaesy-memory"
