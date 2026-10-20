# Setup

## Requirements

- Rust 1.88+ (uses `slice::as_chunks`); on Windows the **MSVC** toolchain (`stable-x86_64-pc-windows-msvc`) with Visual Studio Build Tools
- ~3 GB disk for build artifacts; first build downloads ONNX Runtime binaries (`ort` `download-binaries`)
- Optional: Python (Hermes' own interpreter) for the provider shim

```bash
cargo build --release
cargo test                     # real-ONNX tests run only if the model is installed
```

## Embedding model (one-time)

```bash
make model                     # scripts/install-model.sh
```

Downloads `model.onnx` (fp32, ~90 MB) and `vocab.txt` from `sentence-transformers/all-MiniLM-L6-v2` into
`~/.plaesy-models/all-MiniLM-L6-v2/`. Lookup order at runtime: `embeddings.model_dir` (config),
`$PLAESY_MODEL_DIR`, `~/.plaesy-models/<model_name>`. After the download nothing touches the network.

No model ⇒ keyword-only mode (a warning is logged).

### Multilingual model (optional, for non-English notes)

The default `all-MiniLM-L6-v2` is English-only. `intfloat/multilingual-e5-small` (384 dimensions, same as the default) is
optional and works on Indonesian queries better in the probe we ran (see README, "Quality evidence"). Install it by hand:

```bash
mkdir -p ~/.plaesy-models/multilingual-e5-small
curl -L -o ~/.plaesy-models/multilingual-e5-small/model.onnx \
  https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx      # ~470 MB, fp32
curl -L -o ~/.plaesy-models/multilingual-e5-small/tokenizer.json \
  https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json       # ~17 MB
```

Select it with `MEMORY_MODEL_NAME=multilingual-e5-small`. E5 queries and passages get the `query: ` and `passage: `
prefixes automatically. Switching models re-embeds every document on the next sync (the index records the model it was
built with, and a database built before recording is treated as `all-MiniLM-L6-v2`). A warning is logged when that happens.
Keep one database per model if you want to compare them.

## Configuration

Layering (later wins): defaults → `~/.hermes/memory.toml` → `.plaesy/memory.toml` → environment.
Partial files are fine; only the keys you set are overridden.

```toml
watch_dir = ".plaesy"

[database]
db_path = "memory.db"        # HNSW snapshot is stored beside it as memory.db.hnsw (+ .meta.json, .gen)

[embeddings]
model_dir = "C:/models/all-MiniLM-L6-v2"
max_batch_size = 32
timeout_secs = 30

[search]
default_limit = 10
max_limit = 100
timeout_secs = 30
confidence_threshold = 0.0   # drop results scoring below this (0-1)
stale_after_days = 180       # last_verified older than this => flagged STALE (0 disables)
stale_penalty = 0.85         # score multiplier for stale sources

[logging]
level = "info"               # a bare level also silences ONNX Runtime's INFO chatter
```

Environment overrides: `MEMORY_WATCH_DIR`, `MEMORY_DB_PATH`, `MEMORY_MODEL_DIR`, `MEMORY_MODEL_NAME`, `MEMORY_SEARCH_DEFAULT_LIMIT`,
`MEMORY_SEARCH_MAX_LIMIT`, `MEMORY_SEARCH_TIMEOUT_SECS`, `MEMORY_ENABLE_VECTOR_SEARCH`,
`MEMORY_ENABLE_KEYWORD_SEARCH`, `MEMORY_LOG_LEVEL`, `MEMORY_LOG_JSON`, `MEMORY_LOG_FILE`. `RUST_LOG` overrides the log filter.

`MEMORY_EXCLUDE_DIRS` (also `search.exclude_dirs` in TOML) keeps files out of the index. It is a comma-separated list
of folder runs or file names, matched anywhere in the tree: `.plaesy,.claude,CLAUDE.md` excludes every `.plaesy` and
`.claude` folder and every file named `CLAUDE.md`. Excluded documents are removed from the index on the next sync.

## Hermes plugin

```bash
make install-hermes-plugin
hermes config set memory.provider plaesy-memory
```

This copies `hermes-plugin/plaesy-memory/{plugin.yaml,__init__.py}` and the release binary to
`$HERMES_HOME/plugins/plaesy-memory/` (`bin/plaesy-memory-plugin[.exe]`). `$HERMES_HOME` resolves to
`$HERMES_HOME`, else `%LOCALAPPDATA%\hermes` on Windows, else `~/.hermes`.

**Why not `plugins/plaesy/memory/`?** Hermes' general plugin manager accepts `<category>/<name>/` layouts, but the
memory-provider discovery (`plugins/memory/__init__.py`) iterates only the direct children of `plugins/`.
A nested directory would not be found as a memory provider, so the flat name `plaesy-memory` is used.

The provider:

- starts `plaesy-memory-plugin` with `MEMORY_DB_PATH` and `MEMORY_WATCH_DIR` taken from the environment when set.
  Defaults apply only when unset: `$HERMES_HOME/plaesy-memory/state.db` and `<agent workspace>/.plaesy`.
  For a shared vault, set both as user environment variables (see Configuration). Processes started before the change
  do not see them, so restart Hermes after setting them.
- the binary answers immediately and indexes/watches in the background (searches see what is indexed so far)
- exposes four tools: `plaesy_memory_search`, `plaesy_memory_backlinks`, `plaesy_memory_hygiene`, `plaesy_memory_stats`,
  and per-turn recall via `queue_prefetch` → `prefetch`
- if the binary cannot be started (for example the working directory does not exist, or the binary is missing), the
  provider logs `initialize failed` at DEBUG/WARNING level and every tool returns `provider not initialized`
- binary lookup: `$PLAESY_MEMORY_BIN`, `<plugin dir>/bin/`, then `PATH`

## MCP (any client)

`plaesy-memory-plugin --mcp` is a read-only MCP server (search with tier/trust filters, backlinks, hygiene, stats).
Client configuration for Claude Code, Claude Desktop, Cursor and Hermes: [MCP.md](MCP.md).

## Notes vault

The engine indexes any folder of Markdown. For a ready-made structure (core / episodic / semantic / procedural / archive,
templates, hygiene gate) use the companion `plaesy/brain` repository and point `MEMORY_WATCH_DIR` at its `memory/` folder.

## Benchmark

`make lme DATA=/path/longmemeval_s_cleaned.json STRIDE=5` runs the LongMemEval-S retrieval benchmark through the real
engine (see README, "Quality evidence").

## Upgrading an existing index

Schema v2 adds a `links` table and a `body_start` column. Opening an older database migrates it in place and marks every
document for re-indexing on the next sync (so links and line offsets are filled in). Embeddings are recomputed for re-indexed files.

A sync skips files whose content hash has not changed. A chunking change (for example the heading rule in ARCHITECTURE)
therefore does not reach existing documents on its own. To rebuild, stop every process that uses the database, back up
the `.db`, `.hnsw*` files, delete them, and run `index` again. The index is a cache; the notes are the source of truth.
There is no `--force` flag yet.

If the vault folder structure changes (for example a folder is renamed), run `index` once: documents at the old paths are
removed and the new ones added.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `dlltool.exe: program not found` | GNU Rust toolchain without MinGW — use the MSVC toolchain |
| Link error `nk_*` unresolved | `usearch` must be built with `default-features = false` (already set) |
| `vector search: off (keyword-only)` in `stats` | model not found — `make model` or set `PLAESY_MODEL_DIR` |
| First search after install returns few hits | initial index still running in the background; check `stats` |
| Hermes answers `provider not initialized` but `hermes memory status` says the provider is active | the binary was never started. Run `hermes chat -v -q "..." --oneshot` and look for `initialize failed` (for example `WinError 267`, a missing working directory) |
| Hermes search returns `No results.` for a known note | Hermes may be using the default database, because it was started before `MEMORY_DB_PATH` was set. Restart it, or check `stats` for the expected document count |
| Results show only a heading, or a document that no longer exists | the index predates a chunker change or a folder rename. Rebuild as described under "Upgrading an existing index" |
