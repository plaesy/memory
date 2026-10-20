# plaesy-memory

Offline-first memory engine for AI agents: hybrid semantic + keyword retrieval over Markdown notes, where
**every result cites its source** and says how much the source itself can be trusted.

- Crate `plaesy`, module `plaesy::memory` (`use plaesy::memory::prelude::*;`)
- `plaesy-memory-cli` (index / search / links / hygiene / stats / watch) and `plaesy-memory-plugin` (JSON service, or MCP with `--mcp`)
- Hermes memory provider: `hermes-plugin/plaesy-memory/` (pins core notes into the prompt, per-turn recall)
- Companion repo [`plaesy/brain`](../brain): the note vault template (tiers, templates, hygiene gate)

## How it works

```
 notes/*.md ─▶ Extractor (YAML frontmatter + Markdown) ─▶ Chunker (heading+paragraph / code / quote,
                                                           exact 1-based line ranges, verbatim text)
            ─▶ Embeddings (ONNX all-MiniLM-L6-v2, 384-d) ─▶ SQLite (documents, chunks, FTS5/BM25, links)
                                                         └▶ HNSW (usearch, cosine; snapshot on disk)
 query ─▶ filters (tier, path, trust, status) ─▶ semantic leg + keyword leg ─▶ staleness penalty
       ─▶ weighted fusion ─▶ [optional cross-encoder rerank] ─▶ results + citations
```

Evidence rules, enforced in code and tests:

| Rule | Behaviour |
|------|-----------|
| EV-01 | Every result carries file path, line range, git revision (if any) and index timestamp; `CitationBuilder::build` rejects results it cannot cite |
| EV-02 | `FACT` = lexical match in the source text, `INFERRED` = found by embedding similarity only |
| EV-03 | Result text is a verbatim slice of the source file, frontmatter chunks included (a unit test re-slices the file and compares) |
| NN-01 | No network at runtime, no API keys; models are downloaded once by `make model` |

## What a source can say about itself

Notes may declare `tier`, `confidence`, `last_verified`, `status` in frontmatter (full spec in the `brain` repo's `docs/FORMAT.md`):

- **Tiers** (`core | episodic | semantic | procedural | archive`) filter retrieval; tier is inferred from the folder if omitted.
- **Staleness**: `last_verified` older than 180 days (configurable) flags the result `STALE` and multiplies its score by 0.85.
  It is down-weighted, never hidden. Episodic logs and archives do not go stale.
- **Status**: `superseded | deprecated | archived` notes are hidden unless `include_inactive`.
- **Trust filter**: `min_trust: high` keeps only sources that declare high confidence; undeclared is not "high".
- **Wikilinks**: `[[note]]` links are indexed (code fences, inline code and template directives ignored) for backlinks and dangling-link detection.
- **Hygiene**: `plaesy-memory-cli hygiene --strict` reports stale notes, missing metadata, broken links, superseded-without-replacement, and backticked paths that no longer exist (the last is a heuristic, labelled INFERRED). Read-only.

## Quick start

```bash
cargo build --release
make model                               # one-time ONNX download (~90 MB); `scripts/install-model.sh --rerank` adds the cross-encoder
target/release/plaesy-memory-cli index notes
target/release/plaesy-memory-cli search "how do we deploy" --tier procedural --min-trust medium
target/release/plaesy-memory-cli links runbook          # who links to it
target/release/plaesy-memory-cli hygiene notes --strict
```

```
1. score 0.95  roles/researcher.md:58-60 [FACT] indexed 2026-10-18T17:25:19Z @ 576b7b5e… (tier semantic, trust high, verified 2026-10-01)
     - **Search & retrieval**: Web search for broad discovery; …
```

No model installed ⇒ keyword-only (a warning is logged; fake embeddings are never substituted silently).

## Integrations

```bash
# any MCP client (Claude Code / Desktop, Cursor, Hermes): read-only tools, see docs/MCP.md
plaesy-memory-plugin --mcp

# Hermes memory provider
make install-hermes-plugin && hermes config set memory.provider plaesy-memory
```

Hermes discovers memory providers one directory level under `plugins/`, hence `plugins/plaesy-memory/`, not `plugins/plaesy/memory/`.
Details: [docs/SETUP.md](docs/SETUP.md), [docs/MCP.md](docs/MCP.md), [docs/API.md](docs/API.md).

## Quality evidence

Everything below was measured on one machine (AMD Ryzen 7 5700U, CPU only) by this repository's own tests and benches.

**LongMemEval-S retrieval** ([dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), session level; per question, a fresh index of its ~48 haystack sessions; abstention questions skipped).
Subset: every 5th of the 470 answerable questions = **94 questions**, not the full set (a full run is ~5 h on this CPU). Reproduce: `make lme DATA=longmemeval_s_cleaned.json STRIDE=5`.

| Mode | R@1 | R@3 | R@5 | R@10 | MRR |
|------|-----|-----|-----|------|-----|
| keyword (FTS5/BM25) | 0.798 | 0.904 | 0.936 | 0.979 | 0.861 |
| semantic (MiniLM) | 0.840 | 0.936 | 0.957 | 0.979 | 0.889 |
| **hybrid (default)** | 0.840 | 0.926 | **0.968** | 0.979 | 0.891 |
| hybrid + cross-encoder, weight 0.5 | 0.840 | 0.936 | 0.979 | 0.979 | 0.897 |
| hybrid + cross-encoder, weight 0.7 | 0.862 | 0.936 | 0.968 | 0.979 | 0.908 |

Read this with care: with n = 94, one question is 1.1 points and the 95% interval on R@5 is roughly ±3.5, so hybrid vs semantic and
the reranker gains are **not statistically distinguishable**. Competitors' published figures (e.g. agentmemory BM25+MiniLM 95.2% R@5,
MemPalace 96.6% R@5) are self-reported on the full set with other settings; this table shows we are in the same range, not that we are ahead.
The weak spot is `single-session-preference` (R@5 0.83 hybrid, n = 6). Reranking is therefore **off by default**: it costs one
forward pass per candidate for no demonstrable gain at this sample size.

**Indonesian probe** (`scripts/id_probe.sh all-MiniLM-L6-v2 multilingual-e5-small`; notes and queries in `docs/eval/id-probe/`).
Eight notes written for this probe, eight paraphrased Indonesian queries labelled by the author. Top-1 / top-3 hit counts:

| Model | semantic | hybrid |
|-------|----------|--------|
| all-MiniLM-L6-v2 (default) | 4/8 · 6/8 | 6/8 · 7/8 |
| multilingual-e5-small (`MEMORY_MODEL_NAME`) | 7/8 · 8/8 | 6/8 · 8/8 |

This is a probe, not a benchmark: with n = 8 one query is 12.5 points, so treat the gap as a direction, not a result. The
default stays MiniLM until e5 is checked on English notes (LongMemEval, which is not yet run for e5).

**Latency** (`scripts/latency_probe.py <plugin binary>`; one client, sequential, 200 requests per mode, 20-document vault, same machine as above):
hybrid p50 2.7 ms / p95 4.2 ms; hybrid with `expand_links` p50 3.0 ms / p95 4.2 ms; keyword p50 1.1 ms / p95 1.8 ms.
The stdio protocol serves one request at a time per process, so no concurrent-load figure is given.

**Other checks**

| What | Result |
|------|--------|
| Tests | 129 Rust tests (`cargo test`, all targets) + 7 Python shim tests (`make test-plugin`); real-ONNX tests run when models are installed. Counted 2026-10-20 |
| Lint | `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` clean |
| 20-doc/20-query labeled set (`make eval`) | keyword 1.00 / 0.70 (lexical / paraphrase R@3) · semantic 1.00 / 1.00 · hybrid 1.00 / 1.00 (saturated; a regression guard only) |
| Cold start, 4.6K chunks | 0.37 s with HNSW snapshot (was ~4 s) |
| Indexing `.plaesy/` (145 files, 4 646 chunks, ONNX) | ~2 min 10 s, embedding-bound |
| Criterion | HNSW top-10 search 0.18 ms / 2.8 ms / 3.9 ms at 1K / 10K / 50K random 384-d vectors; ONNX embed 11 ms single, 5.7 ms/text in a batch of 32; hybrid search 0.49 ms on 100 docs |

The design target "p50 < 1 ms on 100K vectors" is **not met** here (3.9 ms at 50K random vectors, a worst case for ANN; real-embedding latency unmeasured).

## Known limits

- **English-only default embedding model.** all-MiniLM-L6-v2 was trained on English. `multilingual-e5-small` is supported
  (`MEMORY_MODEL_NAME`, see SETUP) and did better on the Indonesian probe above, but it is not the default yet. On a 6-note Indonesian probe (`bagaimana merilis aplikasi`, ...)
  hybrid found the right note 5/6, semantic alone 4/6; the miss was a cross-vocabulary paraphrase (`password` vs `kata sandi`). Keyword matching
  carries non-English notes; a multilingual model needs a different tokenizer (SentencePiece) and is not implemented. Six queries is a probe, not a benchmark.
- The tokenizer lower-cases but does not strip accents.
- Only `.md`, `.markdown`, `.txt` are indexed; invalid YAML frontmatter falls back to plain text.
- One SQLite connection behind a mutex: fine for one local agent, not for many concurrent readers.
- `read_only` and `limit` in core notes are conventions the Hermes provider honours; the engine has no write tools, so nothing enforces them elsewhere.
- Staleness is based on declared `last_verified`; the engine cannot know whether a claim is still true.
- `ort` is pinned to a release candidate (`2.0.0-rc.14`).
- The MCP server and Hermes provider were exercised with scripted clients and Hermes' own `MemoryProvider` class, not inside a live Claude/Cursor/Hermes session.

## Development

```bash
make verify        # build + test + python shim tests + clippy + fmt-check
make eval          # relevance table
make lme DATA=...  # LongMemEval-S
make bench         # criterion
```

Docs: [ARCHITECTURE](docs/ARCHITECTURE.md) (how it works and why), [COMPARISON](docs/COMPARISON.md) (versus Mem0, Zep, Letta, Basic Memory, ...),
[BEST_PRACTICES](BEST_PRACTICES.md), [SETUP](docs/SETUP.md), [API](docs/API.md), [MCP](docs/MCP.md), [CONTRIBUTING](CONTRIBUTING.md), [CHANGELOG](CHANGELOG.md). MIT licensed.
