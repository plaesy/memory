# Architecture

This document describes what the code does today. The original design (Tantivy, gRPC/HTTP, Cargo feature tiers,
three "layers") is in git history (`git log -- docs/architecture.md`); where the implementation departs from it,
the reason is in [Decisions](#decisions).

## Scope

A local, single-user memory engine. It indexes a folder of Markdown notes and answers queries with verbatim passages,
each carrying a citation and what the source says about its own reliability. It does not generate text, call a network at
runtime, or write to the notes (the vault is the source of truth; the index is rebuildable).

## Data Flow

**Write Path** (Indexing):

```text
notes/*.md → Extractor → Chunker → EmbeddingEngine (ONNX)
                ↓
          Frontmatter + Body
              ↓
        Chunks (heading+paragraph+code+quote)
            (exact 1-based line ranges)
                ↓
          Database (SQLite)
          + HnswIndex (usearch)
```

**Read Path** (Search):

```text
query
  ├─ FTS5/BM25 (keyword leg)        → Keyword scores
  └─ ONNX + HNSW (semantic leg)     → Vector scores
       ↓
  Weighted Fusion (70% semantic, 30% keyword)
       ↓
  Attach Metadata (tier, trust, staleness)
       ↓
  Apply Filters (tiers, min_trust, date range)
       ↓
  Optional: CrossEncoder Reranking
       ↓
  SearchResult + Citation (path:line:revision)
```

### Modules (`src/memory/`)

| Module | Responsibility |
|--------|----------------|
| `indexer` | `Extractor` (YAML frontmatter + body, exact line offsets), `Chunker`, `Watcher` (debounced `notify` events), `scan_dir` |
| `embeddings` | `EmbeddingEngine`: ONNX Runtime, mean pooling, L2 norm, LRU cache. Tokenizer is a hand-written WordPiece (`vocab.txt`, default all-MiniLM-L6-v2) or the Hugging Face tokenizer (`tokenizer.json`, multilingual E5 via the `tokenizers` crate). E5 prefixes (`query: ` / `passage: `) are applied per model. Process-wide ONNX environment. `Hashing` backend is an explicit lexical-only opt-in for tests |
| `storage` | `Database` (single rusqlite connection behind a mutex; schema v2), `HnswIndex` (usearch), wikilink extraction/resolution |
| `search` | `SearchEngine` (owns the write path so SQLite and HNSW change together), `filter.rs` (tiers, trust, staleness), `Ranker`, `CitationBuilder` |
| `rerank` | Optional `CrossEncoder` (ms-marco-MiniLM-L-6-v2 ONNX) |
| `hygiene` | Read-only audit of the index |
| `service` | Line-delimited JSON protocol over stdio (Hermes shim) |
| `mcp` | MCP server over stdio (read-only tools) |
| `config`, `metrics`, `error` | TOML/env configuration, in-process counters with Prometheus text export, `MemoryError` (`thiserror`) |

Binary: `memory` (`src/bin/cli.rs`). Subcommands cover indexing, search, links, hygiene, stats and watch. `memory serve` runs the JSON service on stdin/stdout (used by the Hermes provider); `memory serve --mcp` runs the MCP server.

## Storage

SQLite (WAL) is the only durable store besides the notes themselves.

| Table | Contents |
|-------|----------|
| `documents` | path, content hash, git revision, frontmatter JSON, body, `body_start` (file line where the body begins), `indexed_at` |
| `chunks` | text, kind, start/end line (1-based, inclusive), embedding BLOB |
| `chunks_fts` | FTS5 external-content index over `chunks.text`, kept in sync by triggers |
| `links` | `[[wikilinks]]` per document (target text, file line); resolved at read time |
| `index_metadata` | `schema_version`, and `generation` |

`generation` is a counter bumped on every write that changes chunks or embeddings. The HNSW graph is saved as
`<db>.hnsw` with the generation it was built from. On start the snapshot is used only if the generation matches, the dimension
matches and the vector count equals the number of stored embeddings; otherwise the graph is rebuilt from SQLite. The snapshot
is a cache, never a source of truth. Measured: 0.37 s start versus about 4 s to rebuild at 4.6 K chunks.

Schema upgrades happen on open. v1 to v2 added `links` and `documents.body_start`, then nulls every document hash so the next
sync re-indexes them.

## Indexing rules

- Change detection is a SHA-256 of the file content; unchanged files are skipped, files gone from disk are removed.
- Line endings are normalised to `\n`; a BOM is dropped. Invalid YAML frontmatter does not fail the file: it is indexed as plain text.
- Chunks are verbatim slices of the file. A heading opens a run that takes the paragraphs, lists, code and quotes under it,
  and also a heading directly under another heading (so a bare title is never a chunk of its own). A run is cut when the next
  block would exceed `chunk_size`, which is why a long list can start a separate chunk. Long blocks split on line boundaries.
  Frontmatter is one chunk holding the raw block.
- Exclusion (`search.exclude_dirs`, `MEMORY_EXCLUDE_DIRS`) applies at every entry point: directory sync, watcher events,
  and the coverage figure. Excluded files are never extracted, and documents already indexed under an excluded path are
  removed on the next sync.
- A document whose embedding fails is still stored and stays keyword-searchable.
- Writers are serialised by one async mutex. Git revision is resolved once per directory sync.

## Retrieval

1. **Candidates.** Keyword: FTS5 BM25, stopwords removed, tokens quoted and OR-ed (so user input cannot inject FTS operators).
   Semantic: embed the query, HNSW top-k. A restrictive filter over-samples 4x.
2. **Refine.** Attach `ResultMeta` (tier, trust, `last_verified`, status, stale). A stale source's score is multiplied by `stale_penalty`.
   The filter drops results; `superseded|deprecated|archived|obsolete` are dropped unless `include_inactive`.
3. **Fuse.** Weighted (default): `(0.7·cosine + 0.3·bm25_strength) / Σ weights of non-empty legs`, with `bm25_strength = s/(s+8)`
   (absolute, so a lone weak keyword hit is not inflated to 1.0). RRF is selectable. Results are de-duplicated by chunk.
4. **Rerank (optional).** The top `rerank_top_n` are rescored: `(1-w)·fused + w·sigmoid(logit)`. Text and evidence label do not change.
5. **Truncate** to `limit`, apply `confidence_threshold`.

A failing semantic leg is non-fatal: keyword results are returned and the failure is counted.

### Evidence labels

`FACT`: the passage matched lexically, so the query terms occur in the source text. `INFERRED`: found by embedding similarity only.
The label describes how the passage was found, not whether its claim is true. What the source declares about itself
(`tier`, `trust`, `last_verified`, `status`, `stale`) travels separately in `Citation.source`.

## Constitution rules

| Rule | Where it is enforced |
|------|----------------------|
| EV-01 every result cites its source | `CitationBuilder::build` returns `CitationError` for empty path, invalid line range or missing timestamp; `SearchHit::from_result` uses it, so service/MCP/CLI cannot emit an uncited hit |
| EV-02 uncertainty labelled | `ConfidenceLevel` on every result; `Fact` assigned only on the keyword leg |
| EV-03 nothing fabricated | chunk text is a slice of the file; `test_chunk_line_numbers_are_exact` re-slices the source and compares, frontmatter included |
| NN-01 offline, no keys | no network code at runtime; models are fetched once by `scripts/install-model.sh` |
| NN-03 inferred never presented as fact | hits found only by vectors stay `INFERRED`; reranking does not change labels |

## Decisions

| Decision | Reason | Cost |
|----------|--------|------|
| FTS5 instead of Tantivy | one store, no second index to keep consistent, sufficient on the evaluations | no custom analyzers; no accent folding |
| stdio JSON and MCP instead of HTTP/gRPC | the engine is a child process of one agent; no port to secure | single client per process |
| Weighted fusion over RRF | RRF discards score magnitude: on the labeled set it lost to semantic-only (MRR 0.90 vs 1.00) | needs comparable score scales (hence the BM25 normalisation) |
| Staleness down-weights, never hides | an unverified note may still be the only source; hiding it is silent loss | stale-but-relevant notes can still appear, flagged |
| `min_trust` excludes undeclared trust | treating unknown as high would defeat the filter | notes without metadata vanish under that filter |
| Reranker off by default | one forward pass per candidate; gain not demonstrable at n = 94 | a possible small gain left unused |
| Index is rebuildable, notes are truth | files stay editable by humans and diffable in git | re-index cost after schema changes |
| One SQLite connection | simplest correct thing for one local agent | concurrent readers serialise |
| `usearch` without default features | the default SIMD backend fails to link on Windows MSVC | slower distance kernels than the default build |
| Crate `plaesy`, module `memory` | room for sibling Plaesy modules (`plaesy::memory::*`) | |
| Hermes provider at `plugins/plaesy-memory/` | memory-provider discovery scans one level only | |
| Undated sources pass an `as_of` filter | a note without `valid_from` / `valid_to` should not vanish under a date filter; a date filter only narrows dated sources | an undated note can answer a question about a period it did not cover; declare the dates to be exact |
| Linked passages appended after direct hits, not merged | the direct hits keep their `limit`; a linked passage is context, not a competitor for the top slots | a linked passage can be less relevant than the direct hits; it is labelled and scored lower |
| Index records the embedding model; a change re-embeds everything | vectors from different models are not comparable and have the same dimension, so nothing would fail; a silent mix would corrupt search | a model switch costs one full re-embed of the vault |
| Default stays all-MiniLM-L6-v2 although E5 did better on the probe | the probe is 8 queries; the English side is unmeasured for E5 | Indonesian queries keep the weaker default until E5 is measured on English notes too |
| Exclusion by folder run or file name, anywhere in the tree | a vault mixes notes with framework and template files (`.plaesy`, `.claude`, `CLAUDE.md`); the agent should see only the notes | a folder that should be indexed under a matching name is also excluded; the match is by name, not by path |
| `cwd` for the provider process only when the directory exists | on Windows a missing directory makes `CreateProcess` fail with WinError 267, and the provider then never starts | none observed |

## Observability

`Metrics` counts searches, errors by kind, vector-leg outcomes, and records indexing/search/embedding latency, exported as
Prometheus text through the `metrics` service method. Logging uses `tracing`; a bare level in config also silences ONNX Runtime
INFO output. Logs go to stderr so they never corrupt the stdio protocols.

## Testing

| Suite | What it proves |
|-------|----------------|
| unit (`src/**`) | chunk line exactness, FTS injection safety, snapshot reuse/invalidations, ranker maths, wikilink parsing, filters, MCP protocol |
| `tests/integration_test.rs` | index, search, citations and persistence together |
| `tests/metadata_test.rs` | tiers, trust, staleness, backlinks and hygiene on a small vault |
| `tests/relevance_eval.rs` | 20-document labeled set per mode (a saturated regression guard), rerank invariants |
| `examples/longmemeval.rs` | LongMemEval-S retrieval benchmark through the real engine |
| `hermes-plugin/tests` | provider helpers (core pinning, formatting) without Hermes; provider start with a missing workspace (needs the release binary, skipped otherwise) |
| `benches/` | criterion: HNSW, embedding, end-to-end |

ONNX-dependent tests skip with a message when the model is not installed.

## Known gaps

- No accent folding in the tokenizer; no Unicode normalisation.
- Graph retrieval is one wikilink hop (`search_expanded`); no entity extraction, multi-hop paths or relation types.
- Temporal validity is per source (`valid_from` / `valid_to`), not per fact: there are no intervals between facts and no
  "what changed" queries.
- Consolidation of episodes into facts is a human/agent workflow documented in the `brain` repo, not an engine feature.
- `ort` is a release candidate; `serde_yaml` is deprecated upstream.
- HNSW target of p50 < 1 ms at 100 K vectors is not met on the reference CPU (3.9 ms at 50 K random vectors).
- The default embedding model is English-only (all-MiniLM-L6-v2, WordPiece). `multilingual-e5-small` is supported and better on
  the Indonesian probe, but is not the default: it has not been measured on the English side.
- `stats` reports `coverage_percent` only when the watch root is known to the engine (set at startup by the CLI and the plugin).
- No `index --force`; a chunking change needs the manual rebuild described in SETUP.
