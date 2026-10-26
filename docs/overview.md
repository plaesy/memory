# plaesy-memory: Offline-First Memory Engine

**Purpose**: Offline-first semantic memory indexing for AI agents. Indexes Markdown notes with hybrid retrieval (semantic + keyword), evidence-based citations, and metadata-aware filtering.

**Target users**: AI agents (Hermes, Claude Code, Cursor), developers building knowledge systems, teams managing research notes or runbooks.

**Core features**:

- **Hybrid retrieval**: ONNX embeddings (all-MiniLM-L6-v2, 384-dim) + SQLite FTS5/BM25 keyword search
- **Evidence-based results**: Every hit carries file path, line range, git revision, confidence label (FACT/INFERRED), and staleness flag
- **Metadata filtering**: Tier (core/episodic/semantic/procedural/archive), trust level, validity dates, status (active/superseded/deprecated)
- **Wikilink indexing**: `[[backlink]]` discovery and dangling-link detection for vault hygiene
- **No network at runtime**: All models downloaded once; no API calls, no cloud dependencies
- **Watch mode**: Debounced file-system monitoring for incremental index updates
- **Multiple protocols**: Library (Rust), line-delimited JSON (Hermes shim), MCP server (Claude Code, Cursor, Kilo Code)

**Main technologies**:

- **Async runtime**: Tokio (1.40)
- **Vector search**: HNSW via usearch (Qdrant engine, 2.15)
- **Embeddings**: ONNX Runtime (2.0.0-rc.14) + all-MiniLM-L6-v2 or multilingual alternatives
- **Full-text search**: SQLite (bundled) with FTS5 and BM25 ranking
- **Storage**: SQLite WAL (write-ahead log) + crash recovery
- **Observability**: tracing + Prometheus-compatible metrics

**Stack summary**: Rust (edition 2021), single-threaded search (one async mutex over SQLite), CPU-only by default (no CUDA/GPU required).

---

## System Flow

**Indexing**: notes → Extract → Chunk → ONNX embed → SQLite + HNSW
**Search**: Query → (Keyword: FTS5/BM25 + Semantic: ONNX/HNSW) → Rank → Metadata filter → Results with citations

---

## Quick Links

- **Installation**: `scripts/install.ps1` (Windows) or `make model && cargo build --release` (Linux/macOS)
- **Quick start**: `memory index notes` → `memory search "your question"`
- **Integration**: `memory serve --mcp` for Claude Code / Cursor / Kilo Code (MCP over stdio or HTTP)
- **Learning**: [architecture.md](architecture.md) (design rationale), [api.md](api.md) (library + protocol reference)
- **Vault setup**: Companion repo [plaesy/brain](../brain) (vault template, hygiene rules, note format)

---

## Retrieval Pipeline

```text
Query
  ├─ Keyword Leg (FTS5/BM25)       → Keyword Scores (30%)
  └─ Semantic Leg (ONNX + HNSW)    → Vector Scores (70%)
       ↓
    Weighted Fusion (70% semantic + 30% keyword)
       ↓
    Metadata Filter (tier, trust, staleness)
       ↓
    Citations (path:line:revision [FACT|INFERRED])
```

**Ranking** (default): 70% cosine similarity + 30% BM25 keyword strength, inverse-weighted by staleness and filters. The `RRF` (reciprocal rank fusion) algorithm remains available as an alternative.

**Evidence rules**:

- **FACT**: Lexical match in source text (keyword hit)
- **INFERRED**: Found by embedding similarity only (semantic match)
- **STALE**: Last verified > 180 days ago; score multiplied by 0.85, not hidden

---

## Design Principles

1. **No unwrap() in production code** – All error paths explicit; failing embeddings don't crash search
2. **Strong typing** – Newtype patterns, enums over strings, compile-time guarantees
3. **Single-user, offline** – One SQLite connection behind a mutex (fine for local agents, not many concurrent readers)
4. **Deterministic latency** – No GC pauses; p50 ~2.7 ms on 20-document vault (hybrid mode)
5. **Verbatim citations** – Results are exact slices from source files with 1-based line numbers

---

## Project Status

**Version**: 0.1.0 (beta) · **License**: MIT · **Active development**: Yes

**Quality metrics**:

- **Tests**: 129 Rust tests (cargo test) + 7 Python shim tests
- **Lint**: `cargo clippy --all-targets` and `cargo fmt --check` pass
- **Retrieval**: LongMemEval-S (94 questions) hybrid R@5 0.968; keyword 0.936; semantic 0.957
- **Cold start**: 0.37 s at 4.6K chunks (with HNSW snapshot cache)
- **Latency**: p50 2.7 ms / p95 4.2 ms hybrid search on 20-document vault

**Known limits**:

- English-default embedding model (all-MiniLM-L6-v2); multilingual E5 available as alternative
- One SQLite connection (OK for one local agent, not concurrent readers)
- `.md` / `.markdown` / `.txt` files only
- Staleness based on declared `last_verified` date (engine cannot verify truth)
