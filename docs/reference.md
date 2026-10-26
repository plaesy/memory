# Reference

Complete API reference for library, CLI, and protocols.

## Architecture Overview

```text
┌─────────────────────────────────────────┐
│  Entry Points                           │
│  • CLI (memory search/index)            │
│  • Library (plaesy::memory)             │
│  • JSON Service (Hermes shim)           │
│  • MCP Server (Claude Code/Cursor)      │
└────────────────┬────────────────────────┘
                 │
         ┌───────▼────────┐
         │ SearchEngine   │
         └───┬────┬───────┘
             │    │
    ┌────────▼─┐ ┌┴─────────┐
    │ SQLite   │ │ HNSW     │
    │ (FTS5)   │ │(usearch) │
    └──────────┘ └──────────┘
         │
    ┌────▼──────────┐
    │ ONNX Runtime  │
    │ (Embeddings)  │
    └───────────────┘
```

## Library (`plaesy::memory`)

### Module Declarations

| Module | Purpose | Location |
|--------|---------|----------|
| `error` | Error types and `Result<T>` alias | `src/memory/error.rs:1` |
| `indexer` | Document extraction and chunking | `src/memory/indexer/mod.rs:1` |
| `embeddings` | ONNX embedding engine | `src/memory/embeddings/mod.rs:1` |
| `storage` | SQLite + HNSW persistence | `src/memory/storage/mod.rs:1` |
| `search` | Query engine and ranking | `src/memory/search/mod.rs:1` |
| `rerank` | Cross-encoder reranking (optional) | `src/memory/rerank.rs:1` |
| `hygiene` | Index health audit | `src/memory/hygiene.rs:1` |
| `config` | Settings and validation | `src/memory/config/mod.rs:1` |
| `metrics` | Prometheus-compatible metrics | `src/memory/metrics/mod.rs:1` |
| `service` | Hermes JSON protocol | `src/memory/service.rs:1` |
| `mcp` | MCP server tools | `src/memory/mcp.rs:1` |

### Prelude

```rust
use plaesy::memory::prelude::*;
use plaesy::memory::config::MemoryConfig;
use plaesy::memory::search::SearchEngine;
```

Imports: `Result`, `MemoryError`, common types.

### Core Types

#### `SearchEngine`

**Location**: `src/memory/search/mod.rs:50` (struct definition)

**Constructor**:

```rust
impl SearchEngine {
    pub async fn new() -> Result<Self>
    pub async fn from_config(config: &MemoryConfig) -> Result<Self>
    pub fn set_watch_root(&mut self, root: PathBuf)
}
```

**Search methods**:

```rust
pub async fn search(
    &self,
    query: &str,
    limit: usize,
) -> Result<Vec<SearchResult>>

pub async fn search_with(
    &self,
    query: &str,
    limit: usize,
    mode: SearchMode,
    filter: &SearchFilter,
) -> Result<Vec<SearchResult>>

pub async fn search_expanded(
    &self,
    query: &str,
    limit: usize,
    mode: SearchMode,
    filter: &SearchFilter,
) -> Result<Vec<SearchResult>>
```

Default search mode is Hybrid (70% cosine + 30% BM25 keyword strength).

**Indexing**:

```rust
pub async fn index_directory(&self, path: &Path) -> Result<()>
pub async fn index_file(&self, path: &Path) -> Result<()>
pub async fn apply_events(&self, events: Vec<FileEvent>) -> Result<()>
```

Change detection via SHA-256 content hash. Unchanged files skipped. Deleted files removed from index.

**Metadata**:

```rust
pub async fn stats(&self) -> Result<IndexStats>
pub async fn metrics(&self) -> Metrics
pub async fn hygiene(&self, root: Option<&Path>) -> Result<HygieneReport>
pub async fn links(&self, path: &str, direction: LinkDirection) -> Result<Vec<Link>>
```

---

#### `SearchResult`

**Location**: `src/memory/storage/mod.rs:150`

```rust
pub struct SearchResult {
    pub chunk_id: u64,
    pub text: String,           // verbatim slice from source file
    pub score: f32,             // 0.0..=1.0
    pub kind: ChunkKind,        // Heading, Paragraph, Code, Quote
    pub line_range: (u32, u32), // 1-based, inclusive
    pub confidence: Confidence, // FACT or INFERRED
    pub source: SourceMetadata, // tier, trust, last_verified, status, stale
    pub revision: Option<String>, // git commit hash (if available)
    pub indexed_at: DateTime<Utc>,
}
```

---

#### `Citation`

**Location**: `src/memory/search/mod.rs:200`

```rust
pub struct Citation {
    pub chunk_id: u64,
    pub file_path: PathBuf,
    pub line_range: (u32, u32),
    pub revision: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub confidence: Confidence, // FACT or INFERRED
    pub source: SourceMetadata,
}

pub struct CitationBuilder;

impl CitationBuilder {
    pub fn new() -> Self
    pub fn build(&self, result: &SearchResult) -> Result<String>
    // Output: "path:line-range [FACT|INFERRED] indexed <timestamp> @ <revision>"
}
```

---

#### `SearchFilter`

**Location**: `src/memory/search/mod.rs:280`

```rust
pub struct SearchFilter {
    pub tiers: Vec<String>,           // ["core", "procedural", …]
    pub min_trust: Option<Trust>,     // high, medium, low
    pub path_prefix: Option<String>,  // path must start with this
    pub include_inactive: bool,       // include superseded/deprecated
    pub valid_from: Option<NaiveDate>,
    pub valid_to: Option<NaiveDate>,
    pub as_of: Option<NaiveDate>,    // keeps sources whose valid_from..valid_to contains this date
}
```

Tier is inferred from directory name if not declared in frontmatter:

- `10-episodic/`, `20-semantic/`, etc. → tier
- `00. Inbox/` → core

---

#### `MemoryConfig`

**Location**: `src/memory/config/mod.rs:30`

```rust
pub struct MemoryConfig {
    pub notes_dir: PathBuf,
    pub db_path: PathBuf,
    pub model_dir: PathBuf,
    pub chunk_size: usize,
    pub enable_vector_search: bool,
    pub stale_after_days: u32,
    pub search: SearchConfig,
    pub ranker: RankerConfig,
}

impl MemoryConfig {
    pub async fn load() -> Result<Self>
    pub async fn load_from_file(path: &Path) -> Result<Self>
    pub fn apply_env(&mut self)
    pub fn validate(&self) -> Result<()>
}
```

Defaults per OS (Windows: `%LOCALAPPDATA%/Plaesy`, Linux: `~/.local/share/plaesy`). Environment variables override:

- `PLAESY_ROOT` — Settings root
- `PLAESY_NOTES_DIR`, `PLAESY_DB_PATH`, `PLAESY_MODEL_DIR`
- `MEMORY_EXCLUDE_DIRS` — Comma-separated paths to skip

---

#### `EmbeddingEngine`

**Location**: `src/memory/embeddings/mod.rs:80`

```rust
pub struct EmbeddingEngine;

impl EmbeddingEngine {
    pub async fn new(model_dir: &Path) -> Result<Self>
    pub async fn with_config(config: &EmbeddingConfig) -> Result<Self>

    pub async fn embed(&self, text: &str) -> Result<EmbeddingVector>
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingVector>>

    pub fn cosine_similarity(v1: &EmbeddingVector, v2: &EmbeddingVector) -> f32
}
```

Default model: `all-MiniLM-L6-v2` (384-dim). Alternative: set `MEMORY_MODEL_NAME=multilingual-e5-small`.

Output: 384-dimensional f32 vector, L2-normalized.

---

#### `Database`

**Location**: `src/memory/storage/mod.rs:60`

```rust
pub struct Database;

impl Database {
    pub async fn new(path: &Path) -> Result<Self>
    pub async fn open(path: &Path) -> Result<Self>

    pub fn get_document(&self, id: u64) -> Result<Option<Document>>
    pub fn list_chunks(&self, doc_id: u64) -> Result<Vec<Chunk>>
    pub fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<Chunk>>
    pub async fn index_document(&self, doc: Document) -> Result<u64>
    pub async fn delete_document(&self, path: &Path) -> Result<()>
}
```

Schema version 2 (documents, chunks, chunks_fts, links, index_metadata).

---

### Error Handling

All public functions return `Result<T> = std::result::Result<T, MemoryError>`.

```rust
pub enum MemoryError {
    EmbeddingError(String),
    IndexError(String),
    StorageError(String),
    SearchError(String),
    NotFound(String),
    InvalidConfig(String),
    Io(std::io::Error),
    Timeout,
    Internal(String),
}

impl std::error::Error for MemoryError { … }
```

Failing embeddings are **non-fatal** for hybrid search: keyword results are returned; failing recorded in metrics.

---

## CLI Commands

**Base**: `memory [--config PATH] [--db PATH] COMMAND`

### index

```bash
memory index [DIR]
```

Syncs directory into index. New/changed files in, deleted files out. Unchanged files skipped (SHA-256 hash).

Output: `Indexed X files, removed Y, failed Z` + detailed list of changes.

---

### search

```bash
memory search QUERY \
  [-n LIMIT]                                          # default 10
  [--mode hybrid|semantic|keyword]                    # default hybrid
  [--format text|json|md]                             # default text
  [--tier TIER] [--tier TIER2]                        # filter by tier
  [--path PREFIX]                                      # path prefix match
  [--min-trust high|medium|low]                       # filter by trust
  [--include-inactive]                                # show superseded/deprecated
  [--as-of YYYY-MM-DD]                                # date validity window
  [--expand-links]                                    # add one-hop linked passages
  [--show-legs]                                       # show per-leg scores
```

Output (text):

```text
1. score 0.95  path/to/file.md:40-52 [FACT] indexed 2026-10-18T17:25:19Z @ 576b7b5e…
   - **Search & retrieval**: Web search for broad discovery; …
```

JSON output: `[{chunk_id, text, score, confidence, citation: {file_path, line_range, …}}]`

Markdown output: Bulleted list with score prefix.

---

### links

```bash
memory links NOTE [--outgoing]
```

Show wikilinks referencing NOTE (incoming, default) or links from NOTE (outgoing).

Output: `[[target]]` references with file path and line number.

---

### hygiene

```bash
memory hygiene [DIR] [--strict] [--json]
```

Audit index for:

- Stale notes (last_verified > 180 days)
- Missing metadata (missing tier, confidence)
- Broken wikilinks
- Superseded without replacement

With `--strict`: fail on first error (exit 1). Default: warnings only.

---

### missing

```bash
memory missing [--all] [--resolve GAP_ID]
```

Show unanswered queries (recorded by agents), broken links, stale notes. Limit 5 per kind unless `--all`.

With `--resolve ID`: mark gap as resolved (writes to gap log).

---

### stats

```bash
memory stats
```

Index metadata: document count, chunk count, index size, last sync, coverage %, cold-start time.

---

### watch

```bash
memory watch [DIR]
```

Keep index current until Ctrl-C. Debounced file-system events (notify crate, 500 ms window).

---

### config

```bash
memory config                              # print effective configuration
memory config mcp [--app claude|n8n|dify]  # integration snippets for MCP clients
```

---

### serve

```bash
memory serve [--mcp]                                   # JSON service on stdin/stdout
memory serve --mcp --transport http --port 8080       # HTTP Streamable transport for n8n/Dify/CrewAI
```

See [API.md](API.md) for protocol reference.

---

### onboard

```bash
memory onboard [--vault PATH] [--yes] [--skip-model-download] [--skip-onboard]
```

First-run setup: ask for vault path, download embedding model (~90 MB), index notes, write settings to `%LOCALAPPDATA%/Plaesy/settings.json` (Windows) or `~/.local/share/plaesy` (Linux/macOS).

---

## MCP Server

**Tools** (read-only):

### memory_search

```json
{"method": "memory_search", "params": {"query": "str", "limit": 10, "mode": "hybrid|semantic|keyword", "tier": "str|[str]", "path_prefix": "str", "min_trust": "high|medium|low", "include_inactive": false, "expand_links": false}}
```

Returns array of hits with citations.

---

### memory_hygiene

```json
{"method": "memory_hygiene", "params": {"root": "str"}}
```

Audit report: stale, missing-metadata, broken-links arrays.

---

### memory_links

```json
{"method": "memory_links", "params": {"path": "str", "direction": "outgoing|incoming"}}
```

Wikilink graph: resolved paths and line numbers.

---

### memory_stats

```json
{"method": "memory_stats"}
```

Index size, coverage %, cold-start time, last sync.

---

### memory_remember

```json
{"method": "memory_remember", "params": {"note": "str", "tags": ["str"]}}
```

Writes to vault inbox (`00. Inbox/`). Write tools require `enable_write_tools: true` in config.

---

## JSON Service Protocol (Hermes)

**Format**: Line-delimited JSON. Request and response identified by `id`.

```json
→ {"id":1,"method":"ping"}
← {"id":1,"ok":true,"result":{"pong":true,"vector_search":true}}
```

**Methods**: `ping`, `search`, `index`, `backlinks`, `links`, `hygiene`, `stats`, `metrics`.

See [API.md](API.md) for full protocol specification.

---

## Type Aliases and Enums

### SearchMode

```rust
pub enum SearchMode {
    Hybrid,       // 70% cosine + 30% BM25 (default)
    Semantic,     // cosine only
    Keyword,      // BM25 only
}
```

### Confidence

```rust
pub enum Confidence {
    Fact,        // lexical match in source
    Inferred,    // embedding similarity only
}
```

### Trust

```rust
pub enum Trust {
    High,
    Medium,
    Low,
}
```

### ChunkKind

```rust
pub enum ChunkKind {
    Heading,
    Paragraph,
    Code,
    Quote,
}
```

### LinkDirection

```rust
pub enum LinkDirection {
    Outgoing,  // links from this note
    Incoming,  // links to this note
}
```

---

## Configuration Keys (TOML)

```toml
[memory]
notes_dir = "C:/_"
db_path = "%LOCALAPPDATA%/Plaesy/memory.db"
model_dir = "%LOCALAPPDATA%/plaesy/models"
chunk_size = 512
enable_vector_search = true
stale_after_days = 180

[search]
exclude_dirs = ["target", ".git", "node_modules"]

[ranker]
ranker_type = "weighted"  # or "rrf"
semantic_weight = 0.7
keyword_weight = 0.3
stale_penalty = 0.85
```

Environment variable overrides: `PLAESY_*`, `MEMORY_*`.

---

## Performance Baselines

**Machine**: AMD Ryzen 7 5700U, CPU only, 20-document vault, 4.6K chunks.

| Operation | p50 | p95 | Measurement |
|-----------|-----|-----|-------------|
| Hybrid search | 2.7 ms | 4.2 ms | sequential, 200 requests |
| Keyword search | 1.1 ms | 1.8 ms | FTS5/BM25 only |
| Embedding (cold) | 11 ms | — | single query, no cache |
| Embedding (batch) | 5.7 ms/text | — | batch of 32 |
| Cold start (4.6K chunks) | 0.37 s | — | with HNSW snapshot cache |
| Index rebuild (4.6K chunks) | ~4 s | — | without HNSW snapshot |
| Index (145 files, 4.6K chunks) | ~2 min 10 s | — | embedding-bound |

**LongMemEval-S** (94 questions, every 5th of 470):

| Mode | R@1 | R@3 | R@5 | MRR |
|------|-----|-----|-----|-----|
| Keyword | 0.798 | 0.904 | 0.936 | 0.861 |
| Semantic | 0.840 | 0.936 | 0.957 | 0.889 |
| Hybrid | 0.840 | 0.926 | 0.968 | 0.891 |

---

## Limits and Trade-offs

- **One SQLite connection** behind async mutex: single-user, local agents only. Not concurrent readers.
- **English-default model**: all-MiniLM-L6-v2. Multilingual E5 available but not default (no multilingual tokenizer yet).
- **No network at runtime**: model download once by `make model` or `memory onboard`.
- **No write-ahead logging for notes**: vault is the source of truth; index is rebuildable.
- **ONNX RC version**: 2.0.0-rc.14 (stable release pending).
