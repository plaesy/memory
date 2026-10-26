# API

## Library (`plaesy::memory`)

```rust
use plaesy::memory::prelude::*;
use plaesy::memory::config::MemoryConfig;

let config = MemoryConfig::load().await?;
let engine = SearchEngine::from_config(&config).await?;   // keyword-only if no model is installed
engine.set_watch_root(config.watch_dir.clone());          // lets stats() report coverage_percent
engine.index_directory(".plaesy".as_ref()).await?;       // new/changed in, deleted out, HNSW snapshot saved

for r in engine.search("memory architecture", 10).await? {
    println!("{}", CitationBuilder::new().build(&r)?);     // path:lines [FACT|INFERRED] indexed <ts> @ <rev>
}
```

| Module | Main types |
|--------|-----------|
| `indexer` | `Extractor`, `Chunker`, `Watcher`, `Document`, `Chunk`, `scan_dir`, `content_hash` |
| `embeddings` | `EmbeddingEngine` (`new`/`with_config` = ONNX, `hashing(dim)` = lexical opt-in), `cosine_similarity` |
| `storage` | `Database` (SQLite + FTS5), `HnswIndex` (usearch), `SearchResult`, `ConfidenceLevel` |
| `search` | `SearchEngine` (`search`, `search_with(mode, filter)`), `SearchFilter`, `SearchMode`, `Ranker` (`Weighted` default, `RRF`), `CitationBuilder`, `Citation` |
| `hygiene` | `SearchEngine::hygiene` -> `HygieneReport`; `SearchEngine::links` (wikilinks / backlinks) |
| `mcp` | Model Context Protocol server, see [MCP.md](MCP.md) |
| `config` | `MemoryConfig::{load, load_from_file, apply_env, validate}` |
| `metrics` | `Metrics` (+ `render_prometheus`), `Timer` |
| `service` | line-delimited JSON protocol (`serve`, `handle_request`, `SearchHit`) |

### Metadata-aware retrieval

```rust
let hits = engine
    .search_with(
        "deploy ritual",
        10,
        SearchMode::Hybrid,
        &SearchFilter {
            tiers: vec!["semantic".into(), "procedural".into()],
            min_trust: Some(Trust::Medium),
            ..SearchFilter::default()
        },
    )
    .await?;
```

`search_expanded(query, limit, mode, filter)` is `search_with` plus one wikilink hop: the `limit` direct hits come first,
then the best passage of each note that the top hits link to or are linked from (at most 5). Linked passages score 0.8×
their own score and keep their own evidence label.
Each `SearchResult` carries them in `meta` and each `Citation` in `source`. Rules:

- Filters run on the candidate lists *before* fusion/truncation (candidates are over-sampled 4x when a filter is set).
- `status: superseded|deprecated|archived|obsolete` is hidden unless `include_inactive`.
- A source with `last_verified` older than `stale_after_days` (default 180; episodic and archive never go stale) is flagged
  `stale` and its score multiplied by `stale_penalty` (default 0.85). It is down-weighted, not hidden.
- `min_trust` excludes sources that declare nothing: unknown is not "high".
- Tier is the frontmatter `tier`, else inferred from a directory name (`10-episodic/`, `knowledge/`, ...).

### Ranking

`Weighted` (default): `score = (0.7·cosine + 0.3·bm25_strength) / Σ weights of non-empty legs`, with
`bm25_strength = s/(s+8)` (absolute, not relative to the best hit). On the bundled labeled set RRF lost to
semantic-only on paraphrase queries because rank-only fusion discards score magnitude; weighted fusion did not.
`RRF` remains available via `RankerConfig { ranker_type: RankerType::RRF, .. }`.

### Errors

Everything returns `plaesy::memory::Result<T>` (`MemoryError`, `thiserror`). Library code does not use `unwrap`.
A failing semantic leg is non-fatal for hybrid search: it is counted in metrics and keyword results are returned.

## Plugin protocol (stdin/stdout, one JSON object per line)

```jsonc
→ {"id":1,"method":"ping"}
← {"id":1,"ok":true,"result":{"pong":true,"vector_search":true}}

→ {"id":2,"method":"search","params":{"query":"evidence labels","limit":5,"mode":"hybrid"}}   // mode: hybrid|semantic|keyword
← {"id":2,"ok":true,"result":{"hits":[{"chunk_id":12,"text":"…verbatim…","score":0.81,"confidence":"FACT",
     "citation":{"chunk_id":12,"file_path":"instructions/x.md","revision":"576b7b5…","line_range":[40,52],
                 "timestamp":"2026-10-18T17:25:19Z","confidence":"FACT",
                 "source":{"tier":"semantic","trust":"high","last_verified":"2026-10-01","status":null,"stale":false}}}]}}

// optional search params: "tier": "semantic" | ["semantic","procedural"], "path_prefix": "20-semantic/",
//                         "min_trust": "high"|"medium"|"low"|0.9, "include_inactive": true,
//                         "as_of": "2026-06-15"   (keeps sources whose valid_from..valid_to contains the date; undated pass),
//                         "expand_links": true    (adds one-hop linked passages after the direct hits)
→ {"id":3,"method":"index","params":{"path":".plaesy"}}      // sync a directory
→ {"id":3,"method":"backlinks","params":{"path":"runbook"}}  // or "links" for outgoing; result {"links":[{source_path,target,line,resolved_path}]}
→ {"id":3,"method":"hygiene","params":{"root":".plaesy"}}    // root optional; enables dead-path checks
→ {"id":4,"method":"stats"}   → {"id":5,"method":"metrics"}  // metrics: Prometheus text
// stats includes coverage_percent once the engine knows its watch root (the plugin sets it at startup)

← {"id":9,"ok":false,"error":{"kind":"bad_request","message":"…"}}   // kinds: bad_request, config, feature_not_enabled, timeout, io, internal
```

Malformed lines yield an error response (`id: null`) and the session continues. Logs go to stderr.
