//! Search subsystem for query execution and ranking.
//!
//! Combines vector search, keyword search, and re-ranking:
//! - **Vector Search**: HNSW semantic similarity
//! - **Keyword Search**: FTS5 BM25 ranking
//! - **Hybrid Ranking**: Reciprocal Rank Fusion (RRF)
//! - **Citations**: Build structured citations for results (EV-01 compliance)
//! - **Indexing**: `SearchEngine` also owns the write path (extract → chunk →
//!   embed → store) so the SQLite and HNSW stores never diverge
//!
//! # Architecture
//!
//! ```text
//! Query: "memory architecture"
//!     │
//!     ├─▶ Embed query (ONNX) ──▶ Vector Search (HNSW)  ─┐
//!     │                                                  ├─▶ RRF merge ─▶ Top-N + citations
//!     └─▶ Tokenize query ──────▶ Keyword Search (FTS5) ─┘
//! ```
//!
//! If no embedding model is available the engine runs keyword-only; if the
//! vector leg fails at query time the keyword results are still returned.
//!
//! # Evidence labels
//!
//! Result text is always verbatim stored text (EV-03). Confidence is `Fact`
//! when the result matched lexically (FTS5), and `Inferred` when it was found
//! by embedding similarity alone (EV-02 / NN-03).
//!
//! # Example
//!
//! ```ignore
//! use plaesy::memory::config::MemoryConfig;
//! use plaesy::memory::search::{CitationBuilder, SearchEngine};
//!
//! let engine = SearchEngine::from_config(&MemoryConfig::default()).await?;
//! engine.index_directory(".plaesy".as_ref()).await?;
//! for r in engine.search("memory architecture", 10).await? {
//!     println!("{}", CitationBuilder::new().build(&r)?);
//! }
//! ```

mod filter;
pub use filter::{
    age_days, build_meta, infer_tier, normalize_tier, parse_date, SearchFilter, Trust, TIERS,
};

pub mod rerank;

use crate::memory::config::{MemoryConfig, SearchConfig};
use crate::memory::embeddings::{EmbeddingBackend, EmbeddingEngine, EmbeddingVector};
use crate::memory::error::{MemoryError, Result};
use crate::memory::search::rerank::{sigmoid, CrossEncoder};
use crate::memory::indexer::{
    is_excluded, is_indexable, scan_dir, Chunker, Document, EventKind, Extractor, FileEvent,
};
use crate::memory::metrics::{IndexHealth, Metrics, SearchType, Timer};
use crate::memory::storage::{
    ConfidenceLevel, Database, DatabaseStats, HnswIndex, ResultMeta, SearchResult,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// RRF damping constant (Cormack et al. 2009 use 60).
const RRF_K: f32 = 60.0;

/// Result of indexing one document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexOutcome {
    /// Document stored. `embedded` is false when no vectors were produced
    /// (no embedding engine, or embedding failed) — it is still keyword-searchable.
    Indexed {
        /// Number of chunks stored.
        chunks: usize,
        /// Whether vectors were stored for the chunks.
        embedded: bool,
    },

    /// Content hash matched the stored document; nothing was written.
    Unchanged,
}

/// Summary of a directory sync.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexSummary {
    /// Files (re)indexed.
    pub indexed: usize,
    /// Files skipped because their content hash was unchanged.
    pub unchanged: usize,
    /// Stored documents removed because the file no longer exists.
    pub removed: usize,
    /// Chunks written during this sync.
    pub chunks: usize,
    /// Files that failed: `(path, error message)`. Failures never abort the sync.
    pub failed: Vec<(String, String)>,
}

/// Top hits whose wikilinks are followed by `search_expanded`.
const LINK_SEED_LIMIT: usize = 3;
/// Linked notes examined per `search_expanded` call.
const LINK_NEIGHBOUR_LIMIT: usize = 5;
/// Candidates fetched per linked note (the note's own passages are few).
const LINK_NEIGHBOUR_SEARCH: usize = 20;
/// Score multiplier for passages reached through a link rather than matched directly.
const LINK_SCORE_FACTOR: f32 = 0.8;

/// Search engine orchestrating vector + keyword search, ranking, and indexing.
pub struct SearchEngine {
    db: Arc<Database>,
    embeddings: Option<Arc<EmbeddingEngine>>,
    hnsw: RwLock<HnswIndex>,
    ranker: Ranker,
    reranker: Option<Arc<CrossEncoder>>,
    extractor: Extractor,
    chunker: Chunker,
    config: SearchConfig,
    metrics: Arc<Metrics>,
    /// Serialises writers so SQLite and HNSW change together.
    write_lock: tokio::sync::Mutex<()>,
    /// Database generation captured by the last HNSW snapshot (`u64::MAX` = none).
    snapshot_generation: AtomicU64,
    /// Folder the index mirrors; set once at startup so `statistics` can report coverage.
    watch_root: std::sync::OnceLock<PathBuf>,
}

fn git_revision(dir: &Path) -> Option<String> {
    std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

impl SearchEngine {
    /// Create a search engine over an open database.
    ///
    /// The HNSW index is rebuilt from the embeddings stored in `db`, so SQLite
    /// is the single source of truth. `embeddings = None` gives keyword-only operation.
    ///
    /// # Errors
    ///
    /// - `HnswIndexCorruption` / `DimensionMismatch` if the index cannot be built from stored vectors
    pub async fn new(
        db: Arc<Database>,
        embeddings: Option<Arc<EmbeddingEngine>>,
        config: SearchConfig,
    ) -> Result<Self> {
        let dim = embeddings.as_ref().map(|e| e.dimension()).unwrap_or(384);
        // Vectors from a different model are not comparable: when the model changed, the stored
        // documents are queued for re-embedding and the HNSW snapshot (built from the old model)
        // is not reused.
        let model_changed = match &embeddings {
            Some(e) if e.backend() == EmbeddingBackend::Onnx => {
                let changed = db.record_embedding_model(e.model_name()).await?;
                if changed {
                    tracing::warn!(
                        model = e.model_name(),
                        "embedding model changed; documents are re-embedded on the next sync"
                    );
                }
                changed
            }
            _ => false,
        };
        let generation = db.generation().await?;
        let snapshot = if model_changed {
            None
        } else {
            Self::load_snapshot(&db, dim, generation).await
        };
        let (hnsw, snapshot_generation) = match snapshot {
            Some(h) => (h, generation),
            None => {
                let mut h = HnswIndex::with_dimensions(db.hnsw_config().clone(), dim)?;
                for (id, v) in db.all_embeddings().await? {
                    h.insert(id, &v)?;
                }
                (h, u64::MAX) // no valid snapshot: write one on the next save_snapshot()
            }
        };
        Ok(SearchEngine {
            db,
            embeddings,
            hnsw: RwLock::new(hnsw),
            ranker: Ranker::new(RankerConfig::default()),
            reranker: None,
            extractor: Extractor::new(),
            chunker: Chunker::new(),
            config,
            metrics: Arc::new(Metrics::new()),
            write_lock: tokio::sync::Mutex::new(()),
            snapshot_generation: AtomicU64::new(snapshot_generation),
            watch_root: std::sync::OnceLock::new(),
        })
    }

    fn snapshot_path(db: &Database) -> Option<PathBuf> {
        db.path().map(|p| {
            let mut s = p.as_os_str().to_owned();
            s.push(".hnsw");
            PathBuf::from(s)
        })
    }

    /// Load the on-disk HNSW snapshot if it is provably in sync with the database:
    /// same change generation, same dimension, and one vector per stored embedding.
    /// Any mismatch or error returns `None` and the caller rebuilds from SQLite.
    async fn load_snapshot(db: &Database, dim: usize, generation: u64) -> Option<HnswIndex> {
        let path = Self::snapshot_path(db)?;
        let gen_file = path.with_extension("hnsw.gen");
        let saved: u64 = std::fs::read_to_string(&gen_file)
            .ok()?
            .trim()
            .parse()
            .ok()?;
        if saved != generation {
            return None;
        }
        let index = HnswIndex::load(&path).ok()?;
        let stored = db.embedding_count().await.ok()?;
        (index.dimensions() == dim && index.len() as u64 == stored).then_some(index)
    }

    /// Persist the HNSW index next to the database so the next start skips the rebuild.
    /// No-op for in-memory databases or when nothing changed since the last snapshot.
    /// Failures are logged, never fatal (the snapshot is only a cache).
    pub async fn save_snapshot(&self) {
        let Some(path) = Self::snapshot_path(&self.db) else {
            return;
        };
        let _guard = self.write_lock.lock().await;
        let Ok(generation) = self.db.generation().await else {
            return;
        };
        if self.snapshot_generation.load(Ordering::Relaxed) == generation {
            return;
        }
        let saved = {
            let h = self.hnsw.read();
            h.save(&path)
        };
        match saved {
            Ok(()) => {
                // Written last: a crash before this leaves a stale generation => rebuild.
                if std::fs::write(path.with_extension("hnsw.gen"), generation.to_string()).is_ok() {
                    self.snapshot_generation
                        .store(generation, Ordering::Relaxed);
                }
            }
            Err(e) => tracing::warn!("could not save HNSW snapshot: {e}"),
        }
    }

    /// Build a search engine from configuration: opens the database and, if
    /// vector search is enabled, tries to load the embedding model. A missing
    /// model downgrades to keyword-only (logged as a warning), never an error.
    pub async fn from_config(config: &MemoryConfig) -> Result<Self> {
        config.validate()?;
        let mut db_cfg = config.database.clone();
        db_cfg.hnsw_config = config.database.hnsw_config.clone();
        let db = Arc::new(Database::new(db_cfg).await?);
        let embeddings = if config.search.enable_vector_search {
            match EmbeddingEngine::with_config(config.embeddings.clone()).await {
                Ok(e) => Some(Arc::new(e)),
                Err(e) => {
                    tracing::warn!("embeddings unavailable, running keyword-only: {e}");
                    None
                }
            }
        } else {
            None
        };
        let mut engine = Self::new(db, embeddings, config.search.clone()).await?;
        if config.search.rerank {
            let model = CrossEncoder::resolve_model_dir(config.search.rerank_model_dir.as_deref())
                .ok_or_else(|| "no cross-encoder model found".to_string())
                .and_then(|dir| CrossEncoder::load(&dir, 256).map_err(|e| e.to_string()));
            match model {
                Ok(m) => engine.reranker = Some(Arc::new(m)),
                Err(e) => {
                    tracing::warn!("reranking requested but unavailable, continuing without: {e}");
                }
            }
        }
        Ok(engine)
    }

    /// Attach a cross-encoder; hybrid searches then rerank their top candidates.
    pub fn with_reranker(mut self, reranker: Arc<CrossEncoder>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Whether hybrid search reranks with a cross-encoder.
    pub fn has_reranker(&self) -> bool {
        self.config.rerank && self.reranker.is_some()
    }

    /// Rescore the first `top_n` of `results` with the cross-encoder and re-sort them.
    ///
    /// Final score = `(1 - weight) * fused + weight * sigmoid(logit)`. Results beyond `top_n`
    /// keep their order after the reranked block. Text stays verbatim (EV-03); the evidence
    /// label is unchanged because reranking does not change *why* a passage matched.
    ///
    /// # Errors
    ///
    /// - `FeatureNotEnabled` if no cross-encoder is attached
    /// - `EmbeddingInferenceError` if scoring fails
    pub async fn rerank_results(
        &self,
        query: &str,
        mut results: Vec<SearchResult>,
        top_n: usize,
        weight: f32,
    ) -> Result<Vec<SearchResult>> {
        let ce = self
            .reranker
            .as_ref()
            .ok_or(MemoryError::FeatureNotEnabled {
                feature: "reranking".to_string(),
            })?;
        let n = top_n.min(results.len());
        if n < 2 {
            return Ok(results);
        }
        let w = weight.clamp(0.0, 1.0);
        let docs: Vec<String> = results[..n].iter().map(|r| r.text.clone()).collect();
        let logits = ce.score(query, docs).await?;
        for (r, l) in results[..n].iter_mut().zip(logits) {
            r.score = (1.0 - w) * r.score + w * sigmoid(l);
        }
        results[..n].sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.chunk_id.cmp(&b.chunk_id))
        });
        Ok(results)
    }

    /// Metrics collector for this engine.
    pub fn metrics(&self) -> Arc<Metrics> {
        Arc::clone(&self.metrics)
    }

    /// Whether the semantic leg is available.
    pub fn has_vector_search(&self) -> bool {
        self.config.enable_vector_search && self.embeddings.is_some()
    }

    /// Search settings in effect.
    pub fn search_config(&self) -> &SearchConfig {
        &self.config
    }

    /// Shared handle to the underlying database.
    pub fn database(&self) -> Arc<Database> {
        Arc::clone(&self.db)
    }

    // -- Indexing ----------------------------------------------------------

    /// Index (or re-index) a document, replacing any previous version.
    ///
    /// Embedding happens first; if it fails the document is still stored for
    /// keyword search and `embedded` is false.
    pub async fn index_document(&self, mut doc: Document) -> Result<IndexOutcome> {
        let timer = Timer::start();
        doc.source_path = PathBuf::from(doc.source_path.to_string_lossy().replace('\\', "/"));
        let path = doc.source_path.to_string_lossy().to_string();
        let mut chunks = self.chunker.chunk(&doc)?;
        if chunks.is_empty() {
            // Nothing indexable (e.g. only blank lines/frontmatter-less empty body).
            chunks = Vec::new();
        }

        let mut vectors: Option<Vec<EmbeddingVector>> = None;
        if let (Some(engine), true) = (&self.embeddings, self.config.enable_vector_search) {
            let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
            let t = Timer::start();
            match engine.embed_batch(&texts).await {
                Ok(v) => {
                    self.metrics.record_embedding_latency(t.elapsed());
                    self.metrics
                        .set_embedding_cache_hit_ratio(engine.cache_hit_ratio());
                    vectors = Some(v);
                }
                Err(e) => {
                    self.metrics.record_error("embedding_failed");
                    tracing::warn!("embedding failed for {path}, indexing keyword-only: {e}");
                }
            }
        }

        let _guard = self.write_lock.lock().await;
        if let Some((old_id, _)) = self.db.find_document(&path).await? {
            let old_chunks = self.db.chunk_ids_for_document(old_id).await?;
            let mut h = self.hnsw.write();
            for id in old_chunks {
                h.remove(id)?;
            }
        }
        let doc_id = self.db.insert_document(&doc).await?;
        let ids = self.db.insert_chunks(doc_id, &chunks).await?;
        if let Some(vecs) = &vectors {
            self.db.update_chunk_embeddings(&ids, vecs).await?;
            let mut h = self.hnsw.write();
            for (id, v) in ids.iter().zip(vecs) {
                h.insert(*id, v)?;
            }
        }
        self.metrics.record_document_indexed(chunks.len());
        self.metrics
            .record_indexing_latency(timer.elapsed(), "index_document");
        Ok(IndexOutcome::Indexed {
            chunks: chunks.len(),
            embedded: vectors.is_some(),
        })
    }

    /// Index one file. `source_path` is stored relative to `root` (forward slashes).
    /// Skips the write when the content hash is unchanged.
    pub async fn index_file(&self, root: &Path, path: &Path) -> Result<IndexOutcome> {
        self.index_file_at(root, path, git_revision(root)).await
    }

    /// Like [`SearchEngine::index_file`] with a caller-supplied git revision
    /// (spawning `git` per file is slow, so directory syncs resolve it once).
    async fn index_file_at(
        &self,
        root: &Path,
        path: &Path,
        revision: Option<String>,
    ) -> Result<IndexOutcome> {
        let mut doc = self.extractor.extract_file(path)?;
        let rel = path.strip_prefix(root).unwrap_or(path);
        doc.source_path = PathBuf::from(rel.to_string_lossy().replace('\\', "/"));
        let key = doc.source_path.to_string_lossy().to_string();
        if let Some((_, Some(h))) = self.db.find_document(&key).await? {
            if Some(&h) == doc.hash.as_ref() {
                return Ok(IndexOutcome::Unchanged);
            }
        }
        if !doc.metadata.contains_key("revision") {
            if let Some(rev) = revision {
                doc.metadata
                    .insert("revision".to_string(), serde_json::Value::String(rev));
            }
        }
        self.index_document(doc).await
    }

    /// Sync a directory: index new/changed files, remove documents whose file is gone.
    /// Per-file failures are collected in the summary and do not abort the sync.
    pub async fn index_directory(&self, root: &Path) -> Result<IndexSummary> {
        let mut summary = IndexSummary::default();
        let files: Vec<PathBuf> = scan_dir(root)
            .into_iter()
            .filter(|f| !is_excluded(root, f, &self.config.exclude_dirs))
            .collect();
        let mut present: HashSet<String> = HashSet::new();
        let revision = git_revision(root);
        for file in &files {
            let rel = file
                .strip_prefix(root)
                .unwrap_or(file)
                .to_string_lossy()
                .replace('\\', "/");
            present.insert(rel.clone());
            match self.index_file_at(root, file, revision.clone()).await {
                Ok(IndexOutcome::Indexed { chunks, .. }) => {
                    summary.indexed += 1;
                    summary.chunks += chunks;
                }
                Ok(IndexOutcome::Unchanged) => summary.unchanged += 1,
                Err(e) => {
                    self.metrics.record_error("index_file_failed");
                    summary.failed.push((rel, e.to_string()));
                }
            }
        }
        for (_, stored) in self.db.document_paths().await? {
            if !present.contains(&stored) && self.remove_path(&stored).await? {
                summary.removed += 1;
            }
        }
        self.save_snapshot().await;
        Ok(summary)
    }

    /// Remove a document (by stored relative path) from both stores.
    /// Returns whether it existed.
    pub async fn remove_path(&self, source_path: &str) -> Result<bool> {
        let _guard = self.write_lock.lock().await;
        let Some((doc_id, _)) = self.db.find_document(source_path).await? else {
            return Ok(false);
        };
        let ids = self.db.chunk_ids_for_document(doc_id).await?;
        {
            let mut h = self.hnsw.write();
            for id in ids {
                h.remove(id)?;
            }
        }
        self.db.delete_document(doc_id).await?;
        Ok(true)
    }

    /// Apply a batch of watcher events under `root`. Returns the number applied;
    /// failures are logged and counted in metrics, not propagated.
    pub async fn apply_events(&self, root: &Path, events: &[FileEvent]) -> usize {
        let mut applied = 0;
        for ev in events {
            if !is_indexable(&ev.path) || is_excluded(root, &ev.path, &self.config.exclude_dirs) {
                continue;
            }
            let res = match ev.kind {
                EventKind::Added | EventKind::Modified => {
                    if ev.path.is_file() {
                        self.index_file(root, &ev.path).await.map(|_| ())
                    } else {
                        continue; // vanished between event and read
                    }
                }
                EventKind::Deleted => {
                    let rel = ev
                        .path
                        .strip_prefix(root)
                        .unwrap_or(&ev.path)
                        .to_string_lossy()
                        .replace('\\', "/");
                    self.remove_path(&rel).await.map(|_| ())
                }
            };
            match res {
                Ok(()) => applied += 1,
                Err(e) => {
                    self.metrics.record_error("apply_event_failed");
                    tracing::warn!(
                        "failed to apply {:?} for {}: {e}",
                        ev.kind,
                        ev.path.display()
                    );
                }
            }
        }
        self.save_snapshot().await;
        applied
    }

    /// The notes root set by [`SearchEngine::set_watch_root`], if any.
    pub fn watch_root(&self) -> Option<&Path> {
        self.watch_root.get().map(PathBuf::as_path)
    }

    /// Record the folder the index mirrors. Call once at startup; later calls are ignored.
    pub fn set_watch_root(&self, root: PathBuf) {
        let _ = self.watch_root.set(root);
    }

    /// Database statistics including the HNSW memory footprint and, once a watch root is set,
    /// the share of files under it that are indexed.
    pub async fn statistics(&self) -> Result<DatabaseStats> {
        let mut s = self.db.statistics().await?;
        s.hnsw_size_bytes = self.hnsw.read().statistics().memory_bytes;
        if let Some(root) = self.watch_root.get() {
            s.coverage_percent = self.coverage(root).await?;
        }
        Ok(s)
    }

    /// Percentage of indexable files under `root` that are stored in the index (100 when none exist).
    async fn coverage(&self, root: &Path) -> Result<f64> {
        let on_disk: HashSet<String> = scan_dir(root)
            .iter()
            .filter(|f| !is_excluded(root, f, &self.config.exclude_dirs))
            .map(|f| {
                f.strip_prefix(root)
                    .unwrap_or(f)
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        let stored: HashSet<String> = self
            .db
            .document_paths()
            .await?
            .into_iter()
            .map(|(_, p)| p)
            .collect();
        Ok(if on_disk.is_empty() {
            100.0
        } else {
            on_disk.intersection(&stored).count() as f64 / on_disk.len() as f64 * 100.0
        })
    }

    /// Compute index health (coverage against files under `root`) and publish it to metrics.
    pub async fn update_health(&self, root: &Path) -> Result<IndexHealth> {
        let stats = self.statistics().await?;
        let coverage = self.coverage(root).await?;
        let health = IndexHealth {
            documents_total: stats.documents_total,
            chunks_total: stats.chunks_total,
            index_size_bytes: stats.db_size_bytes + stats.hnsw_size_bytes,
            coverage_percent: coverage,
            last_indexed: chrono::Utc::now().to_rfc3339(),
            is_corrupted: false,
        };
        self.metrics.update_index_health(&health);
        Ok(health)
    }

    // -- Searching ---------------------------------------------------------

    fn clamp_limit(&self, limit: usize) -> usize {
        limit.clamp(1, self.config.max_limit.max(1))
    }

    async fn with_timeout<T>(
        &self,
        fut: impl std::future::Future<Output = Result<T>>,
    ) -> Result<T> {
        match tokio::time::timeout(Duration::from_secs(self.config.timeout_secs), fut).await {
            Ok(r) => r,
            Err(_) => {
                self.metrics.record_error("search_timeout");
                Err(MemoryError::ResourceLimitExceeded(format!(
                    "search exceeded {}s timeout",
                    self.config.timeout_secs
                )))
            }
        }
    }

    /// Execute a hybrid search query.
    ///
    /// Runs the semantic and keyword legs (whichever are enabled/available),
    /// fuses them with RRF, drops results under `confidence_threshold`, and
    /// returns at most `limit` (capped at `max_limit`).
    ///
    /// # Errors
    ///
    /// - `KeywordSearchError` if keyword search fails or no search method is enabled
    /// - `RankingError` if ranking fails
    /// - `ResourceLimitExceeded` on timeout
    ///
    /// A failing semantic leg is non-fatal: it is recorded in metrics and the
    /// keyword results are returned.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search_filtered(query, limit, &SearchFilter::default(), false)
            .await
    }

    /// Run a search in `mode`, restricted by `filter` (tier, path, trust, status).
    ///
    /// # Errors
    ///
    /// Same as [`SearchEngine::search`] / [`SearchEngine::semantic_search`].
    pub async fn search_with(
        &self,
        query: &str,
        limit: usize,
        mode: SearchMode,
        filter: &SearchFilter,
    ) -> Result<Vec<SearchResult>> {
        self.search_with_explain(query, limit, mode, filter, false)
            .await
    }

    /// Run a search in `mode`, restricted by `filter`, attaching a per-leg score
    /// breakdown to each result when `explain` is true.
    ///
    /// # Errors
    ///
    /// Same as [`SearchEngine::search`] / [`SearchEngine::semantic_search`].
    pub async fn search_with_explain(
        &self,
        query: &str,
        limit: usize,
        mode: SearchMode,
        filter: &SearchFilter,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        match mode {
            SearchMode::Hybrid => self.search_filtered(query, limit, filter, explain).await,
            SearchMode::Semantic => self.semantic_filtered(query, limit, filter, explain).await,
            SearchMode::Keyword => self.keyword_filtered(query, limit, filter, explain).await,
        }
    }

    /// `search_with`, plus one hop along wikilinks. Returns the `limit` direct hits, followed by the
    /// best passage of each note that the top hits link to or are linked from (at most
    /// `LINK_NEIGHBOUR_LIMIT`, ranked among themselves).
    ///
    /// The link only decides which note is looked at. A linked passage keeps its own evidence label
    /// (FACT only if its own text matched the query) and its score is multiplied by
    /// `LINK_SCORE_FACTOR`. It is returned only when it matches the query at all.
    ///
    /// # Errors
    ///
    /// - Any error from the underlying search or the link table
    pub async fn search_expanded(
        &self,
        query: &str,
        limit: usize,
        mode: SearchMode,
        filter: &SearchFilter,
    ) -> Result<Vec<SearchResult>> {
        let mut hits = self.search_with(query, limit, mode, filter).await?;
        let mut seeds: Vec<String> = hits
            .iter()
            .take(LINK_SEED_LIMIT)
            .map(|h| h.source_path.clone())
            .collect();
        seeds.sort();
        seeds.dedup();

        let mut neighbours: Vec<String> = Vec::new();
        for path in &seeds {
            for l in self.links(path, false).await? {
                if let Some(target) = l.resolved_path {
                    neighbours.push(target);
                }
            }
            for l in self.links(path, true).await? {
                neighbours.push(l.source_path);
            }
        }
        neighbours.retain(|p| !seeds.contains(p));
        neighbours.sort();
        neighbours.dedup();
        neighbours.truncate(LINK_NEIGHBOUR_LIMIT);

        let mut seen: HashSet<u64> = hits.iter().map(|h| h.chunk_id).collect();
        let mut linked: Vec<SearchResult> = Vec::new();
        for path in neighbours {
            let scoped = SearchFilter {
                path_prefix: Some(path.clone()),
                ..filter.clone()
            };
            // `path_prefix` is a prefix match; keep only the note itself.
            let best = self
                .search_with(query, LINK_NEIGHBOUR_SEARCH, mode, &scoped)
                .await?
                .into_iter()
                .find(|r| r.source_path == path);
            if let Some(mut r) = best {
                if seen.insert(r.chunk_id) {
                    r.score *= LINK_SCORE_FACTOR;
                    linked.push(r);
                }
            }
        }
        linked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Direct hits keep their `limit`; linked passages follow them, so they are never cut
        // by the limit that the direct hits already filled.
        hits.extend(linked);
        Ok(hits)
    }

    /// Hybrid search restricted by `filter`. When `explain` is true, each
    /// result carries a per-leg score breakdown in `leg_scores`.
    pub async fn search_filtered(
        &self,
        query: &str,
        limit: usize,
        filter: &SearchFilter,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        let timer = Timer::start();
        self.metrics.record_search_query();
        let res = self
            .with_timeout(self.hybrid(query, limit, filter, explain))
            .await;
        self.metrics
            .record_search_latency(timer.elapsed(), SearchType::Hybrid);
        if res.is_err() {
            self.metrics.record_error("search_failed");
        }
        res
    }

    async fn hybrid(
        &self,
        query: &str,
        limit: usize,
        filter: &SearchFilter,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        let limit = self.clamp_limit(limit);
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        // A restrictive filter discards candidates, so look at more of them.
        let oversample = if filter.is_restrictive() { 4 } else { 1 };
        let candidates = (limit * 3 * oversample).max(20);

        let keyword = if self.config.enable_keyword_search {
            let raw = self.db.search_fts(query, candidates).await?;
            Some(self.refine(raw, filter).await?)
        } else {
            None
        };
        let vector = if self.has_vector_search() {
            let leg = match self.vector_candidates(query, candidates).await {
                Ok(raw) => self.refine(raw, filter).await,
                Err(e) => Err(e),
            };
            match leg {
                Ok(v) => {
                    self.metrics.record_vector_search(true);
                    Some(v)
                }
                Err(e) => {
                    self.metrics.record_vector_search(false);
                    self.metrics.record_error("vector_search_failed");
                    tracing::warn!("semantic search failed, using keyword only: {e}");
                    None
                }
            }
        } else {
            None
        };
        if keyword.is_none() && vector.is_none() {
            return Err(MemoryError::KeywordSearchError(
                "no search method available (keyword disabled and semantic unavailable)"
                    .to_string(),
            ));
        }
        let keyword_scores: HashMap<u64, f32> = keyword
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|r| (r.chunk_id, r.score))
            .collect();
        let vector_scores: HashMap<u64, f32> = vector
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|r| (r.chunk_id, r.score))
            .collect();
        let mut merged = self.ranker.merge_and_rank(
            vector.as_deref().unwrap_or(&[]),
            keyword.as_deref().unwrap_or(&[]),
        )?;
        let has_reranker = self.has_reranker();
        if has_reranker {
            let n = self.config.rerank_top_n.max(limit);
            merged.truncate(n);
            match self
                .rerank_results(query, merged.clone(), n, self.config.rerank_weight)
                .await
            {
                Ok(r) => merged = r,
                Err(e) => {
                    self.metrics.record_error("rerank_failed");
                    tracing::warn!("reranking failed, using fused order: {e}");
                }
            }
        }
        if explain {
            use crate::memory::storage::LegScores;
            for r in &mut merged {
                r.explain = Some(LegScores {
                    fts: keyword_scores.get(&r.chunk_id).copied(),
                    vector: vector_scores.get(&r.chunk_id).copied(),
                    fused: Some(r.score),
                    staleness_factor: Some(if r.meta.stale { 0.5 } else { 1.0 }),
                    rerank: if has_reranker { Some(r.score) } else { None },
                });
            }
        }
        merged.retain(|r| r.score >= self.config.confidence_threshold);
        merged.truncate(limit);
        Ok(merged)
    }

    async fn vector_candidates(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        let engine = self
            .embeddings
            .as_ref()
            .ok_or(MemoryError::FeatureNotEnabled {
                feature: "embeddings".to_string(),
            })?;
        let t = Timer::start();
        let q = engine.embed(query).await?;
        self.metrics.record_embedding_latency(t.elapsed());
        self.metrics
            .set_embedding_cache_hit_ratio(engine.cache_hit_ratio());
        let hits = self.hnsw.read().search(&q, k)?;
        let mut out = Vec::with_capacity(hits.len());
        for (id, dist) in hits {
            let score = (1.0 - dist).clamp(0.0, 1.0);
            if let Some(r) = self
                .db
                .result_for_chunk(id, score, ConfidenceLevel::Inferred)
                .await?
            {
                out.push(r);
            }
        }
        Ok(out)
    }

    /// Execute semantic search only (vector similarity).
    ///
    /// # Errors
    ///
    /// - `FeatureNotEnabled` if no embedding engine is available
    pub async fn semantic_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.semantic_filtered(query, limit, &SearchFilter::default(), false)
            .await
    }

    async fn semantic_filtered(
        &self,
        query: &str,
        limit: usize,
        filter: &SearchFilter,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        if !self.has_vector_search() {
            return Err(MemoryError::FeatureNotEnabled {
                feature: "embeddings".to_string(),
            });
        }
        let timer = Timer::start();
        self.metrics.record_search_query();
        let limit = self.clamp_limit(limit);
        let fetch = if filter.is_restrictive() {
            limit * 4
        } else {
            limit
        };
        let mut res = if query.trim().is_empty() {
            Ok(Vec::new())
        } else {
            self.with_timeout(async {
                let raw = self.vector_candidates(query, fetch).await?;
                let mut v = self.refine(raw, filter).await?;
                v.truncate(limit);
                Ok(v)
            })
            .await
        };
        if explain {
            use crate::memory::storage::LegScores;
            if let Ok(v) = &mut res {
                for r in v.iter_mut() {
                    r.explain = Some(LegScores {
                        fts: None,
                        vector: Some(r.score),
                        fused: Some(r.score),
                        staleness_factor: Some(if r.meta.stale { 0.5 } else { 1.0 }),
                        rerank: None,
                    });
                }
            }
        }
        self.metrics
            .record_search_latency(timer.elapsed(), SearchType::Semantic);
        res.map(|mut v| {
            v.retain(|r| r.score >= self.config.confidence_threshold);
            v
        })
    }

    /// Execute keyword search only (full-text search).
    pub async fn keyword_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.keyword_filtered(query, limit, &SearchFilter::default(), false)
            .await
    }

    async fn keyword_filtered(
        &self,
        query: &str,
        limit: usize,
        filter: &SearchFilter,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        let timer = Timer::start();
        self.metrics.record_search_query();
        let limit = self.clamp_limit(limit);
        let fetch = if filter.is_restrictive() {
            limit * 4
        } else {
            limit
        };
        let res = self
            .with_timeout(async {
                let raw = self.db.search_fts(query, fetch).await?;
                let mut v = self.refine(raw, filter).await?;
                if explain {
                    use crate::memory::storage::LegScores;
                    for r in &mut v {
                        r.explain = Some(LegScores {
                            fts: Some(r.score),
                            vector: None,
                            fused: Some(r.score),
                            staleness_factor: Some(if r.meta.stale { 0.5 } else { 1.0 }),
                            rerank: None,
                        });
                    }
                }
                v.truncate(limit);
                Ok(v)
            })
            .await;
        self.metrics
            .record_search_latency(timer.elapsed(), SearchType::Keyword);
        res.map(|mut v| {
            v.retain(|r| r.score >= self.config.confidence_threshold);
            v
        })
    }
}

/// Which retrieval legs a query uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchMode {
    /// Keyword + semantic, fused.
    #[default]
    Hybrid,
    /// Embedding similarity only.
    Semantic,
    /// Full-text (BM25) only.
    Keyword,
}

impl SearchMode {
    /// Parse `hybrid`, `semantic` or `keyword`.
    pub fn parse(s: &str) -> Option<SearchMode> {
        match s {
            "hybrid" => Some(SearchMode::Hybrid),
            "semantic" => Some(SearchMode::Semantic),
            "keyword" => Some(SearchMode::Keyword),
            _ => None,
        }
    }
}

/// Ranker combines results from multiple search methods.
pub struct Ranker {
    config: RankerConfig,
}

/// Ranker configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankerConfig {
    /// Weight for vector search results (0-1).
    pub vector_weight: f32,

    /// Weight for keyword search results (0-1).
    pub keyword_weight: f32,

    /// Fusion algorithm.
    pub ranker_type: RankerType,
}

impl Default for RankerConfig {
    fn default() -> Self {
        RankerConfig {
            vector_weight: 0.7,
            keyword_weight: 0.3,
            ranker_type: RankerType::Weighted,
        }
    }
}

/// Type of ranking algorithm.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RankerType {
    /// Reciprocal Rank Fusion (simple, no ML).
    RRF,

    /// Weighted score fusion: `vector_weight * cosine + keyword_weight * bm25_strength`.
    /// Keeps absolute score magnitude, so a strong semantic match is not outvoted by weak keyword hits.
    Weighted,
}

impl Ranker {
    /// Create a new ranker.
    pub fn new(config: RankerConfig) -> Self {
        Ranker { config }
    }

    /// Weighted fusion of absolute vector scores and relative keyword scores.
    fn weighted(
        vw: f32,
        kw: f32,
        vector_results: &[SearchResult],
        keyword_results: &[SearchResult],
    ) -> Vec<SearchResult> {
        let mut by_chunk: HashMap<u64, (f32, f32, SearchResult)> = HashMap::new();
        for r in keyword_results {
            by_chunk
                .entry(r.chunk_id)
                .or_insert_with(|| (0.0, r.score, r.clone()))
                .1 = r.score;
        }
        for r in vector_results {
            by_chunk
                .entry(r.chunk_id)
                .and_modify(|e| e.0 = r.score)
                .or_insert_with(|| (r.score, 0.0, r.clone()));
        }
        // Only legs that returned something count toward the normaliser.
        let mut denom = 0.0;
        if !vector_results.is_empty() {
            denom += vw;
        }
        if !keyword_results.is_empty() {
            denom += kw;
        }
        if denom <= 0.0 {
            return Vec::new();
        }
        let mut merged: Vec<SearchResult> = by_chunk
            .into_values()
            .map(|(v, k, mut r)| {
                r.score = ((vw * v + kw * k) / denom).clamp(0.0, 1.0);
                r
            })
            .collect();
        merged.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.chunk_id.cmp(&b.chunk_id))
        });
        merged
    }

    /// Merge and re-rank results from multiple sources with weighted RRF.
    ///
    /// `score(d) = Σ weight_s / (60 + rank_s(d))` over the sources containing
    /// `d` (1-based ranks), normalised by the best score a document could get
    /// from the *non-empty* sources, so scores fall in `(0, 1]`. Results are
    /// de-duplicated by chunk. A result found by keyword search keeps
    /// `Fact`; one found only by vectors is `Inferred`. Ties break on chunk id.
    ///
    /// # Errors
    ///
    /// - `RankingError` if weights are negative/non-finite or both zero
    pub fn merge_and_rank(
        &self,
        vector_results: &[SearchResult],
        keyword_results: &[SearchResult],
    ) -> Result<Vec<SearchResult>> {
        let (vw, kw) = (self.config.vector_weight, self.config.keyword_weight);
        if !vw.is_finite() || !kw.is_finite() || vw < 0.0 || kw < 0.0 || vw + kw <= 0.0 {
            return Err(MemoryError::RankingError(format!(
                "invalid weights: vector={vw}, keyword={kw}"
            )));
        }
        if self.config.ranker_type == RankerType::Weighted {
            return Ok(Self::weighted(vw, kw, vector_results, keyword_results));
        }
        let mut max = 0.0f32;
        if !vector_results.is_empty() {
            max += vw / (RRF_K + 1.0);
        }
        if !keyword_results.is_empty() {
            max += kw / (RRF_K + 1.0);
        }
        if max <= 0.0 {
            return Ok(Vec::new());
        }

        let mut by_chunk: HashMap<u64, (f32, SearchResult)> = HashMap::new();
        for (rank, r) in keyword_results.iter().enumerate() {
            let s = kw / (RRF_K + rank as f32 + 1.0);
            by_chunk
                .entry(r.chunk_id)
                .and_modify(|e| e.0 += s)
                .or_insert_with(|| (s, r.clone()));
        }
        for (rank, r) in vector_results.iter().enumerate() {
            let s = vw / (RRF_K + rank as f32 + 1.0);
            by_chunk
                .entry(r.chunk_id)
                .and_modify(|e| e.0 += s)
                .or_insert_with(|| (s, r.clone()));
        }
        let mut merged: Vec<SearchResult> = by_chunk
            .into_values()
            .map(|(s, mut r)| {
                r.score = (s / max).clamp(0.0, 1.0);
                r
            })
            .collect();
        merged.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.chunk_id.cmp(&b.chunk_id))
        });
        Ok(merged)
    }
}

/// Citation builder: constructs structured citations for search results (EV-01 compliance).
pub struct CitationBuilder {}

impl CitationBuilder {
    /// Create a new citation builder.
    pub fn new() -> Self {
        CitationBuilder {}
    }

    /// Build a citation for a search result.
    ///
    /// # Errors
    ///
    /// - `CitationError` if the result lacks a source path, has an invalid
    ///   line range (must be 1-based and `start <= end`), or has no timestamp
    pub fn build(&self, result: &SearchResult) -> Result<Citation> {
        if result.source_path.trim().is_empty() {
            return Err(MemoryError::CitationError(format!(
                "chunk {} has no source path",
                result.chunk_id
            )));
        }
        if result.start_line == 0 || result.end_line < result.start_line {
            return Err(MemoryError::CitationError(format!(
                "chunk {} has invalid line range {}-{}",
                result.chunk_id, result.start_line, result.end_line
            )));
        }
        if result.indexed_at.trim().is_empty() {
            return Err(MemoryError::CitationError(format!(
                "chunk {} has no index timestamp",
                result.chunk_id
            )));
        }
        let docid = docid_for(&result.source_path);
        Ok(Citation {
            chunk_id: result.chunk_id,
            docid,
            file_path: result.source_path.clone(),
            revision: result.revision.clone(),
            line_range: (result.start_line, result.end_line),
            timestamp: result.indexed_at.clone(),
            confidence: match result.confidence {
                ConfidenceLevel::Fact => "FACT",
                ConfidenceLevel::Inferred => "INFERRED",
                ConfidenceLevel::Speculative => "SPECULATIVE",
            }
            .to_string(),
            source: result.meta.clone(),
        })
    }
}

impl Default for CitationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Structured citation for search results (EV-01 compliance).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Citation {
    /// Chunk the citation points at.
    pub chunk_id: u64,

    /// Docid derived from the normalized relative path (first 6 hex chars of SHA-256).
    pub docid: String,

    /// Source file path (relative to .plaesy/).
    pub file_path: String,

    /// Git revision (if available).
    pub revision: Option<String>,

    /// Line range in source file (1-based, inclusive).
    pub line_range: (usize, usize),

    /// Timestamp when content was indexed (RFC 3339).
    pub timestamp: String,

    /// Confidence in the citation (FACT, INFERRED, SPECULATIVE).
    pub confidence: String,

    /// What the source itself declares: tier, trust, last verification, status, staleness.
    #[serde(default)]
    pub source: ResultMeta,
}

impl std::fmt::Display for Citation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}-{} [{}] indexed {}",
            self.file_path, self.line_range.0, self.line_range.1, self.confidence, self.timestamp
        )?;
        if let Some(rev) = &self.revision {
            write!(f, " @ {rev}")?;
        }
        if !self.docid.is_empty() {
            write!(f, " {}", self.docid)?;
        }
        let m = &self.source;
        let mut notes: Vec<String> = Vec::new();
        if let Some(t) = &m.tier {
            notes.push(format!("tier {t}"));
        }
        if let Some(t) = &m.trust {
            notes.push(format!("trust {t}"));
        }
        if let Some(v) = &m.last_verified {
            notes.push(format!("verified {v}"));
        }
        if let Some(s) = &m.status {
            if s != "active" {
                notes.push(format!("status {s}"));
            }
        }
        if m.stale {
            notes.push("STALE".to_string());
        }
        if !notes.is_empty() {
            write!(f, " ({})", notes.join(", "))?;
        }
        Ok(())
    }
}

/// Derive a short docid from a normalized relative path (first 6 hex chars of SHA-256).
fn docid_for(path: &str) -> String {
    use sha2::{Digest, Sha256};
    let normalized = path.replace('\\', "/").trim().to_lowercase();
    let digest = Sha256::digest(normalized.as_bytes());
    digest.iter().map(|b| format!("{b:x}")).collect::<String>()[..6].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::storage::DatabaseConfig;

    fn res(id: u64, conf: ConfidenceLevel) -> SearchResult {
        SearchResult {
            chunk_id: id,
            text: format!("t{id}"),
            source_path: "a.md".into(),
            start_line: 1,
            end_line: 2,
            revision: None,
            indexed_at: "2026-01-01T00:00:00Z".into(),
            score: 0.0,
            confidence: conf,
            meta: Default::default(),
            explain: None,
        }
    }

    async fn engine(with_vectors: bool) -> SearchEngine {
        let db = Arc::new(
            Database::new(DatabaseConfig {
                db_path: PathBuf::from(":memory:"),
                ..DatabaseConfig::default()
            })
            .await
            .expect("db"),
        );
        let emb = with_vectors.then(|| Arc::new(EmbeddingEngine::hashing(128).expect("hash")));
        SearchEngine::new(db, emb, SearchConfig::default())
            .await
            .expect("engine")
    }

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(&p, content).expect("write");
        p
    }

    fn rrf_ranker() -> Ranker {
        Ranker::new(RankerConfig {
            ranker_type: RankerType::RRF,
            ..RankerConfig::default()
        })
    }

    #[test]
    fn test_weighted_keeps_semantic_winner_over_weak_keyword_hit() {
        let r = Ranker::new(RankerConfig::default());
        let mut strong_vec = res(1, ConfidenceLevel::Inferred);
        strong_vec.score = 0.8;
        let mut weak_kw = res(2, ConfidenceLevel::Fact);
        weak_kw.score = 0.2;
        let out = r.merge_and_rank(&[strong_vec], &[weak_kw]).expect("rank");
        assert_eq!(out[0].chunk_id, 1);
        assert!((out[0].score - 0.56).abs() < 1e-5); // 0.7 * 0.8 / (0.7 + 0.3)
        assert!((out[1].score - 0.06).abs() < 1e-5); // 0.3 * 0.2
        assert_eq!(out[1].confidence, ConfidenceLevel::Fact);
        assert_eq!(out[0].confidence, ConfidenceLevel::Inferred);
    }

    #[test]
    fn test_ranker_config_default() {
        let config = RankerConfig::default();
        assert_eq!(config.vector_weight, 0.7);
        assert_eq!(config.keyword_weight, 0.3);
        assert_eq!(config.ranker_type, RankerType::Weighted);
    }

    #[test]
    fn test_ranker_type_serialize() {
        let rt = RankerType::RRF;
        let json = serde_json::to_string(&rt).expect("serialize");
        assert_eq!(json, "\"rrf\"");
    }

    #[test]
    fn test_rrf_prefers_docs_in_both_lists_and_dedupes() {
        let r = rrf_ranker();
        let vec_hits = vec![
            res(1, ConfidenceLevel::Inferred),
            res(2, ConfidenceLevel::Inferred),
        ];
        let kw_hits = vec![res(2, ConfidenceLevel::Fact), res(3, ConfidenceLevel::Fact)];
        let out = r.merge_and_rank(&vec_hits, &kw_hits).expect("rank");
        let ids: Vec<u64> = out.iter().map(|x| x.chunk_id).collect();
        assert_eq!(ids, vec![2, 1, 3]);
        assert!(
            (out[0].score - (0.7 / 62.0 + 0.3 / 61.0) / (0.7 / 61.0 + 0.3 / 61.0)).abs() < 1e-5
        );
        assert!(out.iter().all(|x| x.score > 0.0 && x.score <= 1.0));
        // Fact wins when keyword matched; vector-only stays Inferred.
        assert_eq!(out[0].confidence, ConfidenceLevel::Fact);
        assert_eq!(out[1].confidence, ConfidenceLevel::Inferred);
        assert_eq!(out[2].confidence, ConfidenceLevel::Fact);
    }

    #[test]
    fn test_rrf_single_source_top_score_is_one() {
        let r = rrf_ranker();
        let out = r
            .merge_and_rank(
                &[],
                &[res(5, ConfidenceLevel::Fact), res(6, ConfidenceLevel::Fact)],
            )
            .expect("rank");
        assert!((out[0].score - 1.0).abs() < 1e-6);
        assert!(out[1].score < out[0].score);
        assert!(r.merge_and_rank(&[], &[]).expect("empty").is_empty());
    }

    #[test]
    fn test_ranker_rejects_bad_config() {
        let bad = |vw, kw, t| {
            Ranker::new(RankerConfig {
                vector_weight: vw,
                keyword_weight: kw,
                ranker_type: t,
            })
            .merge_and_rank(&[res(1, ConfidenceLevel::Fact)], &[])
        };
        assert!(matches!(
            bad(-1.0, 1.0, RankerType::RRF),
            Err(MemoryError::RankingError(_))
        ));
        assert!(matches!(
            bad(0.0, 0.0, RankerType::RRF),
            Err(MemoryError::RankingError(_))
        ));
        assert!(matches!(
            bad(f32::NAN, 1.0, RankerType::RRF),
            Err(MemoryError::RankingError(_))
        ));
    }

    #[test]
    fn test_citation_validation_and_display() {
        let b = CitationBuilder::new();
        let mut r = res(9, ConfidenceLevel::Fact);
        r.revision = Some("abc123".into());
        let c = b.build(&r).expect("cite");
        assert_eq!(c.confidence, "FACT");
        assert_eq!(
            c.to_string(),
            format!(
                "a.md:1-2 [FACT] indexed 2026-01-01T00:00:00Z @ abc123 {}",
                docid_for("a.md")
            )
        );
        r.start_line = 0;
        assert!(matches!(b.build(&r), Err(MemoryError::CitationError(_))));
        r.start_line = 3;
        assert!(b.build(&r).is_err());
        let mut r = res(9, ConfidenceLevel::Fact);
        r.source_path.clear();
        assert!(b.build(&r).is_err());
        let mut r = res(9, ConfidenceLevel::Inferred);
        r.indexed_at.clear();
        assert!(b.build(&r).is_err());
    }

    #[test]
    fn test_docid_derived_from_path() {
        let id1 = docid_for("a/b.md");
        let id2 = docid_for("A/b.md"); // case-insensitive normalization
        assert_eq!(id1, id2);
        let id3 = docid_for("a/../b/c.md"); // normalized relative path
        assert!(!id1.eq(&id3));
        assert!(id1.len() == 6);
        // Same path -> same docid; different path -> different docid
        let id4 = docid_for("a/x.md");
        let id5 = docid_for("a/y.md");
        assert_ne!(id4, id5);
    }

    #[test]
    fn test_docid_is_stable_and_path_based() {
        // First 6 hex chars of SHA-256 of normalized path
        let id_a = docid_for("notes/a.md");
        let id_a_again = docid_for("notes/a.md");
        assert_eq!(id_a, id_a_again);
        assert!(id_a.len() == 6);
        assert!(
            id_a.chars().all(|c| c.is_ascii_hexdigit()),
            "docid must be hex"
        );
        // Moving the file changes the docid (acceptable breakage per spec)
        let id_b = docid_for("notes/b.md");
        assert_ne!(id_a, id_b);
        // Backslashes normalized to forward slashes
        let id_win = docid_for(r"notes\a.md");
        assert_eq!(id_a, id_win, "backslash normalized to forward slash");
    }

    #[tokio::test]
    async fn test_index_directory_and_search_with_exact_citations() {
        let dir = tempfile::tempdir().expect("tmp");
        let root = dir.path();
        let content = "---\nid: EV-01\n---\n\n# Vectors\n\nHNSW graphs enable approximate nearest neighbor search.\n";
        write(root, "notes/vectors.md", content);
        write(
            root,
            "cooking.md",
            "# Cake\n\nMix flour and sugar then bake.\n",
        );
        let e = engine(true).await;
        let s = e.index_directory(root).await.expect("index");
        assert_eq!((s.indexed, s.unchanged, s.removed), (2, 0, 0));
        assert!(s.failed.is_empty());

        let hits = e
            .search("nearest neighbor graphs", 5)
            .await
            .expect("search");
        assert!(!hits.is_empty());
        let top = &hits[0];
        assert_eq!(top.source_path, "notes/vectors.md");
        assert_eq!(top.confidence, ConfidenceLevel::Fact);
        // Cited line range must reproduce the chunk text from the real file.
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(top.text, lines[top.start_line - 1..top.end_line].join("\n"));
        // EV-01: every result is citable.
        let b = CitationBuilder::new();
        for r in &hits {
            b.build(r).expect("citation");
        }
    }

    #[tokio::test]
    async fn test_reindex_unchanged_then_modified_replaces_old_content() {
        let dir = tempfile::tempdir().expect("tmp");
        let root = dir.path();
        let p = write(root, "a.md", "alpha unique1 text\n");
        let e = engine(true).await;
        e.index_directory(root).await.expect("1");
        let again = e.index_directory(root).await.expect("2");
        assert_eq!((again.indexed, again.unchanged), (0, 1));

        std::fs::write(&p, "beta unique2 text\n").expect("rewrite");
        let s = e.index_directory(root).await.expect("3");
        assert_eq!(s.indexed, 1);
        assert!(e.keyword_search("unique1", 5).await.expect("kw").is_empty());
        assert_eq!(e.keyword_search("unique2", 5).await.expect("kw").len(), 1);
        // Old vector must be gone too: one chunk, one vector.
        assert_eq!(e.hnsw.read().len(), 1);
        let st = e.statistics().await.expect("stats");
        assert_eq!((st.documents_total, st.chunks_total), (1, 1));
    }

    #[tokio::test]
    async fn test_deleted_file_is_removed_from_both_stores() {
        let dir = tempfile::tempdir().expect("tmp");
        let root = dir.path();
        let p = write(root, "gone.md", "ephemeral content here\n");
        write(root, "stay.md", "persistent content here\n");
        let e = engine(true).await;
        e.index_directory(root).await.expect("index");
        std::fs::remove_file(&p).expect("rm");
        let s = e.index_directory(root).await.expect("resync");
        assert_eq!(s.removed, 1);
        assert!(e
            .keyword_search("ephemeral", 5)
            .await
            .expect("kw")
            .is_empty());
        assert_eq!(e.hnsw.read().len(), 1);
        assert!(!e.remove_path("gone.md").await.expect("again"));
    }

    #[tokio::test]
    async fn test_bad_file_does_not_abort_sync() {
        let dir = tempfile::tempdir().expect("tmp");
        let root = dir.path();
        write(root, "empty.md", "  \n");
        write(root, "good.md", "fine content\n");
        // Broken frontmatter must not make the file unsearchable.
        write(
            root,
            "tmpl.md",
            "---\nid: {{ID}}: x\n---\nplaceholder body\n",
        );
        let e = engine(false).await;
        let s = e.index_directory(root).await.expect("index");
        assert_eq!(s.indexed, 2);
        assert_eq!(s.failed.len(), 1);
        assert_eq!(s.failed[0].0, "empty.md");
        let hits = e.keyword_search("placeholder", 5).await.expect("kw");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source_path, "tmpl.md");
        // Line numbers refer to the real file (body is on line 4).
        assert_eq!((hits[0].start_line, hits[0].end_line), (1, 4));
    }

    #[tokio::test]
    async fn test_semantic_search_and_keyword_only_mode() {
        let dir = tempfile::tempdir().expect("tmp");
        write(dir.path(), "a.md", "vector similarity search\n");
        let sem = engine(true).await;
        sem.index_directory(dir.path()).await.expect("index");
        let hits = sem
            .semantic_search("vector similarity", 3)
            .await
            .expect("sem");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].confidence, ConfidenceLevel::Inferred);
        assert!(hits[0].score > 0.5);

        let kw = engine(false).await;
        kw.index_directory(dir.path()).await.expect("index");
        assert!(!kw.has_vector_search());
        assert!(matches!(
            kw.semantic_search("vector", 3).await,
            Err(MemoryError::FeatureNotEnabled { .. })
        ));
        assert_eq!(kw.search("vector", 3).await.expect("hybrid").len(), 1);
    }

    #[tokio::test]
    async fn test_hybrid_exact_match_is_fact_with_top_score() {
        let dir = tempfile::tempdir().expect("tmp");
        write(dir.path(), "a.md", "rust memory safety\n");
        let e = engine(true).await;
        e.index_directory(dir.path()).await.expect("index");
        let hits = e.search("rust memory safety", 5).await.expect("hybrid");
        assert_eq!(hits[0].confidence, ConfidenceLevel::Fact);
        assert!(hits[0].score >= 0.65, "score {}", hits[0].score);
    }

    #[tokio::test]
    async fn test_onnx_semantic_only_hit_is_inferred() {
        let cfg = crate::memory::embeddings::EmbeddingConfig::default();
        let Ok(emb) = EmbeddingEngine::with_config(cfg).await else {
            eprintln!("SKIPPED test_onnx_semantic_only_hit_is_inferred: no model");
            return;
        };
        let dir = tempfile::tempdir().expect("tmp");
        write(
            dir.path(),
            "nn.md",
            "Nearest neighbor lookup over embeddings\n",
        );
        write(
            dir.path(),
            "cake.md",
            "Baking a chocolate cake with flour\n",
        );
        let db = Arc::new(
            Database::new(DatabaseConfig {
                db_path: PathBuf::from(":memory:"),
                ..DatabaseConfig::default()
            })
            .await
            .expect("db"),
        );
        let e = SearchEngine::new(db, Some(Arc::new(emb)), SearchConfig::default())
            .await
            .expect("engine");
        e.index_directory(dir.path()).await.expect("index");
        // No query word occurs in either file: only the vector leg can match.
        let hits = e
            .search("finding similar vectors quickly", 2)
            .await
            .expect("hybrid");
        assert!(!hits.is_empty());
        assert_eq!(hits[0].source_path, "nn.md");
        assert_eq!(hits[0].confidence, ConfidenceLevel::Inferred);
    }

    #[tokio::test]
    async fn test_limits_empty_query_and_threshold() {
        let dir = tempfile::tempdir().expect("tmp");
        for i in 0..8 {
            write(
                dir.path(),
                &format!("d{i}.md"),
                &format!("shared token number{i}\n"),
            );
        }
        let e = engine(false).await;
        e.index_directory(dir.path()).await.expect("index");
        assert_eq!(e.search("shared", 3).await.expect("3").len(), 3);
        assert_eq!(e.search("shared", 0).await.expect("0 clamps to 1").len(), 1);
        assert!(e.search("   ", 5).await.expect("empty").is_empty());
        assert!(e.search("zzzzqqq", 5).await.expect("none").is_empty());
        let mut strict = engine(false).await;
        strict.config.confidence_threshold = 1.1;
        strict.index_directory(dir.path()).await.expect("index");
        assert!(strict.search("shared", 5).await.expect("thr").is_empty());
    }

    #[tokio::test]
    async fn test_no_search_method_enabled_is_error() {
        let mut e = engine(false).await;
        e.config.enable_keyword_search = false;
        assert!(matches!(
            e.search("x", 3).await,
            Err(MemoryError::KeywordSearchError(_))
        ));
    }

    #[tokio::test]
    async fn test_hnsw_rebuilt_from_database_on_new() {
        let dir = tempfile::tempdir().expect("tmp");
        let docs = tempfile::tempdir().expect("docs");
        write(docs.path(), "a.md", "persistent vector content\n");
        let cfg = DatabaseConfig {
            db_path: dir.path().join("state.db"),
            ..DatabaseConfig::default()
        };
        {
            let db = Arc::new(Database::new(cfg.clone()).await.expect("db"));
            let emb = Some(Arc::new(EmbeddingEngine::hashing(128).expect("h")));
            let e = SearchEngine::new(db, emb, SearchConfig::default())
                .await
                .expect("e");
            e.index_directory(docs.path()).await.expect("index");
        }
        let db = Arc::new(Database::new(cfg).await.expect("reopen"));
        let emb = Some(Arc::new(EmbeddingEngine::hashing(128).expect("h")));
        let e = SearchEngine::new(db, emb, SearchConfig::default())
            .await
            .expect("e2");
        assert_eq!(e.hnsw.read().len(), 1);
        assert_eq!(
            e.semantic_search("persistent vector", 3)
                .await
                .expect("sem")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn test_search_expanded_adds_linked_passage_after_direct_hits() {
        let dir = tempfile::tempdir().expect("tmp");
        let root = dir.path();
        write(
            root,
            "a.md",
            "rollback procedure for the deploy\n\nsee [[b]] for the steps\n",
        );
        write(root, "b.md", "rollback steps: tag the previous build\n");
        write(root, "c.md", "unrelated gardening notes\n");
        let e = engine(false).await;
        e.index_directory(root).await.expect("index");

        let plain = e
            .search_with("rollback", 1, SearchMode::Keyword, &SearchFilter::default())
            .await
            .expect("plain");
        assert_eq!(plain.len(), 1);

        let expanded = e
            .search_expanded("rollback", 1, SearchMode::Keyword, &SearchFilter::default())
            .await
            .expect("expanded");
        // The direct hit keeps its place; the other note of the link pair follows it.
        assert_eq!(expanded.len(), 2, "direct hit plus linked passage");
        assert_eq!(expanded[0].source_path, plain[0].source_path);
        let other = if plain[0].source_path.ends_with("a.md") {
            "b.md"
        } else {
            "a.md"
        };
        assert!(
            expanded[1].source_path.ends_with(other),
            "{:?}",
            expanded[1].source_path
        );
        assert!(
            expanded[1].score < expanded[0].score,
            "linked passage ranks below direct hits"
        );
        // Its evidence label is its own: its text matched the keyword leg.
        assert!(matches!(
            expanded[1].confidence,
            crate::memory::storage::ConfidenceLevel::Fact
        ));
        // Notes with no link and no match stay out.
        assert!(expanded.iter().all(|r| !r.source_path.ends_with("c.md")));
    }

    #[tokio::test]
    async fn test_apply_events_and_health() {
        let dir = tempfile::tempdir().expect("tmp");
        let root = dir.path();
        let p = write(root, "n.md", "event driven content\n");
        let e = engine(false).await;
        let added = FileEvent {
            path: p.clone(),
            kind: EventKind::Added,
        };
        assert_eq!(e.apply_events(root, &[added]).await, 1);
        assert_eq!(e.keyword_search("driven", 5).await.expect("kw").len(), 1);
        let h = e.update_health(root).await.expect("health");
        assert_eq!(h.coverage_percent, 100.0);
        assert_eq!(h.documents_total, 1);

        // `statistics` reports the same coverage once the watch root is known, and drops
        // when a file on disk was never indexed.
        e.set_watch_root(root.to_path_buf());
        assert_eq!(e.statistics().await.expect("stats").coverage_percent, 100.0);
        write(root, "not-indexed.md", "on disk only\n");
        assert_eq!(e.statistics().await.expect("stats").coverage_percent, 50.0);

        std::fs::remove_file(&p).expect("rm");
        let del = FileEvent {
            path: p,
            kind: EventKind::Deleted,
        };
        assert_eq!(e.apply_events(root, &[del]).await, 1);
        assert!(e.keyword_search("driven", 5).await.expect("kw").is_empty());
    }

    #[tokio::test]
    async fn test_metrics_recorded_by_search() {
        let e = engine(true).await;
        e.search("anything", 3).await.expect("search");
        e.keyword_search("anything", 3).await.expect("kw");
        let snap = e.metrics().snapshot();
        assert_eq!(snap.total_queries, 2);
        assert_eq!(snap.vector_search_success_rate, 1.0);
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use crate::memory::storage::DatabaseConfig;

    async fn open(db_path: &Path) -> SearchEngine {
        let db = Arc::new(
            Database::new(DatabaseConfig {
                db_path: db_path.to_path_buf(),
                ..DatabaseConfig::default()
            })
            .await
            .expect("db"),
        );
        let emb = Some(Arc::new(EmbeddingEngine::hashing(64).expect("hash")));
        SearchEngine::new(db, emb, SearchConfig::default())
            .await
            .expect("engine")
    }

    #[tokio::test]
    async fn snapshot_is_written_reused_and_invalidated_by_changes() {
        let docs = tempfile::tempdir().expect("docs");
        std::fs::write(docs.path().join("a.md"), "alpha snapshot content\n").expect("w");
        let store = tempfile::tempdir().expect("store");
        let db_path = store.path().join("state.db");
        let snap = store.path().join("state.db.hnsw");

        let e = open(&db_path).await;
        assert_eq!(
            e.snapshot_generation.load(Ordering::Relaxed),
            u64::MAX,
            "fresh db has no snapshot"
        );
        e.index_directory(docs.path()).await.expect("index");
        assert!(snap.exists(), "index_directory must write a snapshot");
        let gen_after_index = e.db.generation().await.expect("gen");
        assert_eq!(
            e.snapshot_generation.load(Ordering::Relaxed),
            gen_after_index
        );
        drop(e);

        // Reopen: snapshot is valid, so it is adopted (generation recorded, no rebuild needed).
        let e2 = open(&db_path).await;
        assert_eq!(
            e2.snapshot_generation.load(Ordering::Relaxed),
            gen_after_index
        );
        assert_eq!(
            e2.semantic_search("alpha snapshot", 3)
                .await
                .expect("sem")
                .len(),
            1
        );

        // A change makes the next snapshot newer; a stale snapshot must never be trusted.
        std::fs::write(docs.path().join("b.md"), "beta other text\n").expect("w");
        e2.index_directory(docs.path()).await.expect("index2");
        drop(e2);
        std::fs::write(snap.with_extension("hnsw.gen"), "0").expect("corrupt gen");
        let e3 = open(&db_path).await;
        assert_eq!(
            e3.snapshot_generation.load(Ordering::Relaxed),
            u64::MAX,
            "stale gen => rebuild"
        );
        assert_eq!(
            e3.hnsw.read().len(),
            2,
            "rebuilt index must hold both documents' vectors"
        );
    }

    #[tokio::test]
    async fn corrupt_snapshot_falls_back_to_rebuild() {
        let docs = tempfile::tempdir().expect("docs");
        std::fs::write(docs.path().join("a.md"), "gamma content\n").expect("w");
        let store = tempfile::tempdir().expect("store");
        let db_path = store.path().join("state.db");
        let e = open(&db_path).await;
        e.index_directory(docs.path()).await.expect("index");
        drop(e);
        std::fs::write(store.path().join("state.db.hnsw"), b"garbage").expect("corrupt");
        let e2 = open(&db_path).await;
        assert_eq!(e2.hnsw.read().len(), 1);
        assert_eq!(e2.snapshot_generation.load(Ordering::Relaxed), u64::MAX);
    }
}
