//! # plaesy-memory: Offline-first semantic memory indexing for Hermes Agent
//!
//! A high-performance Rust library for indexing and searching `.plaesy/` configuration and memory files.
//! Combines vector search (HNSW), full-text search (SQLite FTS5), and intelligent ranking for
//! semantic retrieval of project knowledge.
//!
//! ## Architecture
//!
//! Three-layer memory system:
//! - **Layer 1 (Native)**: MEMORY.md, USER.md (instant, 2.2K chars)
//! - **Layer 2 (Session)**: SQLite FTS5 (1-2 sec, unlimited capacity)
//! - **Layer 3 (Custom)**: This library - vector + keyword search (3-5 sec, offline-first)
//!
//! ## Tech Stack
//!
//! - **Async Runtime**: Tokio
//! - **Vector Search**: HNSW via usearch (Qdrant engine)
//! - **Embeddings**: ONNX Runtime + all-MiniLM-L6-v2 (384-dim)
//! - **Full-Text Search**: SQLite FTS5 (BM25)
//! - **Storage**: SQLite with WAL (write-ahead log) + crash recovery
//! - **Observability**: tracing + in-process metrics with Prometheus text export
//!
//! ## Design Principles
//!
//! 1. **No `unwrap()` in library code** — All error paths use `Result<T, MemoryError>`
//! 2. **Strong typing** — Newtype patterns, enums over strings, compile-time guarantees
//! 3. **Offline-first** — Works without network, no cloud dependencies
//! 4. **Deterministic latency** — No GC pauses, predictable p99 latency
//! 5. **Evidence-based retrieval** — Every result includes citations (file, line, revision)
//!
//! ## Example Usage
//!
//! ```ignore
//! use plaesy::memory::prelude::*;
//! use plaesy::memory::config::MemoryConfig;
//! use plaesy::memory::search::SearchEngine;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load configuration
//!     let config = MemoryConfig::load().await?;
//!     config.validate()?;
//!
//!     // Initialize search engine
//!     let engine = SearchEngine::new().await?;
//!
//!     // Execute search
//!     let results = engine.search("memory architecture", 10).await?;
//!
//!     for result in results {
//!         println!("{}: {} (confidence: {})", result.score, result.text, result.confidence);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Runtime modes
//!
//! No compile-time features: the engine picks a mode at startup.
//! - **Hybrid** (default): ONNX embeddings + FTS5, when `model.onnx` and `vocab.txt` are installed
//!   (see [`embeddings`]).
//! - **Keyword-only**: automatically, when no model is found or `enable_vector_search = false`.
//!
//! ## Error Handling
//!
//! All public functions return `Result<T>` where errors are explicitly typed:
//!
//! ```ignore
//! use plaesy::memory::Result;
//! use plaesy::memory::error::MemoryError;
//!
//! async fn search(query: &str) -> Result<Vec<SearchResult>> {
//!     // Errors bubble up and can be pattern-matched
//!     let embedding = embed_query(query).await?;
//!     let results = search_vectors(&embedding).await?;
//!     Ok(results)
//! }
//! ```
//!
//! ## Observability
//!
//! - **Structured logging**: `tracing` crate with optional JSON format
//! - **Metrics**: Prometheus-compatible metrics exported on HTTP endpoint
//! - **Diagnostics**: Built-in health checks and index statistics
//!
//! See [`metrics`] module for available metrics.
//!
//! ## Testing
//!
//! Run tests with:
//! ```bash
//! cargo test
//! cargo test --features full  # Test all features
//! ```
//!
//! Run benchmarks with:
//! ```bash
//! cargo bench
//! ```
//!
//! ## References
//!
//! - **Architecture**: `docs/ARCHITECTURE.md`
//! - **Best Practices**: `BEST_PRACTICES.md`
//! - **API Documentation**: <https://docs.rs/plaesy-memory/>

// ============================================================================
// Public Module Declarations
// ============================================================================

/// Error types for the memory system.
///
/// All public functions return `Result<T>` which is an alias for `std::result::Result<T, MemoryError>`.
/// Custom error variants cover indexing, embedding, storage, and search failures.
///
/// See [`error::MemoryError`] for the full error variant documentation.
pub mod error;

/// Document indexing subsystem.
///
/// Handles file watching, text extraction, chunking, and document management.
/// Provides:
/// - [`indexer::Extractor`] for YAML + Markdown parsing
/// - [`indexer::Chunker`] for multi-scale document chunking
/// - [`indexer::Watcher`] for debounced file system monitoring
///
/// See [`indexer`] module documentation for detailed examples.
pub mod indexer;

/// Embedding subsystem for semantic search.
///
/// Loads ONNX models and generates text embeddings (384-dimensional vectors).
/// Provides:
/// - [`embeddings::EmbeddingEngine`] for ONNX inference
/// - [`embeddings::EmbeddingVector`] type alias for embedding vectors
/// - [`embeddings::cosine_similarity`] utility function
///
/// See [`embeddings`] module documentation for configuration and examples.
pub mod embeddings;

/// Storage subsystem for metadata, vectors, and indexes.
///
/// Manages SQLite database (metadata, FTS5) and HNSW graph (vector index).
/// Provides:
/// - [`storage::Database`] for document/chunk storage and FTS queries
/// - [`storage::HnswIndex`] for vector similarity search
/// - WAL (write-ahead log) for crash recovery
///
/// See [`storage`] module documentation for schema and transaction patterns.
pub mod storage;

/// Search subsystem for query execution and ranking.
///
/// Combines vector search, keyword search, and ranking algorithms.
/// Provides:
/// - [`search::SearchEngine`] orchestrating hybrid search
/// - [`search::Ranker`] for combining and ranking results
/// - [`search::CitationBuilder`] for EV-01 compliance (source attribution)
///
/// See [`search`] module documentation for examples.
pub mod search;

/// Configuration management.
///
/// Loads and validates configuration from environment, files, and defaults.
/// Provides:
/// - [`config::MemoryConfig`] top-level configuration
/// - [`config::SearchConfig`] search-specific settings
/// - [`config::LoggingConfig`] logging configuration
///
/// Configuration sources (in precedence order):
/// 1. Environment variables (MEMORY_*)
/// 2. Project config (.plaesy/memory.toml)
/// 3. User config (~/.hermes/memory.toml)
/// 4. Hardcoded defaults
///
/// See [`config`] module documentation for details.
pub mod config;

/// Metrics and observability.
///
/// Exports Prometheus metrics and structured logging for:
/// - Search and indexing latencies
/// - Throughput and query rates
/// - Index health and corruption detection
/// - Cache hit rates and memory usage
///
/// See [`metrics`] module documentation for exported metrics.
pub mod metrics;

/// Line-delimited JSON service protocol (used by the Hermes plugin binary).
///
/// See [`service`] for the request/response format.
pub mod service;

/// Optional cross-encoder reranking of search results.
pub use search::rerank;

/// Model Context Protocol server (stdio).
pub mod mcp;

/// `memory serve`: the long-running service (JSON protocol or MCP) with background sync.
pub mod serve;

/// `memory config mcp`: per-app integration snippets for MCP clients.
pub use config::mcp as config_mcp;

/// `<root>/Plaesy/settings.json`, written by `memory onboard`.
pub mod settings;

/// `memory onboard`: first-run setup (settings, model download, first index).
pub mod onboard;

/// Per-OS default directories for models and data.
pub mod paths;

/// Memory hygiene checks (stale notes, broken links, dead path references).
pub mod hygiene;

/// Memory gaps: questions the memory could not answer, and broken or stale notes (`memory missing`).
pub mod gaps;

/// Writing new notes on behalf of an agent (`memory_remember`); never overwrites.
pub mod notes;

/// Scheduled jobs (`memory schedule once|run`): index, hygiene and missing at fixed intervals.
pub mod schedule;

// ============================================================================
// Re-exports and Convenience Aliases
// ============================================================================

/// Convenience type alias for `Result<T, MemoryError>`.
pub use error::{MemoryError, Result};

// ============================================================================
// Prelude Module
// ============================================================================

/// Prelude module: commonly-used types and traits.
///
/// Import with `use plaesy::memory::prelude::*;` to get:
/// - [`error::Result`] and [`error::MemoryError`]
/// - Core types from each subsystem
/// - Common traits and utilities
///
/// # Example
///
/// ```ignore
/// use plaesy::memory::prelude::*;
///
/// async fn my_search() -> Result<Vec<SearchResult>> {
///     // ... search logic ...
/// }
/// ```
pub mod prelude {
    // Core error handling
    pub use crate::memory::error::{MemoryError, Result};

    // Indexing types
    pub use crate::memory::indexer::{Chunk, ChunkKind, Document, Extractor, Watcher};

    // Embedding types and functions
    pub use crate::memory::embeddings::{cosine_similarity, EmbeddingConfig, EmbeddingVector};

    // Storage types
    pub use crate::memory::storage::{
        ConfidenceLevel, Database, DatabaseStats, HnswIndex, HnswStats, SearchResult,
    };

    // Search types
    pub use crate::memory::search::{
        Citation, CitationBuilder, RankerConfig, RankerType, SearchEngine,
    };

    // Configuration
    pub use crate::memory::config::{LoggingConfig, MemoryConfig, SearchConfig};

    // Metrics
    pub use crate::memory::metrics::{IndexHealth, Metrics, MetricsSnapshot, SearchType, Timer};
}

// ============================================================================
// Runtime Setup (Optional convenience functions)
// ============================================================================

/// Initialize logging and tracing.
///
/// Call this at application startup to set up structured logging.
///
/// # Example
///
/// ```ignore
/// plaesy::memory::init_logging()?;
/// ```
pub fn init_logging() -> Result<()> {
    init_logging_with(&config::LoggingConfig::default())
}

/// Initialize logging from a [`config::LoggingConfig`].
///
/// `RUST_LOG` takes precedence over `config.level` when set. Logs go to
/// `config.log_file` (appended) if given, else stderr. Calling this more than
/// once is harmless: the first subscriber wins.
///
/// # Errors
///
/// - `ConfigError` if `config.level` is not a valid filter directive
/// - `IoError` if the log file cannot be opened
pub fn init_logging_with(cfg: &config::LoggingConfig) -> Result<()> {
    use tracing_subscriber::EnvFilter;

    let filter = match std::env::var("RUST_LOG") {
        Ok(v) if !v.trim().is_empty() => EnvFilter::try_new(v),
        // A bare level ("info") also quiets ONNX Runtime's very chatty INFO logs.
        _ if !cfg.level.contains(['=', ',']) => {
            EnvFilter::try_new(format!("{},ort=warn", cfg.level))
        }
        _ => EnvFilter::try_new(&cfg.level),
    }
    .map_err(|e| MemoryError::ConfigError(format!("invalid log level '{}': {e}", cfg.level)))?;

    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    let result = match (&cfg.log_file, cfg.json_format) {
        (Some(path), json) => {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| MemoryError::IoError {
                    path: path.display().to_string(),
                    source: e,
                })?;
            let b = builder
                .with_ansi(false)
                .with_writer(std::sync::Mutex::new(file));
            if json {
                b.json().try_init()
            } else {
                b.try_init()
            }
        }
        (None, true) => builder.with_writer(std::io::stderr).json().try_init(),
        (None, false) => builder.with_writer(std::io::stderr).try_init(),
    };
    // A subscriber already being installed is fine (idempotent init).
    let _ = result;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_includes_error() {
        // Verify prelude is usable
        let _error_type: std::result::Result<(), MemoryError> = Ok(());
    }

    #[test]
    fn test_module_structure() {
        // Verify all modules are accessible
        let _ = error::MemoryError::InternalError("test".to_string());
    }
}
