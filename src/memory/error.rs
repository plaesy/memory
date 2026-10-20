//! Error types for the plaesy memory system.
//!
//! All public functions return `Result<T, MemoryError>`. The error type uses the `thiserror`
//! crate to provide ergonomic error handling with custom error variants for different failure
//! modes across indexing, embedding, storage, and search operations.
//!
//! # Design Principles
//!
//! - **No unwrap() in library code**: All error paths use `Result` and explicit error handling.
//! - **Contextual errors**: Each variant captures the operation and reason for failure.
//! - **Source chain**: Errors implement `std::error::Error` with optional source tracking via `#[from]`.
//! - **Graceful fallback**: Some errors (e.g., embedding timeout) allow fallback to keyword-only search.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Core error type for all memory system operations.
///
/// Variants cover:
/// - **Indexing**: File watching, extraction, chunking failures
/// - **Embedding**: ONNX model loading, inference, batch processing
/// - **Storage**: SQLite operations, transaction failures, schema violations
/// - **Search**: Vector search, keyword search, ranking failures
/// - **Config**: Configuration validation, missing required fields
#[derive(Error, Debug)]
pub enum MemoryError {
    /// File I/O operation failed (reading, writing, watching).
    #[error("I/O error while accessing {path}: {source}")]
    IoError {
        path: String,
        #[source]
        source: io::Error,
    },

    /// File system watcher encountered an error.
    #[error("File watcher error: {0}")]
    WatcherError(String),

    /// Failed to extract text from a file (invalid format, encoding, YAML parse).
    #[error("Failed to extract text from {path}: {reason}")]
    ExtractionError { path: PathBuf, reason: String },

    /// Failed to parse YAML frontmatter.
    #[error("YAML parse error in {path}: {reason}")]
    YamlError { path: PathBuf, reason: String },

    /// Failed to chunk document (invalid state, split logic failure).
    #[error("Chunking error: {0}")]
    ChunkingError(String),

    /// Failed to load or initialize ONNX embedding model.
    #[error("ONNX embedding model error: {0}")]
    EmbeddingModelError(String),

    /// Failed to run inference on a batch of text.
    #[error("Embedding inference failed for batch size {batch_size}: {reason}")]
    EmbeddingInferenceError { batch_size: usize, reason: String },

    /// Embedding operation timed out (fallback to keyword search allowed).
    #[error("Embedding operation timed out after {timeout_secs}s")]
    EmbeddingTimeout { timeout_secs: u64 },

    /// Vector dimension mismatch (query vs stored vectors).
    #[error("Vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// SQLite database operation failed.
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// Failed to execute SQL query.
    #[error("SQL execution error: {0}")]
    SqlError(String),

    /// Database transaction failed or rolled back.
    #[error("Transaction error: {reason}")]
    TransactionError { reason: String },

    /// Database schema mismatch or migration failure.
    #[error("Schema error: {reason}")]
    SchemaError { reason: String },

    /// WAL (write-ahead log) recovery detected corruption.
    #[error("WAL corruption detected at byte {offset}: {reason}")]
    WalCorruption { offset: u64, reason: String },

    /// HNSW index is corrupted or unreadable.
    #[error("HNSW index corruption: {0}")]
    HnswIndexCorruption(String),

    /// Vector search operation failed.
    #[error("Vector search failed: {0}")]
    VectorSearchError(String),

    /// Keyword (full-text) search operation failed.
    #[error("Keyword search failed: {0}")]
    KeywordSearchError(String),

    /// Ranking/re-ranking operation failed.
    #[error("Ranking error: {0}")]
    RankingError(String),

    /// Failed to build citations for results.
    #[error("Citation builder error: {0}")]
    CitationError(String),

    /// Configuration error (missing field, invalid value, parse failure).
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Feature not enabled in build configuration.
    #[error("Feature \"{feature}\" not enabled; rebuild with --features {feature}")]
    FeatureNotEnabled { feature: String },

    /// Serialization error (JSON, bincode, etc.).
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error (invalid JSON, bincode, etc.).
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Generic internal error (programmer logic, invariant violation).
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Operation cancelled or interrupted by caller.
    #[error("Operation cancelled")]
    Cancelled,

    /// Resource limit exceeded (too many open files, memory, timeout).
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
}

/// Specialized result type for memory system operations.
///
/// # Example
///
/// ```ignore
/// use plaesy::memory::Result;
///
/// async fn search(query: &str) -> Result<Vec<SearchResult>> {
///     let embedding = embed_query(query)?;
///     let results = search_vectors(&embedding)?;
///     Ok(results)
/// }
/// ```
pub type Result<T> = std::result::Result<T, MemoryError>;

impl MemoryError {
    /// Returns `true` if this error is recoverable (search can fall back).
    ///
    /// Examples of recoverable errors:
    /// - `EmbeddingTimeout`: fall back to keyword search
    /// - `VectorSearchError`: use keyword-only results
    ///
    /// Non-recoverable errors (require full failure):
    /// - `DatabaseError`: data corruption
    /// - `ConfigError`: invalid setup
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            MemoryError::EmbeddingTimeout { .. }
                | MemoryError::VectorSearchError(_)
                | MemoryError::EmbeddingInferenceError { .. }
        )
    }

    /// Returns a short description for user-facing error messages.
    pub fn user_message(&self) -> String {
        match self {
            MemoryError::EmbeddingTimeout { .. } => {
                "Semantic search timed out; using keyword search instead".to_string()
            }
            MemoryError::VectorSearchError(_) => {
                "Vector search unavailable; using keyword search instead".to_string()
            }
            MemoryError::DatabaseError(e) => {
                format!("Memory database error: {}", e)
            }
            MemoryError::FeatureNotEnabled { feature } => {
                format!(
                    "Feature '{}' not enabled. Rebuild with: cargo build --features {}",
                    feature, feature
                )
            }
            _ => self.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_is_recoverable() {
        let timeout_err = MemoryError::EmbeddingTimeout { timeout_secs: 5 };
        assert!(timeout_err.is_recoverable());

        let db_err = MemoryError::DatabaseError("corruption".to_string());
        assert!(!db_err.is_recoverable());
    }

    #[test]
    fn test_error_user_message() {
        let timeout_err = MemoryError::EmbeddingTimeout { timeout_secs: 5 };
        let msg = timeout_err.user_message();
        assert!(msg.contains("Semantic search"));
        assert!(msg.contains("keyword search"));
    }

    #[test]
    fn test_error_display() {
        let err = MemoryError::DimensionMismatch {
            expected: 384,
            got: 512,
        };
        assert_eq!(
            err.to_string(),
            "Vector dimension mismatch: expected 384, got 512"
        );
    }
}
