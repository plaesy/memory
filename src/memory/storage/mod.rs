//! Storage subsystem for metadata, vectors, and indexes.
//!
//! Manages two persistent stores:
//! - **SQLite**: Documents, chunks, metadata, FTS5 full-text index (WAL mode)
//! - **HNSW Graph**: Vector index for semantic search (usearch, saved to disk)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────┐
//! │ SQLite Database (state.db)      │
//! ├─────────────────────────────────┤
//! │ Table: documents                │
//! │ Table: chunks                   │
//! │ Table: index_metadata           │
//! │ Virtual: chunks_fts (FTS5)      │
//! └─────────────────────────────────┘
//!
//! ┌─────────────────────────────────┐
//! │ HNSW Graph (usearch)            │
//! ├─────────────────────────────────┤
//! │ key: chunk_id → vector (cosine) │
//! └─────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use plaesy::memory::storage::{Database, DatabaseConfig};
//!
//! let db = Database::new(DatabaseConfig::default()).await?;
//! let doc_id = db.insert_document(&doc).await?;
//! db.insert_chunks(doc_id, &doc.chunks).await?;
//! let results = db.search_fts("keyword", 10).await?;
//! ```

use crate::memory::embeddings::EmbeddingVector;
use crate::memory::error::{MemoryError, Result};
use crate::memory::indexer::{Chunk, ChunkKind, Document};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Database configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Path to SQLite database file (`:memory:` for an in-memory database).
    pub db_path: PathBuf,

    /// Enable WAL mode (write-ahead log) for crash recovery.
    pub wal_mode: bool,

    /// Cache size in pages (negative = KiB, positive = pages).
    pub cache_size: i32,

    /// Maximum number of concurrent readers.
    pub max_readers: usize,

    /// Enable FTS5 full-text search.
    pub enable_fts: bool,

    /// HNSW graph configuration.
    pub hnsw_config: HnswConfig,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        DatabaseConfig {
            db_path: PathBuf::from("memory.db"),
            wal_mode: true,
            cache_size: -64_000, // 64 MB
            max_readers: 10,
            enable_fts: true,
            hnsw_config: HnswConfig::default(),
        }
    }
}

/// HNSW (Hierarchical Navigable Small World) index configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Max neighbors per node (default 32, higher = more accurate).
    pub m: usize,

    /// Construction neighbors to explore (default 256).
    pub ef_construct: usize,

    /// Runtime neighbors to explore (default 128).
    pub ef_search: usize,

    /// Max neighbors for layer 0 (usually 2×m).
    pub max_m0: usize,

    /// Maximum number of layers.
    pub max_layers: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        HnswConfig {
            m: 32,
            ef_construct: 256,
            ef_search: 128,
            max_m0: 64,
            max_layers: 16,
        }
    }
}

const SCHEMA_VERSION: &str = "2";

/// The model every database built before model recording was made with.
const LEGACY_EMBEDDING_MODEL: &str = "all-MiniLM-L6-v2";

/// BM25 value at which the normalised keyword score reaches 0.5.
/// A lone weak term match scores ~0.2; strong multi-term matches 0.5+.
const BM25_HALF_SATURATION: f64 = 8.0;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    hash        TEXT,
    revision    TEXT,
    metadata    TEXT NOT NULL DEFAULT '{}',
    body        TEXT NOT NULL,
    indexed_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    layer       INTEGER NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    metadata    TEXT,
    embedding   BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE TABLE IF NOT EXISTS links (
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target      TEXT NOT NULL,
    line        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_links_document ON links(document_id);
CREATE TABLE IF NOT EXISTS index_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
";

const FTS_SCHEMA: &str = "
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
";

fn db_err(e: rusqlite::Error) -> MemoryError {
    MemoryError::DatabaseError(e.to_string())
}

/// Increment the persistent change counter used to validate HNSW snapshots.
/// Called by every write that adds, replaces or removes chunks or embeddings.
fn bump_generation(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO index_metadata(key, value) VALUES ('generation', '1')
         ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
        [],
    )?;
    Ok(())
}

fn json_err(e: serde_json::Error) -> MemoryError {
    MemoryError::SerializationError(e.to_string())
}

fn kind_to_str(kind: ChunkKind) -> &'static str {
    match kind {
        ChunkKind::Summary => "summary",
        ChunkKind::Detail => "detail",
        ChunkKind::Code => "code",
        ChunkKind::Quote => "quote",
        ChunkKind::Heading => "heading",
    }
}

fn kind_from_str(s: &str) -> ChunkKind {
    match s {
        "summary" => ChunkKind::Summary,
        "code" => ChunkKind::Code,
        "quote" => ChunkKind::Quote,
        "heading" => ChunkKind::Heading,
        _ => ChunkKind::Detail,
    }
}

fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_vec(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Convert a free-text query into a safe FTS5 expression.
///
/// Each alphanumeric token is double-quoted (so FTS5 operators in user input
/// are inert) and tokens are OR-ed. Returns `None` if no tokens remain.
///
/// Common English stopwords are dropped so they don't match every chunk; if
/// the query consists only of stopwords they are kept rather than returning nothing.
pub fn fts_query(query: &str) -> Option<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "in",
        "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "will",
        "with",
    ];
    let all: Vec<&str> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .collect();
    let content: Vec<&str> = all
        .iter()
        .copied()
        .filter(|t| !STOPWORDS.contains(&t.to_lowercase().as_str()))
        .collect();
    let chosen = if content.is_empty() { all } else { content };
    if chosen.is_empty() {
        None
    } else {
        Some(
            chosen
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(" OR "),
        )
    }
}

const RESULT_SELECT: &str = "SELECT c.id, c.text, d.source_path, c.start_line, c.end_line, \
                             d.revision, d.indexed_at FROM chunks c JOIN documents d ON d.id = c.document_id";

fn row_to_result(
    row: &rusqlite::Row<'_>,
    score: f32,
    confidence: ConfidenceLevel,
) -> rusqlite::Result<SearchResult> {
    Ok(SearchResult {
        chunk_id: row.get::<_, i64>(0)? as u64,
        text: row.get(1)?,
        source_path: row.get(2)?,
        start_line: row.get::<_, i64>(3)? as usize,
        end_line: row.get::<_, i64>(4)? as usize,
        revision: row.get(5)?,
        indexed_at: row.get(6)?,
        score,
        confidence,
        meta: ResultMeta::default(),
        explain: None,
    })
}

/// A document as stored in the index.
#[derive(Clone, Debug)]
pub struct StoredDocument {
    /// Path relative to the indexed root.
    pub path: String,

    /// YAML frontmatter.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Markdown body (frontmatter removed).
    pub body: String,

    /// 1-based line of the source file where `body` begins.
    pub body_start_line: usize,
}

/// A `[[wikilink]]` found in a document body.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LinkRef {
    /// Path of the document containing the link.
    pub source_path: String,

    /// Link target exactly as written (without alias or heading part).
    pub target: String,

    /// 1-based line of the link in the source file.
    pub line: usize,

    /// Path of the document the target resolves to, if any.
    pub resolved_path: Option<String>,
}

/// Whether a `[[...]]` body names a note rather than being a template directive.
///
/// Also applied when links are read back, so tightening these rules takes effect on an
/// existing index without re-indexing.
pub fn plausible_target(target: &str) -> bool {
    !target.is_empty()
        && target.len() <= 120
        && !target.contains(": ")
        && !target.contains("{{")
        && !is_directive(target)
        && !target.contains(['\n', '[', ']'])
}

/// `[[BROWNFIELD ONLY]]`-style conditional markers: several words, all upper case.
fn is_directive(target: &str) -> bool {
    target.contains(' ')
        && target.chars().any(char::is_alphabetic)
        && !target.chars().any(char::is_lowercase)
}

/// Extract `[[target]]`, `[[target|alias]]` and `[[target#heading]]` links from Markdown.
///
/// Returns `(target, 1-based line within body)`. Links inside fenced code blocks are ignored.
pub fn extract_wikilinks(body: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for (i, line) in body.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        // Links inside inline code spans are examples, not links.
        let visible: String = line
            .split('`')
            .enumerate()
            .filter(|(n, _)| n % 2 == 0)
            .map(|(_, part)| part)
            .collect::<Vec<_>>()
            .join(" ");
        let mut rest = visible.as_str();
        while let Some(start) = rest.find("[[") {
            let after = &rest[start + 2..];
            let Some(end) = after.find("]]") else { break };
            let raw = &after[..end];
            let target = raw
                .split(['|', '#'])
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            // Template directives such as `[[LLM: do this]]` or `[[{{PLACEHOLDER}}]]` are
            // not links: real note names are short and contain no sentence punctuation.
            if plausible_target(&target) {
                out.push((target, i + 1));
            }
            rest = &after[end + 2..];
        }
    }
    out
}

#[cfg(test)]
mod link_tests {
    use super::*;

    #[test]
    fn wikilinks_parse_aliases_headings_and_skip_noise() {
        let body = "See [[Alpha]] and [[beta/Gamma|the gamma]] plus [[delta#Heading]].\n\
                    ```\n[[in-fence]]\n```\n\
                    [[LLM: verify the claim]] [[{{TOKEN}}]] [[BROWNFIELD ONLY]] [[Real Note]] [[unclosed\n\
                    two on a line [[a]] [[b]] and `[[in-code]]` ignored\n";
        assert_eq!(
            extract_wikilinks(body),
            vec![
                ("Alpha".to_string(), 1),
                ("beta/Gamma".to_string(), 1),
                ("delta".to_string(), 1),
                ("Real Note".to_string(), 5),
                ("a".to_string(), 6),
                ("b".to_string(), 6),
            ]
        );
    }

    #[test]
    fn resolution_is_case_insensitive_and_stable() {
        let paths: Vec<String> = ["notes/Alpha.md", "z/alpha.md", "notes/beta/gamma.md"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            resolve_link("ALPHA", &paths).as_deref(),
            Some("notes/Alpha.md")
        );
        assert_eq!(
            resolve_link("beta/gamma", &paths).as_deref(),
            Some("notes/beta/gamma.md")
        );
        assert_eq!(
            resolve_link("notes/alpha.md", &paths).as_deref(),
            Some("notes/Alpha.md")
        );
        assert_eq!(resolve_link("nope", &paths), None);
    }
}

/// Resolve a wikilink target against known document paths.
///
/// Matches (case-insensitively) the full path, the path without extension, or the file
/// stem. Ambiguous stems resolve to the lexicographically first path so results are stable.
pub fn resolve_link(target: &str, paths: &[String]) -> Option<String> {
    let t = target.trim().trim_end_matches(".md").to_lowercase();
    let t = t.replace('\\', "/");
    let mut hits: Vec<&String> = paths
        .iter()
        .filter(|p| {
            let low = p.to_lowercase();
            let no_ext = low.rsplit_once('.').map(|(a, _)| a).unwrap_or(&low);
            let stem = no_ext.rsplit('/').next().unwrap_or(no_ext);
            low == t
                || no_ext == t
                || no_ext.ends_with(&format!("/{t}"))
                || (!t.contains('/') && stem == t)
        })
        .collect();
    hits.sort();
    hits.first().map(|s| (*s).clone())
}

/// SQLite-backed document storage.
pub struct Database {
    config: DatabaseConfig,
    conn: Mutex<Connection>,
}

impl Database {
    /// Create a new database instance.
    ///
    /// Initializes SQLite schema (documents, chunks, metadata tables) on first open.
    ///
    /// # Errors
    ///
    /// - `DatabaseError` if cannot open or initialize database
    pub async fn new(config: DatabaseConfig) -> Result<Self> {
        Self::open(config)
    }

    fn open(config: DatabaseConfig) -> Result<Self> {
        let in_memory = config.db_path.as_os_str() == ":memory:";
        if !in_memory {
            if let Some(parent) = config.db_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| MemoryError::IoError {
                        path: parent.display().to_string(),
                        source: e,
                    })?;
                }
            }
        }
        let conn = Connection::open(&config.db_path).map_err(db_err)?;
        // The CLI, the MCP server and a scheduled job may write at the same time; wait for the
        // lock instead of failing with "database is locked".
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(db_err)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .map_err(db_err)?;
        if config.wal_mode && !in_memory {
            conn.query_row("PRAGMA journal_mode = WAL;", [], |_| Ok(()))
                .map_err(db_err)?;
        }
        conn.execute_batch(&format!("PRAGMA cache_size = {};", config.cache_size))
            .map_err(db_err)?;
        conn.execute_batch(SCHEMA).map_err(db_err)?;
        if config.enable_fts {
            conn.execute_batch(FTS_SCHEMA).map_err(db_err)?;
        }
        conn.execute(
            "INSERT OR IGNORE INTO index_metadata(key, value) VALUES ('schema_version', ?1)",
            params![SCHEMA_VERSION],
        )
        .map_err(db_err)?;
        let has_body_start: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('documents') WHERE name = 'body_start'")
            .and_then(|mut st| st.exists([]))
            .map_err(db_err)?;
        if !has_body_start {
            conn.execute_batch(
                "ALTER TABLE documents ADD COLUMN body_start INTEGER NOT NULL DEFAULT 1;",
            )
            .map_err(db_err)?;
        }
        // v1 -> v2 added the `links` table. Documents indexed under v1 have no link rows, so
        // clear their hashes: the next sync sees them as changed and re-indexes them.
        let stored: String = conn
            .query_row(
                "SELECT value FROM index_metadata WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .map_err(db_err)?;
        if stored != SCHEMA_VERSION {
            conn.execute_batch("UPDATE documents SET hash = NULL;")
                .map_err(db_err)?;
            conn.execute(
                "UPDATE index_metadata SET value = ?1 WHERE key = 'schema_version'",
                params![SCHEMA_VERSION],
            )
            .map_err(db_err)?;
        }
        Ok(Database {
            config,
            conn: Mutex::new(conn),
        })
    }

    /// Record which embedding model the stored vectors come from.
    ///
    /// A database with documents but no record was built with the original model
    /// (`all-MiniLM-L6-v2`). If a different model was recorded, or the model changed now, every
    /// document's hash is cleared so the next sync re-embeds it. Returns `true` on such a change.
    pub async fn record_embedding_model(&self, model: &str) -> Result<bool> {
        let conn = self.conn.lock();
        let recorded: Option<String> = conn
            .query_row(
                "SELECT value FROM index_metadata WHERE key = 'embedding_model'",
                [],
                |r| r.get(0),
            )
            .optional()
            .map_err(db_err)?;
        let previous = match recorded {
            Some(m) => Some(m),
            None => {
                let docs: i64 = conn
                    .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
                    .map_err(db_err)?;
                (docs > 0).then(|| LEGACY_EMBEDDING_MODEL.to_string())
            }
        };
        let changed = previous.as_deref().is_some_and(|p| p != model);
        if changed {
            conn.execute_batch("UPDATE documents SET hash = NULL;")
                .map_err(db_err)?;
        }
        conn.execute(
            "INSERT INTO index_metadata(key, value) VALUES ('embedding_model', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![model],
        )
        .map_err(db_err)?;
        Ok(changed)
    }

    /// Insert or update a document (keyed by `source_path`).
    ///
    /// On update, all previous chunks of the document are removed (and with
    /// them their FTS rows); callers re-insert chunks afterwards. Returns the
    /// document row ID.
    pub async fn insert_document(&self, doc: &Document) -> Result<u64> {
        let path = doc.source_path.to_string_lossy().replace('\\', "/");
        let metadata = serde_json::to_string(&doc.metadata).map_err(json_err)?;
        let revision = doc
            .metadata
            .get("revision")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let indexed_at = doc.indexed_at.unwrap_or_else(chrono::Utc::now).to_rfc3339();
        let conn = self.conn.lock();
        conn.execute_batch("SAVEPOINT insert_document")
            .map_err(db_err)?;
        let res: rusqlite::Result<i64> = (|| {
            let id: i64 = conn.query_row(
                "INSERT INTO documents(source_path, hash, revision, metadata, body, indexed_at, body_start)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(source_path) DO UPDATE SET
                    hash = excluded.hash, revision = excluded.revision,
                    metadata = excluded.metadata, body = excluded.body,
                    indexed_at = excluded.indexed_at, body_start = excluded.body_start
                 RETURNING id",
                params![
                    path,
                    doc.hash,
                    revision,
                    metadata,
                    doc.body,
                    indexed_at,
                    doc.body_start_line.max(1) as i64
                ],
                |r| r.get(0),
            )?;
            conn.execute("DELETE FROM chunks WHERE document_id = ?1", params![id])?;
            conn.execute("DELETE FROM links WHERE document_id = ?1", params![id])?;
            for (target, line) in extract_wikilinks(&doc.body) {
                // `line` is relative to the body; report it relative to the file.
                let file_line = line + doc.body_start_line.saturating_sub(1);
                conn.execute(
                    "INSERT INTO links(document_id, target, line) VALUES (?1, ?2, ?3)",
                    params![id, target, file_line as i64],
                )?;
            }
            bump_generation(&conn)?;
            Ok(id)
        })();
        match res {
            Ok(id) => {
                conn.execute_batch("RELEASE insert_document")
                    .map_err(db_err)?;
                Ok(id as u64)
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK TO insert_document; RELEASE insert_document");
                Err(db_err(e))
            }
        }
    }

    /// Insert a batch of chunks belonging to `document_id`.
    ///
    /// Returns the assigned chunk IDs in input order. The batch is atomic.
    pub async fn insert_chunks(&self, document_id: u64, chunks: &[Chunk]) -> Result<Vec<u64>> {
        let conn = self.conn.lock();
        conn.execute_batch("SAVEPOINT insert_chunks")
            .map_err(db_err)?;
        let res: Result<Vec<u64>> = (|| {
            let mut stmt = conn
                .prepare_cached(
                    "INSERT INTO chunks(document_id, text, kind, layer, start_line, end_line, metadata)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                )
                .map_err(db_err)?;
            let mut ids = Vec::with_capacity(chunks.len());
            for c in chunks {
                let meta = match &c.metadata {
                    Some(m) => Some(serde_json::to_string(m).map_err(json_err)?),
                    None => None,
                };
                stmt.execute(params![
                    document_id as i64,
                    c.text,
                    kind_to_str(c.kind),
                    c.layer as i64,
                    c.start_line as i64,
                    c.end_line as i64,
                    meta
                ])
                .map_err(db_err)?;
                ids.push(conn.last_insert_rowid() as u64);
            }
            Ok(ids)
        })();
        match res {
            Ok(ids) => {
                conn.execute_batch("RELEASE insert_chunks")
                    .map_err(db_err)?;
                Ok(ids)
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK TO insert_chunks; RELEASE insert_chunks");
                Err(e)
            }
        }
    }

    /// Update chunk embeddings (vectors).
    ///
    /// # Errors
    ///
    /// - `InternalError` if the slices differ in length
    pub async fn update_chunk_embeddings(
        &self,
        chunk_ids: &[u64],
        embeddings: &[EmbeddingVector],
    ) -> Result<()> {
        if chunk_ids.len() != embeddings.len() {
            return Err(MemoryError::InternalError(format!(
                "chunk_ids ({}) and embeddings ({}) length mismatch",
                chunk_ids.len(),
                embeddings.len()
            )));
        }
        let conn = self.conn.lock();
        conn.execute_batch("SAVEPOINT update_emb").map_err(db_err)?;
        let res: rusqlite::Result<()> = (|| {
            let mut stmt = conn.prepare_cached("UPDATE chunks SET embedding = ?1 WHERE id = ?2")?;
            for (id, v) in chunk_ids.iter().zip(embeddings) {
                stmt.execute(params![vec_to_blob(v), *id as i64])?;
            }
            bump_generation(&conn)?;
            Ok(())
        })();
        match res {
            Ok(()) => conn.execute_batch("RELEASE update_emb").map_err(db_err),
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK TO update_emb; RELEASE update_emb");
                Err(db_err(e))
            }
        }
    }

    /// Search using full-text search (FTS5, BM25 ranking).
    ///
    /// Scores are BM25 squashed into (0, 1) via `s / (s + 8)` (absolute, not relative to the best hit). Every hit is
    /// verbatim stored text containing a query term, so confidence is `Fact`.
    pub async fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let Some(expr) = fts_query(query) else {
            return Ok(Vec::new());
        };
        let conn = self.conn.lock();
        // bm25() is negative (more negative = better), so ORDER BY ascending.
        let sql = "SELECT c.id, c.text, d.source_path, c.start_line, c.end_line, \
                   d.revision, d.indexed_at, f.r \
                   FROM (SELECT rowid AS rid, bm25(chunks_fts) AS r FROM chunks_fts \
                         WHERE chunks_fts MATCH ?1 ORDER BY r LIMIT ?2) f \
                   JOIN chunks c ON c.id = f.rid \
                   JOIN documents d ON d.id = c.document_id ORDER BY f.r";
        let kerr = |e: rusqlite::Error| MemoryError::KeywordSearchError(e.to_string());
        let mut stmt = conn.prepare(sql).map_err(kerr)?;
        let rows = stmt
            .query_map(params![expr, limit as i64], |row| {
                let r: f64 = row.get(7)?;
                let s = (-r).max(0.0);
                row_to_result(
                    row,
                    (s / (s + BM25_HALF_SATURATION)) as f32,
                    ConfidenceLevel::Fact,
                )
            })
            .map_err(kerr)?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(kerr)
    }

    /// Build a `SearchResult` (with full citation data) for a stored chunk.
    pub async fn result_for_chunk(
        &self,
        chunk_id: u64,
        score: f32,
        confidence: ConfidenceLevel,
    ) -> Result<Option<SearchResult>> {
        let conn = self.conn.lock();
        conn.query_row(
            &format!("{RESULT_SELECT} WHERE c.id = ?1"),
            params![chunk_id as i64],
            |row| row_to_result(row, score, confidence),
        )
        .optional()
        .map_err(db_err)
    }

    fn row_to_chunk(row: &rusqlite::Row<'_>) -> rusqlite::Result<Chunk> {
        let meta: Option<String> = row.get(7)?;
        let metadata: Option<HashMap<String, String>> =
            meta.and_then(|m| serde_json::from_str(&m).ok());
        Ok(Chunk {
            id: Some(row.get::<_, i64>(0)? as u64),
            document_id: Some(row.get::<_, i64>(1)? as u64),
            text: row.get(2)?,
            kind: kind_from_str(&row.get::<_, String>(3)?),
            layer: row.get::<_, i64>(4)? as usize,
            start_line: row.get::<_, i64>(5)? as usize,
            end_line: row.get::<_, i64>(6)? as usize,
            metadata,
        })
    }

    const CHUNK_COLS: &'static str =
        "id, document_id, text, kind, layer, start_line, end_line, metadata";

    /// Retrieve a chunk by ID.
    pub async fn get_chunk(&self, chunk_id: u64) -> Result<Option<Chunk>> {
        let conn = self.conn.lock();
        conn.query_row(
            &format!("SELECT {} FROM chunks WHERE id = ?1", Self::CHUNK_COLS),
            params![chunk_id as i64],
            Self::row_to_chunk,
        )
        .optional()
        .map_err(db_err)
    }

    /// Get all chunks for a document, in document order.
    pub async fn get_document_chunks(&self, doc_id: u64) -> Result<Vec<Chunk>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {} FROM chunks WHERE document_id = ?1 ORDER BY id",
                Self::CHUNK_COLS
            ))
            .map_err(db_err)?;
        let rows = stmt
            .query_map(params![doc_id as i64], Self::row_to_chunk)
            .map_err(db_err)?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(db_err)
    }

    /// Look up a document by source path: returns `(id, content hash)`.
    pub async fn find_document(&self, source_path: &str) -> Result<Option<(u64, Option<String>)>> {
        let path = source_path.replace('\\', "/");
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT id, hash FROM documents WHERE source_path = ?1",
            params![path],
            |r| Ok((r.get::<_, i64>(0)? as u64, r.get(1)?)),
        )
        .optional()
        .map_err(db_err)
    }

    /// Number of chunks that have a stored embedding.
    pub async fn embedding_count(&self) -> Result<u64> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL",
            [],
            |r| r.get::<_, i64>(0),
        )
        .map(|n| n as u64)
        .map_err(db_err)
    }

    /// All stored documents as `(id, source_path)`, ordered by path.
    pub async fn document_paths(&self) -> Result<Vec<(u64, String)>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare("SELECT id, source_path FROM documents ORDER BY source_path")
            .map_err(db_err)?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)? as u64, r.get(1)?)))
            .map_err(db_err)?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(db_err)
    }

    /// IDs of all chunks belonging to a document.
    pub async fn chunk_ids_for_document(&self, doc_id: u64) -> Result<Vec<u64>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare("SELECT id FROM chunks WHERE document_id = ?1 ORDER BY id")
            .map_err(db_err)?;
        let rows = stmt
            .query_map(params![doc_id as i64], |r| Ok(r.get::<_, i64>(0)? as u64))
            .map_err(db_err)?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(db_err)
    }

    /// All stored `(chunk_id, embedding)` pairs (used to rebuild the HNSW index).
    pub async fn all_embeddings(&self) -> Result<Vec<(u64, EmbeddingVector)>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id")
            .map_err(db_err)?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)? as u64,
                    blob_to_vec(&r.get::<_, Vec<u8>>(1)?),
                ))
            })
            .map_err(db_err)?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(db_err)
    }

    /// Delete a document and its chunks (FK cascade; FTS rows removed by trigger).
    pub async fn delete_document(&self, doc_id: u64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "DELETE FROM documents WHERE id = ?1",
            params![doc_id as i64],
        )
        .map_err(db_err)?;
        bump_generation(&conn).map_err(db_err)?;
        Ok(())
    }

    /// Frontmatter metadata of the given documents, keyed by `source_path`.
    pub async fn document_metadata(
        &self,
        paths: &[String],
    ) -> Result<HashMap<String, HashMap<String, serde_json::Value>>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare_cached("SELECT metadata FROM documents WHERE source_path = ?1")
            .map_err(db_err)?;
        let mut out = HashMap::new();
        for p in paths {
            let raw: Option<String> = stmt
                .query_row(params![p], |r| r.get(0))
                .optional()
                .map_err(db_err)?;
            if let Some(raw) = raw {
                out.insert(p.clone(), serde_json::from_str(&raw).unwrap_or_default());
            }
        }
        Ok(out)
    }

    /// Every stored document, ordered by path.
    pub async fn all_documents(&self) -> Result<Vec<StoredDocument>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT source_path, metadata, body, body_start FROM documents ORDER BY source_path",
            )
            .map_err(db_err)?;
        let rows = stmt
            .query_map([], |r| {
                let raw: String = r.get(1)?;
                Ok(StoredDocument {
                    path: r.get(0)?,
                    metadata: serde_json::from_str(&raw).unwrap_or_default(),
                    body: r.get(2)?,
                    body_start_line: r.get::<_, i64>(3)?.max(1) as usize,
                })
            })
            .map_err(db_err)?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(db_err)
    }

    /// All wikilinks with their resolution against the indexed documents.
    pub async fn all_links(&self) -> Result<Vec<LinkRef>> {
        let (raw, paths): (Vec<(String, String, usize)>, Vec<String>) = {
            let conn = self.conn.lock();
            let mut stmt = conn
                .prepare(
                    "SELECT d.source_path, l.target, l.line FROM links l                      JOIN documents d ON d.id = l.document_id ORDER BY d.source_path, l.line",
                )
                .map_err(db_err)?;
            let raw = stmt
                .query_map([], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get::<_, i64>(2)? as usize))
                })
                .map_err(db_err)?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(db_err)?;
            let mut pstmt = conn
                .prepare("SELECT source_path FROM documents")
                .map_err(db_err)?;
            let paths = pstmt
                .query_map([], |r| r.get(0))
                .map_err(db_err)?
                .collect::<rusqlite::Result<Vec<String>>>()
                .map_err(db_err)?;
            (raw, paths)
        };
        Ok(raw
            .into_iter()
            .filter(|(_, target, _)| plausible_target(target))
            .map(|(source_path, target, line)| {
                let resolved_path = resolve_link(&target, &paths);
                LinkRef {
                    source_path,
                    target,
                    line,
                    resolved_path,
                }
            })
            .collect())
    }

    /// Persistent change counter: increases on every write that can affect the vector index.
    pub async fn generation(&self) -> Result<u64> {
        let conn = self.conn.lock();
        let v: Option<String> = conn
            .query_row(
                "SELECT value FROM index_metadata WHERE key = 'generation'",
                [],
                |r| r.get(0),
            )
            .optional()
            .map_err(db_err)?;
        Ok(v.and_then(|s| s.parse().ok()).unwrap_or(0))
    }

    /// HNSW parameters from the database configuration.
    pub fn hnsw_config(&self) -> &HnswConfig {
        &self.config.hnsw_config
    }

    /// Lock the SQLite connection. Crate-internal so feature tables (see `gaps`) can sit beside the core schema.
    pub(crate) fn conn(&self) -> parking_lot::MutexGuard<'_, Connection> {
        self.conn.lock()
    }

    /// Path of the database file, or `None` for an in-memory database.
    pub fn path(&self) -> Option<&Path> {
        let p = self.config.db_path.as_path();
        (p.as_os_str() != ":memory:").then_some(p)
    }

    /// Get database statistics (diagnostic).
    ///
    /// `hnsw_size_bytes` and `coverage_percent` are not known to the database
    /// and are reported as 0; callers owning the HNSW index / file tree fill them.
    pub async fn statistics(&self) -> Result<DatabaseStats> {
        let conn = self.conn.lock();
        let count = |sql: &str| -> Result<u64> {
            conn.query_row(sql, [], |r| r.get::<_, i64>(0))
                .map(|n| n as u64)
                .map_err(db_err)
        };
        let documents_total = count("SELECT COUNT(*) FROM documents")?;
        let chunks_total = count("SELECT COUNT(*) FROM chunks")?;
        let page_count = count("PRAGMA page_count")?;
        let page_size = count("PRAGMA page_size")?;
        let fts_index_size_bytes = if self.config.enable_fts {
            count("SELECT COALESCE(SUM(LENGTH(block)), 0) FROM chunks_fts_data")?
        } else {
            0
        };
        Ok(DatabaseStats {
            documents_total,
            chunks_total,
            db_size_bytes: page_count * page_size,
            fts_index_size_bytes,
            hnsw_size_bytes: 0,
            coverage_percent: 0.0,
        })
    }

    /// Begin a transaction.
    pub async fn begin_transaction(&self) -> Result<()> {
        self.conn
            .lock()
            .execute_batch("BEGIN")
            .map_err(|e| MemoryError::TransactionError {
                reason: e.to_string(),
            })
    }

    /// Commit current transaction.
    pub async fn commit(&self) -> Result<()> {
        self.conn
            .lock()
            .execute_batch("COMMIT")
            .map_err(|e| MemoryError::TransactionError {
                reason: e.to_string(),
            })
    }

    /// Rollback current transaction.
    pub async fn rollback(&self) -> Result<()> {
        self.conn
            .lock()
            .execute_batch("ROLLBACK")
            .map_err(|e| MemoryError::TransactionError {
                reason: e.to_string(),
            })
    }
}

/// HNSW vector index for semantic search (cosine metric), backed by usearch.
pub struct HnswIndex {
    config: HnswConfig,
    dimensions: usize,
    index: usearch::Index,
}

/// Default embedding dimension (all-MiniLM-L6-v2).
const DEFAULT_DIMENSIONS: usize = 384;

fn hnsw_err(e: impl std::fmt::Display) -> MemoryError {
    MemoryError::HnswIndexCorruption(e.to_string())
}

fn build_usearch(config: &HnswConfig, dimensions: usize) -> Result<usearch::Index> {
    let options = usearch::IndexOptions {
        dimensions,
        metric: usearch::MetricKind::Cos,
        quantization: usearch::ScalarKind::F32,
        connectivity: config.m,
        expansion_add: config.ef_construct,
        expansion_search: config.ef_search,
        multi: false,
    };
    usearch::Index::new(&options).map_err(hnsw_err)
}

impl HnswIndex {
    /// Create a new HNSW index for vectors of `dimensions` length.
    ///
    /// # Errors
    ///
    /// - `HnswIndexCorruption` if the underlying index cannot be created
    pub fn with_dimensions(config: HnswConfig, dimensions: usize) -> Result<Self> {
        let index = build_usearch(&config, dimensions)?;
        Ok(HnswIndex {
            config,
            dimensions,
            index,
        })
    }

    /// Create a new HNSW index with the default 384 dimensions.
    ///
    /// # Errors
    ///
    /// - `HnswIndexCorruption` if the underlying index cannot be created
    pub fn new(config: HnswConfig) -> Result<Self> {
        Self::with_dimensions(config, DEFAULT_DIMENSIONS)
    }

    /// Vector dimension of this index.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Number of vectors stored.
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// Whether the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn ensure_capacity(&self, extra: usize) -> Result<()> {
        let need = self.index.size() + extra;
        if need > self.index.capacity() {
            self.index.reserve((need * 2).max(1024)).map_err(hnsw_err)?;
        }
        Ok(())
    }

    /// Add (or replace) a vector in the index.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the vector length differs from the index dimension
    /// - `HnswIndexCorruption` if index state is invalid
    pub fn insert(&mut self, id: u64, vector: &EmbeddingVector) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }
        if self.index.contains(id) {
            self.index.remove(id).map_err(hnsw_err)?;
        }
        self.ensure_capacity(1)?;
        self.index.add(id, vector).map_err(hnsw_err)
    }

    /// Remove a vector; returns whether it was present.
    pub fn remove(&mut self, id: u64) -> Result<bool> {
        if !self.index.contains(id) {
            return Ok(false);
        }
        self.index.remove(id).map_err(hnsw_err)?;
        Ok(true)
    }

    /// Search for nearest neighbors.
    ///
    /// Returns up to `k` `(id, cosine distance)` pairs, nearest first.
    pub fn search(&self, query: &EmbeddingVector, k: usize) -> Result<Vec<(u64, f32)>> {
        if query.len() != self.dimensions {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }
        if self.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let m = self
            .index
            .search(query, k)
            .map_err(|e| MemoryError::VectorSearchError(e.to_string()))?;
        Ok(m.keys.into_iter().zip(m.distances).collect())
    }

    fn meta_path(path: &Path) -> PathBuf {
        let mut s = path.as_os_str().to_owned();
        s.push(".meta.json");
        PathBuf::from(s)
    }

    /// Load index from disk (the file written by [`HnswIndex::save`] plus its `.meta.json`).
    pub fn load(path: &Path) -> Result<Self> {
        let meta_path = Self::meta_path(path);
        let raw = std::fs::read_to_string(&meta_path).map_err(|e| MemoryError::IoError {
            path: meta_path.display().to_string(),
            source: e,
        })?;
        let (config, dimensions): (HnswConfig, usize) = serde_json::from_str(&raw)
            .map_err(|e| MemoryError::DeserializationError(e.to_string()))?;
        let index = build_usearch(&config, dimensions)?;
        index.load(&path.to_string_lossy()).map_err(hnsw_err)?;
        Ok(HnswIndex {
            config,
            dimensions,
            index,
        })
    }

    /// Save index to disk, plus a `<path>.meta.json` sidecar with config and dimensions.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| MemoryError::IoError {
                    path: parent.display().to_string(),
                    source: e,
                })?;
            }
        }
        self.index.save(&path.to_string_lossy()).map_err(hnsw_err)?;
        let meta = serde_json::to_string(&(&self.config, self.dimensions)).map_err(json_err)?;
        let meta_path = Self::meta_path(path);
        std::fs::write(&meta_path, meta).map_err(|e| MemoryError::IoError {
            path: meta_path.display().to_string(),
            source: e,
        })
    }

    /// Get index statistics.
    pub fn statistics(&self) -> HnswStats {
        HnswStats {
            nodes_total: self.index.size() as u64,
            dimensions: self.dimensions,
            connectivity: self.index.connectivity(),
            memory_bytes: self.index.memory_usage() as u64,
        }
    }
}

/// Result of a search query (single match) with full citation data (EV-01).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// Chunk ID.
    pub chunk_id: u64,

    /// Chunk text (verbatim stored text, never generated).
    pub text: String,

    /// Source file path.
    pub source_path: String,

    /// First line of the chunk in the source document.
    pub start_line: usize,

    /// Last line of the chunk in the source document.
    pub end_line: usize,

    /// Git revision of the source at index time, if known.
    pub revision: Option<String>,

    /// RFC 3339 timestamp when the source was indexed.
    pub indexed_at: String,

    /// Relevance score (0-1).
    pub score: f32,

    /// Confidence level (FACT, INFERRED, SPECULATIVE).
    pub confidence: ConfidenceLevel,

    /// Source-document metadata (tier, trust, freshness). Filled by the search engine.
    #[serde(default)]
    pub meta: ResultMeta,

    /// Per-leg score breakdown, populated only when explain is requested.
    #[serde(default)]
    pub explain: Option<LegScores>,
}

/// Document-level metadata attached to a search result.
///
/// Read from the source file's YAML frontmatter (`tier`, `confidence`, `last_verified`,
/// `status`, `supersedes`); `tier` falls back to the path (see `search::infer_tier`).
/// This is the *source's* declared trust, distinct from the per-result FACT/INFERRED label.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResultMeta {
    /// Memory tier: `core`, `episodic`, `semantic`, `procedural` or `archive`.
    pub tier: Option<String>,

    /// Declared trust of the source: `high`, `medium` or `low`.
    pub trust: Option<String>,

    /// Date the source's claims were last verified (as written in the file).
    pub last_verified: Option<String>,

    /// Lifecycle status, e.g. `active`, `superseded`, `deprecated`.
    pub status: Option<String>,

    /// First day the source's claims hold (`valid_from` in frontmatter), if declared.
    pub valid_from: Option<String>,

    /// Last day the source's claims hold (`valid_to`, inclusive), if declared.
    pub valid_to: Option<String>,

    /// `true` when `last_verified` is older than the configured staleness window.
    pub stale: bool,
}

/// Per-leg score breakdown attached to a search result for `--explain`.
///
/// `None` means that leg did not contribute to this result. Each field is an
/// `Option<f32>` so keyword-only results can omit the vector leg and vice versa.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LegScores {
    /// Raw full-text (BM25) score, if present.
    #[serde(default)]
    pub fts: Option<f32>,
    /// Raw embedding (cosine distance → similarity) score, if present.
    #[serde(default)]
    pub vector: Option<f32>,
    /// Fused score after RRF / weighted fusion, before re-ranking.
    #[serde(default)]
    pub fused: Option<f32>,
    /// Staleness factor: 1.0 when not stale, reduced when the source is stale.
    #[serde(default)]
    pub staleness_factor: Option<f32>,
    /// Re-ranked score, if the reranker was enabled and applied.
    #[serde(default)]
    pub rerank: Option<f32>,
}

/// Confidence level for a search result.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConfidenceLevel {
    /// Directly from source (no inference).
    Fact,

    /// Inferred from context.
    Inferred,

    /// Speculative (model-added content).
    Speculative,
}

/// Database statistics (diagnostic).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatabaseStats {
    /// Total number of documents.
    pub documents_total: u64,

    /// Total number of chunks.
    pub chunks_total: u64,

    /// Database file size in bytes.
    pub db_size_bytes: u64,

    /// FTS index size in bytes.
    pub fts_index_size_bytes: u64,

    /// HNSW graph size in bytes (filled by the owner of the index).
    pub hnsw_size_bytes: u64,

    /// Coverage percentage (indexed files / total files in .plaesy/).
    pub coverage_percent: f64,
}

/// HNSW graph statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HnswStats {
    /// Total number of vectors in the graph.
    pub nodes_total: u64,

    /// Vector dimension.
    pub dimensions: usize,

    /// Configured connectivity (max neighbors per node).
    pub connectivity: usize,

    /// Approximate memory used by the index in bytes.
    pub memory_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn doc(path: &str, body: &str) -> Document {
        Document {
            id: None,
            source_path: PathBuf::from(path),
            hash: Some("h1".into()),
            metadata: HashMap::new(),
            body: body.into(),
            frontmatter_raw: None,
            body_start_line: 1,
            chunks: Vec::new(),
            indexed_at: None,
        }
    }

    fn chunk(text: &str, start: usize, end: usize) -> Chunk {
        Chunk {
            id: None,
            document_id: None,
            text: text.into(),
            kind: ChunkKind::Detail,
            layer: 1,
            start_line: start,
            end_line: end,
            metadata: None,
        }
    }

    async fn mem_db() -> Database {
        let cfg = DatabaseConfig {
            db_path: PathBuf::from(":memory:"),
            ..DatabaseConfig::default()
        };
        Database::new(cfg).await.expect("open db")
    }

    #[test]
    fn test_hnsw_config_default() {
        let config = HnswConfig::default();
        assert_eq!(config.m, 32);
        assert_eq!(config.ef_construct, 256);
        assert_eq!(config.ef_search, 128);
    }

    #[test]
    fn test_database_config_default() {
        let config = DatabaseConfig::default();
        assert!(config.wal_mode);
        assert!(config.enable_fts);
    }

    #[test]
    fn test_confidence_level_serialize() {
        let conf = ConfidenceLevel::Fact;
        let json = serde_json::to_string(&conf).expect("serialize");
        assert_eq!(json, "\"Fact\"");
    }

    #[test]
    fn test_fts_query_sanitizes() {
        assert_eq!(
            fts_query("hello, world"),
            Some("\"hello\" OR \"world\"".into())
        );
        // Operators in user input are inert (quoted); stopwords like "a"/"OR" are dropped.
        assert_eq!(fts_query("a\" OR NEAR("), Some("\"NEAR\"".into()));
        assert_eq!(
            fts_query("cite the SOURCE of results"),
            Some("\"cite\" OR \"SOURCE\" OR \"results\"".into())
        );
        // Stopword-only queries are kept rather than emptied.
        assert_eq!(fts_query("the of"), Some("\"the\" OR \"of\"".into()));
        assert_eq!(fts_query("  !!! "), None);
    }

    #[tokio::test]
    async fn test_model_change_clears_hashes_only_when_it_changes() {
        let db = mem_db().await;
        // Empty database: nothing to re-embed, the model is simply recorded.
        assert!(!db
            .record_embedding_model("multilingual-e5-small")
            .await
            .expect("first"));
        assert!(!db
            .record_embedding_model("multilingual-e5-small")
            .await
            .expect("same"));

        // A document is stored with a hash; switching the model clears that hash.
        db.insert_document(&doc(".plaesy/a.md", "body"))
            .await
            .expect("doc");
        assert!(db
            .record_embedding_model("all-MiniLM-L6-v2")
            .await
            .expect("switch"));
        let hashes: i64 = db
            .conn
            .lock()
            .query_row(
                "SELECT COUNT(*) FROM documents WHERE hash IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(
            hashes, 0,
            "documents must be re-embedded after a model change"
        );
    }

    #[tokio::test]
    async fn test_legacy_database_is_treated_as_minilm() {
        let db = mem_db().await;
        db.insert_document(&doc(".plaesy/old.md", "body"))
            .await
            .expect("doc");
        // Built before model recording: switching to E5 is a change, staying on MiniLM is not.
        assert!(!db
            .record_embedding_model(LEGACY_EMBEDDING_MODEL)
            .await
            .expect("same"));
        assert!(db
            .record_embedding_model("multilingual-e5-small")
            .await
            .expect("change"));
    }

    #[tokio::test]
    async fn test_insert_and_fts_search_with_citation() {
        let db = mem_db().await;
        let d = doc(".plaesy/a.md", "body");
        let did = db.insert_document(&d).await.expect("doc");
        let ids = db
            .insert_chunks(
                did,
                &[
                    chunk("vector embeddings for search", 3, 5),
                    chunk("unrelated cooking text", 6, 8),
                ],
            )
            .await
            .expect("chunks");
        assert_eq!(ids.len(), 2);

        let hits = db.search_fts("embeddings", 10).await.expect("fts");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, ids[0]);
        assert_eq!(hits[0].source_path, ".plaesy/a.md");
        assert_eq!((hits[0].start_line, hits[0].end_line), (3, 5));
        assert_eq!(hits[0].confidence, ConfidenceLevel::Fact);
        assert!(hits[0].score > 0.0 && hits[0].score < 1.0);
        assert!(!hits[0].indexed_at.is_empty());
    }

    #[tokio::test]
    async fn test_fts_rejects_operator_injection() {
        let db = mem_db().await;
        let did = db.insert_document(&doc("a.md", "b")).await.expect("doc");
        db.insert_chunks(did, &[chunk("plain text", 1, 1)])
            .await
            .expect("chunks");
        // Would be an FTS5 syntax error if passed through raw.
        assert!(db.search_fts("\"unbalanced AND (", 5).await.is_ok());
        assert!(db.search_fts("", 5).await.expect("empty").is_empty());
    }

    #[tokio::test]
    async fn test_reinsert_document_replaces_chunks_and_fts() {
        let db = mem_db().await;
        let did = db.insert_document(&doc("a.md", "v1")).await.expect("doc");
        db.insert_chunks(did, &[chunk("oldterm here", 1, 1)])
            .await
            .expect("c");
        let did2 = db.insert_document(&doc("a.md", "v2")).await.expect("doc2");
        assert_eq!(did, did2);
        assert!(db.get_document_chunks(did).await.expect("get").is_empty());
        assert!(db.search_fts("oldterm", 5).await.expect("fts").is_empty());
        let st = db.statistics().await.expect("stats");
        assert_eq!((st.documents_total, st.chunks_total), (1, 0));
    }

    #[tokio::test]
    async fn test_delete_document_cascades() {
        let db = mem_db().await;
        let did = db.insert_document(&doc("a.md", "b")).await.expect("doc");
        let ids = db
            .insert_chunks(did, &[chunk("findme", 1, 1)])
            .await
            .expect("c");
        db.delete_document(did).await.expect("del");
        assert!(db.get_chunk(ids[0]).await.expect("get").is_none());
        assert!(db.search_fts("findme", 5).await.expect("fts").is_empty());
        assert!(db.find_document("a.md").await.expect("find").is_none());
    }

    #[tokio::test]
    async fn test_embeddings_roundtrip_and_length_mismatch() {
        let db = mem_db().await;
        let did = db.insert_document(&doc("a.md", "b")).await.expect("doc");
        let ids = db.insert_chunks(did, &[chunk("x", 1, 1)]).await.expect("c");
        db.update_chunk_embeddings(&ids, &[vec![0.5, -1.0, 2.0]])
            .await
            .expect("emb");
        let all = db.all_embeddings().await.expect("all");
        assert_eq!(all, vec![(ids[0], vec![0.5, -1.0, 2.0])]);
        assert!(db.update_chunk_embeddings(&ids, &[]).await.is_err());
    }

    #[tokio::test]
    async fn test_transaction_rollback() {
        let db = mem_db().await;
        db.begin_transaction().await.expect("begin");
        db.insert_document(&doc("a.md", "b")).await.expect("doc");
        db.rollback().await.expect("rollback");
        assert_eq!(db.statistics().await.expect("stats").documents_total, 0);
    }

    #[tokio::test]
    async fn test_persistent_db_reopens() {
        let dir = tempfile::tempdir().expect("tmp");
        let cfg = DatabaseConfig {
            db_path: dir.path().join("sub").join("state.db"),
            ..DatabaseConfig::default()
        };
        {
            let db = Database::new(cfg.clone()).await.expect("open");
            let did = db.insert_document(&doc("a.md", "b")).await.expect("doc");
            db.insert_chunks(did, &[chunk("persisted", 1, 1)])
                .await
                .expect("c");
        }
        let db = Database::new(cfg).await.expect("reopen");
        assert_eq!(db.search_fts("persisted", 5).await.expect("fts").len(), 1);
    }

    #[test]
    fn test_hnsw_insert_search_remove() {
        let mut idx = HnswIndex::with_dimensions(HnswConfig::default(), 3).expect("idx");
        idx.insert(1, &vec![1.0, 0.0, 0.0]).expect("i1");
        idx.insert(2, &vec![0.0, 1.0, 0.0]).expect("i2");
        idx.insert(3, &vec![0.9, 0.1, 0.0]).expect("i3");
        let r = idx.search(&vec![1.0, 0.0, 0.0], 2).expect("search");
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].0, 1);
        assert_eq!(r[1].0, 3);
        assert!(r[0].1 < r[1].1);
        assert!(idx.remove(1).expect("rm"));
        assert!(!idx.remove(1).expect("rm again"));
        assert_eq!(idx.search(&vec![1.0, 0.0, 0.0], 1).expect("s")[0].0, 3);
        assert_eq!(idx.statistics().nodes_total, 2);
    }

    #[test]
    fn test_hnsw_dimension_mismatch() {
        let mut idx = HnswIndex::with_dimensions(HnswConfig::default(), 3).expect("idx");
        assert!(matches!(
            idx.insert(1, &vec![1.0]),
            Err(MemoryError::DimensionMismatch {
                expected: 3,
                got: 1
            })
        ));
        assert!(idx.search(&vec![1.0], 1).is_err());
    }

    #[test]
    fn test_hnsw_grows_past_initial_capacity() {
        let mut idx = HnswIndex::with_dimensions(HnswConfig::default(), 4).expect("idx");
        for i in 0..2500u64 {
            idx.insert(
                i,
                &vec![(i % 7) as f32 + 1.0, (i % 11) as f32, 1.0, (i % 3) as f32],
            )
            .expect("ins");
        }
        assert_eq!(idx.len(), 2500);
    }

    #[tokio::test]
    async fn test_hnsw_save_load_roundtrip() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("v.usearch");
        let mut idx = HnswIndex::with_dimensions(HnswConfig::default(), 3).expect("idx");
        idx.insert(7, &vec![0.0, 0.0, 1.0]).expect("i");
        idx.save(&path).expect("save");
        let loaded = HnswIndex::load(&path).expect("load");
        assert_eq!(loaded.dimensions(), 3);
        assert_eq!(loaded.search(&vec![0.0, 0.0, 1.0], 1).expect("s")[0].0, 7);
    }
}
