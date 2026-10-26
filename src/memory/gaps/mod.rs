//! Memory gaps (`memory missing`): what the memory was asked for and could not supply, and
//! what it lacks in its own structure.
//!
//! Human analogue. People notice a gap when retrieval fails: a tip-of-the-tongue state or a low
//! feeling-of-knowing is a metacognitive signal that memory expects something it does not have.
//! A gap that keeps returning is more pressing, its pull fades when it is not asked about for a
//! long time (forgetting curve), and a later successful recall closes it.
//!
//! Two sources:
//!
//! - **query gaps**: a search that returned nothing, or whose best hit scores below `WEAK_SCORE`.
//!   Stored in the `query_gaps` table, the only write this module makes. Recorded by callers that
//!   ask on behalf of a user or agent (`memory search`, the service `search` method); the read-only
//!   MCP server does not record.
//! - **structural gaps**: read from the hygiene audit, not stored. `broken_link` (a `[[wikilink]]`
//!   to a note that does not exist) and `stale` (past the verification window).

use crate::memory::error::{MemoryError, Result};
use crate::memory::hygiene::{Finding, HygieneOptions};
use crate::memory::search::SearchEngine;
use crate::memory::storage::{Database, SearchResult};
use chrono::{DateTime, Utc};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A best score below this is weak evidence. Scores are fused and lie in 0-1.
///
/// Calibrated 2026-10-26 on the vault index with 15 queries: on-topic queries scored 0.495 to 0.841,
/// off-topic ones 0.117 to 0.292. 0.35 sits between the two groups with a margin on each side.
pub const WEAK_SCORE: f32 = 0.35;

/// Days without a new miss after which a gap's priority halves.
pub const HALF_LIFE_DAYS: f64 = 30.0;

/// Longest query stored, in characters. Longer input is cut so one call cannot bloat the table.
pub const MAX_QUERY_CHARS: usize = 500;

const GAP_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS query_gaps (
    id             INTEGER PRIMARY KEY,
    key            TEXT NOT NULL UNIQUE,
    query          TEXT NOT NULL,
    misses         INTEGER NOT NULL,
    last_top_score REAL,
    first_seen     TEXT NOT NULL,
    last_seen      TEXT NOT NULL,
    resolved_at    TEXT
);";

/// An unanswered query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct QueryGap {
    /// Gap id, used by `memory missing --resolve`.
    pub id: i64,

    /// The query as first asked.
    pub query: String,

    /// How many times it was asked without useful evidence.
    pub misses: i64,

    /// Best score of the most recent miss (`None` when nothing matched at all).
    pub last_top_score: Option<f64>,

    /// RFC 3339 time of the first miss.
    pub first_seen: String,

    /// RFC 3339 time of the latest miss.
    pub last_seen: String,

    /// `misses` weighted by recency (see [`HALF_LIFE_DAYS`]). Higher means more pressing.
    pub priority: f64,
}

/// Everything the memory is missing.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GapReport {
    /// Open query gaps, most pressing first.
    pub query_gaps: Vec<QueryGap>,

    /// Structural gaps from hygiene: `broken_link` and `stale` findings.
    pub structural: Vec<Finding>,
}

/// Normalised key so that `Deploy  Steps` and `deploy steps` count as one question.
fn key_of(query: &str) -> String {
    query
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// `misses` decayed by the time since the latest miss.
fn priority(misses: i64, last_seen: &str, now: DateTime<Utc>) -> f64 {
    let days = DateTime::parse_from_rfc3339(last_seen)
        .map(|t| (now - t.with_timezone(&Utc)).num_seconds().max(0) as f64 / 86_400.0)
        .unwrap_or(0.0);
    misses as f64 * 0.5_f64.powf(days / HALF_LIFE_DAYS)
}

fn db_err(e: rusqlite::Error) -> MemoryError {
    MemoryError::DatabaseError(e.to_string())
}

impl Database {
    /// Record a query. A best score below `WEAK_SCORE` (or no hit) is a miss; a strong hit closes
    /// an open gap for the same query. A miss on a closed gap reopens it.
    ///
    /// # Errors
    ///
    /// - `DatabaseError` if the gap table cannot be written
    pub fn record_query(&self, query: &str, top_score: Option<f32>) -> Result<()> {
        let query: String = query.chars().take(MAX_QUERY_CHARS).collect();
        let key = key_of(&query);
        if key.is_empty() {
            return Ok(());
        }
        let conn = self.conn();
        conn.execute_batch(GAP_SCHEMA).map_err(db_err)?;
        let now = Utc::now().to_rfc3339();
        // A NaN score counts as a miss, not a hit: NaN compares false with everything.
        if top_score.is_none_or(|s| s.is_nan() || s < WEAK_SCORE) {
            conn.execute(
                "INSERT INTO query_gaps (key, query, misses, last_top_score, first_seen, last_seen)
                 VALUES (?1, ?2, 1, ?3, ?4, ?4)
                 ON CONFLICT(key) DO UPDATE SET
                     misses = misses + 1,
                     last_top_score = excluded.last_top_score,
                     last_seen = excluded.last_seen,
                     resolved_at = NULL",
                params![key, query.trim(), top_score.map(f64::from), now],
            )
            .map_err(db_err)?;
        } else {
            conn.execute(
                "UPDATE query_gaps SET resolved_at = ?2 WHERE key = ?1 AND resolved_at IS NULL",
                params![key, now],
            )
            .map_err(db_err)?;
        }
        Ok(())
    }

    /// Open query gaps, most pressing first, at most `limit`.
    ///
    /// # Errors
    ///
    /// - `DatabaseError` if the gap table cannot be read
    pub fn open_query_gaps(&self, limit: usize, now: DateTime<Utc>) -> Result<Vec<QueryGap>> {
        let conn = self.conn();
        conn.execute_batch(GAP_SCHEMA).map_err(db_err)?;
        let mut st = conn
            .prepare(
                "SELECT id, query, misses, last_top_score, first_seen, last_seen
                 FROM query_gaps WHERE resolved_at IS NULL",
            )
            .map_err(db_err)?;
        let mut gaps = st
            .query_map([], |r| {
                let misses: i64 = r.get(2)?;
                let last_seen: String = r.get(5)?;
                Ok(QueryGap {
                    id: r.get(0)?,
                    query: r.get(1)?,
                    misses,
                    last_top_score: r.get(3)?,
                    first_seen: r.get(4)?,
                    priority: priority(misses, &last_seen, now),
                    last_seen,
                })
            })
            .map_err(db_err)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(db_err)?;
        gaps.sort_by(|a, b| b.priority.total_cmp(&a.priority));
        gaps.truncate(limit);
        Ok(gaps)
    }

    /// Close an open gap by id. Returns `false` when no open gap has that id.
    ///
    /// # Errors
    ///
    /// - `DatabaseError` if the gap table cannot be written
    pub fn resolve_query_gap(&self, id: i64) -> Result<bool> {
        let conn = self.conn();
        conn.execute_batch(GAP_SCHEMA).map_err(db_err)?;
        let n = conn
            .execute(
                "UPDATE query_gaps SET resolved_at = ?2 WHERE id = ?1 AND resolved_at IS NULL",
                params![id, Utc::now().to_rfc3339()],
            )
            .map_err(db_err)?;
        Ok(n > 0)
    }
}

impl SearchEngine {
    /// Record a query and its results. Call it for searches made on someone's behalf.
    ///
    /// # Errors
    ///
    /// - `DatabaseError` if the gap table cannot be written
    pub fn note_query(&self, query: &str, results: &[SearchResult]) -> Result<()> {
        self.database()
            .record_query(query, results.first().map(|r| r.score))
    }

    /// Close an open query gap by id.
    ///
    /// # Errors
    ///
    /// - `DatabaseError` if the gap table cannot be written
    pub fn resolve_gap(&self, id: i64) -> Result<bool> {
        self.database().resolve_query_gap(id)
    }

    /// Everything the memory is missing. `root` enables the file-level checks of hygiene.
    ///
    /// # Errors
    ///
    /// - `DatabaseError` if the index or gap table cannot be read
    pub async fn missing(&self, root: Option<PathBuf>, limit: usize) -> Result<GapReport> {
        let query_gaps = self.database().open_query_gaps(limit, Utc::now())?;
        let report = self
            .hygiene(&HygieneOptions {
                root,
                ..HygieneOptions::default()
            })
            .await?;
        let structural = report
            .findings
            .into_iter()
            .filter(|f| f.kind == "broken_link" || f.kind == "stale")
            .collect();
        Ok(GapReport {
            query_gaps,
            structural,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::storage::DatabaseConfig;
    use chrono::Duration;

    async fn db() -> Database {
        Database::new(DatabaseConfig {
            db_path: ":memory:".into(),
            ..DatabaseConfig::default()
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn weak_queries_accumulate_and_normalise() {
        let db = db().await;
        db.record_query("Deploy  Steps", None).unwrap();
        db.record_query("deploy steps", Some(0.1)).unwrap();
        let gaps = db.open_query_gaps(10, Utc::now()).unwrap();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].misses, 2);
        assert_eq!(gaps[0].query, "Deploy  Steps");
    }

    #[tokio::test]
    async fn strong_hit_closes_and_next_miss_reopens() {
        let db = db().await;
        db.record_query("kata sandi", None).unwrap();
        db.record_query("kata sandi", Some(0.9)).unwrap();
        assert!(db.open_query_gaps(10, Utc::now()).unwrap().is_empty());
        db.record_query("kata sandi", Some(0.05)).unwrap();
        assert_eq!(db.open_query_gaps(10, Utc::now()).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn nan_score_is_a_miss_and_long_query_is_cut() {
        let db = db().await;
        db.record_query("nan case", Some(f32::NAN)).unwrap();
        let gaps = db.open_query_gaps(10, Utc::now()).unwrap();
        assert_eq!(gaps.len(), 1);
        let long = "q".repeat(MAX_QUERY_CHARS + 50);
        db.record_query(&long, None).unwrap();
        let stored = db.open_query_gaps(10, Utc::now()).unwrap();
        assert!(stored
            .iter()
            .any(|g| g.query.chars().count() == MAX_QUERY_CHARS));
    }

    #[tokio::test]
    async fn resolve_by_id_and_priority_decays() {
        let db = db().await;
        db.record_query("old question", None).unwrap();
        db.record_query("new question", None).unwrap();
        let gaps = db.open_query_gaps(10, Utc::now()).unwrap();
        assert!(db.resolve_query_gap(gaps[0].id).unwrap());
        assert!(!db.resolve_query_gap(gaps[0].id).unwrap());
        // One half-life later the same miss counts half as much.
        let later = Utc::now() + Duration::days(HALF_LIFE_DAYS as i64);
        let now = db.open_query_gaps(10, Utc::now()).unwrap();
        let then = db.open_query_gaps(10, later).unwrap();
        assert!((then[0].priority - now[0].priority * 0.5).abs() < 0.01);
    }
}
