//! Metadata-aware retrieval: tiers, declared source trust, lifecycle status and staleness.
//!
//! Sources declare these in YAML frontmatter:
//!
//! ```yaml
//! ---
//! tier: semantic            # core | episodic | semantic | procedural | archive
//! confidence: high          # high | medium | low (or a number 0-1)
//! last_verified: 2026-10-01 # when the claims were last checked against reality
//! status: active            # active | superseded | deprecated | archived
//! ---
//! ```
//!
//! `tier` falls back to the path (`10-episodic/…`, `knowledge/…`). Filters are applied to
//! the candidate lists *before* fusion and truncation, so a restrictive filter does not
//! starve the result list. Stale sources (see `SearchConfig::stale_after_days`) are not
//! hidden: they are flagged and down-weighted, so the caller can still see them but a
//! stale-but-confident note no longer outranks a fresh one by default.

use super::SearchEngine;
use crate::memory::error::Result;
use crate::memory::storage::{ResultMeta, SearchResult};
use chrono::{DateTime, NaiveDate, Utc};
use serde_json::Value;
use std::collections::HashMap;

/// The memory tiers a document can belong to.
pub const TIERS: [&str; 5] = ["core", "episodic", "semantic", "procedural", "archive"];

/// Statuses that mark a source as no longer current. Hidden unless `include_inactive`.
const INACTIVE_STATUSES: [&str; 4] = ["superseded", "deprecated", "archived", "obsolete"];

/// Normalise a tier name (accepts common aliases). Returns `None` for unknown names.
pub fn normalize_tier(name: &str) -> Option<&'static str> {
    match name.trim().to_lowercase().as_str() {
        "core" | "working" | "pinned" => Some("core"),
        "episodic" | "episode" | "daily" | "journal" | "log" => Some("episodic"),
        "semantic" | "knowledge" | "facts" | "fact" => Some("semantic"),
        "procedural" | "procedure" | "skills" | "skill" | "playbook" => Some("procedural"),
        "archive" | "archived" => Some("archive"),
        _ => None,
    }
}

/// Infer a tier from a document path by looking for tier-like path segments.
pub fn infer_tier(path: &str) -> Option<&'static str> {
    path.replace('\\', "/")
        .to_lowercase()
        .split('/')
        .rev()
        .skip(1) // directories only, nearest first
        .find_map(|seg| {
            let seg = seg.trim_start_matches(|c: char| c.is_ascii_digit() || c == '-' || c == '_');
            normalize_tier(seg)
        })
}

/// Declared trust of a source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Trust {
    /// Partially drifted or unverified claims.
    Low,
    /// Not verifiable cheaply.
    Medium,
    /// Verified recently.
    High,
}

impl Trust {
    /// Parse `high|medium|low` (case-insensitive) or a number in 0-1.
    pub fn parse(v: &Value) -> Option<Trust> {
        match v {
            Value::String(s) => match s.trim().to_lowercase().as_str() {
                "high" => Some(Trust::High),
                "medium" | "med" => Some(Trust::Medium),
                "low" => Some(Trust::Low),
                other => other.parse::<f64>().ok().map(Self::from_score),
            },
            Value::Number(n) => n.as_f64().map(Self::from_score),
            _ => None,
        }
    }

    fn from_score(x: f64) -> Trust {
        if x >= 0.8 {
            Trust::High
        } else if x >= 0.5 {
            Trust::Medium
        } else {
            Trust::Low
        }
    }

    /// Lower-case label.
    pub fn as_str(self) -> &'static str {
        match self {
            Trust::High => "high",
            Trust::Medium => "medium",
            Trust::Low => "low",
        }
    }
}

/// Restricts which sources may appear in results.
#[derive(Clone, Debug, Default)]
pub struct SearchFilter {
    /// Keep only these tiers (empty = all). Documents without a tier never match a tier filter.
    pub tiers: Vec<String>,

    /// Keep only documents whose path starts with this prefix.
    pub path_prefix: Option<String>,

    /// Keep only sources declaring at least this trust. Sources declaring none are excluded.
    pub min_trust: Option<Trust>,

    /// Also return superseded / deprecated / archived sources.
    pub include_inactive: bool,

    /// Keep only sources whose declared validity period contains this instant ("as of" a date).
    /// A source with no `valid_from` / `valid_to` is undated and always passes this check.
    pub as_of: Option<DateTime<Utc>>,
}

impl SearchFilter {
    /// `true` if the filter can discard candidates beyond the always-on status check.
    pub fn is_restrictive(&self) -> bool {
        !self.tiers.is_empty()
            || self.path_prefix.is_some()
            || self.min_trust.is_some()
            || self.as_of.is_some()
    }

    /// Whether `r` (with `meta` already filled) passes the filter.
    pub fn accepts(&self, r: &SearchResult) -> bool {
        if !self.include_inactive {
            if let Some(s) = &r.meta.status {
                if INACTIVE_STATUSES.contains(&s.as_str()) {
                    return false;
                }
            }
        }
        if !self.tiers.is_empty() {
            let wanted = |t: &str| {
                self.tiers
                    .iter()
                    .any(|w| normalize_tier(w) == normalize_tier(t) && normalize_tier(t).is_some())
            };
            match &r.meta.tier {
                Some(t) if wanted(t) => {}
                _ => return false,
            }
        }
        if let Some(prefix) = &self.path_prefix {
            let p = r.source_path.replace('\\', "/");
            if !p.starts_with(&prefix.replace('\\', "/")) {
                return false;
            }
        }
        if let Some(min) = self.min_trust {
            let have = r
                .meta
                .trust
                .as_deref()
                .and_then(|t| Trust::parse(&Value::String(t.to_string())));
            if have.is_none_or(|t| t < min) {
                return false;
            }
        }
        if let Some(at) = self.as_of {
            let from = r.meta.valid_from.as_deref().and_then(parse_date);
            let to = r.meta.valid_to.as_deref().and_then(parse_date);
            if from.is_some_and(|f| f > at) || to.is_some_and(|t| t < at) {
                return false;
            }
        }
        true
    }
}

/// Parse `YYYY-MM-DD` or an RFC 3339 timestamp.
pub fn parse_date(s: &str) -> Option<DateTime<Utc>> {
    let s = s.trim();
    if let Ok(d) = DateTime::parse_from_rfc3339(s) {
        return Some(d.with_timezone(&Utc));
    }
    NaiveDate::parse_from_str(s.get(..10)?, "%Y-%m-%d")
        .ok()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|dt| dt.and_utc())
}

/// Whole days between `last_verified` and `now` (None if unparseable).
pub fn age_days(last_verified: &str, now: DateTime<Utc>) -> Option<i64> {
    parse_date(last_verified).map(|d| (now - d).num_days())
}

/// Build result metadata from a document's frontmatter and path.
pub fn build_meta(
    path: &str,
    md: &HashMap<String, Value>,
    now: DateTime<Utc>,
    stale_after_days: u32,
) -> ResultMeta {
    let text = |k: &str| {
        md.get(k)
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
    };
    let tier = text("tier")
        .and_then(|t| normalize_tier(&t))
        .or_else(|| infer_tier(path))
        .map(str::to_string);
    let trust = md
        .get("confidence")
        .or_else(|| md.get("trust"))
        .and_then(Trust::parse)
        .map(|t| t.as_str().to_string());
    let last_verified = text("last_verified");
    let status = text("status").map(|s| s.to_lowercase());
    let valid_from = text("valid_from");
    let valid_to = text("valid_to");
    // Episodic logs and archives record what was true then; they do not go stale.
    let ages = !matches!(tier.as_deref(), Some("episodic") | Some("archive"));
    let stale = ages
        && stale_after_days > 0
        && last_verified
            .as_deref()
            .and_then(|lv| age_days(lv, now))
            .is_some_and(|d| d > i64::from(stale_after_days));
    ResultMeta {
        tier,
        trust,
        last_verified,
        status,
        valid_from,
        valid_to,
        stale,
    }
}

impl SearchEngine {
    /// Attach source metadata to `results`, apply the staleness penalty, drop results the
    /// filter rejects and re-sort by score. Called on each candidate list before fusion.
    pub(super) async fn refine(
        &self,
        mut results: Vec<SearchResult>,
        filter: &SearchFilter,
    ) -> Result<Vec<SearchResult>> {
        if results.is_empty() {
            return Ok(results);
        }
        let mut paths: Vec<String> = results.iter().map(|r| r.source_path.clone()).collect();
        paths.sort();
        paths.dedup();
        let docs = self.db.document_metadata(&paths).await?;
        let now = Utc::now();
        let empty = HashMap::new();
        for r in &mut results {
            let md = docs.get(&r.source_path).unwrap_or(&empty);
            r.meta = build_meta(&r.source_path, md, now, self.config.stale_after_days);
            if r.meta.stale {
                r.score *= self.config.stale_penalty;
            }
        }
        results.retain(|r| filter.accepts(r));
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn now() -> DateTime<Utc> {
        parse_date("2026-10-19").expect("date")
    }

    fn md(v: Value) -> HashMap<String, Value> {
        serde_json::from_value(v).expect("map")
    }

    #[test]
    fn tier_aliases_and_path_inference() {
        assert_eq!(normalize_tier(" Knowledge "), Some("semantic"));
        assert_eq!(normalize_tier("daily"), Some("episodic"));
        assert_eq!(normalize_tier("nonsense"), None);
        assert_eq!(infer_tier("10-episodic/2026/10/a.md"), Some("episodic"));
        assert_eq!(infer_tier("20_semantic/people/x.md"), Some("semantic"));
        assert_eq!(infer_tier("00-core/persona.md"), Some("core"));
        // The file name itself never decides the tier.
        assert_eq!(infer_tier("notes/archive.md"), None);
    }

    #[test]
    fn frontmatter_beats_path_and_numbers_map_to_trust() {
        let m = build_meta(
            "10-episodic/a.md",
            &md(json!({"tier": "procedural", "confidence": 0.9})),
            now(),
            180,
        );
        assert_eq!(m.tier.as_deref(), Some("procedural"));
        assert_eq!(m.trust.as_deref(), Some("high"));
        let low = build_meta("a.md", &md(json!({"confidence": "0.2"})), now(), 180);
        assert_eq!(low.trust.as_deref(), Some("low"));
    }

    #[test]
    fn staleness_rules() {
        let old = json!({"tier": "semantic", "last_verified": "2026-01-01"});
        assert!(build_meta("a.md", &md(old.clone()), now(), 180).stale);
        assert!(!build_meta("a.md", &md(old), now(), 0).stale, "0 disables");
        let fresh = json!({"tier": "semantic", "last_verified": "2026-10-01"});
        assert!(!build_meta("a.md", &md(fresh), now(), 180).stale);
        let log = json!({"tier": "episodic", "last_verified": "2020-01-01"});
        assert!(
            !build_meta("a.md", &md(log), now(), 180).stale,
            "episodic never stale"
        );
        let unverified = json!({"tier": "semantic"});
        assert!(!build_meta("a.md", &md(unverified), now(), 180).stale);
        assert_eq!(age_days("2026-10-09T10:00:00Z", now()), Some(9));
        assert_eq!(age_days("not a date", now()), None);
    }

    fn res(path: &str, meta: ResultMeta) -> SearchResult {
        SearchResult {
            chunk_id: 1,
            text: "t".into(),
            source_path: path.into(),
            start_line: 1,
            end_line: 1,
            revision: None,
            indexed_at: "2026-10-19T00:00:00Z".into(),
            score: 0.5,
            confidence: crate::memory::storage::ConfidenceLevel::Fact,
            meta,
        }
    }

    #[test]
    fn as_of_keeps_sources_valid_on_that_day() {
        let dated = |from: Option<&str>, to: Option<&str>| ResultMeta {
            valid_from: from.map(Into::into),
            valid_to: to.map(Into::into),
            ..ResultMeta::default()
        };
        let f = SearchFilter {
            as_of: parse_date("2026-06-15"),
            ..SearchFilter::default()
        };
        assert!(f.is_restrictive());
        assert!(f.accepts(&res("a.md", dated(Some("2026-01-01"), Some("2026-12-31")))));
        assert!(
            !f.accepts(&res("a.md", dated(Some("2026-07-01"), None))),
            "not yet valid"
        );
        assert!(
            !f.accepts(&res("a.md", dated(None, Some("2026-06-01")))),
            "already expired"
        );
        assert!(f.accepts(&res("a.md", dated(None, None))), "undated passes");
        // The range is inclusive on `valid_to`.
        assert!(f.accepts(&res("a.md", dated(None, Some("2026-06-15")))));

        let m = build_meta(
            "a.md",
            &md(json!({"valid_from": "2026-01-01", "valid_to": "2026-06-30"})),
            now(),
            180,
        );
        assert_eq!(m.valid_from.as_deref(), Some("2026-01-01"));
        assert_eq!(m.valid_to.as_deref(), Some("2026-06-30"));
    }

    #[test]
    fn filter_semantics() {
        let sem = ResultMeta {
            tier: Some("semantic".into()),
            trust: Some("medium".into()),
            ..ResultMeta::default()
        };
        let old = ResultMeta {
            status: Some("superseded".into()),
            ..sem.clone()
        };
        let all = SearchFilter::default();
        assert!(all.accepts(&res("a.md", sem.clone())));
        assert!(
            !all.accepts(&res("a.md", old.clone())),
            "superseded hidden by default"
        );
        let inc = SearchFilter {
            include_inactive: true,
            ..SearchFilter::default()
        };
        assert!(inc.accepts(&res("a.md", old)));

        let by_tier = SearchFilter {
            tiers: vec!["knowledge".into()], // alias of semantic
            ..SearchFilter::default()
        };
        assert!(by_tier.accepts(&res("a.md", sem.clone())));
        assert!(
            !by_tier.accepts(&res("a.md", ResultMeta::default())),
            "no tier never matches"
        );

        let trusted = SearchFilter {
            min_trust: Some(Trust::High),
            ..SearchFilter::default()
        };
        assert!(!trusted.accepts(&res("a.md", sem.clone())));
        assert!(
            !trusted.accepts(&res("a.md", ResultMeta::default())),
            "unknown trust excluded"
        );
        let ok = ResultMeta {
            trust: Some("high".into()),
            ..sem
        };
        assert!(trusted.accepts(&res("a.md", ok)));

        let scoped = SearchFilter {
            path_prefix: Some("20-semantic/".into()),
            ..SearchFilter::default()
        };
        assert!(scoped.accepts(&res("20-semantic\\x.md", ResultMeta::default())));
        assert!(!scoped.accepts(&res("10-episodic/x.md", ResultMeta::default())));
    }
}
