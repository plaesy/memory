//! Memory hygiene: find notes whose trust metadata is missing, stale or contradicted by the
//! current state of the project.
//!
//! The failure this guards against is *stale but confident* memory: a note that was true when
//! written ("the config lives at `x/y.toml`") keeps ranking high after the world changed. The
//! checks are cheap and read-only (the report never edits files):
//!
//! | kind | meaning | label |
//! |------|---------|-------|
//! | `missing_last_verified` / `missing_confidence` | a core/semantic/procedural note declares no verification date or trust | FACT |
//! | `stale` | `last_verified` is older than the staleness window | FACT |
//! | `bad_tier` | frontmatter `tier` is not one of the known tiers | FACT |
//! | `broken_link` | a `[[wikilink]]` resolves to no indexed document | FACT |
//! | `superseded_without_replacement` | `status: superseded` but no `superseded_by` | FACT |
//! | `dead_path_reference` | a backticked relative file path does not exist on disk | INFERRED |
//! | `possible_secret` | a likely credential marker appears in note text; value is never returned | FACT |

//! Secret findings report only kind and line number. They never include the matched value.

//! Set [`HygieneOptions::scan_secrets`] to `false` only for compatibility with a caller that
//! explicitly does not want this check.

//! The check intentionally favours recall over certainty; findings require human review.

//! It ignores fenced code blocks to avoid flagging examples and source snippets.

//! It is not a replacement for a dedicated secret scanner.
//!
//! `dead_path_reference` is a heuristic (the path may legitimately refer to another repository),
//! hence `INFERRED`; every other finding is read straight from the index.

use crate::memory::error::Result;
use crate::memory::search::{age_days, build_meta, normalize_tier, SearchEngine};
use crate::memory::storage::LinkRef;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// File extensions treated as "this backticked token is probably a file path".
const PATH_EXTENSIONS: [&str; 12] = [
    "md", "rs", "toml", "json", "yaml", "yml", "py", "sh", "ts", "js", "txt", "sql",
];

/// Options for [`SearchEngine::hygiene`].
#[derive(Clone, Debug)]
pub struct HygieneOptions {
    /// Directory the indexed paths are relative to. Enables `dead_path_reference` checks
    /// (also resolved against its parent, i.e. the project root for `.plaesy/`).
    pub root: Option<PathBuf>,

    /// Overrides the engine's `stale_after_days`.
    pub stale_after_days: Option<u32>,

    /// Clock override (tests).
    pub now: Option<DateTime<Utc>>,

    /// Scan note text for likely credentials without returning their values.
    pub scan_secrets: bool,
}

impl Default for HygieneOptions {
    fn default() -> Self {
        Self {
            root: None,
            stale_after_days: None,
            now: None,
            scan_secrets: true,
        }
    }
}

/// Return `(kind, 1-based line)` for likely credentials. Values are never returned.
pub fn secret_references(body: &str) -> Vec<(String, usize)> {
    let mut findings = Vec::new();
    let mut in_fence = false;
    for (index, line) in body.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        let kind = if lower.contains("-----begin ") && lower.contains("private key-----") {
            Some("private_key")
        } else if lower.contains("api_key")
            || lower.contains("api-key")
            || lower.contains("api key")
        {
            Some("api_key")
        } else if lower.contains("password") || lower.contains("passwd") {
            Some("password")
        } else if lower.contains("access_token")
            || lower.contains("access-token")
            || lower.contains("bearer ")
        {
            Some("access_token")
        } else if lower.contains("client_secret") || lower.contains("client-secret") {
            Some("client_secret")
        } else {
            None
        };
        if let Some(kind) = kind {
            findings.push((kind.to_string(), index + 1));
        }
    }
    findings
}

/// One problem found in the memory.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Finding {
    /// Finding kind (see module docs).
    pub kind: String,

    /// Document the finding is about.
    pub path: String,

    /// 1-based line, when the finding points at one.
    pub line: Option<usize>,

    /// Human-readable explanation.
    pub detail: String,

    /// `FACT` (read from the index) or `INFERRED` (heuristic).
    pub label: String,
}

/// Result of a hygiene run.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HygieneReport {
    /// Number of documents checked.
    pub checked: usize,

    /// Findings, ordered by path then line.
    pub findings: Vec<Finding>,

    /// Count per finding kind.
    pub summary: BTreeMap<String, usize>,
}

fn finding(kind: &str, path: &str, line: Option<usize>, detail: String, label: &str) -> Finding {
    Finding {
        kind: kind.to_string(),
        path: path.to_string(),
        line,
        detail,
        label: label.to_string(),
    }
}

/// Backticked tokens in `body` that look like relative file paths, with their 1-based line.
pub fn path_references(body: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for (i, line) in body.lines().enumerate() {
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        for (n, part) in line.split('`').enumerate() {
            if n % 2 == 0 || part.is_empty() {
                continue; // outside backticks
            }
            let tok = part.trim();
            let ext_ok = tok
                .rsplit_once('.')
                .is_some_and(|(_, e)| PATH_EXTENSIONS.contains(&e));
            let shape_ok = !tok.contains(char::is_whitespace)
                && !tok.contains([
                    '*', '<', '>', '{', '}', '(', ')', '$', ':', '|', '?', '=', '[', ']',
                ])
                && !tok.starts_with(['/', '~', '.', '@', '#'])
                && !tok.ends_with('/')
                && tok.len() < 200;
            // Require a directory component: a bare `foo.md` is as likely a name as a path.
            if ext_ok && shape_ok && tok.contains('/') {
                out.push((tok.to_string(), i + 1));
            }
        }
    }
    out
}

impl SearchEngine {
    /// Wikilinks of `path`: its outgoing links, or (with `incoming`) the links pointing at it.
    ///
    /// `path` may be a full indexed path, a path without extension, or a bare file stem.
    ///
    /// # Errors
    ///
    /// - `DatabaseError` if the link table cannot be read
    pub async fn links(&self, path: &str, incoming: bool) -> Result<Vec<LinkRef>> {
        let db = self.database();
        let paths: Vec<String> = db
            .document_paths()
            .await?
            .into_iter()
            .map(|(_, p)| p)
            .collect();
        let wanted = crate::memory::storage::resolve_link(path, &paths)
            .unwrap_or_else(|| path.replace('\\', "/"));
        Ok(db
            .all_links()
            .await?
            .into_iter()
            .filter(|l| {
                if incoming {
                    l.resolved_path.as_deref() == Some(wanted.as_str())
                } else {
                    l.source_path == wanted
                }
            })
            .collect())
    }

    /// Run the hygiene checks over every indexed document.
    ///
    /// # Errors
    ///
    /// - `DatabaseError` if documents or links cannot be read
    pub async fn hygiene(&self, opts: &HygieneOptions) -> Result<HygieneReport> {
        let db = self.database();
        let now = opts.now.unwrap_or_else(Utc::now);
        let stale_days = opts
            .stale_after_days
            .unwrap_or(self.search_config().stale_after_days);
        let docs = db.all_documents().await?;
        let mut findings: Vec<Finding> = Vec::new();

        for doc in &docs {
            let (path, md, body) = (&doc.path, &doc.metadata, &doc.body);
            let meta = build_meta(path, md, now, stale_days);
            let tier = meta.tier.as_deref();

            if let Some(raw) = md.get("tier").and_then(|v| v.as_str()) {
                if normalize_tier(raw).is_none() {
                    findings.push(finding(
                        "bad_tier",
                        path,
                        None,
                        format!("tier '{raw}' is not one of core, episodic, semantic, procedural, archive"),
                        "FACT",
                    ));
                }
            }

            if opts.scan_secrets {
                for (kind, line) in secret_references(body) {
                    findings.push(finding(
                        "possible_secret",
                        path,
                        Some(line + doc.body_start_line - 1),
                        format!("possible {kind}; value omitted"),
                        "FACT",
                    ));
                }
            }

            // Verification metadata is only demanded of notes that make claims about the world.
            if matches!(tier, Some("core") | Some("semantic") | Some("procedural")) {
                if meta.last_verified.is_none() {
                    findings.push(finding(
                        "missing_last_verified",
                        path,
                        None,
                        format!("{} note has no `last_verified`", tier.unwrap_or("?")),
                        "FACT",
                    ));
                }
                if meta.trust.is_none() {
                    findings.push(finding(
                        "missing_confidence",
                        path,
                        None,
                        format!("{} note has no `confidence`", tier.unwrap_or("?")),
                        "FACT",
                    ));
                }
            }

            if meta.stale {
                let lv = meta.last_verified.as_deref().unwrap_or("?");
                let days = age_days(lv, now).unwrap_or(0);
                findings.push(finding(
                    "stale",
                    path,
                    None,
                    format!("last verified {lv} ({days} days ago; window is {stale_days})"),
                    "FACT",
                ));
            }

            if meta.status.as_deref() == Some("superseded") && !md.contains_key("superseded_by") {
                findings.push(finding(
                    "superseded_without_replacement",
                    path,
                    None,
                    "status is `superseded` but `superseded_by` is not set".to_string(),
                    "FACT",
                ));
            }

            // Templates reference paths that exist only once instantiated.
            let is_template = path.ends_with(".template.md")
                || path.replace('\\', "/").split('/').any(|s| s == "templates");
            if let (Some(root), false) = (&opts.root, is_template) {
                let bases: Vec<&Path> = std::iter::once(root.as_path())
                    .chain(root.parent())
                    .collect();
                for (token, line) in path_references(body) {
                    if !bases.iter().any(|b| b.join(&token).exists()) {
                        findings.push(finding(
                            "dead_path_reference",
                            path,
                            Some(line + doc.body_start_line - 1),
                            format!("`{token}` does not exist relative to the project"),
                            "INFERRED",
                        ));
                    }
                }
            }
        }

        for l in db.all_links().await? {
            if l.resolved_path.is_none() {
                findings.push(finding(
                    "broken_link",
                    &l.source_path,
                    Some(l.line),
                    format!("[[{}]] matches no indexed document", l.target),
                    "FACT",
                ));
            }
        }

        // Project/decision/context graph: broken relations and conflicting supersession.
        // Read-only and additive — a vault with no `type:`-tagged notes reports nothing here.
        let graph = crate::memory::graph::load_graph(&db).await?;
        for m in &graph.malformed {
            findings.push(finding(
                "malformed_graph_note",
                &m.path,
                None,
                format!("type: {} — {}", m.note_type, m.reason),
                "FACT",
            ));
        }
        let project_ids: std::collections::HashSet<&str> = graph.project_ids().into_iter().collect();
        for n in &graph.notes {
            if n.note_type == "project" {
                continue;
            }
            if let Some(p) = &n.project {
                if !project_ids.contains(p.as_str()) {
                    findings.push(finding(
                        "broken_project_reference",
                        &n.path,
                        None,
                        format!("project '{p}' matches no `type: project` note"),
                        "FACT",
                    ));
                }
            }
            for s in &n.scope {
                if !project_ids.contains(s.as_str()) {
                    findings.push(finding(
                        "broken_scope_reference",
                        &n.path,
                        None,
                        format!("scope '{s}' matches no `type: project` note"),
                        "FACT",
                    ));
                }
            }
        }
        let (conflicts, _) = crate::memory::graph::supersession_map(&graph);
        for c in &conflicts {
            findings.push(finding(
                "conflicting_supersession",
                "(graph)",
                None,
                format!(
                    "decision '{}' is claimed superseded by multiple active decisions: {}",
                    c.target_id,
                    c.claimants.join(", ")
                ),
                "FACT",
            ));
        }

        // Duplicate `id`: two notes (any type) declaring the same id mean lookups such as
        // `resolve_project` silently pick whichever one they see first — every other claimant
        // is invisible to callers with no error raised anywhere in the write path.
        let mut by_id: std::collections::HashMap<&str, Vec<&crate::memory::graph::GraphNote>> =
            std::collections::HashMap::new();
        for n in &graph.notes {
            by_id.entry(n.id.as_str()).or_default().push(n);
        }
        for (id, notes) in &by_id {
            if notes.len() > 1 {
                let paths: Vec<&str> = notes.iter().map(|n| n.path.as_str()).collect();
                findings.push(finding(
                    "duplicate_graph_id",
                    paths[0],
                    None,
                    format!(
                        "id '{id}' is declared by {} notes: {}",
                        notes.len(),
                        paths.join(", ")
                    ),
                    "FACT",
                ));
            }
        }

        // A decision that lists its own id in `supersedes` is a self-referencing cycle: it can
        // never be a legitimate replacement for itself, and it would make `supersession_map`
        // report the decision as superseding itself with no warning otherwise.
        for n in graph.of_type("decision") {
            if n.supersedes.iter().any(|s| s == &n.id) {
                findings.push(finding(
                    "self_superseding_decision",
                    &n.path,
                    None,
                    format!("decision '{}' lists itself in `supersedes`", n.id),
                    "FACT",
                ));
            }
        }

        findings.sort_by(|a, b| {
            (a.path.as_str(), a.line, a.kind.as_str()).cmp(&(
                b.path.as_str(),
                b.line,
                b.kind.as_str(),
            ))
        });
        let mut summary = BTreeMap::new();
        for f in &findings {
            *summary.entry(f.kind.clone()).or_insert(0) += 1;
        }
        Ok(HygieneReport {
            checked: docs.len(),
            findings,
            summary,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_references_filters_noise() {
        let body = "See `src/lib.rs` and `Cargo.toml` and `a b/c.md`.\n\
                    ```\n`inside/fence.rs`\n```\n\
                    Also `https://x.dev/a.md`, `*.rs`, `docs/guide.md`, `dir/`, `tests/[module]/x.py`.\n";
        let refs: Vec<String> = path_references(body).into_iter().map(|(t, _)| t).collect();
        assert_eq!(refs, vec!["src/lib.rs", "docs/guide.md"]);
        let lines: Vec<usize> = path_references(body).into_iter().map(|(_, l)| l).collect();
        assert_eq!(lines, vec![1, 5]);
    }

    #[test]
    fn secret_references_report_kind_and_line_without_value() {
        let body = "safe\nOPENAI_API_KEY=sk-test-value\npassword = \"do-not-return\"\n-----BEGIN PRIVATE KEY-----\n";
        let findings = secret_references(body);
        assert_eq!(findings.len(), 3);
        assert_eq!(findings[0], ("api_key".to_string(), 2));
        assert_eq!(findings[1], ("password".to_string(), 3));
        assert_eq!(findings[2], ("private_key".to_string(), 4));
        assert!(!format!("{findings:?}").contains("do-not-return"));
    }
}
