//! Projects, decisions and contexts: a structured view over notes that declare a `type`.
//!
//! This builds on the metadata every note already has (YAML frontmatter stored verbatim as
//! JSON in `documents.metadata`); no new SQL column or migration is needed. A note opts in by
//! declaring:
//!
//! ```yaml
//! ---
//! type: project            # project | decision | context | knowledge | capture | preference
//! id: studio-a             # stable identifier, referenced by `project` / `scope` / `supersedes`
//! status: active           # see STATUSES below for the allowed values per type
//! project: studio-a        # decision/context: the project it mainly belongs to
//! scope: [studio-a, studio-b]   # optional: also relevant to these projects
//! supersedes: [old-decision-id] # decision: ids of decisions this one replaces
//! ---
//! ```
//!
//! Notes without `type` are unaffected: they keep appearing in `memory search`, just not here.
//! A note that declares `type` but is missing its required `id` is skipped from every listing
//! (never crashes the command) and surfaced instead as a `memory hygiene` finding.

use crate::memory::storage::{Database, StoredDocument};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

/// Recognised values for the `type` field.
pub const NOTE_TYPES: [&str; 6] = [
    "project",
    "decision",
    "context",
    "knowledge",
    "capture",
    "preference",
];

/// Allowed `status` values for `type: project`. Unknown/missing status defaults to `active`.
pub const PROJECT_STATUSES: [&str; 4] = ["active", "paused", "completed", "archived"];
/// Allowed `status` values for `type: decision`. Unknown/missing status defaults to `active`.
pub const DECISION_STATUSES: [&str; 4] = ["proposed", "active", "superseded", "revoked"];
/// Allowed `status` values for `type: context`. Unknown/missing status defaults to `active`.
pub const CONTEXT_STATUSES: [&str; 2] = ["active", "archived"];

/// Statuses hidden by default unless `include_inactive` or an explicit `--status` is given.
fn inactive_statuses(note_type: &str) -> &'static [&'static str] {
    match note_type {
        "project" => &["completed", "archived"],
        "decision" => &["superseded", "revoked"],
        "context" => &["archived"],
        _ => &[],
    }
}

/// One note participating in the project/decision/context graph.
#[derive(Clone, Debug, Serialize)]
pub struct GraphNote {
    /// Path relative to the notes root.
    pub path: String,
    /// Declared `id`. Always `Some` for a note returned by [`load_graph`] (malformed notes with
    /// no id are routed to `malformed` instead).
    pub id: String,
    /// Declared `type`.
    pub note_type: String,
    /// Display title: the `title` frontmatter field, else the declared `id`.
    pub title: String,
    /// Declared `status`, normalised to lowercase. Defaults applied by [`effective_status`].
    pub status: Option<String>,
    /// `project` field (decision/context: the project it mainly belongs to).
    pub project: Option<String>,
    /// `scope` field: other projects this note is also relevant to.
    pub scope: Vec<String>,
    /// `supersedes` field (decision only): ids of decisions this one replaces.
    pub supersedes: Vec<String>,
    /// `updated` or `last_verified`, whichever is declared (for sort order).
    pub updated: Option<String>,
}

impl GraphNote {
    /// `status`, defaulted to `active` when not declared (every type's active-like state).
    pub fn effective_status(&self) -> &str {
        self.status.as_deref().unwrap_or("active")
    }

    /// Whether this note is linked to `project_id`, directly or via `scope`.
    pub fn belongs_to(&self, project_id: &str) -> bool {
        self.project.as_deref() == Some(project_id) || self.scope.iter().any(|s| s == project_id)
    }

    /// `true` when related via `scope` only (not the direct `project` owner) — rendered `(shared)`.
    pub fn is_shared_only(&self, project_id: &str) -> bool {
        self.project.as_deref() != Some(project_id) && self.scope.iter().any(|s| s == project_id)
    }
}

/// A note that declared `type` but could not be loaded into the graph, and why.
#[derive(Clone, Debug, Serialize)]
pub struct MalformedNote {
    pub path: String,
    pub note_type: String,
    pub reason: String,
}

/// Everything decodable from the index's declared metadata.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub notes: Vec<GraphNote>,
    pub malformed: Vec<MalformedNote>,
}

fn as_str_vec(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty())
            .collect(),
        Some(Value::String(s)) if !s.trim().is_empty() => vec![s.trim().to_string()],
        _ => Vec::new(),
    }
}

fn as_str(v: Option<&Value>) -> Option<String> {
    v.and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Decode one stored document into a [`GraphNote`], or record why it was skipped.
fn decode(doc: &StoredDocument) -> Option<Result<GraphNote, String>> {
    let note_type = as_str(doc.metadata.get("type"))?.to_lowercase();
    if !NOTE_TYPES.contains(&note_type.as_str()) {
        return Some(Err(format!(
            "unknown type '{note_type}' (expected one of: {})",
            NOTE_TYPES.join(", ")
        )));
    }
    let Some(id) = as_str(doc.metadata.get("id")) else {
        return Some(Err(format!(
            "type: {note_type} is missing a required 'id' field"
        )));
    };
    let status = as_str(doc.metadata.get("status")).map(|s| s.to_lowercase());
    if let Some(s) = &status {
        let allowed: &[&str] = match note_type.as_str() {
            "project" => &PROJECT_STATUSES,
            "decision" => &DECISION_STATUSES,
            "context" => &CONTEXT_STATUSES,
            _ => &[],
        };
        if !allowed.is_empty() && !allowed.contains(&s.as_str()) {
            return Some(Err(format!(
                "status '{s}' is not valid for type: {note_type} (expected one of: {})",
                allowed.join(", ")
            )));
        }
    }
    let title = as_str(doc.metadata.get("title")).unwrap_or_else(|| id.clone());
    let project = as_str(doc.metadata.get("project"));
    let scope = as_str_vec(doc.metadata.get("scope"));
    let supersedes = as_str_vec(doc.metadata.get("supersedes"));
    let updated = as_str(doc.metadata.get("updated")).or_else(|| as_str(doc.metadata.get("last_verified")));
    Some(Ok(GraphNote {
        path: doc.path.clone(),
        id,
        note_type,
        title,
        status,
        project,
        scope,
        supersedes,
        updated,
    }))
}

/// Load the project/decision/context graph from every indexed document.
///
/// Read-only; never fails the whole listing because of one bad note (see module docs).
pub async fn load_graph(db: &Database) -> crate::memory::error::Result<Graph> {
    let docs = db.all_documents().await?;
    let mut notes = Vec::new();
    let mut malformed = Vec::new();
    for doc in &docs {
        match decode(doc) {
            None => {}
            Some(Ok(note)) => notes.push(note),
            Some(Err(reason)) => {
                // `type` is always present when `decode` returns `Some`.
                let note_type = as_str(doc.metadata.get("type")).unwrap_or_default();
                malformed.push(MalformedNote {
                    path: doc.path.clone(),
                    note_type,
                    reason,
                });
            }
        }
    }
    notes.sort_by(|a, b| b.updated.cmp(&a.updated).then(a.id.cmp(&b.id)));
    Ok(Graph { notes, malformed })
}

impl Graph {
    /// All `type: project` notes.
    pub fn projects(&self) -> Vec<&GraphNote> {
        self.notes.iter().filter(|n| n.note_type == "project").collect()
    }

    /// All notes of `note_type`.
    pub fn of_type(&self, note_type: &str) -> Vec<&GraphNote> {
        self.notes.iter().filter(|n| n.note_type == note_type).collect()
    }

    /// Ids of every active (or any-status) `type: project` note, duplicates removed, for
    /// ambiguity / "did you mean" messages.
    pub fn project_ids(&self) -> Vec<&str> {
        self.projects().iter().map(|p| p.id.as_str()).collect()
    }
}

/// Outcome of resolving a user-supplied project reference (id, id-prefix, or title).
pub enum Resolved<'a> {
    Found(&'a GraphNote),
    NotFound {
        query: String,
        suggestions: Vec<String>,
    },
    Ambiguous {
        query: String,
        candidates: Vec<String>,
    },
}

/// Resolve `query` against the graph's projects: exact id, exact title (case-insensitive),
/// else a unique id-prefix match. Ambiguous prefixes list every match; no match suggests the
/// closest titles/ids by substring.
pub fn resolve_project<'a>(graph: &'a Graph, query: &str) -> Resolved<'a> {
    let projects = graph.projects();
    if let Some(exact) = projects.iter().find(|p| p.id == query) {
        return Resolved::Found(exact);
    }
    if let Some(by_title) = projects
        .iter()
        .find(|p| p.title.eq_ignore_ascii_case(query))
    {
        return Resolved::Found(by_title);
    }
    let prefix_matches: Vec<&&GraphNote> = projects.iter().filter(|p| p.id.starts_with(query)).collect();
    match prefix_matches.len() {
        1 => Resolved::Found(prefix_matches[0]),
        n if n > 1 => Resolved::Ambiguous {
            query: query.to_string(),
            candidates: prefix_matches
                .iter()
                .map(|p| format!("{} ({})", p.id, p.title))
                .collect(),
        },
        _ => {
            let needle = query.to_lowercase();
            let suggestions: Vec<String> = projects
                .iter()
                .filter(|p| {
                    p.title.to_lowercase().contains(&needle) || p.id.to_lowercase().contains(&needle)
                })
                .map(|p| format!("{} ({})", p.id, p.title))
                .collect();
            Resolved::NotFound {
                query: query.to_string(),
                suggestions,
            }
        }
    }
}

/// Filter applied by `decisions` / `contexts` (global or project-scoped) listings.
#[derive(Clone, Debug)]
pub struct ListFilter {
    /// Keep only notes linked (via `project` or `scope`) to one of these project ids. Empty
    /// means no project restriction. A project id that matches no project yields an empty
    /// (not an error) result — see module docs' note on filter vs. direct-target semantics.
    pub projects: Vec<String>,
    /// Keep only notes with one of these statuses. Empty means "apply the default visibility
    /// rule" (see `include_inactive`).
    pub status: Vec<String>,
    /// Also show notes whose status is normally hidden (superseded/revoked/archived/completed).
    pub include_inactive: bool,
    /// Maximum notes returned.
    pub limit: usize,
}

impl Default for ListFilter {
    fn default() -> Self {
        ListFilter {
            projects: Vec::new(),
            status: Vec::new(),
            include_inactive: false,
            limit: 50,
        }
    }
}

/// List every note of `note_type` matching `filter`.
///
/// Status visibility: with no explicit `--status` and no `--include-inactive`, notes whose
/// status is in that type's default-hidden set are left out, but the command never errors.
/// An explicit `--status` always wins (shows exactly those statuses, active or not).
pub fn list<'a>(graph: &'a Graph, note_type: &str, filter: &ListFilter) -> Vec<&'a GraphNote> {
    let hidden = inactive_statuses(note_type);
    let mut out: Vec<&GraphNote> = graph
        .of_type(note_type)
        .into_iter()
        .filter(|n| {
            if !filter.projects.is_empty() {
                let global_requested = filter.projects.iter().any(|p| p == "*");
                if !global_requested && !filter.projects.iter().any(|p| n.belongs_to(p)) {
                    return false;
                }
            }
            if !filter.status.is_empty() {
                return filter.status.iter().any(|s| s == n.effective_status());
            }
            if filter.include_inactive {
                return true;
            }
            !hidden.contains(&n.effective_status())
        })
        .collect();
    out.truncate(filter.limit.max(1));
    out
}

/// One "superseded by …" or conflicting-supersession warning, for display alongside a decision
/// listing. Read-only, never blocks the listing itself (see module docs).
#[derive(Clone, Debug, Serialize)]
pub struct SupersessionWarning {
    pub target_id: String,
    pub claimants: Vec<String>,
}

/// Decision ids claimed as superseded by more than one *active* decision — a data-hygiene
/// smell, not a listing failure. Also returns, for every decision that is itself superseded,
/// the id(s) of the decision(s) that replaced it (for "superseded by" labels).
pub fn supersession_map(graph: &Graph) -> (Vec<SupersessionWarning>, HashMap<String, Vec<String>>) {
    let mut claimants: HashMap<String, Vec<String>> = HashMap::new();
    for d in graph.of_type("decision") {
        if d.effective_status() != "active" {
            continue;
        }
        for target in &d.supersedes {
            claimants.entry(target.clone()).or_default().push(d.id.clone());
        }
    }
    let conflicts = claimants
        .iter()
        .filter(|(_, v)| v.len() > 1)
        .map(|(target_id, v)| SupersessionWarning {
            target_id: target_id.clone(),
            claimants: v.clone(),
        })
        .collect();
    (conflicts, claimants)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as Map;

    fn doc(path: &str, meta: &[(&str, Value)]) -> StoredDocument {
        let mut metadata = Map::new();
        for (k, v) in meta {
            metadata.insert(k.to_string(), v.clone());
        }
        StoredDocument {
            path: path.to_string(),
            metadata,
            body: String::new(),
            body_start_line: 1,
        }
    }

    fn s(v: &str) -> Value {
        Value::String(v.to_string())
    }

    fn graph_of(docs: Vec<StoredDocument>) -> Graph {
        let mut notes = Vec::new();
        let mut malformed = Vec::new();
        for d in &docs {
            match decode(d) {
                None => {}
                Some(Ok(n)) => notes.push(n),
                Some(Err(reason)) => malformed.push(MalformedNote {
                    path: d.path.clone(),
                    note_type: as_str(d.metadata.get("type")).unwrap_or_default(),
                    reason,
                }),
            }
        }
        Graph { notes, malformed }
    }

    #[test]
    fn note_without_type_is_ignored_not_malformed() {
        let g = graph_of(vec![doc("a.md", &[("title", s("plain note"))])]);
        assert!(g.notes.is_empty());
        assert!(g.malformed.is_empty());
    }

    #[test]
    fn decision_missing_id_is_skipped_with_hygiene_reason() {
        let g = graph_of(vec![doc("d.md", &[("type", s("decision"))])]);
        assert!(g.notes.is_empty());
        assert_eq!(g.malformed.len(), 1);
        assert!(g.malformed[0].reason.contains("missing a required 'id'"));
    }

    #[test]
    fn unknown_type_is_malformed_not_silently_dropped() {
        let g = graph_of(vec![doc("x.md", &[("type", s("tasks")), ("id", s("x"))])]);
        assert!(g.notes.is_empty());
        assert_eq!(g.malformed.len(), 1);
    }

    #[test]
    fn invalid_status_for_type_is_malformed() {
        let g = graph_of(vec![doc(
            "p.md",
            &[("type", s("project")), ("id", s("p1")), ("status", s("superseded"))],
        )]);
        assert!(g.notes.is_empty());
        assert_eq!(g.malformed.len(), 1);
    }

    #[test]
    fn empty_vault_lists_nothing() {
        let g = graph_of(vec![]);
        assert!(g.projects().is_empty());
    }

    #[test]
    fn resolve_exact_id() {
        let g = graph_of(vec![doc(
            "a.md",
            &[("type", s("project")), ("id", s("studio-a")), ("title", s("Studio A"))],
        )]);
        assert!(matches!(resolve_project(&g, "studio-a"), Resolved::Found(_)));
    }

    #[test]
    fn resolve_by_title_case_insensitive() {
        let g = graph_of(vec![doc(
            "a.md",
            &[("type", s("project")), ("id", s("studio-a")), ("title", s("Studio A"))],
        )]);
        match resolve_project(&g, "studio a") {
            Resolved::Found(n) => assert_eq!(n.id, "studio-a"),
            _ => panic!("expected Found"),
        }
    }

    #[test]
    fn resolve_unique_prefix() {
        let g = graph_of(vec![doc(
            "a.md",
            &[("type", s("project")), ("id", s("8f3a2c1e")), ("title", s("studio"))],
        )]);
        match resolve_project(&g, "8f3a") {
            Resolved::Found(n) => assert_eq!(n.id, "8f3a2c1e"),
            _ => panic!("expected Found"),
        }
    }

    #[test]
    fn resolve_ambiguous_prefix_lists_every_candidate() {
        let g = graph_of(vec![
            doc("a.md", &[("type", s("project")), ("id", s("8f3a-one")), ("title", s("A"))]),
            doc("b.md", &[("type", s("project")), ("id", s("8f3a-two")), ("title", s("B"))]),
        ]);
        match resolve_project(&g, "8f3a") {
            Resolved::Ambiguous { candidates, .. } => assert_eq!(candidates.len(), 2),
            _ => panic!("expected Ambiguous"),
        }
    }

    #[test]
    fn resolve_not_found_suggests_fuzzy_matches() {
        let g = graph_of(vec![doc(
            "a.md",
            &[("type", s("project")), ("id", s("autonomous-venture-studio")), ("title", s("Autonomous Venture Studio"))],
        )]);
        match resolve_project(&g, "venture") {
            Resolved::NotFound { suggestions, .. } => assert_eq!(suggestions.len(), 1),
            _ => panic!("expected NotFound"),
        }
    }

    #[test]
    fn resolve_not_found_with_no_suggestions_when_nothing_close() {
        let g = graph_of(vec![doc(
            "a.md",
            &[("type", s("project")), ("id", s("studio-a")), ("title", s("Studio A"))],
        )]);
        match resolve_project(&g, "zzzz") {
            Resolved::NotFound { suggestions, .. } => assert!(suggestions.is_empty()),
            _ => panic!("expected NotFound"),
        }
    }

    #[test]
    fn decisions_default_hides_superseded_and_revoked() {
        let g = graph_of(vec![
            doc("d1.md", &[("type", s("decision")), ("id", s("d1")), ("status", s("active"))]),
            doc("d2.md", &[("type", s("decision")), ("id", s("d2")), ("status", s("superseded"))]),
        ]);
        let shown = list(&g, "decision", &ListFilter::default());
        assert_eq!(shown.len(), 1);
        assert_eq!(shown[0].id, "d1");
    }

    #[test]
    fn include_inactive_shows_everything() {
        let g = graph_of(vec![
            doc("d1.md", &[("type", s("decision")), ("id", s("d1")), ("status", s("active"))]),
            doc("d2.md", &[("type", s("decision")), ("id", s("d2")), ("status", s("superseded"))]),
        ]);
        let filter = ListFilter { include_inactive: true, ..ListFilter::default() };
        assert_eq!(list(&g, "decision", &filter).len(), 2);
    }

    #[test]
    fn explicit_status_overrides_default_hiding() {
        let g = graph_of(vec![doc(
            "d2.md",
            &[("type", s("decision")), ("id", s("d2")), ("status", s("superseded"))],
        )]);
        let filter = ListFilter { status: vec!["superseded".into()], ..ListFilter::default() };
        assert_eq!(list(&g, "decision", &filter).len(), 1);
    }

    #[test]
    fn project_filter_matches_direct_owner() {
        let g = graph_of(vec![doc(
            "d.md",
            &[("type", s("decision")), ("id", s("d1")), ("project", s("studio-a"))],
        )]);
        let filter = ListFilter { projects: vec!["studio-a".into()], ..ListFilter::default() };
        assert_eq!(list(&g, "decision", &filter).len(), 1);
    }

    #[test]
    fn project_filter_matches_via_scope() {
        let g = graph_of(vec![doc(
            "d.md",
            &[
                ("type", s("decision")),
                ("id", s("d1")),
                ("project", s("studio-b")),
                ("scope", Value::Array(vec![s("studio-a")])),
            ],
        )]);
        let filter = ListFilter { projects: vec!["studio-a".into()], ..ListFilter::default() };
        let shown = list(&g, "decision", &filter);
        assert_eq!(shown.len(), 1);
        assert!(shown[0].is_shared_only("studio-a"));
    }

    #[test]
    fn multiple_project_flags_are_union_not_intersection() {
        let g = graph_of(vec![
            doc("d1.md", &[("type", s("decision")), ("id", s("d1")), ("project", s("a"))]),
            doc("d2.md", &[("type", s("decision")), ("id", s("d2")), ("project", s("b"))]),
            doc("d3.md", &[("type", s("decision")), ("id", s("d3")), ("project", s("c"))]),
        ]);
        let filter = ListFilter { projects: vec!["a".into(), "b".into()], ..ListFilter::default() };
        assert_eq!(list(&g, "decision", &filter).len(), 2);
    }

    #[test]
    fn project_filter_with_unknown_id_is_empty_not_an_error() {
        let g = graph_of(vec![doc(
            "d.md",
            &[("type", s("decision")), ("id", s("d1")), ("project", s("studio-a"))],
        )]);
        let filter = ListFilter { projects: vec!["zzzz".into()], ..ListFilter::default() };
        assert!(list(&g, "decision", &filter).is_empty());
    }

    #[test]
    fn global_decisions_have_no_project() {
        let g = graph_of(vec![doc("d.md", &[("type", s("decision")), ("id", s("d1"))])]);
        let shown = list(&g, "decision", &ListFilter::default());
        assert_eq!(shown.len(), 1);
        assert!(shown[0].project.is_none());
    }

    #[test]
    fn star_wildcard_means_every_project() {
        let g = graph_of(vec![
            doc("d1.md", &[("type", s("decision")), ("id", s("d1")), ("project", s("a"))]),
            doc("d2.md", &[("type", s("decision")), ("id", s("d2"))]),
        ]);
        let filter = ListFilter { projects: vec!["*".into()], ..ListFilter::default() };
        assert_eq!(list(&g, "decision", &filter).len(), 2);
    }

    #[test]
    fn conflicting_supersession_is_detected() {
        let g = graph_of(vec![
            doc(
                "d1.md",
                &[
                    ("type", s("decision")),
                    ("id", s("d1")),
                    ("status", s("active")),
                    ("supersedes", Value::Array(vec![s("old")])),
                ],
            ),
            doc(
                "d2.md",
                &[
                    ("type", s("decision")),
                    ("id", s("d2")),
                    ("status", s("active")),
                    ("supersedes", Value::Array(vec![s("old")])),
                ],
            ),
        ]);
        let (conflicts, map) = supersession_map(&g);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(map["old"].len(), 2);
    }

    #[test]
    fn supersedes_pointing_nowhere_does_not_panic() {
        let g = graph_of(vec![doc(
            "d1.md",
            &[
                ("type", s("decision")),
                ("id", s("d1")),
                ("status", s("active")),
                ("supersedes", Value::Array(vec![s("ghost")])),
            ],
        )]);
        let (conflicts, map) = supersession_map(&g);
        assert!(conflicts.is_empty());
        assert_eq!(map["ghost"], vec!["d1".to_string()]);
    }

    #[test]
    fn limit_truncates_results() {
        let docs = (0..5)
            .map(|i| doc(&format!("d{i}.md"), &[("type", s("decision")), ("id", s(&format!("d{i}")))]))
            .collect();
        let g = graph_of(docs);
        let filter = ListFilter { limit: 2, ..ListFilter::default() };
        assert_eq!(list(&g, "decision", &filter).len(), 2);
    }

    #[test]
    fn contexts_default_hides_archived() {
        let g = graph_of(vec![
            doc("c1.md", &[("type", s("context")), ("id", s("c1")), ("status", s("active"))]),
            doc("c2.md", &[("type", s("context")), ("id", s("c2")), ("status", s("archived"))]),
        ]);
        assert_eq!(list(&g, "context", &ListFilter::default()).len(), 1);
    }

    #[test]
    fn projects_default_hides_completed_and_archived() {
        let g = graph_of(vec![
            doc("p1.md", &[("type", s("project")), ("id", s("p1")), ("status", s("active"))]),
            doc("p2.md", &[("type", s("project")), ("id", s("p2")), ("status", s("paused"))]),
            doc("p3.md", &[("type", s("project")), ("id", s("p3")), ("status", s("completed"))]),
        ]);
        assert_eq!(list(&g, "project", &ListFilter::default()).len(), 2);
    }

    #[test]
    fn missing_status_defaults_to_active() {
        let g = graph_of(vec![doc("p1.md", &[("type", s("project")), ("id", s("p1"))])]);
        assert_eq!(g.notes[0].effective_status(), "active");
    }
}
