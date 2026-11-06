//! Writing notes on behalf of an agent (`memory_remember` over MCP).
//!
//! The memory's only writes to the notes folder are new Markdown files. A file that already
//! exists is never overwritten, the folder must stay inside the notes root, and the new note is
//! indexed at once so it can be cited straight away.

use crate::memory::error::{MemoryError, Result};
use crate::memory::search::{SearchEngine, TIERS};
use chrono::{NaiveDate, Utc};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::JoinHandle;
use std::time::Duration;

/// Confidence values a note may declare.
pub const CONFIDENCE: [&str; 3] = ["high", "medium", "low"];

/// Longest title, in characters.
pub const MAX_TITLE_CHARS: usize = 200;

/// Largest body an agent may write in one note, in bytes.
pub const MAX_BODY_BYTES: usize = 256 * 1024;

/// A note an agent wants to add.
#[derive(Clone, Debug)]
pub struct NewNote {
    /// Folder relative to the notes root. Empty means `inbox`.
    pub folder: String,
    /// Title; also the basis of the file name.
    pub title: String,
    /// Markdown body.
    pub body: String,
    /// One of [`TIERS`].
    pub tier: String,
    /// One of [`CONFIDENCE`].
    pub confidence: String,
    /// Optional projects/decisions/contexts graph tagging (see `crate::memory::graph`).
    pub graph: Option<GraphFields>,
}

/// Graph frontmatter for a new note, mirroring `crate::memory::graph`'s contract
/// (`type`/`id`/`status`/`project`/`scope`/`supersedes`). Validated against
/// [`crate::memory::graph::NOTE_TYPES`] before the note is written.
#[derive(Clone, Debug, Default)]
pub struct GraphFields {
    /// `type`: project, decision, context, knowledge, capture, preference.
    pub note_type: String,
    /// `id`: stable identifier, referenced by `project` / `scope` / `supersedes`.
    pub id: String,
    /// `status`, if given (else the graph module's per-type default applies).
    pub status: Option<String>,
    /// `project`: decision/context's main project.
    pub project: Option<String>,
    /// `scope`: other projects this note is also relevant to.
    pub scope: Vec<String>,
    /// `supersedes`: ids of decisions this one replaces.
    pub supersedes: Vec<String>,
}

/// File-name slug of a title: lower-case ASCII words joined by `-`, at most 80 characters.
pub fn slugify(title: &str) -> String {
    let mut out = String::new();
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let trimmed: String = out.trim_end_matches('-').chars().take(80).collect();
    if trimmed.is_empty() {
        // Titles with no ASCII letters (for example Japanese or Arabic) would all become the same
        // name; a hash of the whole title keeps them apart. FNV-1a, so the name is stable.
        let hash = title.bytes().fold(0x811c_9dc5_u32, |h, b| {
            (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
        });
        format!("note-{hash:08x}")
    } else {
        trimmed
    }
}

/// Validate a folder: relative, no `..`, no drive or root prefix. Empty means `inbox`.
pub fn safe_folder(folder: &str) -> Result<PathBuf> {
    let folder = folder.trim().trim_matches(|c| c == '/' || c == '\\');
    if folder.is_empty() {
        return Ok(PathBuf::from("inbox"));
    }
    let path = PathBuf::from(folder);
    for part in path.components() {
        match part {
            Component::Normal(_) => {}
            _ => {
                return Err(MemoryError::ConfigError(format!(
                    "folder must be relative to the notes root without '..': {folder}"
                )))
            }
        }
    }
    Ok(path)
}

/// The vault's inbox when no folder is given: an existing top-level folder whose name is `inbox`
/// once any numeric prefix such as `00. ` is ignored (so `00. Inbox` is found), else `inbox`.
pub fn default_inbox(root: &std::path::Path) -> PathBuf {
    let found = std::fs::read_dir(root).ok().and_then(|entries| {
        entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| {
                        n.trim_start_matches(|c: char| {
                            c.is_ascii_digit() || matches!(c, '.' | ' ' | '-' | '_')
                        })
                        .eq_ignore_ascii_case("inbox")
                    })
                    .unwrap_or(false)
            })
    });
    match found.and_then(|p| p.file_name().map(PathBuf::from)) {
        Some(name) => name,
        None => PathBuf::from("inbox"),
    }
}

/// A best-effort cross-process lock for the remember write path.
///
/// The graph preflight and file admission must be serialized: separate processes have separate
/// SQLite connections and can otherwise both pass the duplicate-id check. A lock directory is
/// created atomically by the OS, so it also works when several CLI processes race.
pub(crate) struct WriteLock {
    path: PathBuf,
    stop: Arc<AtomicBool>,
    heartbeat: Option<JoinHandle<()>>,
}

impl WriteLock {
    pub(crate) fn acquire(root: &Path) -> Result<Self> {
        let daemon = root.join(".plaesy-memory-daemon.lock");
        if let Ok(meta) = std::fs::symlink_metadata(&daemon) {
            if !meta.file_type().is_dir() {
                return Err(MemoryError::ConfigError(format!(
                    "daemon lock path is not a directory: {}",
                    daemon.display()
                )));
            }
            let stale = meta
                .modified()
                .ok()
                .and_then(|time| time.elapsed().ok())
                .map(|age| age > Duration::from_secs(600))
                .unwrap_or(false);
            if stale {
                // The daemon heartbeat stopped, so permit recovery after a crash. An active
                // daemon refreshes this directory every second and cannot be reclaimed here.
                let _ = std::fs::remove_dir(&daemon);
            } else {
                return Err(MemoryError::ConfigError(
                    "memory daemon owns this vault; use its service API for writes".into(),
                ));
            }
        }
        Self::acquire_named(root, ".plaesy-memory-writer.lock")
    }

    /// Acquire a cross-process lock with a role-specific path.
    ///
    /// Retries for up to ~30s with capped exponential backoff: a `remember` call that holds the
    /// lock for its whole embed+index duration (roughly 1-2s) must not starve sibling agents
    /// whose turn comes after 5+ others ahead of them in the queue.
    pub(crate) fn acquire_named(root: &Path, name: &str) -> Result<Self> {
        let path = root.join(name);
        let mut wait = Duration::from_millis(10);
        let max_wait = Duration::from_millis(250);
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            match std::fs::create_dir(&path) {
                Ok(()) => {
                    let stop = Arc::new(AtomicBool::new(false));
                    let heartbeat_stop = Arc::clone(&stop);
                    let heartbeat_path = path.clone();
                    let heartbeat = std::thread::spawn(move || {
                        while !heartbeat_stop.load(Ordering::Relaxed) {
                            std::thread::sleep(Duration::from_secs(1));
                            if heartbeat_stop.load(Ordering::Relaxed) {
                                break;
                            }
                            // Create/remove a child so the lock directory mtime advances.
                            // An active writer must never be reclaimed as stale.
                            let heartbeat_file = heartbeat_path.join("heartbeat");
                            if std::fs::OpenOptions::new()
                                .create_new(true)
                                .write(true)
                                .open(&heartbeat_file)
                                .is_ok()
                            {
                                let _ = std::fs::remove_file(heartbeat_file);
                            }
                        }
                    });
                    return Ok(Self {
                        path,
                        stop,
                        heartbeat: Some(heartbeat),
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Never follow a pre-existing symlink at the lock path. Apart from being
                    // ambiguous, it cannot be reclaimed with remove_dir and would create a
                    // permanent denial of service.
                    let lock_meta =
                        std::fs::symlink_metadata(&path).map_err(|error| MemoryError::IoError {
                            path: path.display().to_string(),
                            source: error,
                        })?;
                    if !lock_meta.file_type().is_dir() {
                        return Err(MemoryError::ConfigError(format!(
                            "lock path is not a directory: {}",
                            path.display()
                        )));
                    }
                    // A crashed writer must not permanently disable memory. Only reclaim a lock
                    // whose directory has been untouched for ten minutes.
                    let stale = lock_meta
                        .modified()
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .map(|age| age > Duration::from_secs(600))
                        .unwrap_or(false);
                    if stale {
                        let _ = std::fs::remove_dir(&path);
                        continue;
                    }
                    if std::time::Instant::now() >= deadline {
                        return Err(MemoryError::ConfigError(
                            "another memory write is still in progress".into(),
                        ));
                    }
                    std::thread::sleep(wait);
                    wait = std::cmp::min(wait * 2, max_wait);
                }
                Err(e) => {
                    return Err(MemoryError::IoError {
                        path: path.display().to_string(),
                        source: e,
                    })
                }
            }
        }
    }
}

impl Drop for WriteLock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.heartbeat.take() {
            let _ = handle.join();
        }
        let _ = std::fs::remove_file(self.path.join("heartbeat"));
        let _ = std::fs::remove_dir(&self.path);
    }
}

/// Front matter and body of a new note.
pub fn render(note: &NewNote, today: NaiveDate) -> String {
    // serde_json::to_string on a &str produces a double-quoted, properly escaped string that
    // is also valid YAML (YAML's double-quoted scalar syntax is a superset of JSON's) — this is
    // the only safe way to embed agent-supplied text as a YAML value, since any of these fields
    // may contain ':', '"', '#', or other characters that break bare/naively-quoted YAML.
    //
    // JSON itself is not quite enough: JSON permits raw 0x7F (DEL) and the C1 control range
    // (0x80-0x9F) unescaped inside a string, but YAML's double-quoted scalar grammar forbids
    // raw control characters outside its printable-character set. A title/id/project containing
    // such a byte (e.g. a stray DEL from clipboard noise) would otherwise produce frontmatter
    // that fails to parse as YAML, silently dropping the note from the graph. So every control
    // character is re-escaped as `\xHH` after JSON encoding, which is also valid YAML syntax.
    fn yaml_str(s: &str) -> String {
        let json = serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into());
        let mut out = String::with_capacity(json.len());
        for c in json.chars() {
            let code = c as u32;
            let is_noncharacter = (0xFDD0..=0xFDEF).contains(&code) || (code & 0xFFFF) >= 0xFFFE;
            if is_noncharacter {
                out.push_str(&format!("\\u{:04X}", code));
            } else if c.is_control() && c != '\u{0}' {
                // '\u{0}' never reaches here since JSON already escaped it as \u0000.
                out.push_str(&format!("\\x{:02X}", code));
            } else {
                out.push(c);
            }
        }
        out
    }
    fn yaml_list(items: &[String]) -> String {
        let rendered: Vec<String> = items.iter().map(|s| yaml_str(s)).collect();
        format!("[{}]", rendered.join(", "))
    }
    let title = yaml_str(&note.title);
    let mut graph_lines = String::new();
    if let Some(g) = &note.graph {
        graph_lines.push_str(&format!("type: {}\n", yaml_str(&g.note_type)));
        graph_lines.push_str(&format!("id: {}\n", yaml_str(&g.id)));
        if let Some(status) = &g.status {
            graph_lines.push_str(&format!("status: {}\n", yaml_str(status)));
        }
        if let Some(project) = &g.project {
            graph_lines.push_str(&format!("project: {}\n", yaml_str(project)));
        }
        if !g.scope.is_empty() {
            graph_lines.push_str(&format!("scope: {}\n", yaml_list(&g.scope)));
        }
        if !g.supersedes.is_empty() {
            graph_lines.push_str(&format!("supersedes: {}\n", yaml_list(&g.supersedes)));
        }
    }
    format!(
        "---\ntitle: {title}\ntier: {tier}\nconfidence: {conf}\nlast_verified: {today}\nsource: agent\n{graph_lines}---\n\n# {heading}\n\n{body}\n",
        tier = note.tier,
        conf = note.confidence,
        heading = note.title,
        body = note.body.trim_end(),
    )
}

impl SearchEngine {
    /// Create a new note under the notes root and index it.
    ///
    /// # Errors
    ///
    /// - `ConfigError` when no notes root is set, the tier or confidence is unknown, the folder
    ///   escapes the root, or a note with the same file name already exists
    /// - `IoError` when the file cannot be written
    pub async fn remember(&self, note: &NewNote) -> Result<PathBuf> {
        // An agent cannot vouch for its own note: a `high` claim is capped at `medium`, so a
        // poisoned or wrong note never reads as established fact. The owner may raise it by hand.
        let mut note = note.clone();
        if note.confidence == "high" {
            note.confidence = "medium".into();
        }
        let note = &note;
        let root = self
            .watch_root()
            .ok_or_else(|| MemoryError::ConfigError("no notes root is configured".into()))?
            .to_path_buf();
        let title = note.title.trim();
        if title.is_empty() {
            return Err(MemoryError::ConfigError("title must not be empty".into()));
        }
        if title.chars().count() > MAX_TITLE_CHARS {
            return Err(MemoryError::ConfigError(format!(
                "title is longer than {MAX_TITLE_CHARS} characters"
            )));
        }
        if note.body.trim().is_empty() {
            return Err(MemoryError::ConfigError("body must not be empty".into()));
        }
        if note.body.len() > MAX_BODY_BYTES {
            return Err(MemoryError::ConfigError(format!(
                "body is larger than {MAX_BODY_BYTES} bytes"
            )));
        }
        if !TIERS.contains(&note.tier.as_str()) {
            return Err(MemoryError::ConfigError(format!(
                "unknown tier: {}",
                note.tier
            )));
        }
        if !CONFIDENCE.contains(&note.confidence.as_str()) {
            return Err(MemoryError::ConfigError(format!(
                "confidence must be one of high, medium, low (got {})",
                note.confidence
            )));
        }
        // Serialize the graph preflight with file admission across CLI processes. Without this,
        // two writers can both observe an unused graph id and create duplicate IDs.
        let _write_lock = if self.daemon_mode() {
            None
        } else {
            Some(WriteLock::acquire(&root)?)
        };
        if let Some(g) = &note.graph {
            if !crate::memory::graph::NOTE_TYPES.contains(&g.note_type.as_str()) {
                return Err(MemoryError::ConfigError(format!(
                    "unknown graph type: {} (expected one of {:?})",
                    g.note_type,
                    crate::memory::graph::NOTE_TYPES
                )));
            }
            if g.id.trim().is_empty() {
                return Err(MemoryError::ConfigError(
                    "a graph note (--type given) needs a non-empty --id".into(),
                ));
            }
            if let Some(status) = &g.status {
                let allowed: &[&str] = match g.note_type.as_str() {
                    "project" => &crate::memory::graph::PROJECT_STATUSES,
                    "decision" => &crate::memory::graph::DECISION_STATUSES,
                    "context" => &crate::memory::graph::CONTEXT_STATUSES,
                    _ => &[],
                };
                if !allowed.is_empty() && !allowed.contains(&status.to_lowercase().as_str()) {
                    return Err(MemoryError::ConfigError(format!(
                        "status '{status}' is not valid for type '{}' (expected one of {allowed:?})",
                        g.note_type
                    )));
                }
            }
            // A decision cannot supersede itself: that would be a self-referencing cycle, not a
            // legitimate replacement, and nothing downstream would otherwise catch it.
            if g.supersedes.iter().any(|s| s == &g.id) {
                return Err(MemoryError::ConfigError(format!(
                    "decision '{}' cannot supersede itself",
                    g.id
                )));
            }
            // Two notes (any type) declaring the same id make every id lookup ambiguous: the
            // first match silently wins and every other claimant becomes invisible. Reject at
            // write time rather than letting it surface later as a silent lookup bug.
            let graph = crate::memory::graph::load_graph(&self.database()).await?;
            if let Some(existing) = graph.notes.iter().find(|n| n.id == g.id) {
                return Err(MemoryError::ConfigError(format!(
                    "id '{}' is already used by {}",
                    g.id, existing.path
                )));
            }
        }
        let rel = if note.folder.trim().is_empty() {
            default_inbox(&root)
        } else {
            safe_folder(&note.folder)?
        };
        let dir = root.join(rel);
        std::fs::create_dir_all(&dir).map_err(|e| MemoryError::IoError {
            path: dir.display().to_string(),
            source: e,
        })?;
        // A junction or symlink inside the root can point elsewhere; check where the folder really is.
        let real_root = std::fs::canonicalize(&root).map_err(|e| MemoryError::IoError {
            path: root.display().to_string(),
            source: e,
        })?;
        let real_dir = std::fs::canonicalize(&dir).map_err(|e| MemoryError::IoError {
            path: dir.display().to_string(),
            source: e,
        })?;
        if !real_dir.starts_with(&real_root) {
            return Err(MemoryError::ConfigError(format!(
                "folder resolves outside the notes root: {}",
                real_dir.display()
            )));
        }
        let path = dir.join(format!("{}.md", slugify(title)));
        let text = render(note, Utc::now().date_naive());
        // Publish through a same-directory temporary file. On Linux, every step from here is
        // performed relative to a single O_NOFOLLOW directory descriptor (`DirHandle`) opened
        // once against `dir`: this closes the window between the `canonicalize` check above and
        // the actual write/link/unlink, where a path-based re-open could otherwise follow a
        // symlink swapped in after the check. `hard_link`/`linkat` is used for admission: unlike
        // rename, it fails when the destination already exists and therefore cannot overwrite a
        // note created by another writer. sync_all makes the complete note durable before it
        // becomes visible to indexers.
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp_name = format!(".{}.tmp-{}-{}", slugify(title), std::process::id(), nonce);
        let dest_name = format!("{}.md", slugify(title));
        let tmp = dir.join(&tmp_name);
        let result = (|| -> std::io::Result<()> {
            let handle = crate::memory::fs_safety::DirHandle::open_nofollow(&dir)?;
            let mut file = handle.create_new_file(Path::new(&tmp_name))?;
            file.write_all(text.as_bytes())?;
            file.sync_all()?;
            drop(file);
            // Test-only fault injection: widen the window between the temp file becoming
            // visible on disk and its publication (hard_link + unlink) so a crash-recovery
            // test's SIGKILL can deterministically land inside it. Gated behind an env var that
            // is never set in production, so this has zero effect outside tests.
            #[cfg(debug_assertions)]
            if std::env::var_os("PLAESY_MEMORY_TEST_SLOW_PUBLISH").is_some() {
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
            handle.link_within(Path::new(&tmp_name), Path::new(&dest_name))?;
            let _ = handle.remove_file(Path::new(&tmp_name));
            let _ = handle.sync();
            Ok(())
        })();
        if let Err(e) = result {
            let _ = std::fs::remove_file(&tmp);
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                // Idempotency: a retried `remember` call (e.g. after a client-side timeout on a
                // call that actually succeeded) must not be treated as a conflict when the note
                // already on disk is exactly the one this call would have written. Compare
                // everything except `last_verified`, which legitimately differs by the day.
                if let Ok(existing) = std::fs::read_to_string(&path) {
                    if strip_last_verified(&existing) == strip_last_verified(&text) {
                        return Ok(path);
                    }
                }
                return Err(MemoryError::ConfigError(format!(
                    "note already exists with different content: {}",
                    path.display()
                )));
            }
            return Err(MemoryError::IoError {
                path: path.display().to_string(),
                source: e,
            });
        }
        let relative = path
            .strip_prefix(&root)
            .map_err(|_| MemoryError::ConfigError("published note escaped notes root".into()))?
            .to_string_lossy()
            .replace('\\', "/");
        // The outbox is durable before indexing starts. If the process dies after publication
        // (or during SQLite/HNSW replacement), startup reconciliation replays this path.
        self.database().enqueue_index(&relative).await?;
        self.index_file_at(&root, &path, None).await?;
        self.database().dequeue_index(&relative).await?;
        Ok(path)
    }
}

/// Remove the `last_verified: YYYY-MM-DD` line so two renders of the same note on different
/// days still compare equal for idempotency purposes.
fn strip_last_verified(text: &str) -> String {
    text.lines()
        .filter(|line| !line.starts_with("last_verified:"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::config::SearchConfig;
    use crate::memory::storage::{Database, DatabaseConfig};
    use std::sync::Arc;

    async fn engine(root: &std::path::Path) -> SearchEngine {
        let db = Arc::new(
            Database::new(DatabaseConfig {
                db_path: ":memory:".into(),
                ..DatabaseConfig::default()
            })
            .await
            .expect("db"),
        );
        let e = SearchEngine::new(db, None, SearchConfig::default())
            .await
            .expect("engine");
        e.set_watch_root(root.to_path_buf());
        e
    }

    /// Make a directory link; `false` when the OS refuses (Windows needs developer mode or admin).
    fn dir_link(target: &std::path::Path, link: &std::path::Path) -> bool {
        #[cfg(unix)]
        let made = std::os::unix::fs::symlink(target, link);
        #[cfg(windows)]
        let made = std::os::windows::fs::symlink_dir(target, link);
        made.is_ok()
    }

    #[test]
    fn slugs_are_safe_and_bounded() {
        assert_eq!(slugify("Deploy: Steps!!"), "deploy-steps");
        assert!(slugify("***").starts_with("note-"));
        assert!(slugify(&"a ".repeat(100)).len() <= 80);
    }

    #[test]
    fn non_ascii_titles_do_not_collide() {
        assert_ne!(slugify("漢字"), slugify("العربية"));
        assert_eq!(slugify("漢字"), slugify("漢字"));
    }

    #[tokio::test]
    async fn agent_cannot_claim_high_confidence() {
        let dir = tempfile::tempdir().expect("tmp");
        let engine = engine(dir.path()).await;
        let note = NewNote {
            folder: "ops".into(),
            title: "Claimed high".into(),
            body: "trust me".into(),
            tier: "procedural".into(),
            confidence: "high".into(),
            graph: None,
        };
        let path = engine.remember(&note).await.expect("saved");
        let text = std::fs::read_to_string(path).expect("read");
        assert!(text.contains("confidence: medium"));
        assert!(!text.contains("confidence: high"));
    }

    #[tokio::test]
    async fn remember_rejects_bad_titles_and_oversized_bodies() {
        let dir = tempfile::tempdir().expect("tmp");
        let engine = engine(dir.path()).await;
        let base = NewNote {
            folder: "ops".into(),
            title: "ok".into(),
            body: "x".into(),
            tier: "procedural".into(),
            confidence: "high".into(),
            graph: None,
        };
        let blank = NewNote {
            title: "   ".into(),
            ..base.clone()
        };
        assert!(engine.remember(&blank).await.is_err());
        let blank_body = NewNote {
            body: "  \n".into(),
            ..base.clone()
        };
        assert!(engine.remember(&blank_body).await.is_err());
        let long = NewNote {
            title: "t".repeat(MAX_TITLE_CHARS + 1),
            ..base.clone()
        };
        assert!(engine.remember(&long).await.is_err());
        let big = NewNote {
            body: "x".repeat(MAX_BODY_BYTES + 1),
            ..base.clone()
        };
        assert!(engine.remember(&big).await.is_err());
        assert!(!dir.path().join("ops").join("ok.md").exists());
    }

    #[tokio::test]
    async fn remember_rejects_symlink_lock_path() {
        let root = tempfile::tempdir().expect("root");
        let outside = tempfile::tempdir().expect("outside");
        if !dir_link(
            outside.path(),
            &root.path().join(".plaesy-memory-writer.lock"),
        ) {
            return;
        }
        let engine = engine(root.path()).await;
        let note = NewNote {
            folder: "ops".into(),
            title: "lock symlink".into(),
            body: "x".into(),
            tier: "procedural".into(),
            confidence: "medium".into(),
            graph: None,
        };
        let err = engine
            .remember(&note)
            .await
            .expect_err("symlink lock must fail");
        assert!(err.to_string().contains("lock path is not a directory"));
        assert!(!outside.path().join("ops").exists());
    }

    #[tokio::test]
    async fn remember_refuses_link_out_of_root() {
        let root = tempfile::tempdir().expect("tmp");
        let outside = tempfile::tempdir().expect("tmp");
        if !dir_link(outside.path(), &root.path().join("link")) {
            return; // no link privilege on this machine; nothing to test
        }
        let engine = engine(root.path()).await;
        let note = NewNote {
            folder: "link".into(),
            title: "escape".into(),
            body: "x".into(),
            tier: "procedural".into(),
            confidence: "high".into(),
            graph: None,
        };
        assert!(engine.remember(&note).await.is_err());
        assert!(!outside.path().join("escape.md").exists());
    }

    #[test]
    fn folders_must_stay_inside_root() {
        assert_eq!(safe_folder("").unwrap(), PathBuf::from("inbox"));
        assert_eq!(
            safe_folder("ops/runbooks").unwrap(),
            PathBuf::from("ops/runbooks")
        );
        assert!(safe_folder("../outside").is_err());
        assert!(safe_folder("a/../../b").is_err());
        assert!(safe_folder("/etc").is_ok()); // leading slash is trimmed, then relative
    }

    #[test]
    fn default_inbox_finds_numbered_vault_inbox() {
        let dir = tempfile::tempdir().expect("tmp");
        assert_eq!(default_inbox(dir.path()), PathBuf::from("inbox"));
        std::fs::create_dir(dir.path().join("00. Inbox")).expect("mkdir");
        assert_eq!(default_inbox(dir.path()), PathBuf::from("00. Inbox"));
    }

    #[test]
    fn rendered_note_has_provenance() {
        let note = NewNote {
            folder: String::new(),
            title: "Rollback \"plan\"".into(),
            body: "Revert the release.".into(),
            tier: "procedural".into(),
            confidence: "medium".into(),
            graph: None,
        };
        let text = render(&note, NaiveDate::from_ymd_opt(2026, 10, 26).unwrap());
        assert!(text.contains("source: agent"));
        assert!(text.contains("last_verified: 2026-10-26"));
        assert!(text.contains("title: \"Rollback \\\"plan\\\"\""));
    }

    #[tokio::test]
    async fn remember_is_idempotent_on_identical_retry() {
        let dir = tempfile::tempdir().expect("tmp");
        let engine = engine(dir.path()).await;
        let note = NewNote {
            folder: "".into(),
            title: "Retry Note".into(),
            body: "Same content both times.".into(),
            tier: "semantic".into(),
            confidence: "medium".into(),
            graph: None,
        };
        let first = engine.remember(&note).await.expect("first call");
        // Simulate a client retry after e.g. a timeout: identical call must succeed again with
        // the same path, not error as a conflict.
        let second = engine.remember(&note).await.expect("idempotent retry");
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn remember_rejects_same_title_with_different_content() {
        let dir = tempfile::tempdir().expect("tmp");
        let engine = engine(dir.path()).await;
        let mut note = NewNote {
            folder: "".into(),
            title: "Conflict Note".into(),
            body: "Original content.".into(),
            tier: "semantic".into(),
            confidence: "medium".into(),
            graph: None,
        };
        engine.remember(&note).await.expect("first call");
        note.body = "Different content, not a retry.".into();
        let err = engine
            .remember(&note)
            .await
            .expect_err("must not silently overwrite differing content");
        assert!(err.to_string().contains("different content"));
    }

    #[tokio::test]
    async fn remember_writes_valid_graph_fields_and_they_are_queryable() {
        let dir = tempfile::tempdir().expect("tmp");
        let engine = engine(dir.path()).await;
        let note = NewNote {
            folder: "01. Projects".into(),
            title: "Studio C".into(),
            body: "A new studio.".into(),
            tier: "semantic".into(),
            confidence: "medium".into(),
            graph: Some(GraphFields {
                note_type: "project".into(),
                id: "studio-c".into(),
                status: Some("active".into()),
                project: None,
                scope: vec![],
                supersedes: vec![],
            }),
        };
        let path = engine.remember(&note).await.expect("saved");
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.contains("type: \"project\""));
        assert!(text.contains("id: \"studio-c\""));
        assert!(text.contains("status: \"active\""));
        let graph = crate::memory::graph::load_graph(&engine.database())
            .await
            .expect("graph loads");
        assert!(graph.notes.iter().any(|n| n.id == "studio-c"));
    }

    #[tokio::test]
    async fn remember_rejects_duplicate_graph_id() {
        let dir = tempfile::tempdir().expect("tmp");
        let engine = engine(dir.path()).await;
        let first = NewNote {
            folder: "01. Projects".into(),
            title: "Studio D".into(),
            body: "First.".into(),
            tier: "semantic".into(),
            confidence: "medium".into(),
            graph: Some(GraphFields {
                note_type: "project".into(),
                id: "dup-id".into(),
                status: Some("active".into()),
                project: None,
                scope: vec![],
                supersedes: vec![],
            }),
        };
        engine.remember(&first).await.expect("first saved");
        let second = NewNote {
            folder: "01. Projects".into(),
            title: "Studio E".into(),
            body: "Second, different title, same id.".into(),
            tier: "semantic".into(),
            confidence: "medium".into(),
            graph: Some(GraphFields {
                note_type: "project".into(),
                id: "dup-id".into(),
                status: Some("active".into()),
                project: None,
                scope: vec![],
                supersedes: vec![],
            }),
        };
        let err = engine
            .remember(&second)
            .await
            .expect_err("duplicate id rejected");
        assert!(err.to_string().contains("dup-id"));
    }

    #[tokio::test]
    async fn remember_rejects_self_superseding_decision() {
        let dir = tempfile::tempdir().expect("tmp");
        let engine = engine(dir.path()).await;
        let note = NewNote {
            folder: "02. Areas".into(),
            title: "Self Ref".into(),
            body: "Supersedes itself.".into(),
            tier: "semantic".into(),
            confidence: "medium".into(),
            graph: Some(GraphFields {
                note_type: "decision".into(),
                id: "self-1".into(),
                status: None,
                project: None,
                scope: vec![],
                supersedes: vec!["self-1".into()],
            }),
        };
        let err = engine
            .remember(&note)
            .await
            .expect_err("self-supersede rejected");
        assert!(err.to_string().contains("supersede itself"));
    }

    #[tokio::test]
    async fn remember_rejects_unknown_graph_type_and_missing_id() {
        let dir = tempfile::tempdir().expect("tmp");
        let engine = engine(dir.path()).await;
        let bad_type = NewNote {
            folder: String::new(),
            title: "X".into(),
            body: "x".into(),
            tier: "semantic".into(),
            confidence: "medium".into(),
            graph: Some(GraphFields {
                note_type: "not-a-type".into(),
                id: "x".into(),
                status: None,
                project: None,
                scope: vec![],
                supersedes: vec![],
            }),
        };
        assert!(engine.remember(&bad_type).await.is_err());
        let missing_id = NewNote {
            folder: String::new(),
            title: "Y".into(),
            body: "y".into(),
            tier: "semantic".into(),
            confidence: "medium".into(),
            graph: Some(GraphFields {
                note_type: "decision".into(),
                id: String::new(),
                status: None,
                project: None,
                scope: vec![],
                supersedes: vec![],
            }),
        };
        assert!(engine.remember(&missing_id).await.is_err());
        let bad_status = NewNote {
            folder: String::new(),
            title: "Z".into(),
            body: "z".into(),
            tier: "semantic".into(),
            confidence: "medium".into(),
            graph: Some(GraphFields {
                note_type: "decision".into(),
                id: "z1".into(),
                status: Some("not-a-status".into()),
                project: None,
                scope: vec![],
                supersedes: vec![],
            }),
        };
        assert!(engine.remember(&bad_status).await.is_err());
    }

    #[tokio::test]
    async fn graph_fields_with_yaml_breaking_characters_still_parse() {
        // Regression test: id/status/project/scope with ':' or '"' used to be written as bare
        // or naively single-quoted YAML scalars, producing invalid frontmatter that silently
        // fell back to plain-text indexing (the note vanished from the graph without error).
        let dir = tempfile::tempdir().expect("tmp");
        let engine = engine(dir.path()).await;
        let note = NewNote {
            folder: String::new(),
            title: "Edge case".into(),
            body: "x".into(),
            tier: "semantic".into(),
            confidence: "medium".into(),
            graph: Some(GraphFields {
                note_type: "decision".into(),
                id: "weird: id \"with\" quotes".into(),
                status: Some("active".into()),
                project: Some("studio: a".into()),
                scope: vec!["studio\"b".into(), "plain".into()],
                supersedes: vec![],
            }),
        };
        let path = engine.remember(&note).await.expect("saved");
        let text = std::fs::read_to_string(&path).expect("read");
        // The written file must be valid YAML frontmatter, not a fallback-to-plain-text note.
        let fm = text
            .strip_prefix("---\n")
            .and_then(|s| s.split_once("\n---\n"))
            .map(|(fm, _)| fm)
            .expect("has frontmatter delimiters");
        let parsed: serde_yaml::Value =
            serde_yaml::from_str(fm).expect("frontmatter must be valid YAML");
        assert_eq!(
            parsed.get("id").and_then(|v| v.as_str()),
            Some("weird: id \"with\" quotes")
        );
        assert_eq!(
            parsed.get("project").and_then(|v| v.as_str()),
            Some("studio: a")
        );
        let graph = crate::memory::graph::load_graph(&engine.database())
            .await
            .expect("graph loads");
        assert!(graph
            .notes
            .iter()
            .any(|n| n.id == "weird: id \"with\" quotes"));
    }
}

/// Property-based adversarial fuzzing for the pure, boundary-facing helpers in this module.
/// These take arbitrary agent/user-controlled strings and must never panic, never escape the
/// notes root, and must always round-trip through valid YAML.
#[cfg(test)]
mod fuzz {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // slugify must never panic, is always <= 80 chars, and only ever produces
        // lowercase ascii alphanumerics and single, non-leading/trailing hyphens.
        #[test]
        fn slugify_never_panics_and_is_bounded(title in ".*") {
            let slug = slugify(&title);
            prop_assert!(slug.chars().count() <= 80);
            prop_assert!(!slug.is_empty());
            prop_assert!(slug.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'));
            prop_assert!(!slug.starts_with('-'));
            prop_assert!(!slug.ends_with('-'));
            prop_assert!(!slug.contains("--"));
        }

        // safe_folder must reject every attempt to leave the root (via '..', absolute paths,
        // or anything that is not a plain relative Normal-component path), for any input.
        #[test]
        fn safe_folder_never_escapes_root(folder in ".*") {
            if let Ok(path) = safe_folder(&folder) {
                for part in path.components() {
                    prop_assert!(matches!(part, std::path::Component::Normal(_)));
                }
            }
            // Either outcome (Ok with a safe relative path, or Err) is acceptable; a panic is not.
        }

        #[test]
        fn safe_folder_rejects_all_parent_traversal(
            segments in prop::collection::vec("[a-zA-Z0-9_. ]{0,8}", 1..6)
        ) {
            let folder = segments.join("/") + "/../escape";
            prop_assert!(safe_folder(&folder).is_err());
        }

        // render() embeds arbitrary agent-supplied strings into YAML frontmatter via
        // serde_json::to_string (JSON double-quoted scalars are valid YAML scalars). The
        // produced frontmatter must always parse back as YAML, for any title/body/graph text,
        // including strings containing ':', '"', '#', control characters, or unicode.
        #[test]
        fn render_always_produces_parseable_yaml(
            title in ".{0,200}",
            body in ".{0,200}",
            id in ".{0,100}",
            project in ".{0,100}",
        ) {
            let note = NewNote {
                folder: String::new(),
                title: if title.trim().is_empty() { "x".to_string() } else { title },
                body: if body.trim().is_empty() { "x".to_string() } else { body },
                tier: "semantic".into(),
                confidence: "medium".into(),
                graph: Some(GraphFields {
                    note_type: "decision".into(),
                    id: if id.trim().is_empty() { "id".to_string() } else { id },
                    status: None,
                    project: if project.is_empty() { None } else { Some(project) },
                    scope: vec![],
                    supersedes: vec![],
                }),
            };
            let text = render(&note, NaiveDate::from_ymd_opt(2026, 1, 1).unwrap());
            let fm = text
                .strip_prefix("---\n")
                .and_then(|s| s.split_once("\n---\n"))
                .map(|(fm, _)| fm);
            prop_assert!(fm.is_some(), "frontmatter delimiters must always be present");
            let parsed: std::result::Result<serde_yaml::Value, _> = serde_yaml::from_str(fm.unwrap());
            prop_assert!(parsed.is_ok(), "rendered frontmatter must always be valid YAML: {:?}", parsed.err());
        }

        // strip_last_verified must be stable (idempotent) and must never panic on arbitrary
        // multi-line text, including text that itself contains lines starting with the same
        // prefix in positions other than the real date line.
        #[test]
        fn strip_last_verified_is_idempotent(text in ".{0,500}") {
            let once = strip_last_verified(&text);
            let twice = strip_last_verified(&once);
            prop_assert_eq!(once, twice);
        }
    }
}
