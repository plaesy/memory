//! Writing notes on behalf of an agent (`memory_remember` over MCP).
//!
//! The memory's only writes to the notes folder are new Markdown files. A file that already
//! exists is never overwritten, the folder must stay inside the notes root, and the new note is
//! indexed at once so it can be cited straight away.

use crate::memory::error::{MemoryError, Result};
use crate::memory::search::{SearchEngine, TIERS};
use chrono::{NaiveDate, Utc};
use std::io::Write;
use std::path::{Component, PathBuf};

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

/// Front matter and body of a new note.
pub fn render(note: &NewNote, today: NaiveDate) -> String {
    let title = serde_json::to_string(&note.title).unwrap_or_else(|_| "\"\"".into());
    format!(
        "---\ntitle: {title}\ntier: {tier}\nconfidence: {conf}\nlast_verified: {today}\nsource: agent\n---\n\n# {heading}\n\n{body}\n",
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
        // create_new: fails rather than overwriting a note that is already there.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    MemoryError::ConfigError(format!("note already exists: {}", path.display()))
                } else {
                    MemoryError::IoError {
                        path: path.display().to_string(),
                        source: e,
                    }
                }
            })?;
        file.write_all(text.as_bytes())
            .map_err(|e| MemoryError::IoError {
                path: path.display().to_string(),
                source: e,
            })?;
        drop(file);
        self.index_file(&root, &path).await?;
        Ok(path)
    }
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
        };
        let blank = NewNote {
            title: "   ".into(),
            ..base.clone()
        };
        assert!(engine.remember(&blank).await.is_err());
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
        };
        let text = render(&note, NaiveDate::from_ymd_opt(2026, 10, 26).unwrap());
        assert!(text.contains("source: agent"));
        assert!(text.contains("last_verified: 2026-10-26"));
        assert!(text.contains("title: \"Rollback \\\"plan\\\"\""));
    }
}
