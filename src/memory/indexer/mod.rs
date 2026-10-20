//! Document indexing subsystem.
//!
//! Handles file watching, text extraction, chunking, and batch operations:
//! - **File Watching**: Monitors `.plaesy/` directory for changes (added, modified, deleted)
//! - **Extraction**: Parses YAML frontmatter + Markdown body from text files
//! - **Chunking**: Splits documents into semantic chunks with metadata preservation
//! - **Watcher**: Debounced async file system events with atomic batch updates
//!
//! # Architecture
//!
//! ```text
//! .plaesy/ files
//!     │
//!     ▼
//! Watcher (notify crate, debounced)
//!     │
//!     ▼
//! Extractor (YAML + Markdown)
//!     │
//!     ▼
//! Chunker (headings / paragraphs / code / quotes, exact line ranges)
//!     │
//!     ▼
//! Vec<Chunk>  → embeddings + storage
//! ```
//!
//! Chunk text is always a verbatim slice of the source (EV-03: no generated
//! content), and every chunk carries the 1-based line range it came from (EV-01).
//!
//! # Example
//!
//! ```ignore
//! use plaesy::memory::indexer::{Chunker, Extractor, Watcher};
//!
//! let extractor = Extractor::new();
//! let chunker = Chunker::new();
//! let mut watcher = Watcher::new(".plaesy").await?;
//!
//! for event in watcher.changes().await? {
//!     let content = std::fs::read_to_string(&event.path)?;
//!     let mut doc = extractor.extract(event.path, &content)?;
//!     doc.chunks = chunker.chunk(&doc)?;
//! }
//! ```

use crate::memory::error::{MemoryError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

/// A document represents a single indexed file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    /// Unique document ID (typically from database).
    pub id: Option<u64>,

    /// Path to the original file (relative to `.plaesy/`).
    pub source_path: PathBuf,

    /// SHA256 hash of file content (for change detection).
    pub hash: Option<String>,

    /// YAML frontmatter as key-value pairs.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Markdown body text (frontmatter removed, line endings normalised to `\n`).
    pub body: String,

    /// The frontmatter block exactly as written, delimiters included (EV-03: the
    /// frontmatter chunk is a verbatim slice of the file, not a re-serialisation).
    #[serde(default)]
    pub frontmatter_raw: Option<String>,

    /// 1-based line number in the source file where `body` begins.
    #[serde(default = "default_body_start")]
    pub body_start_line: usize,

    /// Extracted chunks from this document.
    pub chunks: Vec<Chunk>,

    /// Timestamp when document was indexed.
    pub indexed_at: Option<chrono::DateTime<chrono::Utc>>,
}

fn default_body_start() -> usize {
    1
}

/// A chunk represents a semantic unit of a document (paragraph, section, etc.).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique chunk ID.
    pub id: Option<u64>,

    /// Owning document ID (set once stored).
    #[serde(default)]
    pub document_id: Option<u64>,

    /// Chunk text content (verbatim from the source).
    pub text: String,

    /// Type of chunk (summary, detail, code, etc.).
    pub kind: ChunkKind,

    /// Multi-scale layer (0 = summary/heading, 1+ = detail layers).
    pub layer: usize,

    /// Start line in original file (1-based, inclusive).
    pub start_line: usize,

    /// End line in original file (1-based, inclusive).
    pub end_line: usize,

    /// Optional metadata attached to chunk (e.g. `section` = enclosing heading).
    pub metadata: Option<HashMap<String, String>>,
}

/// Type of chunk content.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChunkKind {
    /// Summary (YAML frontmatter or document header).
    Summary,

    /// Detailed content (main body).
    Detail,

    /// Code block or example.
    Code,

    /// Quoted text or reference.
    Quote,

    /// Heading or section title.
    Heading,
}

/// Compute the hex SHA-256 of `content`.
pub fn content_hash(content: &str) -> String {
    let digest = Sha256::digest(content.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Whether a path is a file type the indexer handles (by extension).
pub fn is_indexable(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("md" | "markdown" | "txt")
    )
}

/// Recursively list indexable files under `root`, sorted.
pub fn scan_dir(root: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && is_indexable(e.path()))
        .map(|e| e.into_path())
        .collect();
    files.sort();
    files
}

/// True when `path`, relative to `root`, lies inside one of `exclude_dirs`.
///
/// Each entry is a run of folder names matched anywhere in the tree, so `.plaesy/templates`
/// excludes both `<root>/.plaesy/templates/x.md` and `<root>/Projects/A/.plaesy/templates/x.md`.
pub fn is_excluded(root: &Path, path: &Path, exclude_dirs: &[String]) -> bool {
    if exclude_dirs.is_empty() {
        return false;
    }
    let rel = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    let rel = format!("/{rel}/");
    exclude_dirs.iter().any(|d| {
        let d = d.trim_matches('/');
        !d.is_empty() && rel.contains(&format!("/{d}/"))
    })
}

/// Document extractor: parses files into structured documents.
pub struct Extractor {
    // Configuration for extraction behavior
}

impl Extractor {
    /// Create a new extractor.
    pub fn new() -> Self {
        Extractor {}
    }

    /// Extract a document from file content.
    ///
    /// A leading `---` line starts YAML frontmatter, closed by the next `---`
    /// (or `...`) line. A file without frontmatter is all body.
    ///
    /// # Errors
    ///
    /// - `ExtractionError` if the content is empty
    /// - `YamlError` if frontmatter is unterminated, malformed, or not a mapping
    pub fn extract(&self, path: PathBuf, content: &str) -> Result<Document> {
        if content.trim().is_empty() {
            return Err(MemoryError::ExtractionError {
                path,
                reason: "file is empty".to_string(),
            });
        }
        let hash = content_hash(content);
        let normalized = content.trim_start_matches('\u{feff}').replace("\r\n", "\n");
        let lines: Vec<&str> = normalized.split('\n').collect();

        let mut metadata = HashMap::new();
        let mut body_start = 0usize; // 0-based index into `lines`
        let mut frontmatter_raw: Option<String> = None;

        if lines.first().map(|l| l.trim_end()) == Some("---") {
            let close = lines
                .iter()
                .enumerate()
                .skip(1)
                .find(|(_, l)| matches!(l.trim_end(), "---" | "..."))
                .map(|(i, _)| i);
            let Some(close) = close else {
                return Err(MemoryError::YamlError {
                    path,
                    reason: "frontmatter opened with '---' but never closed".to_string(),
                });
            };
            let yaml = lines[1..close].join("\n");
            if !yaml.trim().is_empty() {
                let value: serde_yaml::Value =
                    serde_yaml::from_str(&yaml).map_err(|e| MemoryError::YamlError {
                        path: path.clone(),
                        reason: e.to_string(),
                    })?;
                let json = serde_json::to_value(value).map_err(|e| MemoryError::YamlError {
                    path: path.clone(),
                    reason: e.to_string(),
                })?;
                match json {
                    serde_json::Value::Object(map) => metadata = map.into_iter().collect(),
                    serde_json::Value::Null => {}
                    _ => {
                        return Err(MemoryError::YamlError {
                            path,
                            reason: "frontmatter must be a YAML mapping".to_string(),
                        })
                    }
                }
            }
            body_start = close + 1;
            frontmatter_raw = Some(lines[..=close].join("\n"));
        }

        let body = lines[body_start..].join("\n");
        Ok(Document {
            id: None,
            source_path: path,
            hash: Some(hash),
            metadata,
            body,
            frontmatter_raw,
            body_start_line: body_start + 1,
            chunks: Vec::new(),
            indexed_at: Some(chrono::Utc::now()),
        })
    }

    /// Extract `content` as plain text, ignoring any frontmatter block.
    ///
    /// Fallback for files whose frontmatter is not valid YAML (e.g. templates
    /// with `{{placeholders}}`): the whole file stays searchable, with line
    /// numbers intact, and `frontmatter_error` records why it was not parsed.
    pub fn extract_plain(&self, path: PathBuf, content: &str, reason: &str) -> Result<Document> {
        if content.trim().is_empty() {
            return Err(MemoryError::ExtractionError {
                path,
                reason: "file is empty".to_string(),
            });
        }
        let mut metadata = HashMap::new();
        metadata.insert(
            "frontmatter_error".to_string(),
            serde_json::Value::String(reason.to_string()),
        );
        Ok(Document {
            id: None,
            source_path: path,
            hash: Some(content_hash(content)),
            metadata,
            body: content.trim_start_matches('\u{feff}').replace("\r\n", "\n"),
            frontmatter_raw: None,
            body_start_line: 1,
            chunks: Vec::new(),
            indexed_at: Some(chrono::Utc::now()),
        })
    }

    /// Read a file from disk and extract it.
    ///
    /// # Errors
    ///
    /// - `IoError` if the file cannot be read
    /// - `ExtractionError` if the file is empty
    ///
    /// Invalid frontmatter is not an error here: the file is extracted with
    /// [`Extractor::extract_plain`] so it stays searchable.
    pub fn extract_file(&self, path: &Path) -> Result<Document> {
        let content = std::fs::read_to_string(path).map_err(|e| MemoryError::IoError {
            path: path.display().to_string(),
            source: e,
        })?;
        match self.extract(path.to_path_buf(), &content) {
            Err(MemoryError::YamlError { reason, .. }) => {
                tracing::warn!(
                    "{}: invalid frontmatter, indexing as plain text: {reason}",
                    path.display()
                );
                self.extract_plain(path.to_path_buf(), &content, &reason)
            }
            other => other,
        }
    }
}

impl Default for Extractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Document chunker: splits documents into multi-scale chunks.
pub struct Chunker {
    /// Target chunk size in characters.
    pub chunk_size: usize,

    /// Overlap between chunks when a single line is longer than `chunk_size`.
    pub overlap: usize,
}

/// A raw block of consecutive body lines before size-splitting.
struct Block {
    kind: ChunkKind,
    start: usize, // 0-based index into body lines
    end: usize,   // inclusive
}

impl Chunker {
    /// Create a new chunker with default settings.
    pub fn new() -> Self {
        Chunker {
            chunk_size: 512,
            overlap: 50,
        }
    }

    /// Chunk a document into semantic units.
    ///
    /// - Frontmatter (if any) becomes one `Summary` chunk (layer 0).
    /// - Each Markdown heading becomes a `Heading` chunk (layer 0).
    /// - Fenced code → `Code`, `>` blocks → `Quote`, other paragraphs → `Detail` (layer 1).
    /// - Adjacent small `Detail` paragraphs are merged up to `chunk_size`;
    ///   larger blocks are split on line boundaries (and on characters, with
    ///   `overlap`, if a single line exceeds `chunk_size`).
    /// - Chunks under a heading carry `section` metadata with that heading.
    ///
    /// # Errors
    ///
    /// - `ChunkingError` if `chunk_size` is zero or `overlap >= chunk_size`
    pub fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>> {
        if self.chunk_size == 0 || self.overlap >= self.chunk_size {
            return Err(MemoryError::ChunkingError(format!(
                "invalid chunker settings: chunk_size={}, overlap={}",
                self.chunk_size, self.overlap
            )));
        }
        let mut out = Vec::new();

        if doc.body_start_line > 1 && !doc.metadata.is_empty() {
            // Frontmatter occupied lines 1..body_start_line-1 (delimiters included).
            let text = doc.frontmatter_raw.clone().unwrap_or_else(|| {
                serde_yaml::to_string(&doc.metadata)
                    .unwrap_or_default()
                    .trim_end()
                    .to_string()
            });
            out.push(Chunk {
                id: None,
                document_id: doc.id,
                text,
                kind: ChunkKind::Summary,
                layer: 0,
                start_line: 1,
                end_line: doc.body_start_line - 1,
                metadata: None,
            });
        }

        let lines: Vec<&str> = doc.body.split('\n').collect();
        let blocks = Self::blocks(&lines);
        let base = doc.body_start_line; // line number of lines[0]
        let mut section: Option<String> = None;
        // Merged run of lines: (start, end, text, heading_only). A heading opens a
        // run so the paragraphs under it share its chunk (a bare title is a poor
        // search hit); if nothing follows, it is emitted as a `Heading` chunk.
        type Run = Option<(usize, usize, String, bool)>;
        let mut pending: Run = None;

        let flush = |pending: &mut Run, out: &mut Vec<Chunk>, section: &Option<String>| {
            if let Some((s, e, text, heading_only)) = pending.take() {
                let (kind, layer) = if heading_only {
                    (ChunkKind::Heading, 0)
                } else {
                    (ChunkKind::Detail, 1)
                };
                out.push(Self::make(doc, kind, layer, s, e, text, section));
            }
        };

        for b in &blocks {
            let text = lines[b.start..=b.end].join("\n");
            match b.kind {
                ChunkKind::Heading => {
                    let title = text.trim_start_matches('#').trim().to_string();
                    // A heading directly under another heading (no body yet) joins that run, so
                    // a bare title never becomes its own chunk; the run is flushed when it has a body.
                    let mergeable = match pending.as_ref() {
                        Some((_, _, t, true)) => {
                            t.chars().count() + 1 + text.chars().count() <= self.chunk_size
                        }
                        _ => false,
                    };
                    if mergeable {
                        if let Some((_, e, t, _)) = pending.as_mut() {
                            let gap = (base + b.start) - *e - 1;
                            t.push_str(&"\n".repeat(gap + 1));
                            t.push_str(&text);
                            *e = base + b.end;
                        }
                    } else {
                        flush(&mut pending, &mut out, &section);
                        pending = Some((base + b.start, base + b.end, text, true));
                    }
                    section = Some(title);
                }
                ChunkKind::Detail => {
                    let fits = text.chars().count() <= self.chunk_size;
                    if fits {
                        match pending.as_mut() {
                            Some((_, e, t, heading_only))
                                if t.chars().count() + 2 + text.chars().count()
                                    <= self.chunk_size =>
                            {
                                // Keep original separation: pad with the blank lines in between.
                                let gap = (base + b.start) - *e - 1;
                                t.push_str(&"\n".repeat(gap + 1));
                                t.push_str(&text);
                                *e = base + b.end;
                                *heading_only = false;
                            }
                            _ => {
                                flush(&mut pending, &mut out, &section);
                                pending = Some((base + b.start, base + b.end, text, false));
                            }
                        }
                    } else {
                        flush(&mut pending, &mut out, &section);
                        for (s, e, t) in self.split_block(&lines, b, base) {
                            out.push(Self::make(doc, ChunkKind::Detail, 1, s, e, t, &section));
                        }
                    }
                }
                kind => {
                    // A bare heading run also takes a following list, code block or quote, so
                    // the title is searchable together with its body.
                    let joins = match pending.as_ref() {
                        Some((_, _, t, true)) => {
                            t.chars().count() + 2 + text.chars().count() <= self.chunk_size
                        }
                        _ => false,
                    };
                    if joins {
                        if let Some((_, e, t, heading_only)) = pending.as_mut() {
                            let gap = (base + b.start) - *e - 1;
                            t.push_str(&"\n".repeat(gap + 1));
                            t.push_str(&text);
                            *e = base + b.end;
                            *heading_only = false;
                        }
                        continue;
                    }
                    flush(&mut pending, &mut out, &section);
                    if text.chars().count() <= self.chunk_size {
                        out.push(Self::make(
                            doc,
                            kind,
                            1,
                            base + b.start,
                            base + b.end,
                            text,
                            &section,
                        ));
                    } else {
                        for (s, e, t) in self.split_block(&lines, b, base) {
                            out.push(Self::make(doc, kind, 1, s, e, t, &section));
                        }
                    }
                }
            }
        }
        flush(&mut pending, &mut out, &section);
        Ok(out)
    }

    fn make(
        doc: &Document,
        kind: ChunkKind,
        layer: usize,
        start_line: usize,
        end_line: usize,
        text: String,
        section: &Option<String>,
    ) -> Chunk {
        Chunk {
            id: None,
            document_id: doc.id,
            text,
            kind,
            layer,
            start_line,
            end_line,
            metadata: section.as_ref().filter(|s| !s.is_empty()).map(|s| {
                let mut m = HashMap::new();
                m.insert("section".to_string(), s.clone());
                m
            }),
        }
    }

    /// Group body lines into typed blocks.
    fn blocks(lines: &[&str]) -> Vec<Block> {
        let mut blocks = Vec::new();
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];
            let t = line.trim_start();
            if line.trim().is_empty() {
                i += 1;
            } else if t.starts_with("```") || t.starts_with("~~~") {
                let fence = &t[..3];
                let mut j = i + 1;
                while j < lines.len() && !lines[j].trim_start().starts_with(fence) {
                    j += 1;
                }
                let end = j.min(lines.len() - 1); // unterminated fence runs to EOF
                blocks.push(Block {
                    kind: ChunkKind::Code,
                    start: i,
                    end,
                });
                i = end + 1;
            } else if t.starts_with('#') && t.trim_start_matches('#').starts_with(' ') {
                blocks.push(Block {
                    kind: ChunkKind::Heading,
                    start: i,
                    end: i,
                });
                i += 1;
            } else {
                let quote = t.starts_with('>');
                let mut j = i;
                while j + 1 < lines.len() {
                    let next = lines[j + 1];
                    let nt = next.trim_start();
                    let stop = next.trim().is_empty()
                        || nt.starts_with("```")
                        || nt.starts_with("~~~")
                        || (nt.starts_with('#') && nt.trim_start_matches('#').starts_with(' '))
                        || nt.starts_with('>') != quote;
                    if stop {
                        break;
                    }
                    j += 1;
                }
                blocks.push(Block {
                    kind: if quote {
                        ChunkKind::Quote
                    } else {
                        ChunkKind::Detail
                    },
                    start: i,
                    end: j,
                });
                i = j + 1;
            }
        }
        blocks
    }

    /// Split an oversized block into `(start_line, end_line, text)` pieces.
    fn split_block(&self, lines: &[&str], b: &Block, base: usize) -> Vec<(usize, usize, String)> {
        let mut pieces = Vec::new();
        let mut cur: Option<(usize, usize, String)> = None;
        for (idx, &line) in lines.iter().enumerate().take(b.end + 1).skip(b.start) {
            let line_no = base + idx;
            if line.chars().count() > self.chunk_size {
                if let Some(p) = cur.take() {
                    pieces.push(p);
                }
                let chars: Vec<char> = line.chars().collect();
                let step = self.chunk_size - self.overlap;
                let mut s = 0;
                loop {
                    let e = (s + self.chunk_size).min(chars.len());
                    pieces.push((line_no, line_no, chars[s..e].iter().collect()));
                    if e == chars.len() {
                        break;
                    }
                    s += step;
                }
                continue;
            }
            match cur.as_mut() {
                Some((_, e, t))
                    if t.chars().count() + 1 + line.chars().count() <= self.chunk_size =>
                {
                    t.push('\n');
                    t.push_str(line);
                    *e = line_no;
                }
                _ => {
                    if let Some(p) = cur.take() {
                        pieces.push(p);
                    }
                    cur = Some((line_no, line_no, line.to_string()));
                }
            }
        }
        if let Some(p) = cur.take() {
            pieces.push(p);
        }
        pieces
    }
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new()
    }
}

/// Debounce window for coalescing bursts of file events.
const DEBOUNCE: Duration = Duration::from_millis(300);

/// File system watcher for `.plaesy/` directory.
pub struct Watcher {
    // Kept alive for the lifetime of the Watcher; dropping it stops watching.
    _watcher: notify::RecommendedWatcher,
    rx: UnboundedReceiver<notify::Result<notify::Event>>,
    debounce: Duration,
}

impl Watcher {
    /// Create a new watcher for the given directory (recursive).
    ///
    /// # Errors
    ///
    /// - `WatcherError` if cannot access directory or initialize file watcher
    pub async fn new(path: &str) -> Result<Self> {
        use notify::Watcher as _;
        let root = Path::new(path);
        if !root.is_dir() {
            return Err(MemoryError::WatcherError(format!(
                "{path} is not an accessible directory"
            )));
        }
        let (tx, rx) = unbounded_channel();
        let mut watcher = notify::recommended_watcher(move |res| {
            // Receiver dropped => watcher is being torn down; ignore.
            let _ = tx.send(res);
        })
        .map_err(|e| MemoryError::WatcherError(e.to_string()))?;
        watcher
            .watch(root, notify::RecursiveMode::Recursive)
            .map_err(|e| MemoryError::WatcherError(e.to_string()))?;
        Ok(Watcher {
            _watcher: watcher,
            rx,
            debounce: DEBOUNCE,
        })
    }

    /// Override the debounce window.
    pub fn with_debounce(mut self, debounce: Duration) -> Self {
        self.debounce = debounce;
        self
    }

    /// Get the next batch of file changes (debounced and coalesced per path).
    ///
    /// Waits for the first relevant event, then keeps collecting until no
    /// event arrives for the debounce window. Returns an empty vec only if
    /// the underlying watcher shut down.
    pub async fn changes(&mut self) -> Result<Vec<FileEvent>> {
        let mut batch: Vec<FileEvent> = Vec::new();
        loop {
            let next = if batch.is_empty() {
                self.rx.recv().await
            } else {
                match tokio::time::timeout(self.debounce, self.rx.recv()).await {
                    Ok(v) => v,
                    Err(_) => break, // quiet period elapsed
                }
            };
            match next {
                None => break,
                Some(Err(e)) => return Err(MemoryError::WatcherError(e.to_string())),
                Some(Ok(ev)) => batch.extend(Self::translate(&ev)),
            }
        }
        Ok(Self::coalesce(batch))
    }

    fn translate(ev: &notify::Event) -> Vec<FileEvent> {
        use notify::event::{ModifyKind, RenameMode};
        use notify::EventKind as NK;
        let mk = |p: &PathBuf, kind| {
            is_indexable(p).then(|| FileEvent {
                path: p.clone(),
                kind,
            })
        };
        match &ev.kind {
            NK::Create(_) => ev
                .paths
                .iter()
                .filter_map(|p| mk(p, EventKind::Added))
                .collect(),
            NK::Remove(_) => ev
                .paths
                .iter()
                .filter_map(|p| mk(p, EventKind::Deleted))
                .collect(),
            NK::Modify(ModifyKind::Name(RenameMode::From)) => ev
                .paths
                .iter()
                .filter_map(|p| mk(p, EventKind::Deleted))
                .collect(),
            NK::Modify(ModifyKind::Name(RenameMode::To)) => ev
                .paths
                .iter()
                .filter_map(|p| mk(p, EventKind::Added))
                .collect(),
            NK::Modify(ModifyKind::Name(RenameMode::Both)) if ev.paths.len() == 2 => {
                let mut v = Vec::new();
                v.extend(mk(&ev.paths[0], EventKind::Deleted));
                v.extend(mk(&ev.paths[1], EventKind::Added));
                v
            }
            NK::Modify(_) => ev
                .paths
                .iter()
                .filter_map(|p| mk(p, EventKind::Modified))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Collapse a burst of events to one event per path (first-seen order).
    fn coalesce(events: Vec<FileEvent>) -> Vec<FileEvent> {
        let mut order: Vec<PathBuf> = Vec::new();
        let mut state: HashMap<PathBuf, EventKind> = HashMap::new();
        for ev in events {
            let merged = match (state.get(&ev.path), ev.kind) {
                (None, k) => Some(k),
                (Some(EventKind::Added), EventKind::Modified) => Some(EventKind::Added),
                (Some(EventKind::Deleted), EventKind::Added) => Some(EventKind::Modified),
                (Some(_), k) => Some(k),
            };
            if !state.contains_key(&ev.path) {
                order.push(ev.path.clone());
            }
            if let Some(k) = merged {
                state.insert(ev.path, k);
            }
        }
        order
            .into_iter()
            .filter_map(|p| state.remove(&p).map(|kind| FileEvent { path: p, kind }))
            .collect()
    }
}

/// A file system event (added, modified, deleted).
#[derive(Clone, Debug)]
pub struct FileEvent {
    /// Path to the file.
    pub path: PathBuf,

    /// Type of change.
    pub kind: EventKind,
}

/// Type of file system change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    /// File was created or first noticed.
    Added,

    /// File content changed.
    Modified,

    /// File was deleted.
    Deleted,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(content: &str) -> Document {
        Extractor::new()
            .extract(PathBuf::from("t.md"), content)
            .expect("extract")
    }

    #[test]
    fn test_chunk_kind_serialize() {
        let kind = ChunkKind::Summary;
        let json = serde_json::to_string(&kind).expect("serialize");
        assert_eq!(json, "\"summary\"");
    }

    #[test]
    fn test_chunker_default() {
        let chunker = Chunker::default();
        assert_eq!(chunker.chunk_size, 512);
        assert_eq!(chunker.overlap, 50);
    }

    #[test]
    fn test_extract_frontmatter_and_body_offset() {
        let d = extract("---\nid: EV-01\ntags: [a, b]\n---\n\n# Title\n\nbody text\n");
        assert_eq!(d.metadata["id"], serde_json::json!("EV-01"));
        assert_eq!(d.metadata["tags"], serde_json::json!(["a", "b"]));
        assert_eq!(d.body_start_line, 5);
        assert!(d.body.starts_with("\n# Title"));
        assert_eq!(d.hash.as_deref().map(str::len), Some(64));
    }

    #[test]
    fn test_extract_no_frontmatter_and_crlf() {
        let d = extract("# Hi\r\n\r\ntext\r\n");
        assert!(d.metadata.is_empty());
        assert_eq!(d.body_start_line, 1);
        assert!(!d.body.contains('\r'));
    }

    #[test]
    fn test_extract_errors() {
        let ex = Extractor::new();
        assert!(matches!(
            ex.extract("a.md".into(), "---\nid: x\nno close"),
            Err(MemoryError::YamlError { .. })
        ));
        assert!(matches!(
            ex.extract("a.md".into(), "---\n: : :\n  - [\n---\nbody"),
            Err(MemoryError::YamlError { .. })
        ));
        assert!(matches!(
            ex.extract("a.md".into(), "---\n- a\n- b\n---\nbody"),
            Err(MemoryError::YamlError { .. })
        ));
        assert!(matches!(
            ex.extract("a.md".into(), "  \n"),
            Err(MemoryError::ExtractionError { .. })
        ));
    }

    #[test]
    fn test_hash_changes_with_content() {
        assert_ne!(extract("a").hash, extract("b").hash);
        assert_eq!(extract("a").hash, extract("a").hash);
    }

    #[test]
    fn test_nested_headings_do_not_leave_a_bare_title_chunk() {
        // "## Overview" has no body of its own; its body sits under the "###" sub-heading.
        let src = "## Overview\n\n### Phases\n- one\n- two\n";
        let d = extract(src);
        let chunks = Chunker::new().chunk(&d).expect("chunk");
        assert!(
            chunks
                .iter()
                .all(|c| c.kind != ChunkKind::Heading || c.text.lines().count() > 1),
            "no heading-only chunk: {:?}",
            chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>()
        );
        let joined: String = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("- two"), "body kept: {joined}");
    }

    #[test]
    fn test_chunk_line_numbers_are_exact() {
        let src = "---\nid: 1\n---\n# Intro\n\nfirst para\n\n```rust\nfn x() {}\n```\n\n> quoted\n";
        let d = extract(src);
        let chunks = Chunker::new().chunk(&d).expect("chunk");
        let src_lines: Vec<&str> = src.lines().collect();
        // Every non-summary chunk text must equal the source lines it claims.
        for c in &chunks {
            let slice = src_lines[c.start_line - 1..c.end_line].join("\n");
            assert_eq!(c.text, slice, "chunk {:?}", c.kind);
        }
        let kinds: Vec<_> = chunks.iter().map(|c| c.kind).collect();
        assert_eq!(
            kinds,
            vec![
                ChunkKind::Summary,
                ChunkKind::Detail,
                ChunkKind::Code,
                ChunkKind::Quote
            ]
        );
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 3));
        assert_eq!(chunks[1].start_line, 4); // heading + paragraph share one chunk
        assert_eq!(chunks[1].end_line, 6);
        assert!(chunks[1].text.starts_with("# Intro\n\nfirst para"));
        assert_eq!(chunks[2].start_line, 8);
        assert_eq!(chunks[2].end_line, 10);
    }

    #[test]
    fn test_chunk_section_metadata() {
        let d = extract("# Alpha\n\ntext one\n\n## Beta\n\ntext two\n");
        let chunks = Chunker::new().chunk(&d).expect("chunk");
        let detail: Vec<_> = chunks
            .iter()
            .filter(|c| c.kind == ChunkKind::Detail)
            .collect();
        assert_eq!(detail.len(), 2);
        assert_eq!(detail[0].metadata.as_ref().expect("m")["section"], "Alpha");
        assert_eq!(detail[1].metadata.as_ref().expect("m")["section"], "Beta");
    }

    #[test]
    fn test_chunk_merges_small_paragraphs_and_splits_large() {
        let d = extract("one\n\ntwo\n\nthree");
        let merged = Chunker::new().chunk(&d).expect("chunk");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "one\n\ntwo\n\nthree");
        assert_eq!((merged[0].start_line, merged[0].end_line), (1, 5));

        let big = (0..40)
            .map(|i| format!("line number {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let d = extract(&big);
        let c = Chunker {
            chunk_size: 100,
            overlap: 10,
        };
        let chunks = c.chunk(&d).expect("chunk");
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.text.chars().count() <= 100));
        // Contiguous, non-overlapping line coverage for line-based splits.
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks.last().expect("last").end_line, 40);
        for w in chunks.windows(2) {
            assert_eq!(w[0].end_line + 1, w[1].start_line);
        }
    }

    #[test]
    fn test_chunk_long_single_line_uses_overlap() {
        let d = extract(&"x".repeat(250));
        let c = Chunker {
            chunk_size: 100,
            overlap: 20,
        };
        let chunks = c.chunk(&d).expect("chunk");
        assert_eq!(chunks.len(), 3); // 0-100, 80-180, 160-250
        assert!(chunks.iter().all(|c| c.start_line == 1 && c.end_line == 1));
    }

    #[test]
    fn test_chunk_unicode_safe() {
        let d = extract(&"สวัสดี".repeat(60));
        let c = Chunker {
            chunk_size: 50,
            overlap: 5,
        };
        assert!(c.chunk(&d).is_ok());
    }

    #[test]
    fn test_chunker_rejects_bad_settings() {
        let d = extract("x");
        assert!(Chunker {
            chunk_size: 0,
            overlap: 0
        }
        .chunk(&d)
        .is_err());
        assert!(Chunker {
            chunk_size: 10,
            overlap: 10
        }
        .chunk(&d)
        .is_err());
    }

    #[test]
    fn test_chunk_unterminated_fence_runs_to_eof() {
        let d = extract("```\ncode\nmore");
        let chunks = Chunker::new().chunk(&d).expect("chunk");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Code);
        assert_eq!(chunks[0].end_line, 3);
    }

    #[test]
    fn test_coalesce_events() {
        let p = |s: &str| PathBuf::from(s);
        let ev = |s: &str, k| FileEvent {
            path: p(s),
            kind: k,
        };
        let out = Watcher::coalesce(vec![
            ev("a.md", EventKind::Added),
            ev("a.md", EventKind::Modified),
            ev("b.md", EventKind::Modified),
            ev("b.md", EventKind::Deleted),
            ev("c.md", EventKind::Deleted),
            ev("c.md", EventKind::Added),
        ]);
        let got: Vec<_> = out
            .iter()
            .map(|e| (e.path.to_str().expect("p"), e.kind))
            .collect();
        assert_eq!(
            got,
            vec![
                ("a.md", EventKind::Added),
                ("b.md", EventKind::Deleted),
                ("c.md", EventKind::Modified)
            ]
        );
    }

    #[test]
    fn test_is_indexable_and_scan_dir() {
        assert!(is_indexable(Path::new("a.MD")));
        assert!(!is_indexable(Path::new("a.json")));
        let dir = tempfile::tempdir().expect("tmp");
        std::fs::create_dir(dir.path().join("sub")).expect("mkdir");
        std::fs::write(dir.path().join("b.md"), "x").expect("w");
        std::fs::write(dir.path().join("sub").join("a.md"), "x").expect("w");
        std::fs::write(dir.path().join("c.json"), "x").expect("w");
        let files = scan_dir(dir.path());
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_is_excluded_matches_folder_and_children_only() {
        let root = Path::new("/vault");
        let ex = vec![".plaesy/templates".to_string()];
        assert!(is_excluded(
            root,
            Path::new("/vault/.plaesy/templates/a.md"),
            &ex
        ));
        assert!(is_excluded(
            root,
            Path::new("/vault/.plaesy/templates"),
            &ex
        ));
        assert!(is_excluded(
            root,
            Path::new("/vault/01 Projects/X/.plaesy/templates/a.md"),
            &ex
        ));
        assert!(!is_excluded(
            root,
            Path::new("/vault/.plaesy/memory/a.md"),
            &ex
        ));
        // A prefix of a folder name is not the folder itself.
        assert!(!is_excluded(
            root,
            Path::new("/vault/.plaesy/templates-old/a.md"),
            &ex
        ));
        assert!(!is_excluded(
            root,
            Path::new("/vault/.plaesy/templates/a.md"),
            &[]
        ));
    }

    #[tokio::test]
    async fn test_watcher_rejects_missing_dir() {
        assert!(matches!(
            Watcher::new("definitely/not/a/dir").await,
            Err(MemoryError::WatcherError(_))
        ));
    }

    #[tokio::test]
    async fn test_watcher_reports_new_file() {
        let dir = tempfile::tempdir().expect("tmp");
        let mut w = Watcher::new(dir.path().to_str().expect("utf8"))
            .await
            .expect("watcher")
            .with_debounce(Duration::from_millis(100));
        let file = dir.path().join("new.md");
        let f2 = file.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            std::fs::write(f2, "hello").expect("write");
        });
        let events = tokio::time::timeout(Duration::from_secs(15), w.changes())
            .await
            .expect("timed out waiting for event")
            .expect("changes");
        assert!(
            events
                .iter()
                .any(|e| e.path.file_name() == file.file_name()),
            "events: {events:?}"
        );
    }
}
