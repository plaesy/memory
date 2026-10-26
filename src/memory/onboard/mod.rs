//! `memory onboard`: first-run setup.
//!
//! Asks where the notes are, writes `<root>/Plaesy/settings.json`, downloads the models that are
//! missing and indexes the notes. Every question has a default, and `--yes` accepts all defaults so
//! the same steps run unattended (`--vault <DIR>` is then required).

use crate::memory::config::MemoryConfig;
use crate::memory::error::MemoryError;
use crate::memory::paths;
use crate::memory::search::SearchEngine;
use crate::memory::settings::{self, Settings, SETTINGS_VERSION};
use crate::memory::Result;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

/// Default embedding model: `<models root>/all-MiniLM-L6-v2`.
const EMBEDDING_MODEL: &str = "all-MiniLM-L6-v2";
/// Files that make up the embedding model, with their download URLs.
const EMBEDDING_FILES: &[(&str, &str)] = &[
    (
        "model.onnx",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
    ),
    (
        "vocab.txt",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
    ),
];
/// Optional reranker: `<models root>/ms-marco-MiniLM-L-6-v2`.
const RERANK_MODEL: &str = "ms-marco-MiniLM-L-6-v2";
const RERANK_FILES: &[(&str, &str)] = &[
    (
        "model.onnx",
        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx",
    ),
    (
        "vocab.txt",
        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/vocab.txt",
    ),
];

/// Command-line options of `memory onboard`.
#[derive(Debug, Default, Clone)]
pub struct OnboardOptions {
    /// Accept every default without asking.
    pub yes: bool,
    /// Vault: the folder with the notes to index.
    pub vault: Option<PathBuf>,
    /// Install root that holds `Plaesy/settings.json` (default: the platform data folder).
    pub root: Option<PathBuf>,
    /// Skip every model download.
    pub no_model: bool,
    /// Enable reranking (also downloads the reranker model).
    pub rerank: bool,
    /// Do not index the notes at the end.
    pub no_index: bool,
}

/// Asks questions on the terminal. With `--yes` every question takes its default.
struct Prompt {
    yes: bool,
}

impl Prompt {
    fn line(&self, question: &str) -> Result<String> {
        print!("{question}: ");
        io::stdout().flush().map_err(io_err)?;
        let mut buf = String::new();
        io::stdin().lock().read_line(&mut buf).map_err(io_err)?;
        Ok(buf.trim().to_string())
    }

    fn confirm(&self, question: &str, default: bool) -> Result<bool> {
        if self.yes {
            return Ok(true);
        }
        let hint = if default { "[Y/n]" } else { "[y/N]" };
        loop {
            match self
                .line(&format!("{question} {hint}"))?
                .to_ascii_lowercase()
                .as_str()
            {
                "" => return Ok(default),
                "y" | "yes" => return Ok(true),
                "n" | "no" => return Ok(false),
                _ => println!("  please answer y or n"),
            }
        }
    }

    fn path(&self, question: &str, default: &Path) -> Result<PathBuf> {
        if self.yes {
            return Ok(default.to_path_buf());
        }
        let answer = self.line(&format!("{question} [{}]", default.display()))?;
        Ok(if answer.is_empty() {
            default.to_path_buf()
        } else {
            PathBuf::from(answer)
        })
    }

    /// A folder that must exist. Asks again on the terminal; errors with `--yes`.
    fn folder(&self, question: &str) -> Result<PathBuf> {
        if self.yes {
            return Err(MemoryError::ConfigError(
                "the notes folder is missing: pass --vault <DIR> with --yes".into(),
            ));
        }
        loop {
            let path = PathBuf::from(self.line(question)?.trim_matches('"'));
            if path.is_dir() {
                return Ok(path);
            }
            println!("  not a folder: {}", path.display());
        }
    }
}

fn io_err(e: io::Error) -> MemoryError {
    MemoryError::ConfigError(format!("terminal input/output failed: {e}"))
}

/// Run the setup. Returns once settings are written, models are present and (unless skipped) notes are indexed.
///
/// # Errors
///
/// - `ConfigError` when run without a terminal and without `--yes`, when `--yes` has no `--vault`,
///   or when settings cannot be written
/// - `DownloadError` when a model file cannot be downloaded
pub async fn run(opts: OnboardOptions) -> Result<()> {
    if !opts.yes && !io::stdin().is_terminal() {
        return Err(MemoryError::ConfigError(
            "memory onboard needs a terminal; run it in one, or pass --yes --vault <DIR>".into(),
        ));
    }
    let ask = Prompt { yes: opts.yes };
    let default_root = dirs::data_local_dir()
        .ok_or_else(|| MemoryError::ConfigError("no platform data folder; pass --root".into()))?;

    println!("Plaesy Memory setup");
    println!();

    // 1. Where settings live
    let root = match &opts.root {
        Some(r) => r.clone(),
        None => ask.path(
            "Install folder (settings and data go under <folder>\\Plaesy)",
            &default_root,
        )?,
    };
    let settings_path = settings::file_in_root(&root);

    // 2. Settings: keep the existing ones unless the user wants to change them
    let existing = settings::load(&settings_path)?;
    let reconfigure = match &existing {
        None => true,
        Some(_) => {
            opts.vault.is_some() || ask.confirm("Settings already exist. Change them?", false)?
        }
    };
    let current = if reconfigure {
        let previous = existing.clone().unwrap_or_default();
        let notes = match opts.vault.clone().or(previous.notes_dir.clone()) {
            Some(p) if p.is_dir() => p,
            Some(p) => {
                return Err(MemoryError::ConfigError(format!(
                    "notes folder not found: {}",
                    p.display()
                )))
            }
            None => ask.folder("Vault folder (the notes to index)")?,
        };
        let rerank = if opts.rerank {
            true
        } else if let Some(v) = previous.rerank {
            v
        } else {
            ask.confirm("Enable reranking? (needs an extra ~90 MB model)", false)?
        };
        let next = Settings {
            version: SETTINGS_VERSION,
            notes_dir: Some(notes),
            db_path: previous.db_path,
            model_name: Some(
                previous
                    .model_name
                    .unwrap_or_else(|| EMBEDDING_MODEL.to_string()),
            ),
            rerank: Some(rerank),
            exclude_dirs: previous.exclude_dirs,
        };
        settings::save(&settings_path, &next)?;
        println!("Saved {}", settings_path.display());
        next
    } else {
        let s = existing.unwrap_or_default();
        println!("Using existing settings {}", settings_path.display());
        s
    };

    // 3. Models
    let models = paths::models_root()
        .ok_or_else(|| MemoryError::ConfigError("no platform cache folder for models".into()))?;
    let model_name = current
        .model_name
        .clone()
        .unwrap_or_else(|| EMBEDDING_MODEL.to_string());
    let downloadable = model_name == EMBEDDING_MODEL;
    ensure_model(
        &ask,
        &models.join(&model_name),
        EMBEDDING_FILES,
        &model_name,
        "embedding model",
        opts.no_model,
        downloadable,
    )?;
    if current.rerank == Some(true) {
        ensure_model(
            &ask,
            &models.join(RERANK_MODEL),
            RERANK_FILES,
            RERANK_MODEL,
            "reranker",
            opts.no_model,
            true,
        )?;
    }

    // 4. First index
    let notes = current.notes_dir.clone();
    match (&notes, opts.no_index) {
        (_, true) => println!("Skipped indexing (--no-index)."),
        (None, _) => println!("No notes folder set: nothing to index."),
        (Some(dir), false) => {
            if ask.confirm(&format!("Index {} now?", dir.display()), true)? {
                index_now(&current, dir).await?;
            }
        }
    }

    // 5. Summary
    let exe = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "memory".into());
    println!();
    println!("Setup done.");
    println!("  settings: {}", settings_path.display());
    println!("  search:   memory search \"your question\"");
    println!("  MCP:      claude mcp add plaesy-memory -- \"{exe}\" serve --mcp");
    if root != default_root {
        println!(
            "  note:     set PLAESY_ROOT to {} so memory finds these settings",
            root.display()
        );
    }
    Ok(())
}

/// Download `files` into `dir` when any is missing. Only the default embedding model has known URLs,
/// so other model names are left alone.
#[allow(clippy::too_many_arguments)]
fn ensure_model(
    ask: &Prompt,
    dir: &Path,
    files: &[(&str, &str)],
    name: &str,
    label: &str,
    skip: bool,
    downloadable: bool,
) -> Result<()> {
    let present = |f: &str| dir.join(f).is_file();
    if files.iter().all(|(f, _)| present(f)) {
        println!("ok    {label} ({name}) is installed");
        return Ok(());
    }
    if skip {
        println!("skip  {label} ({name}): --no-model");
        return Ok(());
    }
    if !downloadable {
        println!(
            "skip  {label} ({name}): no download source known, install it into {}",
            dir.display()
        );
        return Ok(());
    }
    if !ask.confirm(&format!("Download the {label} (~90 MB, one time)?"), true)? {
        println!("skip  {label}: keyword search only until it is installed");
        return Ok(());
    }
    std::fs::create_dir_all(dir).map_err(io_err)?;
    for (file, url) in files {
        if present(file) {
            continue;
        }
        println!("get   {file}");
        download(url, &dir.join(file))?;
    }
    println!("ok    {label} installed in {}", dir.display());
    Ok(())
}

/// Download `url` to `dest` through a `.part` file, so an interrupted download is never mistaken for a model.
fn download(url: &str, dest: &Path) -> Result<()> {
    let fail = |e: &dyn std::fmt::Display| MemoryError::DownloadError(format!("{url}: {e}"));
    let mut response = ureq::get(url).call().map_err(|e| fail(&e))?;
    let part = dest.with_extension("part");
    let mut file = std::fs::File::create(&part).map_err(io_err)?;
    let mut reader = response.body_mut().as_reader();
    io::copy(&mut reader, &mut file).map_err(|e| fail(&e))?;
    drop(file);
    std::fs::rename(&part, dest).map_err(io_err)
}

/// Index `dir` with the settings just written, so this run does not depend on the environment.
async fn index_now(current: &Settings, dir: &Path) -> Result<()> {
    let mut config = MemoryConfig::load().await?;
    current.apply_to(&mut config);
    config.validate()?;
    let engine = SearchEngine::from_config(&config).await?;
    engine.set_watch_root(config.watch_dir.clone());
    let s = engine.index_directory(dir).await?;
    println!(
        "indexed {} | unchanged {} | removed {} | failed {}",
        s.indexed,
        s.unchanged,
        s.removed,
        s.failed.len()
    );
    for (path, error) in &s.failed {
        eprintln!("  failed: {path}: {error}");
    }
    Ok(())
}
