//! Command-line interface for plaesy-memory.
//!
//! ```text
//! plaesy-memory-cli index [DIR]              sync DIR (default: configured watch dir) into the index
//! plaesy-memory-cli search "query" [-n 10]   hybrid/semantic/keyword search with citations
//! plaesy-memory-cli links NOTE [--outgoing]  notes linking to NOTE (or linked from it)
//! plaesy-memory-cli hygiene [DIR] [--json]   stale notes, missing metadata, broken links
//! plaesy-memory-cli stats                    index statistics
//! plaesy-memory-cli watch [DIR]              keep the index current until Ctrl-C
//! plaesy-memory-cli config                   print the effective configuration
//! ```

use clap::{Parser, Subcommand, ValueEnum};
use plaesy::memory::config::MemoryConfig;
use plaesy::memory::hygiene::HygieneOptions;
use plaesy::memory::indexer::Watcher;
use plaesy::memory::search::{
    normalize_tier, CitationBuilder, SearchEngine, SearchFilter, SearchMode, Trust, TIERS,
};
use plaesy::memory::service::SearchHit;
use plaesy::memory::{init_logging_with, MemoryError, Result};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "plaesy-memory-cli",
    version,
    about = "Offline-first memory index for Hermes Agent"
)]
struct Cli {
    /// Config file (TOML) layered over defaults instead of the standard lookup.
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// SQLite database path (overrides config).
    #[arg(long, global = true)]
    db: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    Hybrid,
    Semantic,
    Keyword,
}

#[derive(Subcommand)]
enum Command {
    /// Sync a directory into the index (new/changed files in, deleted files out).
    Index { dir: Option<PathBuf> },
    /// Search the index.
    Search {
        query: String,
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
        #[arg(long, value_enum, default_value_t = Mode::Hybrid)]
        mode: Mode,
        /// Emit JSON (same shape as the plugin protocol's hits).
        #[arg(long)]
        json: bool,
        /// Only these tiers (repeatable): core, episodic, semantic, procedural, archive.
        #[arg(long = "tier")]
        tiers: Vec<String>,
        /// Only notes whose path starts with this prefix.
        #[arg(long)]
        path: Option<String>,
        /// Only notes declaring at least this trust (high, medium, low).
        #[arg(long)]
        min_trust: Option<String>,
        /// Also return superseded / deprecated notes.
        #[arg(long)]
        include_inactive: bool,
        /// Only notes whose declared valid_from..valid_to period contains this date (YYYY-MM-DD).
        #[arg(long)]
        as_of: Option<String>,
        /// Also add the best passage of notes linked to or from the top hits (one hop).
        #[arg(long)]
        expand_links: bool,
    },
    /// Wikilinks: notes linking to NOTE (default) or linked from it.
    Links {
        note: String,
        /// List the links going out of NOTE instead.
        #[arg(long)]
        outgoing: bool,
    },
    /// Audit the memory: stale notes, missing verification metadata, broken links, dead paths.
    Hygiene {
        /// Directory the notes live in (enables dead-path checks; default: configured watch dir).
        dir: Option<PathBuf>,
        /// Emit JSON.
        #[arg(long)]
        json: bool,
        /// Exit with status 1 when any finding is reported (for CI).
        #[arg(long)]
        strict: bool,
    },
    /// Show index statistics.
    Stats,
    /// Watch a directory and keep the index current.
    Watch { dir: Option<PathBuf> },
    /// Print the effective configuration.
    Config,
}

#[tokio::main]
async fn main() {
    if let Err(e) = run(Cli::parse()).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<()> {
    let mut config = match &cli.config {
        Some(p) => MemoryConfig::load_from_file(p).await?,
        None => MemoryConfig::load().await?,
    };
    if let Some(db) = cli.db {
        config.database.db_path = db;
    }
    init_logging_with(&config.logging)?;

    if let Command::Config = cli.command {
        println!("{}", config.to_diagnostic_string());
        return Ok(());
    }

    let engine = SearchEngine::from_config(&config).await?;
    engine.set_watch_root(config.watch_dir.clone());
    match cli.command {
        Command::Config => unreachable!("handled above"),
        Command::Index { dir } => {
            let root = dir.unwrap_or_else(|| config.watch_dir.clone());
            require_dir(&root)?;
            let s = engine.index_directory(&root).await?;
            println!(
                "indexed {} | unchanged {} | removed {} | chunks written {} | failed {}",
                s.indexed,
                s.unchanged,
                s.removed,
                s.chunks,
                s.failed.len()
            );
            for (path, error) in &s.failed {
                eprintln!("  failed: {path}: {error}");
            }
            if !s.failed.is_empty() {
                std::process::exit(2);
            }
        }
        Command::Search {
            query,
            limit,
            mode,
            json,
            tiers,
            path,
            min_trust,
            include_inactive,
            as_of,
            expand_links,
        } => {
            for t in &tiers {
                if normalize_tier(t).is_none() {
                    return Err(MemoryError::ConfigError(format!(
                        "unknown tier '{t}' (expected one of: {})",
                        TIERS.join(", ")
                    )));
                }
            }
            let min_trust = match min_trust {
                None => None,
                Some(t) => Some(
                    Trust::parse(&serde_json::Value::String(t.clone())).ok_or_else(|| {
                        MemoryError::ConfigError(format!(
                            "unknown trust '{t}' (expected high, medium or low)"
                        ))
                    })?,
                ),
            };
            let as_of = as_of
                .map(|s| {
                    plaesy::memory::search::parse_date(&s).ok_or_else(|| {
                        MemoryError::ConfigError(format!(
                            "--as-of '{s}' is not a date (expected YYYY-MM-DD)"
                        ))
                    })
                })
                .transpose()?;
            let filter = SearchFilter {
                tiers,
                path_prefix: path,
                min_trust,
                include_inactive,
                as_of,
            };
            let mode = match mode {
                Mode::Hybrid => SearchMode::Hybrid,
                Mode::Semantic => SearchMode::Semantic,
                Mode::Keyword => SearchMode::Keyword,
            };
            let results = if expand_links {
                engine.search_expanded(&query, limit, mode, &filter).await?
            } else {
                engine.search_with(&query, limit, mode, &filter).await?
            };
            if json {
                let hits: Vec<SearchHit> = results
                    .iter()
                    .map(SearchHit::from_result)
                    .collect::<Result<_>>()?;
                let out = serde_json::to_string_pretty(&hits)
                    .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
                println!("{out}");
            } else if results.is_empty() {
                println!("no results");
            } else {
                let b = CitationBuilder::new();
                for (i, r) in results.iter().enumerate() {
                    println!("{}. score {:.3}  {}", i + 1, r.score, b.build(r)?);
                    for line in r.text.lines().take(4) {
                        println!("     {line}");
                    }
                }
            }
        }
        Command::Links { note, outgoing } => {
            let links = engine.links(&note, !outgoing).await?;
            if links.is_empty() {
                println!("no links");
            }
            for l in links {
                println!(
                    "{}:{}  [[{}]] -> {}",
                    l.source_path,
                    l.line,
                    l.target,
                    l.resolved_path.as_deref().unwrap_or("(unresolved)")
                );
            }
        }
        Command::Hygiene { dir, json, strict } => {
            let root = dir.unwrap_or_else(|| config.watch_dir.clone());
            if root.is_dir() {
                engine.index_directory(&root).await?; // audit what is on disk now
            }
            let report = engine
                .hygiene(&HygieneOptions {
                    root: root.is_dir().then_some(root),
                    ..HygieneOptions::default()
                })
                .await?;
            if json {
                let out = serde_json::to_string_pretty(&report)
                    .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
                println!("{out}");
            } else {
                println!(
                    "{} notes checked, {} findings",
                    report.checked,
                    report.findings.len()
                );
                for f in &report.findings {
                    let at = f.line.map(|l| format!(":{l}")).unwrap_or_default();
                    println!(
                        "  [{}] {:<30} {}{at}  {}",
                        f.label, f.kind, f.path, f.detail
                    );
                }
                for (kind, n) in &report.summary {
                    println!("  {kind}: {n}");
                }
            }
            if strict && !report.findings.is_empty() {
                std::process::exit(1);
            }
        }
        Command::Stats => {
            let s = engine.statistics().await?;
            println!("documents:  {}", s.documents_total);
            println!("chunks:     {}", s.chunks_total);
            println!("db size:    {} bytes", s.db_size_bytes);
            println!("fts size:   {} bytes", s.fts_index_size_bytes);
            println!("hnsw size:  {} bytes", s.hnsw_size_bytes);
            println!(
                "vector search: {}",
                if engine.has_vector_search() {
                    "on"
                } else {
                    "off (keyword-only)"
                }
            );
        }
        Command::Watch { dir } => {
            let root = dir.unwrap_or_else(|| config.watch_dir.clone());
            require_dir(&root)?;
            let s = engine.index_directory(&root).await?;
            println!(
                "initial sync: indexed {} unchanged {} removed {}",
                s.indexed, s.unchanged, s.removed
            );
            let mut watcher = Watcher::new(&root.to_string_lossy()).await?;
            println!("watching {} (Ctrl-C to stop)", root.display());
            loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => break,
                    changes = watcher.changes() => {
                        let events = changes?;
                        if events.is_empty() { break; }
                        let n = engine.apply_events(&root, &events).await;
                        println!("applied {n}/{} file events", events.len());
                    }
                }
            }
        }
    }
    Ok(())
}

fn require_dir(path: &std::path::Path) -> Result<()> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(MemoryError::ConfigError(format!(
            "{} is not a directory",
            path.display()
        )))
    }
}
