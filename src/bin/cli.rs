//! Command-line interface for plaesy-memory.
//!
//! ```text
//! memory index [DIR]              sync DIR (default: configured watch dir) into the index
//! memory search "query" [-n 10]   hybrid/semantic/keyword search with citations
//! memory links NOTE [--outgoing]  notes linking to NOTE (or linked from it)
//! memory hygiene [DIR] [--json]   stale notes, missing metadata, broken links
//! memory missing [--resolve ID]   unanswered queries and broken or stale notes
//! memory stats                    index statistics
//! memory watch [DIR]              keep the index current until Ctrl-C
//! memory config                   print the effective configuration
//! memory config mcp [--app X]     integration snippets for MCP clients (Claude Code, n8n, Dify, Kilo Code, …)
//! memory serve [--mcp]            JSON service on stdin/stdout, or MCP server with --mcp
//! memory serve --mcp --transport http --port 8080   MCP over Streamable HTTP
//! ```

use clap::{Parser, Subcommand, ValueEnum};
use plaesy::memory::config::MemoryConfig;
use plaesy::memory::hygiene::HygieneOptions;
use plaesy::memory::indexer::Watcher;
use plaesy::memory::search::{
    normalize_tier, CitationBuilder, SearchEngine, SearchFilter, SearchMode, Trust, TIERS,
};
use plaesy::memory::serve::Transport;
use plaesy::memory::service::SearchHit;
use plaesy::memory::{
    self, init_logging_with, onboard, serve, settings, MemoryError, Result,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "memory", version, about = "Offline-first memory index")]
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

/// Output format for search results.
#[derive(Clone, Copy, Default, ValueEnum)]
enum OutputFormat {
    /// Plain text listing.
    #[default]
    Text,
    /// JSON array of hits.
    Json,
    /// Markdown list of hits.
    Md,
}

/// Structural gaps shown per kind by `memory missing` unless `--all` is given.
const SAMPLE_PER_KIND: usize = 5;

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
        /// Output format (text, json, md).
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
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
        /// Also print per-leg score breakdown (fts, vector, fused, staleness, rerank).
        #[arg(long)]
        explain: bool,
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
    /// What the memory is missing: queries it could not answer, and broken or stale notes.
    Missing {
        /// Close the query gap with this id (it reopens if the query misses again).
        #[arg(long)]
        resolve: Option<i64>,
        /// Maximum number of query gaps to list.
        #[arg(short = 'n', long, default_value_t = 20)]
        limit: usize,
        /// Emit JSON.
        #[arg(long)]
        json: bool,
        /// List every structural gap, not a sample per kind.
        #[arg(long)]
        all: bool,
        /// Directory the notes live in (default: configured watch dir).
        dir: Option<PathBuf>,
    },
    /// Show index statistics.
    Stats,
    /// Watch a directory and keep the index current.
    Watch { dir: Option<PathBuf> },
    /// Print the effective configuration.
    Config {
        /// Subcommand: `mcp` prints integration snippets for MCP clients.
        #[command(subcommand)]
        sub: Option<ConfigSubcommand>,
    },
    /// Run the long-running service: JSON on stdin/stdout (Hermes), or MCP with --mcp.
    Serve {
        /// Speak the Model Context Protocol instead of the line-delimited JSON service.
        #[arg(long)]
        mcp: bool,
        /// Transport: `stdio` (default) or `http` (Streamable HTTP for n8n, Dify, Kilo Code, …).
        #[arg(long, value_enum, default_value_t = Transport::Stdio)]
        transport: Transport,
        /// TCP port to listen on when `--transport http` is set (default: 8080).
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// First-run setup: vault folder, settings.json, model download, first index.
    Onboard {
        /// Accept every default (needs --vault unless settings already exist).
        #[arg(short = 'y', long)]
        yes: bool,
        /// Vault: the folder with the notes to index.
        #[arg(long)]
        vault: Option<PathBuf>,
        /// Folder that holds Plaesy/settings.json (default: your local app data folder).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Do not download the embedding model.
        #[arg(long)]
        no_model: bool,
        /// Enable reranking and download the reranker model.
        #[arg(long)]
        rerank: bool,
        /// Do not index the notes at the end.
        #[arg(long)]
        no_index: bool,
    },
    /// Run scheduled jobs from the config: `once` runs what is due and exits (for Task Scheduler);
    /// `run` keeps going until Ctrl-C.
    Schedule {
        /// `once` or `run`.
        #[arg(value_enum)]
        mode: ScheduleMode,
    },
}

/// Subcommands for `memory config`.
#[derive(Subcommand, Debug, Clone)]
enum ConfigSubcommand {
    /// Print integration snippets for MCP clients (Claude Code, CrewAI, n8n, Dify, Kilo Code, Paperclip).
    ///
    /// Use `--app` to print only one client's snippet, or `--list` to print every app's snippet.
    Mcp {
        /// Print snippets for only this app (e.g. `claude-code`, `crewai`, `n8n`, `dify`, `kilocode`, `paperclip`).
        #[arg(long)]
        app: Option<String>,
        /// Print every app's snippet.
        #[arg(long)]
        list: bool,
        /// Path to the `memory` executable to embed in the snippets.
        #[arg(long, default_value = "memory")]
        binary: String,
        /// Notes directory to embed in the snippets.
        #[arg(long, default_value = "./notes")]
        notes_dir: String,
        /// HTTP URL of the MCP server (when using `--transport http`).
        #[arg(long)]
        http_url: Option<String>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ScheduleMode {
    Once,
    Run,
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

    if let Command::Onboard {
        yes,
        vault,
        root,
        no_model,
        rerank,
        no_index,
    } = cli.command
    {
        return onboard::run(onboard::OnboardOptions {
            yes,
            vault,
            root,
            no_model,
            rerank,
            no_index,
        })
        .await;
    }
    if !matches!(
        cli.command,
        Command::Config { .. } | Command::Serve { .. } | Command::Schedule { .. }
    ) && settings::default_path().is_some_and(|p| !p.is_file())
    {
        eprintln!(
            "Tip: run `memory onboard` once to choose your vault folder and download the model."
        );
    }

    if let Command::Config { sub } = cli.command {
        if let Some(ConfigSubcommand::Mcp {
            app,
            list,
            binary,
            notes_dir,
            http_url,
        }) = sub
        {
            return memory::config::mcp::run(app, list, binary, notes_dir, http_url).await;
        }
        println!("{}", config.to_diagnostic_string());
        return Ok(());
    }
    if let Command::Serve {
        mcp,
        transport,
        port,
    } = cli.command
    {
        return serve::run(config, mcp, transport, port).await;
    }

    let engine = SearchEngine::from_config(&config).await?;
    engine.set_watch_root(config.watch_dir.clone());
    match cli.command {
        Command::Config { sub: _ } | Command::Serve { .. } | Command::Onboard { .. } => {
            unreachable!("handled above")
        }
        Command::Schedule { mode } => {
            let jobs = plaesy::memory::schedule::jobs_from_config(&config.schedule)?;
            let root = config.watch_dir.clone();
            require_dir(&root)?;
            let log = plaesy::memory::schedule::log_path();
            match mode {
                ScheduleMode::Once => {
                    let now = chrono::Utc::now();
                    for o in plaesy::memory::schedule::run_due(
                        &engine,
                        &root,
                        &jobs,
                        now,
                        log.as_deref(),
                    )
                    .await?
                    {
                        println!("{}", o.line(now));
                    }
                }
                ScheduleMode::Run => {
                    plaesy::memory::schedule::run_loop(&engine, &root, &jobs, log.as_deref())
                        .await?;
                }
            }
        }
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
            format,
            tiers,
            path,
            min_trust,
            include_inactive,
            as_of,
            expand_links,
            explain,
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
                engine
                    .search_with_explain(&query, limit, mode, &filter, explain)
                    .await?
            };
            engine.note_query(&query, &results)?;
            match format {
                OutputFormat::Json => {
                    let hits: Vec<SearchHit> = results
                        .iter()
                        .map(SearchHit::from_result)
                        .collect::<Result<_>>()?;
                    let out = serde_json::to_string_pretty(&hits)
                        .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
                    println!("{out}");
                }
                OutputFormat::Md => {
                    if results.is_empty() {
                        println!("no results");
                    } else {
                        for r in &results {
                            let c = CitationBuilder::new().build(r)?;
                            println!("- {c}");
                        }
                    }
                }
                OutputFormat::Text => {
                    if results.is_empty() {
                        println!("no results");
                    } else {
                        let b = CitationBuilder::new();
                        for (i, r) in results.iter().enumerate() {
                            println!("{}. score {:.3}  {}", i + 1, r.score, b.build(r)?);
                            if explain {
                                if let Some(ex) = &r.explain {
                                    print!("     explain: fused={:.4}", ex.fused.unwrap_or(0.0));
                                    if let Some(v) = ex.fts {
                                        print!(" fts={:.4}", v);
                                    }
                                    if let Some(v) = ex.vector {
                                        print!(" vector={:.4}", v);
                                    }
                                    if let Some(v) = ex.staleness_factor {
                                        print!(" staleness={:.1}", v);
                                    }
                                    println!();
                                }
                            }
                            for line in r.text.lines().take(4) {
                                println!("     {line}");
                            }
                        }
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
        Command::Missing {
            resolve,
            limit,
            json,
            all,
            dir,
        } => {
            if let Some(id) = resolve {
                if engine.resolve_gap(id)? {
                    println!("resolved gap {id}");
                } else {
                    println!("no open gap with id {id}");
                }
                return Ok(());
            }
            if let Some(d) = &dir {
                if !d.is_dir() {
                    return Err(MemoryError::ConfigError(format!(
                        "not a directory: {}",
                        d.display()
                    )));
                }
            }
            let root = dir.unwrap_or_else(|| config.watch_dir.clone());
            if root.is_dir() {
                engine.index_directory(&root).await?;
            }
            let report = engine.missing(root.is_dir().then_some(root), limit).await?;
            if json {
                let out = serde_json::to_string_pretty(&report)
                    .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
                println!("{out}");
            } else if report.query_gaps.is_empty() && report.structural.is_empty() {
                println!("no gaps");
            }
            if !json {
                for g in &report.query_gaps {
                    let score = g
                        .last_top_score
                        .map_or("no hit".to_string(), |s| format!("top {s:.2}"));
                    // One line per gap, whatever the query held.
                    let flat = g.query.split_whitespace().collect::<Vec<_>>().join(" ");
                    let shown: String = flat.chars().take(80).collect();
                    let ellipsis = if flat.chars().count() > 80 { "…" } else { "" };
                    println!(
                        "  #{:<4} pri {:>5.2}  x{:<3} {score:<10} {shown}{ellipsis}",
                        g.id, g.priority, g.misses
                    );
                }
                // Structural gaps can run to hundreds of lines: show a count per kind and a
                // sample of each, unless --all asks for everything.
                let mut shown: std::collections::BTreeMap<&str, usize> = Default::default();
                for f in &report.structural {
                    let n = shown.entry(f.kind.as_str()).or_insert(0);
                    *n += 1;
                    if !all && *n > SAMPLE_PER_KIND {
                        continue;
                    }
                    let at = f.line.map(|l| format!(":{l}")).unwrap_or_default();
                    println!(
                        "  [{}] {:<14} {}{at}  {}",
                        f.label, f.kind, f.path, f.detail
                    );
                }
                let total: usize = report.structural.len();
                if !report.structural.is_empty() {
                    println!("structural gaps: {total}");
                    for (kind, n) in &shown {
                        if !all && *n > SAMPLE_PER_KIND {
                            println!("  {kind}: {n} (showing {SAMPLE_PER_KIND}, use --all)");
                        } else {
                            println!("  {kind}: {n}");
                        }
                    }
                }
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
