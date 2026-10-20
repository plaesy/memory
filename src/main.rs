//! Hermes plugin entry point.
//!
//! Speaks the line-delimited JSON protocol of [`plaesy::memory::service`] on
//! stdin/stdout (logs go to stderr). It answers requests immediately, while a
//! background task syncs the configured watch directory into the index and then
//! keeps it current with a file watcher. Configuration: see
//! [`plaesy::memory::config::MemoryConfig::load`].

use plaesy::memory::config::MemoryConfig;
use plaesy::memory::indexer::Watcher;
use plaesy::memory::search::SearchEngine;
use plaesy::memory::{init_logging_with, mcp, service};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("plaesy-memory-plugin: {e}");
        std::process::exit(1);
    }
}

async fn run() -> plaesy::memory::Result<()> {
    let mut mcp_mode = false;
    if let Some(arg) = std::env::args().nth(1) {
        match arg.as_str() {
            "--mcp" => mcp_mode = true,
            "--version" | "-V" => {
                println!("plaesy-memory-plugin {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--help" | "-h" => {
                println!(
                    "plaesy-memory-plugin {}\n\nLine-delimited JSON service on stdin/stdout \
                     (methods: ping, search, index, backlinks, hygiene, stats, metrics).\n\
                                          With --mcp: Model Context Protocol, read-only tools.\n\
                                          Configuration: MEMORY_* \
                     environment variables or .plaesy/memory.toml.",
                    env!("CARGO_PKG_VERSION")
                );
                return Ok(());
            }
            other => {
                return Err(plaesy::memory::MemoryError::ConfigError(format!(
                    "unknown argument '{other}' (try --help)"
                )))
            }
        }
    }
    let config = MemoryConfig::load().await?;
    init_logging_with(&config.logging)?;
    let engine = Arc::new(SearchEngine::from_config(&config).await?);
    engine.set_watch_root(config.watch_dir.clone());

    let root = config.watch_dir.clone();
    if root.is_dir() {
        // The watcher starts first so edits made during the initial sync are not lost.
        // Sync + watching run in the background: the service answers requests immediately
        // (searches see whatever is indexed so far) instead of blocking on a long first index.
        let mut watcher = Watcher::new(&root.to_string_lossy()).await?;
        let bg_engine = Arc::clone(&engine);
        tokio::spawn(async move {
            match bg_engine.index_directory(&root).await {
                Ok(s) => {
                    tracing::info!(
                        indexed = s.indexed,
                        unchanged = s.unchanged,
                        removed = s.removed,
                        failed = s.failed.len(),
                        "initial sync of {} done",
                        root.display()
                    );
                    for (path, error) in &s.failed {
                        tracing::warn!("could not index {path}: {error}");
                    }
                }
                Err(e) => tracing::warn!("initial sync failed: {e}"),
            }
            loop {
                match watcher.changes().await {
                    Ok(events) if events.is_empty() => break, // watcher shut down
                    Ok(events) => {
                        let n = bg_engine.apply_events(&root, &events).await;
                        tracing::info!("applied {n}/{} file events", events.len());
                    }
                    Err(e) => tracing::warn!("watcher error: {e}"),
                }
            }
        });
    } else {
        tracing::warn!(
            "watch dir {} not found; serving existing index only",
            root.display()
        );
    }

    if mcp_mode {
        mcp::serve(&engine, tokio::io::stdin(), tokio::io::stdout()).await
    } else {
        service::serve(&engine, tokio::io::stdin(), tokio::io::stdout()).await
    }
}
