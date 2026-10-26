//! `memory serve`: the long-running entry point used by the Hermes provider (JSON service on
//! stdin/stdout) and by MCP clients (`memory serve --mcp`).
//!
//! The service answers requests immediately, while a background task syncs the configured watch
//! directory into the index and then keeps it current with a file watcher. Logging must already be
//! initialised by the caller; stdout is reserved for the protocol.

use crate::memory::config::MemoryConfig;
use crate::memory::error::{MemoryError, Result};
use crate::memory::indexer::Watcher;
use crate::memory::search::SearchEngine;
use crate::memory::{mcp, service};
use clap::ValueEnum;
use std::net::SocketAddr;
use std::sync::Arc;

/// Transport selection for `memory serve`.
#[derive(Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum Transport {
    /// Stdio: newline-delimited JSON-RPC 2.0 (default).
    #[default]
    Stdio,
    /// HTTP Streamable: POST /mcp for requests, SSE for notifications.
    Http,
}

impl std::str::FromStr for Transport {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "stdio" => Ok(Transport::Stdio),
            "http" => Ok(Transport::Http),
            other => Err(format!(
                "unknown transport '{other}' (expected stdio or http)"
            )),
        }
    }
}

/// Run the service until stdin closes or the HTTP server stops. `mcp` selects the Model Context
/// Protocol transport; `transport` selects stdio vs HTTP.
pub async fn run(config: MemoryConfig, mcp: bool, transport: Transport, port: u16) -> Result<()> {
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

    if !mcp {
        // JSON service (Hermes provider) is stdio-only.
        return service::serve(&engine, tokio::io::stdin(), tokio::io::stdout()).await;
    }

    match transport {
        Transport::Stdio => mcp::serve(&engine, tokio::io::stdin(), tokio::io::stdout()).await,
        Transport::Http => {
            let addr: SocketAddr = format!("127.0.0.1:{port}")
                .parse()
                .map_err(|e| MemoryError::ConfigError(format!("bad port {port}: {e}")))?;
            mcp::serve_http(engine, addr).await
        }
    }
}
