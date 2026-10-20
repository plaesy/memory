//! Model Context Protocol server (stdio transport, newline-delimited JSON-RPC 2.0).
//!
//! Lets any MCP client (Claude Code / Desktop, Cursor, Hermes, …) use the memory without a
//! per-client shim. All tools are read-only: the index is built and kept current by the
//! background sync of the plugin binary (`plaesy-memory-plugin --mcp`), never by a tool call.
//!
//! Tools:
//!
//! | tool | purpose |
//! |------|---------|
//! | `memory_search` | hybrid / semantic / keyword search with tier, path, trust and status filters; every hit is cited |
//! | `memory_backlinks` | which notes link to a note (or which notes it links to) |
//! | `memory_hygiene` | stale notes, missing verification metadata, broken links, dead path references |
//! | `memory_stats` | index size and search modes available |
//!
//! Supported protocol versions: 2025-06-18, 2025-03-26, 2024-11-05.

use crate::memory::error::{MemoryError, Result};
use crate::memory::hygiene::HygieneOptions;
use crate::memory::search::{SearchEngine, SearchMode};
use crate::memory::service::{parse_filter, SearchHit};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};

const LATEST: &str = "2025-06-18";
const SUPPORTED: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

fn tools() -> Value {
    let read_only =
        json!({ "readOnlyHint": true, "destructiveHint": false, "openWorldHint": false });
    json!([
        {
            "name": "memory_search",
            "description": "Search the project's indexed memory notes. Returns verbatim passages, each with a citation (file, line range, git revision, index time), an evidence label (FACT = lexical match in the source, INFERRED = found by meaning only) and what the source declares about itself (tier, trust, last verified, STALE). Prefer FACT hits, treat STALE or low-trust hits as leads to verify, and cite the file:lines when you use a passage.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to look for, in natural language or keywords." },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 50, "default": 5 },
                    "mode": { "type": "string", "enum": ["hybrid", "semantic", "keyword"], "default": "hybrid" },
                    "tier": {
                        "description": "Restrict to tiers: core, episodic, semantic, procedural, archive.",
                        "oneOf": [{ "type": "string" }, { "type": "array", "items": { "type": "string" } }]
                    },
                    "path_prefix": { "type": "string", "description": "Only notes whose path starts with this." },
                    "min_trust": { "type": "string", "enum": ["high", "medium", "low"], "description": "Only notes declaring at least this trust." },
                    "include_inactive": { "type": "boolean", "default": false, "description": "Also return superseded / deprecated notes." },
                    "as_of": { "type": "string", "format": "date", "description": "Only notes whose valid_from..valid_to period contains this date (YYYY-MM-DD). Undated notes always pass." },
                    "expand_links": { "type": "boolean", "default": false, "description": "Also add the best passage of notes linked to or from the top hits (one hop)." }
                },
                "required": ["query"]
            },
            "annotations": read_only
        },
        {
            "name": "memory_backlinks",
            "description": "List [[wikilinks]] for a note: the notes that link to it (default) or the notes it links to. Dangling links have no resolved_path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Note path, path without extension, or bare file name." },
                    "direction": { "type": "string", "enum": ["incoming", "outgoing"], "default": "incoming" }
                },
                "required": ["path"]
            },
            "annotations": read_only
        },
        {
            "name": "memory_hygiene",
            "description": "Audit the memory for problems: stale notes, missing last_verified/confidence on claim-bearing tiers, superseded notes without a replacement, broken wikilinks, and backticked file paths that no longer exist. Read-only.",
            "inputSchema": { "type": "object", "properties": {} },
            "annotations": read_only
        },
        {
            "name": "memory_stats",
            "description": "Index statistics: documents, chunks, embeddings, and whether semantic search is available.",
            "inputSchema": { "type": "object", "properties": {} },
            "annotations": read_only
        }
    ])
}

fn rpc_ok(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn rpc_err(id: &Value, code: i64, message: impl Into<String>) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message.into() } })
}

fn tool_text(text: String, structured: Value) -> Value {
    json!({ "content": [{ "type": "text", "text": text }], "structuredContent": structured, "isError": false })
}

fn tool_error(message: impl Into<String>) -> Value {
    json!({ "content": [{ "type": "text", "text": message.into() }], "isError": true })
}

/// Render hits for a model: one block per hit with its citation first.
fn render_hits(hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return "No matching notes.".to_string();
    }
    let mut out = String::new();
    for (i, h) in hits.iter().enumerate() {
        out.push_str(&format!(
            "{}. score {:.2}  {}\n",
            i + 1,
            h.score,
            h.citation
        ));
        for line in h.text.lines() {
            out.push_str("   > ");
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

async fn call_tool(engine: &SearchEngine, name: &str, args: &Value) -> Value {
    match run_tool(engine, name, args).await {
        Ok(v) => v,
        Err(ToolError::Invalid(m)) => tool_error(format!("invalid arguments: {m}")),
        Err(ToolError::Engine(e)) => tool_error(format!("{e}")),
    }
}

enum ToolError {
    Invalid(String),
    Engine(MemoryError),
}

impl From<MemoryError> for ToolError {
    fn from(e: MemoryError) -> Self {
        ToolError::Engine(e)
    }
}

async fn run_tool(
    engine: &SearchEngine,
    name: &str,
    args: &Value,
) -> std::result::Result<Value, ToolError> {
    match name {
        "memory_search" => {
            let query = args
                .get("query")
                .and_then(Value::as_str)
                .filter(|q| !q.trim().is_empty())
                .ok_or_else(|| {
                    ToolError::Invalid("`query` (non-empty string) is required".into())
                })?;
            let limit = args
                .get("limit")
                .and_then(Value::as_u64)
                .map_or(5, |n| n.clamp(1, 50) as usize);
            let mode_name = args.get("mode").and_then(Value::as_str).unwrap_or("hybrid");
            let mode = SearchMode::parse(mode_name)
                .ok_or_else(|| ToolError::Invalid(format!("unknown mode '{mode_name}'")))?;
            let filter = parse_filter(args).map_err(ToolError::Invalid)?;
            let expand = args
                .get("expand_links")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let results = if expand {
                engine.search_expanded(query, limit, mode, &filter).await?
            } else {
                engine.search_with(query, limit, mode, &filter).await?
            };
            let hits: Vec<SearchHit> = results
                .iter()
                .map(SearchHit::from_result)
                .collect::<Result<_>>()?;
            Ok(tool_text(render_hits(&hits), json!({ "hits": hits })))
        }
        "memory_backlinks" => {
            let path = args
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| ToolError::Invalid("`path` (string) is required".into()))?;
            let incoming = args
                .get("direction")
                .and_then(Value::as_str)
                .is_none_or(|d| d != "outgoing");
            let links = engine.links(path, incoming).await?;
            let text = if links.is_empty() {
                "No links.".to_string()
            } else {
                links
                    .iter()
                    .map(|l| {
                        format!(
                            "{}:{}  [[{}]] -> {}",
                            l.source_path,
                            l.line,
                            l.target,
                            l.resolved_path.as_deref().unwrap_or("(unresolved)")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            Ok(tool_text(text, json!({ "links": links })))
        }
        "memory_hygiene" => {
            let report = engine.hygiene(&HygieneOptions::default()).await?;
            let mut text = format!(
                "{} notes checked, {} findings\n",
                report.checked,
                report.findings.len()
            );
            for f in &report.findings {
                text.push_str(&format!(
                    "- [{}] {} {}{}: {}\n",
                    f.label,
                    f.kind,
                    f.path,
                    f.line.map(|l| format!(":{l}")).unwrap_or_default(),
                    f.detail
                ));
            }
            let structured = serde_json::to_value(&report)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
            Ok(tool_text(text, structured))
        }
        "memory_stats" => {
            let s = engine.statistics().await?;
            let mut v = serde_json::to_value(&s)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
            if let Some(o) = v.as_object_mut() {
                o.insert("semantic_search".into(), json!(engine.has_vector_search()));
            }
            Ok(tool_text(v.to_string(), v))
        }
        other => Err(ToolError::Invalid(format!("unknown tool '{other}'"))),
    }
}

/// Handle one JSON-RPC message. Returns `None` for notifications (nothing to send back).
pub async fn handle_message(engine: &SearchEngine, msg: &Value) -> Option<Value> {
    let id = msg.get("id").cloned();
    let method = msg.get("method").and_then(Value::as_str);
    let Some(method) = method else {
        // A response or garbage: servers that never send requests have nothing to do with it.
        return id.map(|id| rpc_err(&id, -32600, "invalid request: missing method"));
    };
    let Some(id) = id else {
        return None; // notification (e.g. notifications/initialized)
    };
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    Some(match method {
        "initialize" => {
            let asked = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(LATEST);
            let version = if SUPPORTED.contains(&asked) {
                asked
            } else {
                LATEST
            };
            rpc_ok(
                &id,
                json!({
                    "protocolVersion": version,
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": { "name": "plaesy-memory", "version": env!("CARGO_PKG_VERSION") },
                    "instructions": "Read-only memory over the project's notes. Use memory_search before answering from memory; cite the returned file:lines; treat STALE or INFERRED hits as unverified."
                }),
            )
        }
        "ping" => rpc_ok(&id, json!({})),
        "tools/list" => rpc_ok(&id, json!({ "tools": tools() })),
        "tools/call" => {
            let Some(name) = params.get("name").and_then(Value::as_str) else {
                return Some(rpc_err(&id, -32602, "tools/call requires params.name"));
            };
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            rpc_ok(&id, call_tool(engine, name, &args).await)
        }
        other => rpc_err(&id, -32601, format!("method not found: {other}")),
    })
}

/// Serve MCP over `input`/`output` until EOF (one JSON message per line).
///
/// # Errors
///
/// - `IoError` if reading or writing the streams fails
pub async fn serve<R, W>(engine: &SearchEngine, input: R, mut output: W) -> Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let io_err = |e: std::io::Error| MemoryError::IoError {
        path: "<stdio>".to_string(),
        source: e,
    };
    let mut lines = BufReader::new(input).lines();
    while let Some(line) = lines.next_line().await.map_err(io_err)? {
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => handle_message(engine, &msg).await,
            Err(e) => Some(rpc_err(&Value::Null, -32700, format!("parse error: {e}"))),
        };
        if let Some(reply) = reply {
            let mut bytes = serde_json::to_vec(&reply)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
            bytes.push(b'\n');
            output.write_all(&bytes).await.map_err(io_err)?;
            output.flush().await.map_err(io_err)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::config::SearchConfig;
    use crate::memory::storage::{Database, DatabaseConfig};
    use std::path::PathBuf;
    use std::sync::Arc;

    async fn engine() -> (SearchEngine, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tmp");
        std::fs::write(
            dir.path().join("a.md"),
            "---\ntier: semantic\nconfidence: high\nlast_verified: 2099-01-01\n---\n# Cats\n\nCats purr when content. See [[b]].\n",
        )
        .expect("w");
        std::fs::write(dir.path().join("b.md"), "# Dogs\n\nDogs bark.\n").expect("w");
        let db = Arc::new(
            Database::new(DatabaseConfig {
                db_path: PathBuf::from(":memory:"),
                ..DatabaseConfig::default()
            })
            .await
            .expect("db"),
        );
        let e = SearchEngine::new(db, None, SearchConfig::default())
            .await
            .expect("engine");
        e.index_directory(dir.path()).await.expect("index");
        (e, dir)
    }

    async fn rpc(e: &SearchEngine, msg: Value) -> Value {
        handle_message(e, &msg).await.expect("reply")
    }

    #[tokio::test]
    async fn handshake_and_tool_listing() {
        let (e, _d) = engine().await;
        let init = rpc(
            &e,
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}),
        )
        .await;
        assert_eq!(init["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(init["result"]["serverInfo"]["name"], "plaesy-memory");
        let future = rpc(
            &e,
            json!({"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2031-01-01"}}),
        )
        .await;
        assert_eq!(
            future["result"]["protocolVersion"], LATEST,
            "unknown version -> ours"
        );

        assert!(handle_message(
            &e,
            &json!({"jsonrpc":"2.0","method":"notifications/initialized"})
        )
        .await
        .is_none());
        let list = rpc(&e, json!({"jsonrpc":"2.0","id":3,"method":"tools/list"})).await;
        let names: Vec<&str> = list["result"]["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert_eq!(
            names,
            [
                "memory_search",
                "memory_backlinks",
                "memory_hygiene",
                "memory_stats"
            ]
        );
        let bad = rpc(
            &e,
            json!({"jsonrpc":"2.0","id":4,"method":"resources/list"}),
        )
        .await;
        assert_eq!(bad["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn search_tool_returns_cited_text_and_structured_hits() {
        let (e, _d) = engine().await;
        let r = rpc(
            &e,
            json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"memory_search","arguments":{"query":"purr","mode":"keyword","tier":"semantic","min_trust":"high"}}}),
        )
        .await;
        let res = &r["result"];
        assert_eq!(res["isError"], false);
        let text = res["content"][0]["text"].as_str().expect("text");
        assert!(
            text.contains("a.md:") && text.contains("[FACT]") && text.contains("tier semantic"),
            "{text}"
        );
        assert!(text.contains("> Cats purr when content"), "{text}");
        assert_eq!(
            res["structuredContent"]["hits"][0]["citation"]["file_path"],
            "a.md"
        );

        let none = rpc(
            &e,
            json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"memory_search","arguments":{"query":"bark","tier":"semantic"}}}),
        )
        .await;
        assert_eq!(none["result"]["content"][0]["text"], "No matching notes.");
    }

    #[tokio::test]
    async fn tool_errors_are_results_not_protocol_errors() {
        let (e, _d) = engine().await;
        for args in [
            json!({"name":"memory_search","arguments":{}}),
            json!({"name":"memory_search","arguments":{"query":"x","mode":"fuzzy"}}),
            json!({"name":"memory_search","arguments":{"query":"x","tier":"bogus"}}),
            json!({"name":"nope","arguments":{}}),
        ] {
            let r = rpc(
                &e,
                json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":args}),
            )
            .await;
            assert_eq!(r["result"]["isError"], true, "{r}");
            assert!(r.get("error").is_none());
        }
    }

    #[tokio::test]
    async fn backlinks_hygiene_and_stats_tools() {
        let (e, _d) = engine().await;
        let b = rpc(
            &e,
            json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"memory_backlinks","arguments":{"path":"b"}}}),
        )
        .await;
        assert!(b["result"]["content"][0]["text"]
            .as_str()
            .expect("t")
            .starts_with("a.md:"));
        let h = rpc(
            &e,
            json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"memory_hygiene","arguments":{}}}),
        )
        .await;
        assert_eq!(h["result"]["structuredContent"]["checked"], 2);
        let s = rpc(
            &e,
            json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"memory_stats","arguments":{}}}),
        )
        .await;
        assert_eq!(s["result"]["structuredContent"]["documents_total"], 2);
    }

    #[tokio::test]
    async fn serve_loop_survives_garbage_lines() {
        let (e, _d) = engine().await;
        let input = b"not json\n{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n";
        let mut out: Vec<u8> = Vec::new();
        serve(&e, &input[..], &mut out).await.expect("serve");
        let lines: Vec<Value> = String::from_utf8(out)
            .expect("utf8")
            .lines()
            .map(|l| serde_json::from_str(l).expect("json"))
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["error"]["code"], -32700);
        assert_eq!(lines[1]["id"], 7);
    }
}
