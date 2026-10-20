//! Line-delimited JSON service protocol used by the Hermes plugin binary.
//!
//! One request per line on the input stream, one response per line on the
//! output stream (stdout is reserved for the protocol; logs go to stderr).
//!
//! ```text
//! → {"id":1,"method":"search","params":{"query":"memory layers","limit":5,"mode":"hybrid"}}
//! ← {"id":1,"ok":true,"result":{"hits":[{"chunk_id":3,"text":"…","score":0.93,
//!     "confidence":"FACT","citation":{"file_path":"…","line_range":[4,9],…}}]}}
//! → {"id":2,"method":"index","params":{"path":".plaesy"}}
//! → {"id":3,"method":"stats"}   → {"id":4,"method":"metrics"}   → {"id":5,"method":"ping"}
//! ```
//!
//! Every hit carries its citation (EV-01); errors are returned as
//! `{"ok":false,"error":{"kind":…,"message":…}}` and never end the session.

use crate::memory::error::{MemoryError, Result};
use crate::memory::search::{
    normalize_tier, Citation, CitationBuilder, SearchEngine, SearchFilter, SearchMode, Trust, TIERS,
};
use crate::memory::storage::SearchResult;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};

/// A protocol request.
#[derive(Debug, Deserialize)]
pub struct Request {
    /// Caller-chosen correlation id, echoed back verbatim.
    #[serde(default)]
    pub id: Value,

    /// Method name: `search`, `index`, `stats`, `metrics`, `ping`.
    pub method: String,

    /// Method parameters.
    #[serde(default)]
    pub params: Value,
}

/// A protocol response.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Response {
    /// The request's id (`null` if it could not be parsed).
    pub id: Value,

    /// Whether the call succeeded.
    pub ok: bool,

    /// Result payload on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,

    /// Error payload on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorBody>,
}

/// Error payload.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ErrorBody {
    /// Machine-readable category.
    pub kind: String,

    /// Human-readable message.
    pub message: String,
}

/// A search hit with its mandatory citation.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchHit {
    /// Chunk ID.
    pub chunk_id: u64,

    /// Verbatim chunk text.
    pub text: String,

    /// Relevance score (0-1).
    pub score: f32,

    /// `FACT`, `INFERRED` or `SPECULATIVE`.
    pub confidence: String,

    /// Source citation.
    pub citation: Citation,
}

impl SearchHit {
    /// Build a hit from a search result, validating its citation.
    pub fn from_result(r: &SearchResult) -> Result<Self> {
        let citation = CitationBuilder::new().build(r)?;
        Ok(SearchHit {
            chunk_id: r.chunk_id,
            text: r.text.clone(),
            score: r.score,
            confidence: citation.confidence.clone(),
            citation,
        })
    }
}

fn ok(id: Value, result: Value) -> Response {
    Response {
        id,
        ok: true,
        result: Some(result),
        error: None,
    }
}

fn err(id: Value, kind: &str, message: impl Into<String>) -> Response {
    Response {
        id,
        ok: false,
        result: None,
        error: Some(ErrorBody {
            kind: kind.to_string(),
            message: message.into(),
        }),
    }
}

fn error_kind(e: &MemoryError) -> &'static str {
    match e {
        MemoryError::ConfigError(_) => "config",
        MemoryError::FeatureNotEnabled { .. } => "feature_not_enabled",
        MemoryError::ResourceLimitExceeded(_) => "timeout",
        MemoryError::IoError { .. } => "io",
        _ => "internal",
    }
}

/// Handle one request against `engine`.
pub async fn handle_request(engine: &SearchEngine, req: Request) -> Response {
    let id = req.id.clone();
    match dispatch(engine, &req).await {
        Ok(v) => ok(id, v),
        Err(DispatchError::BadRequest(m)) => err(id, "bad_request", m),
        Err(DispatchError::Engine(e)) => err(id, error_kind(&e), e.to_string()),
    }
}

enum DispatchError {
    BadRequest(String),
    Engine(MemoryError),
}

impl From<MemoryError> for DispatchError {
    fn from(e: MemoryError) -> Self {
        DispatchError::Engine(e)
    }
}

async fn dispatch(
    engine: &SearchEngine,
    req: &Request,
) -> std::result::Result<Value, DispatchError> {
    match req.method.as_str() {
        "ping" => Ok(json!({ "pong": true, "vector_search": engine.has_vector_search() })),
        "search" => {
            let query = req
                .params
                .get("query")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    DispatchError::BadRequest("params.query (string) is required".into())
                })?;
            let limit = req
                .params
                .get("limit")
                .map(|v| {
                    v.as_u64().map(|n| n as usize).ok_or_else(|| {
                        DispatchError::BadRequest(
                            "params.limit must be a non-negative integer".into(),
                        )
                    })
                })
                .transpose()?
                .unwrap_or(10);
            let mode_name = req
                .params
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or("hybrid");
            let mode = SearchMode::parse(mode_name).ok_or_else(|| {
                DispatchError::BadRequest(format!(
                    "unknown mode '{mode_name}' (expected hybrid, semantic or keyword)"
                ))
            })?;
            let filter = parse_filter(&req.params).map_err(DispatchError::BadRequest)?;
            let expand = req
                .params
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
            Ok(json!({ "hits": hits }))
        }
        "index" => {
            let path = req
                .params
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    DispatchError::BadRequest("params.path (string) is required".into())
                })?;
            let root = PathBuf::from(path);
            if !root.is_dir() {
                return Err(DispatchError::BadRequest(format!(
                    "{path} is not a directory"
                )));
            }
            let s = engine.index_directory(&root).await?;
            Ok(json!({
                "indexed": s.indexed, "unchanged": s.unchanged, "removed": s.removed,
                "chunks": s.chunks,
                "failed": s.failed.iter().map(|(p, e)| json!({"path": p, "error": e})).collect::<Vec<_>>(),
            }))
        }
        "backlinks" | "links" => {
            let path = req
                .params
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    DispatchError::BadRequest("params.path (string) is required".into())
                })?;
            let links = engine.links(path, req.method == "backlinks").await?;
            Ok(json!({ "links": links }))
        }
        "hygiene" => {
            let report = engine
                .hygiene(&crate::memory::hygiene::HygieneOptions {
                    root: req
                        .params
                        .get("root")
                        .and_then(Value::as_str)
                        .map(PathBuf::from),
                    ..Default::default()
                })
                .await?;
            Ok(serde_json::to_value(report)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?)
        }
        "stats" => {
            let s = engine.statistics().await?;
            Ok(serde_json::to_value(s)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?)
        }
        "metrics" => Ok(json!({ "prometheus": engine.metrics().render_prometheus() })),
        other => Err(DispatchError::BadRequest(format!(
            "unknown method '{other}'"
        ))),
    }
}

/// Build a [`SearchFilter`] from `params.tier` (string or array), `params.path_prefix`,
/// `params.min_trust` and `params.include_inactive`.
pub fn parse_filter(params: &Value) -> std::result::Result<SearchFilter, String> {
    let mut f = SearchFilter::default();
    match params.get("tier") {
        None | Some(Value::Null) => {}
        Some(Value::String(t)) => f.tiers.push(t.clone()),
        Some(Value::Array(a)) => {
            for t in a {
                f.tiers.push(
                    t.as_str()
                        .ok_or("params.tier must be a string or array of strings")?
                        .to_string(),
                );
            }
        }
        Some(_) => return Err("params.tier must be a string or array of strings".into()),
    }
    for t in &f.tiers {
        if normalize_tier(t).is_none() {
            return Err(format!(
                "unknown tier '{t}' (expected one of: {})",
                TIERS.join(", ")
            ));
        }
    }
    if let Some(v) = params.get("as_of").filter(|v| !v.is_null()) {
        let s = v
            .as_str()
            .ok_or("params.as_of must be a date string (YYYY-MM-DD)")?;
        f.as_of = Some(
            crate::memory::search::parse_date(s)
                .ok_or("params.as_of must be a date (YYYY-MM-DD) or an RFC 3339 timestamp")?,
        );
    }
    if let Some(p) = params.get("path_prefix").and_then(Value::as_str) {
        f.path_prefix = Some(p.to_string());
    }
    if let Some(v) = params.get("min_trust").filter(|v| !v.is_null()) {
        f.min_trust = Some(
            Trust::parse(v).ok_or("params.min_trust must be high, medium, low or a number 0-1")?,
        );
    }
    f.include_inactive = params
        .get("include_inactive")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(f)
}

async fn write_line<W: AsyncWrite + Unpin>(out: &mut W, resp: &Response) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(resp).unwrap_or_else(|_| {
        br#"{"id":null,"ok":false,"error":{"kind":"internal","message":"response serialization failed"}}"#.to_vec()
    });
    line.push(b'\n');
    out.write_all(&line).await?;
    out.flush().await
}

/// Serve requests from `input` until EOF, writing responses to `output`.
///
/// Malformed lines produce an error response and the session continues.
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
        let resp = match serde_json::from_str::<Request>(&line) {
            Ok(req) => handle_request(engine, req).await,
            Err(e) => err(
                Value::Null,
                "bad_request",
                format!("invalid request JSON: {e}"),
            ),
        };
        write_line(&mut output, &resp).await.map_err(io_err)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::config::SearchConfig;
    use crate::memory::storage::{Database, DatabaseConfig};
    use std::sync::Arc;

    async fn engine_with_doc() -> (SearchEngine, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tmp");
        std::fs::write(
            dir.path().join("a.md"),
            "# Layers\n\nlayer three holds vectors\n",
        )
        .expect("w");
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
            .expect("e");
        e.index_directory(dir.path()).await.expect("index");
        (e, dir)
    }

    async fn roundtrip(e: &SearchEngine, input: &str) -> Vec<Response> {
        let mut out = Vec::new();
        serve(e, input.as_bytes(), &mut out).await.expect("serve");
        String::from_utf8(out)
            .expect("utf8")
            .lines()
            .map(|l| serde_json::from_str(l).expect("response json"))
            .collect()
    }

    #[tokio::test]
    async fn test_search_returns_hits_with_citations() {
        let (e, _d) = engine_with_doc().await;
        let r = roundtrip(&e, "{\"id\":7,\"method\":\"search\",\"params\":{\"query\":\"vectors\",\"mode\":\"keyword\"}}\n").await;
        assert_eq!(r.len(), 1);
        assert!(r[0].ok, "{:?}", r[0]);
        assert_eq!(r[0].id, json!(7));
        let hits = r[0].result.as_ref().expect("result")["hits"]
            .as_array()
            .expect("hits")
            .clone();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["confidence"], "FACT");
        assert_eq!(hits[0]["citation"]["file_path"], "a.md");
        assert_eq!(hits[0]["citation"]["line_range"], json!([1, 3]));
    }

    #[tokio::test]
    async fn test_errors_do_not_end_session() {
        let (e, _d) = engine_with_doc().await;
        let input = "not json\n\
                     {\"id\":1,\"method\":\"nope\"}\n\
                     {\"id\":2,\"method\":\"search\",\"params\":{}}\n\
                     {\"id\":3,\"method\":\"search\",\"params\":{\"query\":\"x\",\"mode\":\"semantic\"}}\n\
                     {\"id\":4,\"method\":\"search\",\"params\":{\"query\":\"x\",\"limit\":\"ten\"}}\n\
                     {\"id\":5,\"method\":\"index\",\"params\":{\"path\":\"no/such/dir\"}}\n\
                     \n\
                     {\"id\":6,\"method\":\"ping\"}\n";
        let r = roundtrip(&e, input).await;
        let kinds: Vec<_> = r
            .iter()
            .map(|x| x.error.as_ref().map(|b| b.kind.as_str()).unwrap_or("ok"))
            .collect();
        assert_eq!(
            kinds,
            vec![
                "bad_request",
                "bad_request",
                "bad_request",
                "feature_not_enabled",
                "bad_request",
                "bad_request",
                "ok"
            ]
        );
        assert_eq!(r[0].id, Value::Null);
        assert_eq!(r[6].result.as_ref().expect("pong")["pong"], true);
    }

    #[tokio::test]
    async fn test_index_stats_metrics() {
        let (e, d) = engine_with_doc().await;
        let path = d.path().to_string_lossy().replace('\\', "\\\\");
        let input = format!(
            "{{\"id\":1,\"method\":\"index\",\"params\":{{\"path\":\"{path}\"}}}}\n\
             {{\"id\":2,\"method\":\"stats\"}}\n\
             {{\"id\":3,\"method\":\"metrics\"}}\n"
        );
        let r = roundtrip(&e, &input).await;
        assert!(r.iter().all(|x| x.ok), "{r:?}");
        assert_eq!(r[0].result.as_ref().expect("r")["unchanged"], 1);
        assert_eq!(r[1].result.as_ref().expect("r")["documents_total"], 1);
        assert!(r[2].result.as_ref().expect("r")["prometheus"]
            .as_str()
            .expect("s")
            .contains("plaesy_memory_"));
    }
}

#[cfg(test)]
mod filter_param_tests {
    use super::*;

    #[test]
    fn parse_filter_accepts_valid_and_rejects_invalid() {
        let f = parse_filter(&json!({
            "tier": ["knowledge", "procedural"], "path_prefix": "20-semantic/",
            "min_trust": "high", "include_inactive": true
        }))
        .expect("valid");
        assert_eq!(f.tiers, vec!["knowledge", "procedural"]);
        assert_eq!(f.path_prefix.as_deref(), Some("20-semantic/"));
        assert_eq!(f.min_trust, Some(Trust::High));
        assert!(f.include_inactive);

        assert!(parse_filter(&json!({})).expect("empty").tiers.is_empty());
        assert!(parse_filter(&json!({"tier": "bogus"})).is_err());
        assert!(parse_filter(&json!({"tier": 3})).is_err());
        assert!(parse_filter(&json!({"min_trust": "very"})).is_err());
        let asof = parse_filter(&json!({"as_of": "2026-06-15"})).expect("as_of");
        assert!(asof.as_of.is_some());
        assert!(parse_filter(&json!({"as_of": "not a date"})).is_err());
        assert!(parse_filter(&json!({"as_of": 20260615})).is_err());
        assert_eq!(
            parse_filter(&json!({"min_trust": 0.9}))
                .expect("num")
                .min_trust,
            Some(Trust::High)
        );
    }
}
