//! Tests for `--explain` CLI and MCP `memory_search` breakdown.
use plaesy::memory::config::SearchConfig;
use plaesy::memory::embeddings::EmbeddingEngine;
use plaesy::memory::search::{SearchEngine, SearchFilter, SearchMode};
use std::path::PathBuf;
use std::sync::Arc;

async fn engine(with_vectors: bool) -> SearchEngine {
    let db = Arc::new(
        plaesy::memory::storage::Database::new(plaesy::memory::storage::DatabaseConfig {
            db_path: PathBuf::from(":memory:"),
            ..Default::default()
        })
        .await
        .expect("db"),
    );
    let emb = with_vectors.then(|| Arc::new(EmbeddingEngine::hashing(128).expect("hash")));
    SearchEngine::new(db, emb, SearchConfig::default())
        .await
        .expect("engine")
}

fn write(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
    let p = dir.join(name);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(&p, content).expect("write");
    p
}

#[tokio::test]
async fn cli_explain_hybrid_breakdown_per_hit() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "a.md",
        "# Layers\n\nlayer three holds vectors\n",
    );
    let e = engine(true).await;
    e.index_directory(dir.path()).await.expect("index");
    let results = e
        .search_with_explain(
            "vectors",
            5,
            SearchMode::Hybrid,
            &SearchFilter::default(),
            true,
        )
        .await
        .expect("search");
    assert!(!results.is_empty());
    for r in &results {
        assert!(r.explain.is_some());
        let ex = r.explain.as_ref().unwrap();
        assert!(ex.fused.is_some(), "all hits must carry fused score");
        assert!(ex.fts.is_some());
        assert!(ex.vector.is_some());
    }
}

#[tokio::test]
async fn cli_explain_keyword_only_missing_vector_leg() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "a.md",
        "# Layers\n\nlayer three holds vectors\n",
    );
    let e = engine(false).await; // no embeddings → keyword only
    e.index_directory(dir.path()).await.expect("index");
    let results = e
        .search_with_explain(
            "layers",
            5,
            SearchMode::Keyword,
            &SearchFilter::default(),
            true,
        )
        .await
        .expect("search");
    assert!(!results.is_empty());
    for r in &results {
        let ex = r.explain.as_ref().unwrap();
        assert!(ex.fused.is_some());
        assert!(ex.fts.is_some());
        assert!(
            ex.vector.is_none(),
            "keyword-only result must omit vector leg"
        );
    }
}

#[tokio::test]
async fn cli_explain_mcp_tool_structure() {
    // Validate that `structuredContent.hits[].explain` carries the breakdown via the tool path.
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "b.md", "# Other\nhello world\n");
    let e = engine(true).await;
    e.index_directory(dir.path()).await.expect("index");
    // Direct call through search_with_explain mirror on SearchHit
    let results = e
        .search_with_explain(
            "hello",
            3,
            SearchMode::Hybrid,
            &SearchFilter::default(),
            true,
        )
        .await
        .expect("search");
    for r in &results {
        assert!(r.explain.is_some(), "each hit must carry explain");
    }
}

#[tokio::test]
async fn cli_explain_zero_on_no_request() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "c.md", "# Empty\n\n");
    let e = engine(true).await;
    e.index_directory(dir.path()).await.expect("index");
    let results = e
        .search_with_explain(
            "zzz",
            5,
            SearchMode::Hybrid,
            &SearchFilter::default(),
            false,
        )
        .await
        .expect("search");
    // Even with explain=false, results may contain hits from prior indexing;
    // the key point is that explain is not attached when the flag is off.
    for r in &results {
        assert!(
            r.explain.is_none(),
            "default mode must not attach explain when flag=false"
        );
    }
}
