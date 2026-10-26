//! Tests for `--format {text,json,md}` CLI output.
use plaesy::memory::config::SearchConfig;
use plaesy::memory::embeddings::EmbeddingEngine;
use plaesy::memory::error::Result as MemoryResult;
use plaesy::memory::search::{CitationBuilder, SearchEngine, SearchFilter, SearchMode};
use plaesy::memory::service::SearchHit;
use std::path::PathBuf;
use std::sync::Arc;

async fn engine() -> SearchEngine {
    let db = Arc::new(
        plaesy::memory::storage::Database::new(plaesy::memory::storage::DatabaseConfig {
            db_path: PathBuf::from(":memory:"),
            ..Default::default()
        })
        .await
        .expect("db"),
    );
    let emb = Arc::new(EmbeddingEngine::hashing(128).expect("hash"));
    SearchEngine::new(db, Some(emb), SearchConfig::default())
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
async fn cli_format_json_parses_for_nonempty_results() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "a.md",
        "# Layers\n\nlayer three holds vectors\n",
    );
    let e = engine().await;
    e.index_directory(dir.path()).await.expect("index");
    let results = e
        .search_with_explain(
            "vectors",
            5,
            SearchMode::Hybrid,
            &SearchFilter::default(),
            false,
        )
        .await
        .expect("search");
    assert!(!results.is_empty());
    let hits: Vec<SearchHit> = results
        .iter()
        .map(SearchHit::from_result)
        .collect::<MemoryResult<Vec<SearchHit>>>()
        .expect("hits");
    let json = serde_json::to_string_pretty(&hits).expect("serialize");
    let parsed: Vec<SearchHit> = serde_json::from_str(&json).expect("parse");
    assert_eq!(parsed.len(), hits.len());
    for h in &parsed {
        assert!(
            !h.citation.file_path.is_empty(),
            "citation must carry file_path"
        );
        assert!(
            !h.citation.confidence.is_empty(),
            "citation must carry confidence"
        );
    }
}

#[tokio::test]
async fn cli_format_md_has_one_list_item_per_hit_with_path_and_lines() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "a.md",
        "# Layers\n\nlayer three holds vectors\n",
    );
    let e = engine().await;
    e.index_directory(dir.path()).await.expect("index");
    let results = e
        .search_with_explain(
            "vectors",
            5,
            SearchMode::Hybrid,
            &SearchFilter::default(),
            false,
        )
        .await
        .expect("search");
    assert!(!results.is_empty());
    let mut md = String::new();
    for r in &results {
        let c = CitationBuilder::new().build(r).expect("cite");
        md.push_str(&format!("- {c}\n"));
    }
    assert!(
        md.contains("- "),
        "md must start each line with a list marker"
    );
    assert!(md.contains("a.md:"), "md must include the file path");
    assert!(
        md.contains("[FACT]") || md.contains("[INFERRED]"),
        "md must include confidence label"
    );
}

#[tokio::test]
async fn cli_format_default_text_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "a.md",
        "# Layers\n\nlayer three holds vectors\n",
    );
    let e = engine().await;
    e.index_directory(dir.path()).await.expect("index");
    let results = e
        .search_with_explain(
            "vectors",
            5,
            SearchMode::Hybrid,
            &SearchFilter::default(),
            false,
        )
        .await
        .expect("search");
    let b = CitationBuilder::new();
    let mut text = String::new();
    for (i, r) in results.iter().enumerate() {
        text.push_str(&format!(
            "{}. score {:.3}  {}\n",
            i + 1,
            r.score,
            b.build(r).expect("cite")
        ));
        for line in r.text.lines().take(4) {
            text.push_str(&format!("     {line}\n"));
        }
    }
    assert!(text.contains("score "), "default text must include score");
    assert!(text.contains("a.md:"), "default text must include citation");
    assert!(
        text.contains("layer three"),
        "default text must include hit text"
    );
}
