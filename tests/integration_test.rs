//! Integration test: full pipeline against the public API.
//!
//! 100 documents on disk → index → 10 queries → verify relevance, latency and
//! EV-01 (every result cites a source whose line range reproduces the text).
//! Runs with the real ONNX model when installed (see `embeddings` docs), and
//! always with the lexical hashing backend plus keyword search.

use plaesy::memory::config::SearchConfig;
use plaesy::memory::embeddings::{EmbeddingConfig, EmbeddingEngine};
use plaesy::memory::prelude::*;
use plaesy::memory::search::CitationBuilder;
use plaesy::memory::storage::DatabaseConfig;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tempfile::TempDir;

/// (topic keyword unique to the group, body sentence)
const TOPICS: [(&str, &str); 10] = [
    ("embeddings", "Vector embeddings enable semantic similarity search across high-dimensional spaces using cosine distance."),
    ("hnsw", "HNSW graphs provide approximate nearest neighbor search with logarithmic complexity for million-vector datasets."),
    ("bm25", "Full-text search with BM25 ranking delivers keyword retrieval via inverted indices and term frequency analysis."),
    ("percentiles", "Performance optimization requires measuring latency percentiles: p50, p95, p99 for accurate SLA compliance."),
    ("btrees", "Data structures like LSM trees, B-trees, and skip lists balance read and write throughput for database engines."),
    ("raft", "Distributed systems need consensus protocols such as Raft and Paxos for fault tolerance and consistency."),
    ("hybrid", "Information retrieval combines sparse BM25 and dense embeddings for hybrid ranking effectiveness."),
    ("allocators", "Memory management involves allocators, garbage collection, and reference counting for efficient heap usage."),
    ("batching", "Throughput optimization uses batching, pipelining, and vectorization to maximize operations per second."),
    ("wal", "Write-ahead logging gives databases crash recovery by replaying committed transactions after a failure."),
];

fn write_corpus(root: &Path) -> Vec<PathBuf> {
    (0..100)
        .map(|i| {
            let (topic, body) = TOPICS[i % 10];
            let content = format!(
                "---\nid: EV-{i:03}\ntitle: Note {i}\n---\n\n# Note {i}\n\n{body}\n\nTopic tag: {topic}-{i}\n"
            );
            let path = root.join(format!("note_{i:03}.md"));
            std::fs::write(&path, content).expect("write doc");
            path
        })
        .collect()
}

async fn build_engine(embeddings: Option<Arc<EmbeddingEngine>>) -> SearchEngine {
    let db = Arc::new(
        Database::new(DatabaseConfig {
            db_path: PathBuf::from(":memory:"),
            ..DatabaseConfig::default()
        })
        .await
        .expect("db"),
    );
    SearchEngine::new(db, embeddings, SearchConfig::default())
        .await
        .expect("engine")
}

/// EV-01: every result is citable and its citation reproduces the chunk text from disk.
fn assert_ev01(root: &Path, results: &[SearchResult]) {
    let builder = CitationBuilder::new();
    for r in results {
        let c = builder.build(r).expect("EV-01: citation must build");
        assert!(!c.file_path.is_empty());
        assert!(!c.timestamp.is_empty());
        let on_disk = std::fs::read_to_string(root.join(&c.file_path)).expect("cited file exists");
        let lines: Vec<&str> = on_disk.lines().collect();
        let cited = lines[c.line_range.0 - 1..c.line_range.1].join("\n");
        assert_eq!(r.text, cited, "EV-03: result text must be verbatim source");
    }
}

#[tokio::test]
async fn full_pipeline_keyword_and_hashing() {
    let tmp = TempDir::new().expect("tmp");
    let root = tmp.path();
    write_corpus(root);
    let engine = build_engine(Some(Arc::new(EmbeddingEngine::hashing(256).expect("hash")))).await;

    let t = Instant::now();
    let summary = engine.index_directory(root).await.expect("index");
    assert_eq!(summary.indexed, 100);
    assert!(summary.failed.is_empty(), "{:?}", summary.failed);
    eprintln!(
        "indexed 100 docs / {} chunks in {:?}",
        summary.chunks,
        t.elapsed()
    );

    let mut latencies = Vec::new();
    for (topic, body) in TOPICS {
        // Query with a distinctive word from the group's sentence.
        let query = format!("{topic} {}", body.split_whitespace().nth(2).unwrap_or(""));
        let t = Instant::now();
        let results = engine.search(&query, 10).await.expect("search");
        latencies.push(t.elapsed());
        assert!(!results.is_empty(), "no results for {query}");
        assert!(results.len() <= 10);
        assert_ev01(root, &results);
        // Top hit must be from the right topic group.
        assert!(
            results[0]
                .text
                .contains(body.split('.').next().unwrap_or(body))
                || results[0].text.to_lowercase().contains(topic),
            "query {query:?} top hit was {:?}",
            results[0].text
        );
        assert!(results[0].score > 0.0 && results[0].score <= 1.0);
        // Scores are sorted descending.
        assert!(results.windows(2).all(|w| w[0].score >= w[1].score));
    }
    latencies.sort();
    let p95 = latencies[(latencies.len() * 95 / 100).min(latencies.len() - 1)];
    assert!(p95.as_secs_f64() < 5.0, "p95 {p95:?} exceeds 5s target");

    let stats = engine.statistics().await.expect("stats");
    assert_eq!(stats.documents_total, 100);
    assert!(
        engine
            .update_health(root)
            .await
            .expect("health")
            .coverage_percent
            > 99.9
    );
}

#[tokio::test]
async fn incremental_sync_after_edit_and_delete() {
    let tmp = TempDir::new().expect("tmp");
    let root = tmp.path();
    let paths = write_corpus(root);
    let engine = build_engine(Some(Arc::new(EmbeddingEngine::hashing(256).expect("hash")))).await;
    engine.index_directory(root).await.expect("index");

    std::fs::write(&paths[0], "# Rewritten\n\nzebrafish unique marker\n").expect("edit");
    std::fs::remove_file(&paths[1]).expect("delete");
    let s = engine.index_directory(root).await.expect("resync");
    assert_eq!((s.indexed, s.unchanged, s.removed), (1, 98, 1));

    let hits = engine.keyword_search("zebrafish", 5).await.expect("kw");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].source_path, "note_000.md");
    assert_ev01(root, &hits);
    assert_eq!(
        engine.statistics().await.expect("stats").documents_total,
        99
    );
}

#[tokio::test]
async fn onnx_pipeline_when_model_installed() {
    let Ok(emb) = EmbeddingEngine::with_config(EmbeddingConfig::default()).await else {
        eprintln!("SKIPPED onnx_pipeline_when_model_installed: no ONNX model installed");
        return;
    };
    let tmp = TempDir::new().expect("tmp");
    let root = tmp.path();
    write_corpus(root);
    let engine = build_engine(Some(Arc::new(emb))).await;
    engine.index_directory(root).await.expect("index");

    // Paraphrased queries sharing no keywords with the corpus: only vectors can answer.
    let cases = [
        (
            "finding the closest points in a big collection of vectors",
            "HNSW graphs",
        ),
        ("how databases survive crashes", "Write-ahead logging"),
    ];
    for (query, expected_prefix) in cases {
        let results = engine.semantic_search(query, 5).await.expect("semantic");
        assert!(
            results
                .iter()
                .take(3)
                .any(|r| r.text.contains(expected_prefix)),
            "query {query:?}: top-3 {:?}",
            results.iter().take(3).map(|r| &r.text).collect::<Vec<_>>()
        );
        assert!(results
            .iter()
            .all(|r| r.confidence == ConfidenceLevel::Inferred));
        assert_ev01(root, &results);
    }
}
