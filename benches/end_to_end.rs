//! Criterion benchmark: end-to-end indexing and hybrid search on a 100-document corpus
//! (hashing embeddings, in-memory SQLite). Target: hybrid query p95 < 5 s (in practice ms).
//! Run: `cargo bench --bench end_to_end`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use plaesy::memory::config::SearchConfig;
use plaesy::memory::embeddings::EmbeddingEngine;
use plaesy::memory::search::SearchEngine;
use plaesy::memory::storage::{Database, DatabaseConfig};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;

const TOPICS: [&str; 5] = [
    "Vector embeddings enable semantic similarity search across high-dimensional spaces.",
    "HNSW graphs provide approximate nearest neighbor search for million-vector datasets.",
    "BM25 ranking delivers keyword retrieval via inverted indices and term frequency.",
    "Latency percentiles p50 p95 p99 are needed for accurate SLA compliance.",
    "Write-ahead logging gives databases crash recovery by replaying transactions.",
];

fn corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tmp");
    for i in 0..100 {
        let body = TOPICS[i % TOPICS.len()];
        std::fs::write(
            dir.path().join(format!("doc_{i:03}.md")),
            format!("---\nid: D-{i}\n---\n\n# Doc {i}\n\n{body}\n\nMarker token{i}.\n"),
        )
        .expect("write");
    }
    dir
}

async fn engine() -> SearchEngine {
    let db = Arc::new(
        Database::new(DatabaseConfig {
            db_path: PathBuf::from(":memory:"),
            ..DatabaseConfig::default()
        })
        .await
        .expect("db"),
    );
    let emb = Arc::new(EmbeddingEngine::hashing(256).expect("hashing"));
    SearchEngine::new(db, Some(emb), SearchConfig::default())
        .await
        .expect("engine")
}

fn bench_pipeline(c: &mut Criterion) {
    let rt = Runtime::new().expect("runtime");
    let dir = corpus();

    c.bench_function("index_100_docs", |b| {
        b.iter(|| {
            rt.block_on(async {
                let e = engine().await;
                black_box(e.index_directory(dir.path()).await.expect("index"))
            })
        })
    });

    let e = rt.block_on(async {
        let e = engine().await;
        e.index_directory(dir.path()).await.expect("index");
        e
    });
    c.bench_function("hybrid_search_top10", |b| {
        b.iter(|| {
            black_box(
                rt.block_on(e.search("nearest neighbor search graphs", 10))
                    .expect("search"),
            )
        })
    });
    c.bench_function("keyword_search_top10", |b| {
        b.iter(|| {
            black_box(
                rt.block_on(e.keyword_search("crash recovery transactions", 10))
                    .expect("kw"),
            )
        })
    });
    c.bench_function("reindex_unchanged_100_docs", |b| {
        b.iter(|| black_box(rt.block_on(e.index_directory(dir.path())).expect("resync")))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_pipeline
}
criterion_main!(benches);
