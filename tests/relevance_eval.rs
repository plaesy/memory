//! Relevance evaluation on a small labeled corpus: Recall@3 and MRR per search mode.
//!
//! 20 documents (one topic each), 20 queries: 10 lexical (share words with the
//! target) and 10 paraphrases (share none). The semantic and hybrid thresholds
//! are only enforced when the ONNX model is installed; keyword thresholds always.
//! Run with `--nocapture` to see the table.

use plaesy::memory::config::SearchConfig;
use plaesy::memory::embeddings::{EmbeddingConfig, EmbeddingEngine};
use plaesy::memory::search::SearchEngine;
use plaesy::memory::storage::{Database, DatabaseConfig, SearchResult};
use std::path::PathBuf;
use std::sync::Arc;

/// (file, body)
const DOCS: [(&str, &str); 20] = [
    ("hnsw.md", "# HNSW\n\nHierarchical navigable small world graphs give approximate nearest neighbor search over vectors with logarithmic complexity."),
    ("bm25.md", "# BM25\n\nBM25 ranks documents by term frequency and inverse document frequency using an inverted index."),
    ("wal.md", "# Write-ahead log\n\nA write-ahead log records changes before they are applied so a database can recover after a crash."),
    ("raft.md", "# Raft\n\nRaft is a consensus protocol where a leader replicates a log to followers to keep a cluster consistent."),
    ("gc.md", "# Garbage collection\n\nA garbage collector reclaims heap memory that is no longer reachable by the running program."),
    ("btree.md", "# B-tree\n\nA B-tree keeps keys sorted in wide nodes so disk-based databases can find a record with few page reads."),
    ("tls.md", "# TLS\n\nTransport Layer Security encrypts network traffic and authenticates the server with certificates."),
    ("docker.md", "# Containers\n\nContainers package an application with its dependencies and run isolated from the host using namespaces."),
    ("git.md", "# Git rebase\n\nRebasing replays commits on top of another branch to produce a linear project history."),
    ("cache.md", "# Caching\n\nAn LRU cache evicts the least recently used entry when it is full to keep hot data in memory."),
    ("oauth.md", "# OAuth\n\nOAuth lets an application obtain limited access to a user account through access tokens without sharing the password."),
    ("sql.md", "# SQL joins\n\nAn inner join returns rows with matching keys in both tables while a left join keeps unmatched left rows."),
    ("async.md", "# Async runtime\n\nAn async runtime schedules many lightweight tasks on a few threads and wakes them when IO is ready."),
    ("cdn.md", "# CDN\n\nA content delivery network serves static files from edge servers close to the user to reduce latency."),
    ("ci.md", "# Continuous integration\n\nContinuous integration builds and tests every commit automatically to catch regressions early."),
    ("shard.md", "# Sharding\n\nSharding splits a large dataset across machines by a partition key so each node holds a subset."),
    ("regex.md", "# Regular expressions\n\nA regular expression describes a text pattern that can be matched against strings."),
    ("ssd.md", "# Flash storage\n\nSolid state drives store data in flash cells and have no moving parts, giving fast random reads."),
    ("dns.md", "# DNS\n\nThe domain name system translates human readable host names into IP addresses."),
    ("onnx.md", "# ONNX\n\nONNX is an open format for machine learning models so they can run in different runtimes."),
];

/// (query, expected file, lexical?)
const QUERIES: [(&str, &str, bool); 20] = [
    ("approximate nearest neighbor graph", "hnsw.md", true),
    ("inverted index term frequency ranking", "bm25.md", true),
    ("recover database after crash log", "wal.md", true),
    ("leader replicates log to followers", "raft.md", true),
    ("reclaims heap memory", "gc.md", true),
    ("sorted keys wide nodes disk", "btree.md", true),
    ("encrypts network traffic certificates", "tls.md", true),
    ("replays commits linear history", "git.md", true),
    ("evicts least recently used entry", "cache.md", true),
    (
        "access tokens without sharing the password",
        "oauth.md",
        true,
    ),
    (
        "finding vectors that are close to a query point",
        "hnsw.md",
        false,
    ),
    (
        "how does a database survive a power failure",
        "wal.md",
        false,
    ),
    ("agreement between servers in a cluster", "raft.md", false),
    ("freeing unused objects automatically", "gc.md", false),
    (
        "secure connection between browser and website",
        "tls.md",
        false,
    ),
    ("shipping software in isolated boxes", "docker.md", false),
    ("combining tables on a shared column", "sql.md", false),
    (
        "serve images from a server near the visitor",
        "cdn.md",
        false,
    ),
    ("automatic testing of every code change", "ci.md", false),
    ("turning a website name into a number", "dns.md", false),
];

fn rank_of(results: &[SearchResult], expected: &str) -> Option<usize> {
    results.iter().position(|r| r.source_path == expected)
}

#[derive(Default)]
struct Scores {
    hits3: [usize; 2],
    rr: [f64; 2],
    n: [usize; 2],
}

impl Scores {
    fn add(&mut self, lexical: bool, rank: Option<usize>) {
        let i = usize::from(!lexical);
        self.n[i] += 1;
        if let Some(r) = rank {
            if r < 3 {
                self.hits3[i] += 1;
            }
            self.rr[i] += 1.0 / (r as f64 + 1.0);
        }
    }
    fn recall3(&self, i: usize) -> f64 {
        self.hits3[i] as f64 / self.n[i] as f64
    }
    fn mrr(&self, i: usize) -> f64 {
        self.rr[i] / self.n[i] as f64
    }
}

async fn engine(emb: Option<Arc<EmbeddingEngine>>) -> (SearchEngine, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tmp");
    for (f, body) in DOCS {
        std::fs::write(dir.path().join(f), body).expect("write");
    }
    let db = Arc::new(
        Database::new(DatabaseConfig {
            db_path: PathBuf::from(":memory:"),
            ..DatabaseConfig::default()
        })
        .await
        .expect("db"),
    );
    let e = SearchEngine::new(db, emb, SearchConfig::default())
        .await
        .expect("engine");
    e.index_directory(dir.path()).await.expect("index");
    (e, dir)
}

async fn evaluate(e: &SearchEngine, mode: &str) -> Scores {
    let mut s = Scores::default();
    for (q, expected, lexical) in QUERIES {
        let results = match mode {
            "keyword" => e.keyword_search(q, 10).await,
            "semantic" => e.semantic_search(q, 10).await,
            _ => e.search(q, 10).await,
        }
        .expect("search");
        let rank = rank_of(&results, expected);
        if rank != Some(0) {
            let top: Vec<_> = results
                .iter()
                .take(3)
                .map(|r| (r.source_path.as_str(), (r.score * 100.0).round() / 100.0))
                .collect();
            eprintln!("  [{mode}] {q:?}: expected {expected} at {rank:?}; top {top:?}");
        }
        s.add(lexical, rank);
    }
    s
}

fn report(mode: &str, s: &Scores) {
    eprintln!(
        "{mode:<9} lexical: R@3 {:.2} MRR {:.2} | paraphrase: R@3 {:.2} MRR {:.2}",
        s.recall3(0),
        s.mrr(0),
        s.recall3(1),
        s.mrr(1)
    );
}

#[tokio::test]
async fn keyword_mode_finds_lexical_queries() {
    let (e, _d) = engine(None).await;
    let s = evaluate(&e, "keyword").await;
    report("keyword", &s);
    assert!(s.recall3(0) >= 0.9, "lexical R@3 {}", s.recall3(0));
    assert!(s.mrr(0) >= 0.8, "lexical MRR {}", s.mrr(0));
}

#[tokio::test]
async fn onnx_hybrid_beats_each_leg() {
    let Ok(emb) = EmbeddingEngine::with_config(EmbeddingConfig::default()).await else {
        eprintln!("SKIPPED onnx_hybrid_beats_each_leg: no ONNX model installed");
        return;
    };
    let (e, _d) = engine(Some(Arc::new(emb))).await;
    let kw = evaluate(&e, "keyword").await;
    let sem = evaluate(&e, "semantic").await;
    let hyb = evaluate(&e, "hybrid").await;
    report("keyword", &kw);
    report("semantic", &sem);
    report("hybrid", &hyb);

    // Overall MRR across all 20 queries.
    let overall = |s: &Scores| (s.rr[0] + s.rr[1]) / 20.0;
    assert!(
        overall(&hyb) >= overall(&kw),
        "hybrid must not lose to keyword"
    );
    assert!(
        overall(&hyb) >= overall(&sem) - 0.03,
        "hybrid must be within 0.03 of semantic"
    );
    assert!(
        hyb.recall3(0) >= 0.9,
        "hybrid lexical R@3 {}",
        hyb.recall3(0)
    );
    assert!(
        hyb.recall3(1) >= 0.7,
        "hybrid paraphrase R@3 {}",
        hyb.recall3(1)
    );
    assert!(overall(&hyb) >= 0.8, "overall hybrid MRR {}", overall(&hyb));
}

#[tokio::test]
async fn onnx_rerank_does_not_hurt_and_keeps_text_verbatim() {
    use plaesy::memory::rerank::CrossEncoder;
    let Ok(emb) = EmbeddingEngine::with_config(EmbeddingConfig::default()).await else {
        eprintln!("SKIPPED onnx_rerank: no embedding model installed");
        return;
    };
    let Some(dir) = CrossEncoder::resolve_model_dir(None) else {
        eprintln!("SKIPPED onnx_rerank: no cross-encoder model installed");
        return;
    };
    let ce = Arc::new(CrossEncoder::load(&dir, 256).expect("load cross-encoder"));
    let (e, _d) = engine(Some(Arc::new(emb))).await;
    let e = e.with_reranker(ce);

    let mut base = Scores::default();
    let mut reranked = Scores::default();
    for (q, expected, lexical) in QUERIES {
        let fused = e.search(q, 10).await.expect("search");
        let r = e
            .rerank_results(q, fused.clone(), 10, 0.5)
            .await
            .expect("rerank");
        assert_eq!(r.len(), fused.len());
        for hit in &r {
            let original = fused
                .iter()
                .find(|f| f.chunk_id == hit.chunk_id)
                .expect("same chunks");
            assert_eq!(hit.text, original.text, "EV-03: text is never altered");
            assert_eq!(hit.confidence, original.confidence);
        }
        base.add(lexical, rank_of(&fused, expected));
        reranked.add(lexical, rank_of(&r, expected));
    }
    report("hybrid", &base);
    report("+rerank", &reranked);
    let overall = |s: &Scores| (s.rr[0] + s.rr[1]) / 20.0;
    assert!(
        overall(&reranked) >= overall(&base) - 0.05,
        "rerank MRR {} vs hybrid {}",
        overall(&reranked),
        overall(&base)
    );
}
