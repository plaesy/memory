//! LongMemEval-S retrieval benchmark (session-level recall) for the real search engine.
//!
//! Dataset: `longmemeval_s_cleaned.json` from <https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned>
//! (500 questions, ~48 chat sessions each; the 30 abstention questions are skipped because they have
//! no evidence session). For every question a fresh index holds only that question's haystack; each
//! session is written as one Markdown file, the question is run in keyword / semantic / hybrid mode,
//! chunk hits are collapsed to sessions (best rank wins) and `Recall@k` asks: is any evidence session
//! in the top k sessions?
//!
//! ```text
//! cargo run --release --example longmemeval -- <dataset.json> [--stride N] [--limit N] [--out results.json]
//! ```
//!
//! Embeddings are cached across questions (sessions repeat), so the first pass is the slow one.

use plaesy::memory::config::SearchConfig;
use plaesy::memory::embeddings::{EmbeddingConfig, EmbeddingEngine};
use plaesy::memory::rerank::CrossEncoder;
use plaesy::memory::search::SearchEngine;
use plaesy::memory::storage::{Database, DatabaseConfig, SearchResult};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Deserialize)]
struct Turn {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct Question {
    question_id: String,
    question_type: String,
    question: String,
    answer_session_ids: Vec<String>,
    haystack_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<Turn>>,
}

const KS: [usize; 4] = [1, 3, 5, 10];
/// `rerank@w` = hybrid top 30 rescored by the cross-encoder with weight `w` (needs its model).
const MODES: [&str; 6] = [
    "keyword",
    "semantic",
    "hybrid",
    "rerank@0.3",
    "rerank@0.5",
    "rerank@0.7",
];

#[derive(Default, Clone)]
struct Tally {
    n: usize,
    hits: [usize; 4],
    rr: f64,
}

impl Tally {
    fn add(&mut self, rank: Option<usize>) {
        self.n += 1;
        if let Some(r) = rank {
            for (i, k) in KS.iter().enumerate() {
                if r < *k {
                    self.hits[i] += 1;
                }
            }
            self.rr += 1.0 / (r as f64 + 1.0);
        }
    }
    fn recall(&self, i: usize) -> f64 {
        self.hits[i] as f64 / self.n.max(1) as f64
    }
    fn mrr(&self) -> f64 {
        self.rr / self.n.max(1) as f64
    }
}

fn session_name(r: &SearchResult) -> String {
    let p = r.source_path.replace('\\', "/");
    let file = p.rsplit('/').next().unwrap_or(&p);
    file.strip_suffix(".md").unwrap_or(file).to_string()
}

/// 0-based rank of the first evidence session among distinct sessions, if any.
fn first_rank(results: &[SearchResult], answers: &[String]) -> Option<usize> {
    let mut seen: Vec<String> = Vec::new();
    for r in results {
        let s = session_name(r);
        if !seen.contains(&s) {
            seen.push(s);
        }
    }
    seen.iter().position(|s| answers.contains(s))
}

fn file_safe(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: longmemeval <dataset.json> [--stride N] [--limit N] [--out file.json]");
        std::process::exit(2);
    };
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let stride: usize = opt("--stride").and_then(|v| v.parse().ok()).unwrap_or(1);
    let limit: usize = opt("--limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let out = opt("--out");

    eprintln!("loading {path} ...");
    let raw = std::fs::read_to_string(path).expect("read dataset");
    let all: Vec<Question> = serde_json::from_str(&raw).expect("parse dataset");
    drop(raw);
    let questions: Vec<&Question> = all
        .iter()
        .filter(|q| !q.question_id.ends_with("_abs"))
        .step_by(stride.max(1))
        .take(limit)
        .collect();
    eprintln!(
        "{} questions (of {}; stride {stride})",
        questions.len(),
        all.len()
    );

    let emb = Arc::new(
        EmbeddingEngine::with_config(EmbeddingConfig {
            cache_size: 1_000_000,
            ..EmbeddingConfig::default()
        })
        .await
        .expect("ONNX model required: run scripts/install-model.sh"),
    );

    let ce = match CrossEncoder::resolve_model_dir(None) {
        None => {
            eprintln!("no cross-encoder model installed: rerank modes are skipped");
            None
        }
        Some(dir) => match CrossEncoder::load(&dir, 256) {
            Ok(m) => Some(Arc::new(m)),
            Err(e) => {
                eprintln!(
                    "cross-encoder in {} failed to load ({e}): rerank modes are skipped",
                    dir.display()
                );
                None
            }
        },
    };

    let mut overall: BTreeMap<&str, Tally> = BTreeMap::new();
    let mut by_type: BTreeMap<(String, &str), Tally> = BTreeMap::new();
    let started = Instant::now();

    for (qi, q) in questions.iter().enumerate() {
        let dir = tempfile::tempdir().expect("tmp");
        for (sid, turns) in q.haystack_session_ids.iter().zip(&q.haystack_sessions) {
            let mut body = String::new();
            for t in turns {
                body.push_str(&format!("**{}**: {}\n\n", t.role, t.content.trim()));
            }
            std::fs::write(dir.path().join(format!("{}.md", file_safe(sid))), body).expect("write");
        }
        let db = Arc::new(
            Database::new(DatabaseConfig {
                db_path: PathBuf::from(":memory:"),
                ..DatabaseConfig::default()
            })
            .await
            .expect("db"),
        );
        let mut engine = SearchEngine::new(db, Some(Arc::clone(&emb)), SearchConfig::default())
            .await
            .expect("engine");
        if let Some(ce) = &ce {
            engine = engine.with_reranker(Arc::clone(ce));
        }
        let mut hybrid_cache: Option<Vec<SearchResult>> = None;
        engine.index_directory(dir.path()).await.expect("index");

        let answers: Vec<String> = q.answer_session_ids.iter().map(|s| file_safe(s)).collect();
        for mode in MODES {
            let res = match mode {
                "keyword" => engine.keyword_search(&q.question, 60).await,
                "semantic" => engine.semantic_search(&q.question, 60).await,
                "hybrid" => engine.search(&q.question, 60).await,
                rerank => {
                    let Some(w) = rerank.strip_prefix("rerank@").and_then(|w| w.parse().ok())
                    else {
                        continue;
                    };
                    if ce.is_none() {
                        continue;
                    }
                    let base = match &hybrid_cache {
                        Some(b) => b.clone(),
                        None => continue,
                    };
                    engine.rerank_results(&q.question, base, 30, w).await
                }
            }
            .unwrap_or_default();
            if mode == "hybrid" {
                hybrid_cache = Some(res.clone());
            }
            let rank = first_rank(&res, &answers);
            overall.entry(mode).or_default().add(rank);
            by_type
                .entry((q.question_type.clone(), mode))
                .or_default()
                .add(rank);
        }
        if (qi + 1) % 10 == 0 || qi + 1 == questions.len() {
            let h = &overall["hybrid"];
            eprintln!(
                "[{}/{}] {:>6.0}s  hybrid R@5 so far {:.3}  (embed cache hit {:.0}%)",
                qi + 1,
                questions.len(),
                started.elapsed().as_secs_f64(),
                h.recall(2),
                emb.cache_hit_ratio() * 100.0
            );
        }
    }

    println!(
        "\nLongMemEval-S retrieval, session level, {} questions",
        questions.len()
    );
    println!(
        "{:<10} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "mode", "R@1", "R@3", "R@5", "R@10", "MRR"
    );
    let mut json = serde_json::Map::new();
    for mode in MODES {
        let Some(t) = overall.get(mode) else { continue };
        println!(
            "{:<10} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3}",
            mode,
            t.recall(0),
            t.recall(1),
            t.recall(2),
            t.recall(3),
            t.mrr()
        );
        json.insert(
            mode.to_string(),
            serde_json::json!({"n": t.n, "r1": t.recall(0), "r3": t.recall(1), "r5": t.recall(2), "r10": t.recall(3), "mrr": t.mrr()}),
        );
    }
    println!("\nR@5 by question type");
    let types: std::collections::BTreeSet<&String> = by_type.keys().map(|(t, _)| t).collect();
    for ty in types {
        let line: Vec<String> = MODES
            .iter()
            .filter_map(|m| {
                let t = by_type.get(&(ty.clone(), *m))?;
                Some(format!("{m} {:.3}", t.recall(2)))
            })
            .collect();
        let n = by_type[&(ty.clone(), "hybrid")].n;
        println!("  {ty:<26} n={n:<4} {}", line.join("  "));
    }
    if let Some(p) = out {
        std::fs::write(&p, serde_json::to_string_pretty(&json).expect("json")).expect("write out");
        eprintln!("wrote {p}");
    }
}
