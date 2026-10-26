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
//! cargo run --release --example longmemeval -- <dataset.json> [--stride N] [--limit N]
//!     [--workers N] [--checkpoint results.jsonl] [--out results.json]
//! ```
//!
//! - `--workers N` (or `LME_WORKERS`, default 2): questions run in parallel, each worker with its own
//!   embedding engine (ONNX sessions serialize internally, so one engine per worker is what gives
//!   parallelism). Memory grows with the number of workers.
//! - `--checkpoint file.jsonl`: each finished question is appended as one JSON line. A run that is
//!   stopped can be started again with the same file; questions already recorded are skipped.
//! - `MEMORY_MODEL_NAME` selects the embedding model; `LME_SKIP_RERANK=1` skips the cross-encoder.

use plaesy::memory::config::SearchConfig;
use plaesy::memory::embeddings::{EmbeddingConfig, EmbeddingEngine};
use plaesy::memory::rerank::CrossEncoder;
use plaesy::memory::search::SearchEngine;
use plaesy::memory::storage::{Database, DatabaseConfig, SearchResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;

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

/// One finished question: the rank of its first evidence session per mode (`None` = not found).
/// Modes that were not run (no cross-encoder) are absent.
#[derive(Serialize, Deserialize, Clone)]
struct Record {
    qid: String,
    qtype: String,
    ranks: BTreeMap<String, Option<usize>>,
}

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

/// Index one question's haystack in a fresh database and run every mode against it.
async fn run_question(
    q: &Question,
    emb: Arc<EmbeddingEngine>,
    ce: Option<Arc<CrossEncoder>>,
) -> BTreeMap<String, Option<usize>> {
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
    let mut engine = SearchEngine::new(db, Some(emb), SearchConfig::default())
        .await
        .expect("engine");
    if let Some(ce) = &ce {
        engine = engine.with_reranker(Arc::clone(ce));
    }
    let mut hybrid_cache: Option<Vec<SearchResult>> = None;
    engine.index_directory(dir.path()).await.expect("index");

    let answers: Vec<String> = q.answer_session_ids.iter().map(|s| file_safe(s)).collect();
    let mut ranks = BTreeMap::new();
    for mode in MODES {
        let res = match mode {
            "keyword" => engine.keyword_search(&q.question, 60).await,
            "semantic" => engine.semantic_search(&q.question, 60).await,
            "hybrid" => engine.search(&q.question, 60).await,
            rerank => {
                let Some(w) = rerank.strip_prefix("rerank@").and_then(|w| w.parse().ok()) else {
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
        ranks.insert(mode.to_string(), first_rank(&res, &answers));
    }
    ranks
}

fn make_config(model_name: &str) -> EmbeddingConfig {
    EmbeddingConfig {
        cache_size: 1_000_000,
        model_name: model_name.to_string(),
        ..EmbeddingConfig::default()
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!(
            "usage: longmemeval <dataset.json> [--stride N] [--limit N] [--workers N] \
             [--checkpoint file.jsonl] [--out file.json]"
        );
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
    let workers: usize = opt("--workers")
        .or_else(|| std::env::var("LME_WORKERS").ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(2)
        .max(1);
    let checkpoint = opt("--checkpoint");
    let out = opt("--out");

    eprintln!("loading {path} ...");
    let raw = std::fs::read_to_string(path).expect("read dataset");
    let all: Arc<Vec<Question>> = Arc::new(serde_json::from_str(&raw).expect("parse dataset"));
    drop(raw);
    let selected: Vec<usize> = (0..all.len())
        .filter(|&i| !all[i].question_id.ends_with("_abs"))
        .step_by(stride.max(1))
        .take(limit)
        .collect();
    let selected_ids: HashSet<String> = selected
        .iter()
        .map(|&i| all[i].question_id.clone())
        .collect();
    eprintln!(
        "{} questions (of {}; stride {stride}; workers {workers})",
        selected.len(),
        all.len()
    );

    // Records from an earlier run (checkpoint), restricted to the questions selected now.
    let mut records: Vec<Record> = Vec::new();
    if let Some(cp) = &checkpoint {
        if let Ok(text) = std::fs::read_to_string(cp) {
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                match serde_json::from_str::<Record>(line) {
                    Ok(r) if selected_ids.contains(&r.qid) => records.push(r),
                    Ok(_) => {}
                    Err(e) => eprintln!("skipping unreadable checkpoint line: {e}"),
                }
            }
        }
    }
    let done: HashSet<String> = records.iter().map(|r| r.qid.clone()).collect();
    if !done.is_empty() {
        eprintln!(
            "resuming: {} of {} already in checkpoint",
            done.len(),
            selected.len()
        );
    }
    let pending: VecDeque<usize> = selected
        .iter()
        .copied()
        .filter(|&i| !done.contains(&all[i].question_id))
        .collect();
    let total_pending = pending.len();

    // MEMORY_MODEL_NAME selects the model (same variable as the CLI); the name is printed so
    // every result file records which embedder produced it.
    let model_name = std::env::var("MEMORY_MODEL_NAME")
        .unwrap_or_else(|_| EmbeddingConfig::default().model_name);
    eprintln!("embedding model: {model_name}");

    // LME_SKIP_RERANK=1 skips the cross-encoder (and its three rerank modes per question).
    let skip_rerank = std::env::var_os("LME_SKIP_RERANK").is_some();
    let ce = match CrossEncoder::resolve_model_dir(None).filter(|_| !skip_rerank) {
        None => {
            eprintln!("rerank modes are skipped (no cross-encoder, or LME_SKIP_RERANK set)");
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

    let queue = Arc::new(Mutex::new(pending));
    let (tx, mut rx) = mpsc::unbounded_channel::<Record>();
    for _ in 0..workers {
        let emb = Arc::new(
            EmbeddingEngine::with_config(make_config(&model_name))
                .await
                .expect("ONNX model required: run scripts/install-model.sh"),
        );
        let (all, queue, tx, ce) = (Arc::clone(&all), Arc::clone(&queue), tx.clone(), ce.clone());
        tokio::spawn(async move {
            loop {
                let next = queue.lock().expect("queue lock").pop_front();
                let Some(i) = next else { break };
                let q = &all[i];
                let ranks = run_question(q, Arc::clone(&emb), ce.clone()).await;
                let record = Record {
                    qid: q.question_id.clone(),
                    qtype: q.question_type.clone(),
                    ranks,
                };
                if tx.send(record).is_err() {
                    break;
                }
            }
        });
    }
    drop(tx);

    let mut checkpoint_file = checkpoint.as_ref().map(|cp| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(cp)
            .expect("open checkpoint")
    });
    let started = Instant::now();
    let mut finished = 0usize;
    let mut hybrid_so_far = Tally::default();
    while let Some(record) = rx.recv().await {
        if let Some(f) = checkpoint_file.as_mut() {
            let line = serde_json::to_string(&record).expect("json");
            writeln!(f, "{line}").expect("write checkpoint");
            f.flush().expect("flush checkpoint");
        }
        hybrid_so_far.add(record.ranks.get("hybrid").copied().flatten());
        finished += 1;
        records.push(record);
        if finished.is_multiple_of(10) || finished == total_pending {
            eprintln!(
                "[{}/{}] {:>6.0}s  hybrid R@5 so far {:.3}",
                finished,
                total_pending,
                started.elapsed().as_secs_f64(),
                hybrid_so_far.recall(2)
            );
        }
    }

    // Aggregate over every recorded question (earlier runs included), once.
    let mut overall: BTreeMap<String, Tally> = BTreeMap::new();
    let mut by_type: BTreeMap<(String, String), Tally> = BTreeMap::new();
    for r in &records {
        for (mode, rank) in &r.ranks {
            overall.entry(mode.clone()).or_default().add(*rank);
            by_type
                .entry((r.qtype.clone(), mode.clone()))
                .or_default()
                .add(*rank);
        }
    }
    let n_questions = records.len();

    println!(
        "\nLongMemEval-S retrieval, session level, {n_questions} questions (model {model_name})"
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
    let types: BTreeSet<&String> = by_type.keys().map(|(t, _)| t).collect();
    for ty in types {
        let line: Vec<String> = MODES
            .iter()
            .filter_map(|m| {
                let t = by_type.get(&(ty.clone(), m.to_string()))?;
                Some(format!("{m} {:.3}", t.recall(2)))
            })
            .collect();
        let n = by_type
            .get(&(ty.clone(), "hybrid".to_string()))
            .map_or(0, |t| t.n);
        println!("  {ty:<26} n={n:<4} {}", line.join("  "));
    }
    if let Some(p) = out {
        std::fs::write(&p, serde_json::to_string_pretty(&json).expect("json")).expect("write out");
        eprintln!("wrote {p}");
    }
}
