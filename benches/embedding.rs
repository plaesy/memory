//! Criterion benchmark: embedding generation.
//!
//! Always measures the hashing backend; additionally measures the ONNX model
//! (single text and batch of 32) when `model.onnx` + `vocab.txt` are installed
//! (see `plaesy::memory::embeddings` docs). Target: ONNX p95 < 150 ms/text.
//! Run: `cargo bench --bench embedding`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use plaesy::memory::embeddings::{EmbeddingConfig, EmbeddingEngine};
use tokio::runtime::Runtime;

const SENTENCE: &str = "Hierarchical navigable small world graphs give approximate nearest neighbor search \
                        with logarithmic complexity, which is why memory retrieval can stay fast offline.";

fn texts(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{SENTENCE} variant {i}")).collect()
}

fn bench_embeddings(c: &mut Criterion) {
    let rt = Runtime::new().expect("runtime");

    let hashing = EmbeddingEngine::hashing(384).expect("hashing engine");
    c.bench_function("hashing_embed_single", |b| {
        // Unique text each iteration so the LRU cache never short-circuits the work.
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            let t = format!("{SENTENCE} {i}");
            black_box(rt.block_on(hashing.embed(&t)).expect("embed"))
        })
    });

    match rt.block_on(EmbeddingEngine::with_config(EmbeddingConfig::default())) {
        Ok(onnx) => {
            let mut i = 0u64;
            c.bench_function("onnx_embed_single", |b| {
                b.iter(|| {
                    i += 1;
                    let t = format!("{SENTENCE} {i}");
                    black_box(rt.block_on(onnx.embed(&t)).expect("embed"))
                })
            });
            let mut round = 0usize;
            c.bench_function("onnx_embed_batch32", |b| {
                b.iter(|| {
                    round += 1;
                    let owned: Vec<String> = texts(32)
                        .into_iter()
                        .map(|t| format!("{t} r{round}"))
                        .collect();
                    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
                    black_box(rt.block_on(onnx.embed_batch(&refs)).expect("batch"))
                })
            });
        }
        Err(e) => eprintln!("ONNX benchmarks skipped: {e}"),
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_embeddings
}
criterion_main!(benches);
