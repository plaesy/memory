//! Criterion benchmark: HNSW vector search (real `HnswIndex`, 384-dim, cosine).
//!
//! Measures insert throughput and k-NN search latency at several index sizes.
//! Target (docs/ARCHITECTURE.md): p50 < 1 ms for top-10 on 100K vectors.
//! Run: `cargo bench --bench hnsw_search`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use plaesy::memory::storage::{HnswConfig, HnswIndex};

const DIM: usize = 384;

/// Deterministic pseudo-random unit-ish vector (xorshift), no `rand` dependency.
fn vector(seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIM)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn build(n: u64) -> HnswIndex {
    let mut idx = HnswIndex::with_dimensions(HnswConfig::default(), DIM).expect("index");
    for i in 0..n {
        idx.insert(i, &vector(i)).expect("insert");
    }
    idx
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_top10");
    for n in [1_000u64, 10_000, 50_000] {
        let idx = build(n);
        let query = vector(n + 7);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(idx.search(black_box(&query), 10).expect("search")))
        });
    }
    group.finish();
}

fn bench_insert(c: &mut Criterion) {
    let vectors: Vec<Vec<f32>> = (0..1_000).map(vector).collect();
    c.bench_function("hnsw_insert_1000", |b| {
        b.iter(|| {
            let mut idx = HnswIndex::with_dimensions(HnswConfig::default(), DIM).expect("index");
            for (i, v) in vectors.iter().enumerate() {
                idx.insert(i as u64, v).expect("insert");
            }
            black_box(idx.len())
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30);
    targets = bench_search, bench_insert
}
criterion_main!(benches);
