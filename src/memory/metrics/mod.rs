//! Metrics and observability for the memory system.
//!
//! In-process metrics (lock-light atomics plus bounded latency windows) with
//! a Prometheus text-format renderer (`render_prometheus()`):
//! - **Latency Metrics**: Search, embedding, indexing operations
//! - **Throughput Metrics**: Documents/chunks indexed, queries executed
//! - **Index Health**: Size, coverage, corruption flag
//! - **Cache Metrics**: Embedding cache hit ratio
//! - **Error Metrics**: Error counts by type
//!
//! # Exported Metrics
//!
//! ```text
//! plaesy_memory_queries_total 120
//! plaesy_memory_search_latency_ms{quantile="0.95"} 4.2
//! plaesy_memory_errors_total{type="embedding_timeout"} 1
//! plaesy_memory_chunks_total 45000
//! plaesy_memory_documents_total 1200
//! plaesy_memory_coverage_percent 92.5
//! ```
//!
//! # Example
//!
//! ```ignore
//! use plaesy::memory::metrics::{Metrics, SearchType, Timer};
//!
//! let metrics = Metrics::new();
//! let timer = Timer::start();
//! // ... do work ...
//! metrics.record_search_latency(timer.elapsed(), SearchType::Hybrid);
//! println!("{}", metrics.render_prometheus());
//! ```

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Number of most-recent search latencies kept for percentile computation.
const LATENCY_WINDOW: usize = 10_000;

/// Metrics collector for the memory system.
pub struct Metrics {
    search_latencies_ms: Mutex<VecDeque<f64>>,
    search_by_type: Mutex<BTreeMap<&'static str, u64>>,
    queries: AtomicU64,
    vector_attempts: AtomicU64,
    vector_successes: AtomicU64,
    embedding_count: AtomicU64,
    embedding_total_us: AtomicU64,
    indexing_ops: Mutex<BTreeMap<String, (u64, f64)>>,
    documents_indexed: AtomicU64,
    chunks_indexed: AtomicU64,
    errors: Mutex<BTreeMap<String, u64>>,
    cache_hit_ratio_bits: AtomicU64,
    health: Mutex<Option<IndexHealth>>,
}

fn search_type_label(t: SearchType) -> &'static str {
    match t {
        SearchType::Semantic => "semantic",
        SearchType::Keyword => "keyword",
        SearchType::Hybrid => "hybrid",
    }
}

/// Nearest-rank percentile of an ascending-sorted slice (`p` in 0..=100).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

impl Metrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Metrics {
            search_latencies_ms: Mutex::new(VecDeque::new()),
            search_by_type: Mutex::new(BTreeMap::new()),
            queries: AtomicU64::new(0),
            vector_attempts: AtomicU64::new(0),
            vector_successes: AtomicU64::new(0),
            embedding_count: AtomicU64::new(0),
            embedding_total_us: AtomicU64::new(0),
            indexing_ops: Mutex::new(BTreeMap::new()),
            documents_indexed: AtomicU64::new(0),
            chunks_indexed: AtomicU64::new(0),
            errors: Mutex::new(BTreeMap::new()),
            cache_hit_ratio_bits: AtomicU64::new(0f64.to_bits()),
            health: Mutex::new(None),
        }
    }

    /// Record a search operation latency.
    pub fn record_search_latency(&self, latency: Duration, search_type: SearchType) {
        let mut window = self.search_latencies_ms.lock();
        if window.len() == LATENCY_WINDOW {
            window.pop_front();
        }
        window.push_back(latency.as_secs_f64() * 1000.0);
        *self
            .search_by_type
            .lock()
            .entry(search_type_label(search_type))
            .or_insert(0) += 1;
    }

    /// Record an embedding operation latency.
    pub fn record_embedding_latency(&self, latency: Duration) {
        self.embedding_count.fetch_add(1, Ordering::Relaxed);
        self.embedding_total_us
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record an indexing operation latency.
    pub fn record_indexing_latency(&self, latency: Duration, operation: &str) {
        let mut ops = self.indexing_ops.lock();
        let entry = ops.entry(operation.to_string()).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += latency.as_secs_f64();
    }

    /// Record a document indexing event.
    pub fn record_document_indexed(&self, chunk_count: usize) {
        self.documents_indexed.fetch_add(1, Ordering::Relaxed);
        self.chunks_indexed
            .fetch_add(chunk_count as u64, Ordering::Relaxed);
    }

    /// Record a search query.
    pub fn record_search_query(&self) {
        self.queries.fetch_add(1, Ordering::Relaxed);
    }

    /// Record the outcome of a vector (semantic) search attempt.
    pub fn record_vector_search(&self, success: bool) {
        self.vector_attempts.fetch_add(1, Ordering::Relaxed);
        if success {
            self.vector_successes.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record an error occurrence.
    pub fn record_error(&self, error_type: &str) {
        *self
            .errors
            .lock()
            .entry(error_type.to_string())
            .or_insert(0) += 1;
    }

    /// Record the current embedding cache hit ratio (0-1).
    pub fn set_embedding_cache_hit_ratio(&self, ratio: f64) {
        self.cache_hit_ratio_bits
            .store(ratio.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    /// Update index health metrics.
    pub fn update_index_health(&self, health: &IndexHealth) {
        *self.health.lock() = Some(health.clone());
    }

    /// Get current metrics snapshot for diagnostic output.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let mut sorted: Vec<f64> = self.search_latencies_ms.lock().iter().copied().collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let avg = if sorted.is_empty() {
            0.0
        } else {
            sorted.iter().sum::<f64>() / sorted.len() as f64
        };
        let attempts = self.vector_attempts.load(Ordering::Relaxed);
        let successes = self.vector_successes.load(Ordering::Relaxed);
        MetricsSnapshot {
            total_queries: self.queries.load(Ordering::Relaxed),
            avg_search_latency_ms: avg,
            p95_search_latency_ms: percentile(&sorted, 95.0),
            p99_search_latency_ms: percentile(&sorted, 99.0),
            vector_search_success_rate: if attempts == 0 {
                0.0
            } else {
                successes as f64 / attempts as f64
            },
            embedding_cache_hit_ratio: f64::from_bits(
                self.cache_hit_ratio_bits.load(Ordering::Relaxed),
            ),
            total_errors: self.errors.lock().values().sum(),
            index_health: self.health.lock().clone().unwrap_or_else(|| IndexHealth {
                documents_total: self.documents_indexed.load(Ordering::Relaxed),
                chunks_total: self.chunks_indexed.load(Ordering::Relaxed),
                index_size_bytes: 0,
                coverage_percent: 0.0,
                last_indexed: String::new(),
                is_corrupted: false,
            }),
        }
    }

    /// Render all metrics in Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        use std::fmt::Write;
        let s = self.snapshot();
        let mut out = String::new();
        let mut line = |name: &str, labels: &str, value: f64| {
            let _ = writeln!(out, "plaesy_memory_{name}{labels} {value}");
        };
        line("queries_total", "", s.total_queries as f64);
        line(
            "search_latency_ms",
            "{quantile=\"avg\"}",
            s.avg_search_latency_ms,
        );
        line(
            "search_latency_ms",
            "{quantile=\"0.95\"}",
            s.p95_search_latency_ms,
        );
        line(
            "search_latency_ms",
            "{quantile=\"0.99\"}",
            s.p99_search_latency_ms,
        );
        line(
            "vector_search_success_rate",
            "",
            s.vector_search_success_rate,
        );
        line("embedding_cache_hit_ratio", "", s.embedding_cache_hit_ratio);
        line(
            "embedding_operations_total",
            "",
            self.embedding_count.load(Ordering::Relaxed) as f64,
        );
        line("documents_total", "", s.index_health.documents_total as f64);
        line("chunks_total", "", s.index_health.chunks_total as f64);
        line(
            "index_size_bytes",
            "",
            s.index_health.index_size_bytes as f64,
        );
        line("coverage_percent", "", s.index_health.coverage_percent);
        line(
            "index_corrupted",
            "",
            if s.index_health.is_corrupted {
                1.0
            } else {
                0.0
            },
        );
        for (t, n) in self.search_by_type.lock().iter() {
            line(
                "searches_total",
                &format!("{{search_type=\"{t}\"}}"),
                *n as f64,
            );
        }
        for (op, (n, secs)) in self.indexing_ops.lock().iter() {
            line(
                "index_operations_total",
                &format!("{{operation=\"{op}\"}}"),
                *n as f64,
            );
            line(
                "index_duration_seconds_sum",
                &format!("{{operation=\"{op}\"}}"),
                *secs,
            );
        }
        for (ty, n) in self.errors.lock().iter() {
            line("errors_total", &format!("{{type=\"{ty}\"}}"), *n as f64);
        }
        out
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of search operation (for latency tracking).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchType {
    /// Vector similarity search.
    Semantic,

    /// Full-text keyword search.
    Keyword,

    /// Hybrid (vector + keyword).
    Hybrid,
}

/// Index health metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexHealth {
    /// Total number of documents.
    pub documents_total: u64,

    /// Total number of chunks.
    pub chunks_total: u64,

    /// Index size in bytes.
    pub index_size_bytes: u64,

    /// Coverage percentage (0-100).
    pub coverage_percent: f64,

    /// Last indexed timestamp (ISO 8601).
    pub last_indexed: String,

    /// Corruption detected?
    pub is_corrupted: bool,
}

/// Metrics snapshot for diagnostic output.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Total search queries executed.
    pub total_queries: u64,

    /// Average search latency (milliseconds).
    pub avg_search_latency_ms: f64,

    /// P95 search latency (milliseconds).
    pub p95_search_latency_ms: f64,

    /// P99 search latency (milliseconds).
    pub p99_search_latency_ms: f64,

    /// Vector search success rate (0-1).
    pub vector_search_success_rate: f64,

    /// Embedding cache hit ratio (0-1).
    pub embedding_cache_hit_ratio: f64,

    /// Total errors recorded.
    pub total_errors: u64,

    /// Index health.
    pub index_health: IndexHealth,
}

/// Timer for measuring operation latency.
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Create a new timer.
    pub fn start() -> Self {
        Timer {
            start: Instant::now(),
        }
    }

    /// Get elapsed time since timer started.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get elapsed milliseconds.
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_elapsed() {
        let timer = Timer::start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn test_timer_elapsed_ms() {
        let timer = Timer::start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed_ms = timer.elapsed_ms();
        assert!(elapsed_ms >= 10.0);
    }

    #[test]
    fn test_metrics_default() {
        let snap = Metrics::default().snapshot();
        assert_eq!(snap.total_queries, 0);
        assert_eq!(snap.p95_search_latency_ms, 0.0);
        assert_eq!(snap.vector_search_success_rate, 0.0);
    }

    #[test]
    fn test_percentile_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&v, 95.0), 95.0);
        assert_eq!(percentile(&v, 99.0), 99.0);
        assert_eq!(percentile(&v, 100.0), 100.0);
        assert_eq!(percentile(&[7.0], 50.0), 7.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn test_snapshot_reflects_records() {
        let m = Metrics::new();
        for ms in 1..=100u64 {
            m.record_search_query();
            m.record_search_latency(Duration::from_millis(ms), SearchType::Hybrid);
        }
        m.record_vector_search(true);
        m.record_vector_search(false);
        m.record_error("embedding_timeout");
        m.record_error("embedding_timeout");
        m.record_document_indexed(5);
        m.set_embedding_cache_hit_ratio(0.25);
        let s = m.snapshot();
        assert_eq!(s.total_queries, 100);
        assert!((s.avg_search_latency_ms - 50.5).abs() < 0.5);
        assert!((s.p95_search_latency_ms - 95.0).abs() < 1.0);
        assert_eq!(s.vector_search_success_rate, 0.5);
        assert_eq!(s.total_errors, 2);
        assert_eq!(s.embedding_cache_hit_ratio, 0.25);
        assert_eq!(s.index_health.chunks_total, 5);
    }

    #[test]
    fn test_latency_window_is_bounded() {
        let m = Metrics::new();
        for _ in 0..(LATENCY_WINDOW + 50) {
            m.record_search_latency(Duration::from_millis(1), SearchType::Keyword);
        }
        assert_eq!(m.search_latencies_ms.lock().len(), LATENCY_WINDOW);
    }

    #[test]
    fn test_render_prometheus() {
        let m = Metrics::new();
        m.record_search_query();
        m.record_error("x");
        m.record_indexing_latency(Duration::from_millis(500), "write_chunks");
        let text = m.render_prometheus();
        assert!(text.contains("plaesy_memory_queries_total 1\n"));
        assert!(text.contains("plaesy_memory_errors_total{type=\"x\"} 1\n"));
        assert!(
            text.contains("plaesy_memory_index_operations_total{operation=\"write_chunks\"} 1\n")
        );
    }
}
