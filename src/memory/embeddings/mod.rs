//! Embedding subsystem for semantic search.
//!
//! Handles ONNX model loading, batch inference, and vector caching:
//! - **Model Loading**: Load `all-MiniLM-L6-v2` ONNX model (22MB quantised / 90MB fp32, 384-dim)
//! - **Tokenizing**: BERT WordPiece tokenizer built from the model's `vocab.txt`
//! - **Inference**: Batch text→vectors conversion via ONNX Runtime, mean-pooled and L2-normalised
//! - **Caching**: LRU cache to avoid re-embedding identical texts
//! - **Offline**: no network access at runtime; model files are read from disk
//!
//! # Model files
//!
//! The engine needs a directory containing `model.onnx` and `vocab.txt`
//! (e.g. from `sentence-transformers/all-MiniLM-L6-v2`, `onnx/model.onnx`).
//! Lookup order: `EmbeddingConfig::model_dir`, the `PLAESY_MODEL_DIR`
//! environment variable, then `~/.plaesy-models/<model_name>`.
//!
//! If no model is found, [`EmbeddingEngine::new`] fails with
//! `EmbeddingModelError` and callers fall back to keyword search (the
//! search engine does this automatically). The lexical
//! [`EmbeddingEngine::hashing`] backend is an explicit opt-in, never a silent
//! substitute: it is a deterministic bag-of-words feature hash, not a neural model.
//!
//! # Example
//!
//! ```ignore
//! use plaesy::memory::embeddings::EmbeddingEngine;
//!
//! let engine = EmbeddingEngine::new().await?;
//! let vectors = engine.embed_batch(&["semantic search", "vector database"]).await?;
//! assert_eq!(vectors[0].len(), 384);
//! ```

use crate::memory::error::{MemoryError, Result};
use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Configuration for embedding engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model name (e.g., "all-MiniLM-L6-v2").
    pub model_name: String,

    /// Dimension of output vectors.
    pub dimension: usize,

    /// Maximum batch size for inference.
    pub max_batch_size: usize,

    /// Embedding cache size (number of entries).
    pub cache_size: usize,

    /// Timeout for embedding operation (seconds).
    pub timeout_secs: u64,

    /// Directory holding `model.onnx` and `vocab.txt` (see module docs for lookup order).
    #[serde(default)]
    pub model_dir: Option<PathBuf>,

    /// Maximum token sequence length (longer inputs are truncated).
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// Text prepended to every query before embedding. `None` takes the model's default
    /// (`"query: "` for E5 models, nothing otherwise).
    #[serde(default)]
    pub query_prefix: Option<String>,

    /// Text prepended to every indexed passage. `None` takes the model's default
    /// (`"passage: "` for E5 models, nothing otherwise).
    #[serde(default)]
    pub passage_prefix: Option<String>,
}

fn default_max_seq_len() -> usize {
    256
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            model_name: "all-MiniLM-L6-v2".to_string(),
            dimension: 384,
            max_batch_size: 32,
            cache_size: 10_000,
            timeout_secs: 30,
            model_dir: None,
            max_seq_len: default_max_seq_len(),
            query_prefix: None,
            passage_prefix: None,
        }
    }
}

/// `(query, passage)` prefixes a model was trained with. E5 models (`multilingual-e5-*`, `e5-*`)
/// expect `query: ` and `passage: `; the other supported models expect none.
pub fn default_prefixes(model_name: &str) -> (&'static str, &'static str) {
    if model_name.to_ascii_lowercase().contains("e5") {
        ("query: ", "passage: ")
    } else {
        ("", "")
    }
}

/// Embedding vector (f32, dimension-specific).
pub type EmbeddingVector = Vec<f32>;

/// Which implementation produces the vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingBackend {
    /// ONNX Runtime neural model (semantic).
    Onnx,

    /// Deterministic feature hashing of words/bigrams (lexical only, no model).
    Hashing,
}

// ---------------------------------------------------------------------------
// WordPiece tokenizer (BERT uncased)
// ---------------------------------------------------------------------------

/// BERT-uncased WordPiece tokenizer.
///
/// Lower-cases, splits on whitespace/punctuation, isolates CJK characters,
/// then greedy longest-match WordPiece. Accent stripping is not applied.
pub(crate) struct WordPiece {
    vocab: HashMap<String, i64>,
    unk: i64,
    cls: i64,
    sep: i64,
    pub(crate) pad: i64,
}

const MAX_WORD_CHARS: usize = 100;

impl WordPiece {
    pub(crate) fn from_vocab_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| MemoryError::IoError {
            path: path.display().to_string(),
            source: e,
        })?;
        let vocab: HashMap<String, i64> = raw
            .lines()
            .enumerate()
            .map(|(i, l)| (l.trim_end().to_string(), i as i64))
            .collect();
        let get = |t: &str| {
            vocab.get(t).copied().ok_or_else(|| {
                MemoryError::EmbeddingModelError(format!(
                    "vocab {} is missing special token {t}",
                    path.display()
                ))
            })
        };
        Ok(WordPiece {
            unk: get("[UNK]")?,
            cls: get("[CLS]")?,
            sep: get("[SEP]")?,
            pad: get("[PAD]")?,
            vocab,
        })
    }

    fn is_cjk(c: char) -> bool {
        matches!(c as u32,
            0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
    }

    /// Basic tokenization: lowercase, split into words and punctuation.
    fn basic(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let flush = |cur: &mut String, out: &mut Vec<String>| {
            if !cur.is_empty() {
                out.push(std::mem::take(cur));
            }
        };
        for c in text.chars() {
            if c.is_control() && !c.is_whitespace() {
                continue;
            }
            if c.is_whitespace() {
                flush(&mut cur, &mut out);
            } else if c.is_ascii_punctuation() || Self::is_cjk(c) {
                flush(&mut cur, &mut out);
                out.push(c.to_lowercase().collect());
            } else {
                cur.extend(c.to_lowercase());
            }
        }
        flush(&mut cur, &mut out);
        out
    }

    fn wordpiece(&self, word: &str, ids: &mut Vec<i64>) {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > MAX_WORD_CHARS {
            ids.push(self.unk);
            return;
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while start < end {
                let mut sub: String = chars[start..end].iter().collect();
                if start > 0 {
                    sub.insert_str(0, "##");
                }
                if let Some(&id) = self.vocab.get(&sub) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => {
                    pieces.push(id);
                    start = end;
                }
                None => {
                    ids.push(self.unk);
                    return;
                }
            }
        }
        ids.extend(pieces);
    }

    /// Encode a (query, document) pair as `[CLS] query [SEP] document [SEP]` for cross-encoders.
    ///
    /// Returns `(input_ids, token_type_ids)` of equal length <= `max_len`. The query keeps at
    /// most a third of the budget; the document takes the rest and is truncated.
    pub(crate) fn encode_pair(
        &self,
        query: &str,
        doc: &str,
        max_len: usize,
    ) -> (Vec<i64>, Vec<i64>) {
        let max_len = max_len.max(8);
        let q_budget = max_len / 3;
        let mut q: Vec<i64> = Vec::new();
        for w in Self::basic(query) {
            self.wordpiece(&w, &mut q);
            if q.len() >= q_budget {
                break;
            }
        }
        q.truncate(q_budget);
        let d_budget = max_len - q.len() - 3;
        let mut d: Vec<i64> = Vec::new();
        for w in Self::basic(doc) {
            self.wordpiece(&w, &mut d);
            if d.len() >= d_budget {
                break;
            }
        }
        d.truncate(d_budget);
        let mut ids = Vec::with_capacity(q.len() + d.len() + 3);
        ids.push(self.cls);
        ids.extend(&q);
        ids.push(self.sep);
        let first_len = ids.len();
        ids.extend(&d);
        ids.push(self.sep);
        let mut types = vec![0i64; first_len];
        types.resize(ids.len(), 1);
        (ids, types)
    }

    /// Encode to `[CLS] tokens… [SEP]`, truncated to `max_len`.
    fn encode(&self, text: &str, max_len: usize) -> Vec<i64> {
        let mut ids = vec![self.cls];
        for w in Self::basic(text) {
            self.wordpiece(&w, &mut ids);
            if ids.len() >= max_len - 1 {
                break;
            }
        }
        ids.truncate(max_len - 1);
        ids.push(self.sep);
        ids
    }
}

// ---------------------------------------------------------------------------
// ONNX model
// ---------------------------------------------------------------------------

struct OnnxModel {
    /// Keeps the ONNX Runtime environment alive for the session's lifetime.
    _env: Arc<ort::environment::Environment>,
    session: Mutex<ort::session::Session>,
    /// BERT WordPiece (`vocab.txt`), used when the model dir has no `tokenizer.json`.
    wordpiece: Option<WordPiece>,
    /// Hugging Face tokenizer (`tokenizer.json`, e.g. XLM-R SentencePiece for multilingual E5).
    hf: Option<tokenizers::Tokenizer>,
    pad: i64,
    needs_token_type_ids: bool,
    max_seq_len: usize,
    dimension: usize,
}

/// The process-wide ONNX Runtime environment.
///
/// ONNX Runtime allows exactly one environment per process, so every model (embedding model,
/// cross-encoder, repeated engine construction in tests) shares this one.
pub(crate) fn ort_environment() -> Result<Arc<ort::environment::Environment>> {
    static ENV: std::sync::OnceLock<
        std::result::Result<Arc<ort::environment::Environment>, String>,
    > = std::sync::OnceLock::new();
    ENV.get_or_init(|| {
        ort::init()
            .with_name("plaesy-memory")
            .build()
            .map(Arc::new)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(model_err)
}

fn model_err(e: impl std::fmt::Display) -> MemoryError {
    MemoryError::EmbeddingModelError(e.to_string())
}

impl OnnxModel {
    fn load(dir: &Path, config: &EmbeddingConfig) -> Result<Self> {
        let env = ort_environment()?;
        let session = ort::session::Session::builder(&env)
            .map_err(model_err)?
            .commit_from_file(dir.join("model.onnx"))
            .map_err(model_err)?;
        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        let (wordpiece, hf, pad) = if dir.join("tokenizer.json").is_file() {
            let tok =
                tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(model_err)?;
            let pad = tok.token_to_id("<pad>").unwrap_or(1) as i64;
            (None, Some(tok), pad)
        } else {
            let wp = WordPiece::from_vocab_file(&dir.join("vocab.txt"))?;
            let pad = wp.pad;
            (Some(wp), None, pad)
        };
        Ok(OnnxModel {
            _env: env,
            session: Mutex::new(session),
            wordpiece,
            hf,
            pad,
            needs_token_type_ids,
            max_seq_len: config.max_seq_len.max(3),
            dimension: config.dimension,
        })
    }

    /// Token ids for one text, truncated to `max_seq_len`.
    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        if let Some(hf) = &self.hf {
            let enc = hf.encode(text, true).map_err(model_err)?;
            return Ok(enc
                .get_ids()
                .iter()
                .take(self.max_seq_len)
                .map(|&i| i as i64)
                .collect());
        }
        Ok(self
            .wordpiece
            .as_ref()
            .map(|wp| wp.encode(text, self.max_seq_len))
            .unwrap_or_default())
    }

    fn infer(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        let batch = texts.len();
        let encoded: Vec<Vec<i64>> = texts
            .iter()
            .map(|t| self.encode(t))
            .collect::<Result<_>>()?;
        let seq = encoded.iter().map(Vec::len).max().unwrap_or(0);
        let mut ids = vec![self.pad; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        for (b, e) in encoded.iter().enumerate() {
            ids[b * seq..b * seq + e.len()].copy_from_slice(e);
            mask[b * seq..b * seq + e.len()].fill(1);
        }
        let infer_err = |e: &dyn std::fmt::Display| MemoryError::EmbeddingInferenceError {
            batch_size: batch,
            reason: e.to_string(),
        };
        let shape = [batch as i64, seq as i64];
        let ids_t = ort::value::Tensor::from_array((shape, ids)).map_err(|e| infer_err(&e))?;
        let mask_t =
            ort::value::Tensor::from_array((shape, mask.clone())).map_err(|e| infer_err(&e))?;
        let mut inputs = ort::inputs! {
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
        };
        if self.needs_token_type_ids {
            let tt = ort::value::Tensor::from_array((shape, vec![0i64; batch * seq]))
                .map_err(|e| infer_err(&e))?;
            inputs.push(("token_type_ids".into(), tt.into()));
        }
        let mut session = self.session.lock();
        let outputs = session.run(inputs).map_err(|e| infer_err(&e))?;
        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| infer_err(&e))?;
        let dims: Vec<i64> = out_shape.iter().copied().collect();
        if dims.len() != 3 || dims[0] as usize != batch || dims[1] as usize != seq {
            return Err(infer_err(&format!("unexpected output shape {dims:?}")));
        }
        let hidden = dims[2] as usize;
        if hidden != self.dimension {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dimension,
                got: hidden,
            });
        }
        // Mean pooling over attended tokens, then L2 normalise.
        let mut result = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut v = vec![0f32; hidden];
            let mut count = 0f32;
            for s in 0..seq {
                if mask[b * seq + s] == 0 {
                    continue;
                }
                count += 1.0;
                let row = &data[(b * seq + s) * hidden..(b * seq + s + 1) * hidden];
                for (acc, x) in v.iter_mut().zip(row) {
                    *acc += x;
                }
            }
            let count = count.max(1.0);
            v.iter_mut().for_each(|x| *x /= count);
            l2_normalize(&mut v);
            result.push(v);
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Hashing backend
// ---------------------------------------------------------------------------

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Signed feature hashing of lower-cased word unigrams and bigrams.
fn hash_embed(text: &str, dim: usize) -> EmbeddingVector {
    let words: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    let mut v = vec![0f32; dim];
    let mut add = |feature: &str, weight: f32| {
        let h = fnv1a(feature.as_bytes());
        let idx = (h % dim as u64) as usize;
        let sign = if (h >> 63) & 1 == 0 { 1.0 } else { -1.0 };
        v[idx] += sign * weight;
    };
    for w in &words {
        add(w, 1.0);
    }
    for pair in words.windows(2) {
        add(&format!("{} {}", pair[0], pair[1]), 0.5);
    }
    l2_normalize(&mut v);
    v
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

enum Backend {
    Onnx(Arc<OnnxModel>),
    Hashing,
}

/// Embedding engine: loads ONNX model and generates embeddings.
pub struct EmbeddingEngine {
    config: EmbeddingConfig,
    backend: Backend,
    cache: Mutex<LruCache<String, EmbeddingVector>>,
    hits: AtomicU64,
    misses: AtomicU64,
    query_prefix: String,
    passage_prefix: String,
}

impl EmbeddingEngine {
    /// Create a new embedding engine with default configuration.
    ///
    /// # Errors
    ///
    /// - `EmbeddingModelError` if model files cannot be found or loaded
    pub async fn new() -> Result<Self> {
        Self::with_config(EmbeddingConfig::default()).await
    }

    fn resolve_model_dir(config: &EmbeddingConfig) -> Option<PathBuf> {
        let mut candidates = Vec::new();
        if let Some(d) = &config.model_dir {
            candidates.push(d.clone());
        }
        if let Some(d) = std::env::var_os("PLAESY_MODEL_DIR") {
            candidates.push(PathBuf::from(d));
        }
        if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            candidates.push(
                PathBuf::from(home)
                    .join(".plaesy-models")
                    .join(&config.model_name),
            );
        }
        candidates.into_iter().find(|d| {
            d.join("model.onnx").is_file()
                && (d.join("vocab.txt").is_file() || d.join("tokenizer.json").is_file())
        })
    }

    fn build(config: EmbeddingConfig, backend: Backend) -> Result<Self> {
        let cap = NonZeroUsize::new(config.cache_size.max(1)).ok_or_else(|| {
            MemoryError::ConfigError("embedding cache_size must be > 0".to_string())
        })?;
        let (default_query, default_passage) = default_prefixes(&config.model_name);
        let query_prefix = config
            .query_prefix
            .clone()
            .unwrap_or_else(|| default_query.to_string());
        let passage_prefix = config
            .passage_prefix
            .clone()
            .unwrap_or_else(|| default_passage.to_string());
        Ok(EmbeddingEngine {
            config,
            backend,
            cache: Mutex::new(LruCache::new(cap)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            query_prefix,
            passage_prefix,
        })
    }

    /// Create a new embedding engine with custom configuration (ONNX backend).
    ///
    /// # Errors
    ///
    /// - `EmbeddingModelError` if `model.onnx` / `vocab.txt` are not found or fail to load
    /// - `ConfigError` if `dimension` or `max_batch_size` is zero
    pub async fn with_config(config: EmbeddingConfig) -> Result<Self> {
        if config.dimension == 0 || config.max_batch_size == 0 {
            return Err(MemoryError::ConfigError(
                "embedding dimension and max_batch_size must be > 0".to_string(),
            ));
        }
        let dir = Self::resolve_model_dir(&config).ok_or_else(|| {
            MemoryError::EmbeddingModelError(format!(
                "no ONNX model found for '{}': expected model.onnx and vocab.txt in \
                 config.model_dir, $PLAESY_MODEL_DIR, or ~/.plaesy-models/{}",
                config.model_name, config.model_name
            ))
        })?;
        let cfg = config.clone();
        let model = tokio::task::spawn_blocking(move || OnnxModel::load(&dir, &cfg))
            .await
            .map_err(|e| MemoryError::InternalError(e.to_string()))??;
        Self::build(config, Backend::Onnx(Arc::new(model)))
    }

    /// Create a lexical feature-hashing engine (no model required).
    ///
    /// Deterministic and offline, but only captures word overlap, not meaning.
    /// Intended for tests and degraded operation; check [`EmbeddingEngine::backend`].
    pub fn hashing(dimension: usize) -> Result<Self> {
        let config = EmbeddingConfig {
            model_name: "feature-hashing".to_string(),
            dimension,
            ..EmbeddingConfig::default()
        };
        if dimension == 0 {
            return Err(MemoryError::ConfigError(
                "embedding dimension must be > 0".to_string(),
            ));
        }
        Self::build(config, Backend::Hashing)
    }

    /// Which backend produces vectors.
    pub fn backend(&self) -> EmbeddingBackend {
        match self.backend {
            Backend::Onnx(_) => EmbeddingBackend::Onnx,
            Backend::Hashing => EmbeddingBackend::Hashing,
        }
    }

    /// Generate embeddings for a single text.
    ///
    /// # Errors
    ///
    /// - `EmbeddingInferenceError` if inference fails
    /// - `EmbeddingTimeout` if operation exceeds timeout
    pub async fn embed(&self, text: &str) -> Result<EmbeddingVector> {
        let prefixed = vec![format!("{}{}", self.query_prefix, text)];
        self.embed_prefixed(prefixed).await.and_then(|mut v| {
            v.pop().ok_or_else(|| {
                MemoryError::InternalError("unexpected empty embedding batch result".to_string())
            })
        })
    }

    /// Generate embeddings for a batch of texts.
    ///
    /// Results are in input order. Cached texts are not re-embedded; the rest
    /// are run in sub-batches of at most `config.max_batch_size`.
    ///
    /// # Errors
    ///
    /// - `EmbeddingInferenceError` if inference fails for any item
    /// - `EmbeddingTimeout` if operation exceeds timeout
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingVector>> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{}", self.passage_prefix, t))
            .collect();
        self.embed_prefixed(prefixed).await
    }

    /// Embed texts that already carry their model prefix; the cache is keyed by that text.
    async fn embed_prefixed(&self, texts: Vec<String>) -> Result<Vec<EmbeddingVector>> {
        let mut results: Vec<Option<EmbeddingVector>> = vec![None; texts.len()];
        let mut missing: Vec<(usize, String)> = Vec::new();
        {
            let mut cache = self.cache.lock();
            for (i, t) in texts.iter().enumerate() {
                if let Some(v) = cache.get(t.as_str()) {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    results[i] = Some(v.clone());
                } else {
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    missing.push((i, t.clone()));
                }
            }
        }

        // Similar lengths per batch => less padding => less wasted ONNX compute.
        missing.sort_by_key(|(_, t)| t.len());
        for group in missing.chunks(self.config.max_batch_size) {
            let batch: Vec<String> = group.iter().map(|(_, t)| t.clone()).collect();
            let vectors = self.run_backend(batch).await?;
            let mut cache = self.cache.lock();
            for ((i, text), v) in group.iter().zip(vectors) {
                cache.put(text.clone(), v.clone());
                results[*i] = Some(v);
            }
        }

        results
            .into_iter()
            .map(|v| v.ok_or_else(|| MemoryError::InternalError("embedding missing".to_string())))
            .collect()
    }

    async fn run_backend(&self, batch: Vec<String>) -> Result<Vec<EmbeddingVector>> {
        match &self.backend {
            Backend::Hashing => Ok(batch
                .iter()
                .map(|t| hash_embed(t, self.config.dimension))
                .collect()),
            Backend::Onnx(model) => {
                let model = Arc::clone(model);
                let size = batch.len();
                let job = tokio::task::spawn_blocking(move || model.infer(&batch));
                match tokio::time::timeout(Duration::from_secs(self.config.timeout_secs), job).await
                {
                    Err(_) => Err(MemoryError::EmbeddingTimeout {
                        timeout_secs: self.config.timeout_secs,
                    }),
                    Ok(Err(join)) => Err(MemoryError::EmbeddingInferenceError {
                        batch_size: size,
                        reason: join.to_string(),
                    }),
                    Ok(Ok(res)) => res,
                }
            }
        }
    }

    /// Name of the model that produces the vectors (as configured).
    pub fn model_name(&self) -> &str {
        &self.config.model_name
    }

    /// Get the dimension of embeddings produced by this engine.
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Cache hit ratio `hits / (hits + misses)`; 0.0 before any lookup.
    pub fn cache_hit_ratio(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let total = hits + self.misses.load(Ordering::Relaxed) as f64;
        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }

    /// Clear the embedding cache and reset hit/miss counters.
    pub fn clear_cache(&mut self) {
        self.cache.lock().clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

/// Vectorizer: wrapper providing simpler API for embedding operations.
pub struct Vectorizer {
    engine: EmbeddingEngine,
}

impl Vectorizer {
    /// Create a new vectorizer.
    pub async fn new() -> Result<Self> {
        let engine = EmbeddingEngine::new().await?;
        Ok(Vectorizer { engine })
    }

    /// Vectorize a single text.
    pub async fn vectorize(&self, text: &str) -> Result<EmbeddingVector> {
        self.engine.embed(text).await
    }

    /// Vectorize multiple texts.
    pub async fn vectorize_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingVector>> {
        self.engine.embed_batch(texts).await
    }
}

/// Utility function for computing cosine similarity between two vectors.
///
/// Returns a value in [-1, 1], where 1 is identical, 0 is orthogonal, -1 is opposite.
pub fn cosine_similarity(a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (av, bv) in a.iter().zip(b.iter()) {
        dot_product += (av * bv) as f64;
        norm_a += (av * av) as f64;
        norm_b += (bv * bv) as f64;
    }

    norm_a = f64::sqrt(norm_a);
    norm_b = f64::sqrt(norm_b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> Option<PathBuf> {
        EmbeddingEngine::resolve_model_dir(&EmbeddingConfig::default())
    }

    #[tokio::test]
    async fn test_e5_prefixes_apply_to_queries_and_passages() {
        assert_eq!(default_prefixes("all-MiniLM-L6-v2"), ("", ""));
        assert_eq!(
            default_prefixes("multilingual-e5-small"),
            ("query: ", "passage: ")
        );
        let config = EmbeddingConfig {
            model_name: "multilingual-e5-test".into(),
            dimension: 64,
            ..EmbeddingConfig::default()
        };
        let e5 = EmbeddingEngine::build(config, Backend::Hashing).expect("build");
        let plain = EmbeddingEngine::hashing(64).expect("plain");
        // A query is embedded as "query: <text>", a passage as "passage: <text>".
        assert_eq!(
            e5.embed("cara deploy").await.expect("q"),
            plain.embed("query: cara deploy").await.expect("plain q")
        );
        assert_eq!(
            e5.embed_batch(&["isi catatan"]).await.expect("p")[0],
            plain.embed("passage: isi catatan").await.expect("plain p")
        );
    }

    #[tokio::test]
    async fn test_multilingual_e5_embeds_when_installed() {
        let config = EmbeddingConfig {
            model_name: "multilingual-e5-small".into(),
            dimension: 384,
            ..EmbeddingConfig::default()
        };
        let Ok(e) = EmbeddingEngine::with_config(config).await else {
            eprintln!("SKIPPED: multilingual-e5-small is not installed");
            return;
        };
        let v = e.embed("cara membuat dokumen").await.expect("embed");
        assert_eq!(v.len(), 384);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "L2-normalised, got {norm}");
    }

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.dimension, 384);
        assert_eq!(config.model_name, "all-MiniLM-L6-v2");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_wordpiece_basic_and_encode() {
        let mut vocab = HashMap::new();
        for (i, t) in [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "##s", "!",
        ]
        .iter()
        .enumerate()
        {
            vocab.insert((*t).to_string(), i as i64);
        }
        let wp = WordPiece {
            vocab,
            unk: 1,
            cls: 2,
            sep: 3,
            pad: 0,
        };
        assert_eq!(wp.encode("Hello worlds!", 16), vec![2, 4, 5, 6, 7, 3]);
        assert_eq!(wp.encode("zzz", 16), vec![2, 1, 3]);
        // Truncation keeps [CLS] ... [SEP].
        assert_eq!(wp.encode("hello world hello world", 4), vec![2, 4, 5, 3]);
        assert_eq!(
            WordPiece::basic("a-b 日本"),
            vec!["a", "-", "b", "日", "本"]
        );
    }

    #[tokio::test]
    async fn test_hashing_engine_properties() {
        let e = EmbeddingEngine::hashing(64).expect("engine");
        assert_eq!(e.backend(), EmbeddingBackend::Hashing);
        let v = e.embed("vector search engine").await.expect("embed");
        assert_eq!(v.len(), 64);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        // Deterministic.
        assert_eq!(v, e.embed("vector search engine").await.expect("again"));
        // Word overlap scores higher than none.
        let near = e.embed("vector search library").await.expect("near");
        let far = e.embed("chocolate cake recipe").await.expect("far");
        assert!(cosine_similarity(&v, &near) > cosine_similarity(&v, &far));
    }

    #[tokio::test]
    async fn test_cache_hits_and_clear() {
        let mut e = EmbeddingEngine::hashing(32).expect("engine");
        assert_eq!(e.cache_hit_ratio(), 0.0);
        e.embed_batch(&["a b", "c d"]).await.expect("first");
        e.embed_batch(&["a b", "c d", "e f"]).await.expect("second");
        // 5 lookups: 2 hits, 3 misses.
        assert!((e.cache_hit_ratio() - 0.4).abs() < 1e-9);
        e.clear_cache();
        assert_eq!(e.cache_hit_ratio(), 0.0);
        e.embed("a b").await.expect("after clear");
        assert_eq!(e.cache_hit_ratio(), 0.0);
    }

    #[tokio::test]
    async fn test_batch_order_and_subbatching() {
        let mut e = EmbeddingEngine::hashing(16).expect("engine");
        e.config.max_batch_size = 2;
        let texts: Vec<String> = (0..7).map(|i| format!("text number {i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let batch = e.embed_batch(&refs).await.expect("batch");
        assert_eq!(batch.len(), 7);
        for (t, v) in refs.iter().zip(&batch) {
            assert_eq!(*v, hash_embed(t, 16));
        }
    }

    #[tokio::test]
    async fn test_missing_model_is_error_not_fallback() {
        let cfg = EmbeddingConfig {
            model_name: "no-such-model".to_string(),
            model_dir: Some(PathBuf::from("definitely/missing")),
            ..EmbeddingConfig::default()
        };
        assert!(matches!(
            EmbeddingEngine::with_config(cfg).await,
            Err(MemoryError::EmbeddingModelError(_))
        ));
        let bad = EmbeddingConfig {
            dimension: 0,
            ..EmbeddingConfig::default()
        };
        assert!(matches!(
            EmbeddingEngine::with_config(bad).await,
            Err(MemoryError::ConfigError(_))
        ));
    }

    #[tokio::test]
    async fn test_onnx_semantic_similarity() {
        let Some(dir) = model_dir() else {
            eprintln!("SKIPPED test_onnx_semantic_similarity: no model.onnx/vocab.txt found");
            return;
        };
        let cfg = EmbeddingConfig {
            model_dir: Some(dir),
            ..EmbeddingConfig::default()
        };
        let e = EmbeddingEngine::with_config(cfg).await.expect("load onnx");
        assert_eq!(e.backend(), EmbeddingBackend::Onnx);
        let v = e
            .embed_batch(&[
                "How do I find similar documents with vectors?",
                "Nearest neighbor search over embeddings",
                "Recipe for baking chocolate cake",
            ])
            .await
            .expect("embed");
        assert!(v.iter().all(|x| x.len() == 384));
        let norm: f32 = v[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
        let related = cosine_similarity(&v[0], &v[1]);
        let unrelated = cosine_similarity(&v[0], &v[2]);
        assert!(
            related > unrelated + 0.1,
            "related={related} unrelated={unrelated}"
        );
        // Padding must not change a text's embedding: alone vs. batched with a longer text.
        let alone = e
            .embed("Recipe for baking chocolate cake")
            .await
            .expect("alone");
        assert!(cosine_similarity(&alone, &v[2]) > 0.999);
    }
}
