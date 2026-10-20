//! Cross-encoder reranking.
//!
//! A bi-encoder (the embedding model) scores query and passage independently, which is fast but
//! coarse. A cross-encoder reads `[CLS] query [SEP] passage [SEP]` together and scores the pair,
//! which is slower (one forward pass per candidate) but markedly better at ordering a short list.
//! The engine therefore uses it only on the top fused candidates, and only when enabled
//! (`search.rerank = true`).
//!
//! Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` exported to ONNX (`model.onnx`), with its BERT
//! `vocab.txt` (identical to the embedding model's). Lookup order: `search.rerank_model_dir`,
//! `$PLAESY_RERANK_MODEL_DIR`, `~/.plaesy-models/ms-marco-MiniLM-L-6-v2`. `scripts/install-model.sh
//! --rerank` downloads it.

use crate::memory::embeddings::WordPiece;
use crate::memory::error::{MemoryError, Result};
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Default model directory name under `~/.plaesy-models/`.
pub const DEFAULT_MODEL_NAME: &str = "ms-marco-MiniLM-L-6-v2";

/// Pairs scored per forward pass.
const BATCH: usize = 16;

fn model_err(e: impl std::fmt::Display) -> MemoryError {
    MemoryError::EmbeddingModelError(format!("reranker: {e}"))
}

/// ONNX cross-encoder producing one relevance logit per (query, passage) pair.
pub struct CrossEncoder {
    /// Keeps the ONNX Runtime environment alive for the session's lifetime.
    _env: Arc<ort::environment::Environment>,
    session: Mutex<ort::session::Session>,
    tokenizer: WordPiece,
    needs_token_type_ids: bool,
    max_seq_len: usize,
}

/// Logistic function: maps a relevance logit to (0, 1).
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl CrossEncoder {
    /// Locate the model directory (see module docs). Returns `None` if no candidate has `model.onnx`.
    pub fn resolve_model_dir(explicit: Option<&Path>) -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(d) = explicit {
            candidates.push(d.to_path_buf());
        }
        if let Some(d) = std::env::var_os("PLAESY_RERANK_MODEL_DIR") {
            candidates.push(PathBuf::from(d));
        }
        if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            candidates.push(
                PathBuf::from(home)
                    .join(".plaesy-models")
                    .join(DEFAULT_MODEL_NAME),
            );
        }
        candidates
            .into_iter()
            .find(|d| d.join("model.onnx").is_file() && d.join("vocab.txt").is_file())
    }

    /// Load `model.onnx` + `vocab.txt` from `dir`.
    ///
    /// # Errors
    ///
    /// - `EmbeddingModelError` if the model cannot be loaded
    /// - `IoError` if the vocabulary cannot be read
    pub fn load(dir: &Path, max_seq_len: usize) -> Result<Self> {
        let env = crate::memory::embeddings::ort_environment()?;
        let session = ort::session::Session::builder(&env)
            .map_err(model_err)?
            .commit_from_file(dir.join("model.onnx"))
            .map_err(model_err)?;
        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        Ok(CrossEncoder {
            _env: env,
            session: Mutex::new(session),
            tokenizer: WordPiece::from_vocab_file(&dir.join("vocab.txt"))?,
            needs_token_type_ids,
            max_seq_len: max_seq_len.max(16),
        })
    }

    /// Relevance logits for `docs` against `query`, in input order (blocking).
    ///
    /// # Errors
    ///
    /// - `EmbeddingInferenceError` if the forward pass fails or returns an unexpected shape
    pub fn score_blocking(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(docs.len());
        for group in docs.chunks(BATCH) {
            out.extend(self.score_batch(query, group)?);
        }
        Ok(out)
    }

    fn score_batch(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        let batch = docs.len();
        let infer_err = |e: &dyn std::fmt::Display| MemoryError::EmbeddingInferenceError {
            batch_size: batch,
            reason: format!("reranker: {e}"),
        };
        let pairs: Vec<(Vec<i64>, Vec<i64>)> = docs
            .iter()
            .map(|d| self.tokenizer.encode_pair(query, d, self.max_seq_len))
            .collect();
        let seq = pairs.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
        let mut ids = vec![self.tokenizer.pad; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        let mut types = vec![0i64; batch * seq];
        for (b, (i, t)) in pairs.iter().enumerate() {
            ids[b * seq..b * seq + i.len()].copy_from_slice(i);
            mask[b * seq..b * seq + i.len()].fill(1);
            types[b * seq..b * seq + t.len()].copy_from_slice(t);
        }
        let shape = [batch as i64, seq as i64];
        let ids_t = ort::value::Tensor::from_array((shape, ids)).map_err(|e| infer_err(&e))?;
        let mask_t = ort::value::Tensor::from_array((shape, mask)).map_err(|e| infer_err(&e))?;
        let mut inputs = ort::inputs! {
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
        };
        if self.needs_token_type_ids {
            let tt = ort::value::Tensor::from_array((shape, types)).map_err(|e| infer_err(&e))?;
            inputs.push(("token_type_ids".into(), tt.into()));
        }
        let mut session = self.session.lock();
        let outputs = session.run(inputs).map_err(|e| infer_err(&e))?;
        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| infer_err(&e))?;
        // Expected [batch, 1] (or [batch]): one logit per pair.
        if data.len() != batch {
            let dims: Vec<i64> = out_shape.iter().copied().collect();
            return Err(infer_err(&format!(
                "expected {batch} logits, got shape {dims:?}"
            )));
        }
        Ok(data.to_vec())
    }

    /// Async wrapper: runs the forward passes on a blocking thread.
    ///
    /// # Errors
    ///
    /// Same as [`CrossEncoder::score_blocking`]; `InternalError` if the worker thread panics.
    pub async fn score(self: &Arc<Self>, query: &str, docs: Vec<String>) -> Result<Vec<f32>> {
        let me = Arc::clone(self);
        let q = query.to_string();
        tokio::task::spawn_blocking(move || me.score_blocking(&q, &docs))
            .await
            .map_err(|e| MemoryError::InternalError(format!("reranker task failed: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_is_monotonic_and_bounded() {
        assert!(sigmoid(-20.0) < 1e-6);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(20.0) > 1.0 - 1e-6);
        assert!(sigmoid(1.0) > sigmoid(0.5));
    }

    fn model() -> Option<Arc<CrossEncoder>> {
        let dir = CrossEncoder::resolve_model_dir(None)?;
        Some(Arc::new(CrossEncoder::load(&dir, 256).ok()?))
    }

    #[tokio::test]
    async fn relevant_passage_outscores_irrelevant_one() {
        let Some(ce) = model() else {
            eprintln!("SKIPPED: no cross-encoder model installed");
            return;
        };
        let docs = vec![
            "Raft elects a leader that replicates a log to followers.".to_string(),
            "Bananas are rich in potassium and grow in tropical climates.".to_string(),
        ];
        let s = ce
            .score("how does leader election work in a replicated log", docs)
            .await
            .expect("score");
        assert_eq!(s.len(), 2);
        assert!(
            s[0] > s[1] + 2.0,
            "relevant {} vs irrelevant {}",
            s[0],
            s[1]
        );
    }

    #[tokio::test]
    async fn scoring_is_independent_of_batch_composition() {
        let Some(ce) = model() else {
            eprintln!("SKIPPED: no cross-encoder model installed");
            return;
        };
        let a = "Sorting algorithms like quicksort partition an array around a pivot.".to_string();
        let filler = "short".to_string();
        let alone = ce
            .score("quicksort pivot", vec![a.clone()])
            .await
            .expect("alone");
        let mixed = ce
            .score("quicksort pivot", vec![filler, a])
            .await
            .expect("mixed");
        assert!(
            (alone[0] - mixed[1]).abs() < 1e-3,
            "padding must not change scores: {} vs {}",
            alone[0],
            mixed[1]
        );
    }
}
