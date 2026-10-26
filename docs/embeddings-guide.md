# Embeddings Guide: Models, Performance, and Setup

This document covers ONNX embedding model selection, performance tuning, and multi-language support for plaesy-memory.

## What Are Embeddings?

Embeddings are dense vector representations of text. For example:

```text
"how do we deploy?"
    ↓ [ONNX all-MiniLM-L6-v2]
[0.123, -0.456, 0.789, …, 0.012]  ← 384-dimensional vector
```

Two texts with similar meaning produce vectors with high cosine similarity:

```text
"deployment ritual"    → [0.121, -0.458, 0.791, …, 0.015]
                            ↑ cosine_sim ≈ 0.99 ✓ (very similar)
```

plaesy-memory uses embeddings for semantic search: text with similar meaning to your query, even if vocabulary differs.

---

## Model Selection

### Default: all-MiniLM-L6-v2

**When to use**: English-language vaults, balanced latency/quality, no GPU available.

| Property | Value |
|----------|-------|
| Dimension | 384 |
| Language | English |
| Size | 86 MB (ONNX, CPU only) |
| Latency | 11 ms (cold, CPU) / 5.7 ms/text (batch of 32) |
| Training | Contrastive learning on 1B sentence pairs (BERT base) |
| Strengths | Well-established, optimized for CPU, semantic accuracy |
| Weaknesses | English-only, sensitive to tokenization across languages |

**Setup** (automatic via `memory onboard`):

```bash
MEMORY_MODEL_NAME=all-MiniLM-L6-v2
```

Model downloaded to `%LOCALAPPDATA%/plaesy/models/` (Windows) or `~/.local/share/plaesy/models` (Linux/macOS).

### Multilingual: multilingual-e5-small

**When to use**: Non-English vaults (Indonesian, Chinese, German, etc.), or code mixed with natural language.

| Property | Value |
|----------|-------|
| Dimension | 384 |
| Language | 100+ (including Indonesian, Chinese, German, French, Japanese) |
| Size | 200 MB (ONNX) |
| Latency | ~15 ms (cold, CPU, SentencePiece tokenization slower) |
| Training | Contrastive learning on multilingual text pairs |
| Strengths | Multilingual, handles accent-aware matches, cross-vocabulary |
| Weaknesses | Slower tokenization (SentencePiece), larger model file |

**Setup**:

```bash
export MEMORY_MODEL_NAME=multilingual-e5-small
memory onboard --vault C:/_
```

Or in config:

```toml
[embeddings]
model_name = "multilingual-e5-small"
```

**Indonesian probe** (8 notes, 8 Indonesian queries):

| Model | Semantic | Hybrid |
|-------|----------|--------|
| all-MiniLM-L6-v2 | 4/8 · 6/8 | 6/8 · 7/8 |
| multilingual-e5-small | 7/8 · 8/8 | 6/8 · 8/8 |

With n = 8, results show direction, not statistical significance. E5 semantic excels on paraphrases, but hybrid fusion needs tuning for non-English.

### Domain-Specific: all-indo-e5-small-v4, indonesian-embedding-small

**When to use**: High-performance Indonesian or other specialized language vaults.

| Model | Language | Size | Latency | Probe R@1/3 |
|-------|----------|------|---------|------------|
| all-indo-e5-small-v4 | Indonesian | 180 MB | ~12 ms | 7/8 · 8/8 |
| indonesian-embedding-small (q8) | Indonesian | 90 MB (quantized) | ~8 ms | 7/8 · 8/8 |

**Setup**:

```bash
export MEMORY_MODEL_NAME=all-indo-e5-small-v4
# or
export MEMORY_MODEL_NAME=indonesian-embedding-small
```

Trade-off: Specialization (better on Indonesian) vs generalization (E5 handles Indonesian + English + other languages).

---

## Tokenization

### Built-in WordPiece (all-MiniLM)

- **Vocab**: `vocab.txt` (hardcoded, ~30K tokens)
- **Normalization**: Lowercase, no accent stripping
- **Performance**: ~0.5 ms per query (negligible compared to embedding)
- **Language limitation**: English; non-ASCII characters may be tokenized as [UNK] (unknown)

### Hugging Face SentencePiece (E5, multilingual)

- **Vocab**: `tokenizer.json` (downloaded with model)
- **Normalization**: Language-aware, handles accents and scripts
- **Performance**: ~1–2 ms per query (slower than WordPiece, but acceptable)
- **Language support**: Multilingual by design

**Implication**: On non-English text, e5 tokenizes more accurately:

```text
German: "Überraschung" (surprise)
  WordPiece: [UNK] [UNK] [UNK] (three unknown tokens)
  SentencePiece: [Ü] [ber] [rasch] [ung] (meaningful subword tokens)
```

SentencePiece produces better embeddings for non-English text.

---

## Performance Characteristics

### Latency Breakdown

On AMD Ryzen 7 5700U (CPU only, no GPU):

```text
Query: "how do we deploy?"

Tokenization:     0.5 ms  (WordPiece for MiniLM)
Embedding (ONNX): 10.5 ms (forward pass, mean pooling, L2 norm)
L2 Normalization: 0 ms    (negligible)
HNSW k-NN:        0.2 ms  (20-document vault)
Fusion:           0.1 ms  (score combination)
───────────────────────────
Total:            11.3 ms (cold, unbatched)
                   0.2 ms (subsequent queries, if cached)
```

**Batching**: If you embed multiple queries at once:

```text
32 queries × 384-dim:
  Total:   5.7 ms per query (32x throughput)
  Amortized: ~180 ms for batch / 32 queries
```

### Memory Usage

```text
Model loading:           ~200 MB (all-MiniLM in RAM)
Embedding cache (LRU):   ~2 MB   (1000 queries × 384 dims × 4 bytes)
HNSW index:              ~20 MB  (20-document vault, 4.6K chunks)
SQLite database:         ~5 MB   (metadata + FTS5 index)
───────────────────────────────
Total:                   ~230 MB (comfortable on modern machines)
```

---

## Scaling

### Index Size vs Latency

HNSW k-NN search latency (usearch, cosine distance) on random 384-dimensional vectors:

| Index Size | p50 | p95 |
|-----------|-----|-----|
| 1K chunks | 0.18 ms | — |
| 10K chunks | 2.8 ms | — |
| 50K chunks | 3.9 ms | — |

**Real embedding latency** varies by content and model:

- Short, generic queries: p50 ~0.5 ms (cache hit)
- Long, specific queries: p50 ~11 ms (cold, ONNX)

**Throughput**: Sequential (single-threaded): ~90 queries/sec (hybrid mode).

---

## HNSW vs Alternatives

### HNSW (Hierarchical Navigable Small World)

**Used by**: plaesy-memory (usearch library, Qdrant engine)

**Strengths**:

- Fast k-NN search (0.18 ms on 1K vectors)
- Incremental updates (no rebuild needed on insertion)
- Snapshot on disk (cache, 0.37 s cold start vs ~4 s rebuild)

**Weaknesses**:

- High-dimension penalty (384-dim degrades to flat graph performance on some workloads)
- Rebuild cost when many vectors change (resource-intensive)

### Alternatives (Not Used)

**Flat search** (brute-force, O(n)):

- Fast on high-dim (384-dim × 384-dim @ 50K vectors ≈ 1–2 ms)
- No index build; always fresh
- Memory-hungry (full vector BLOB scan)

**ScaNN** (Google):

- ~2× throughput of HNSW at given recall
- Slower index build, no incremental updates
- Not evaluated in project

**Decision**: HNSW default; flat search as future optimization for 384-dim workloads.

---

## Setup and Configuration

### Automatic (Recommended)

```bash
memory onboard --vault C:/_
```

Prompts for:

1. Vault path (notes directory)
2. Embedding model (default: all-MiniLM-L6-v2, option to choose multilingual)
3. Model download location (default: OS standard)

Writes `settings.json` and downloads model.

### Manual

```bash
# 1. Create directories
mkdir -p ~/.local/share/plaesy/models

# 2. Download model
curl -O https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/all-MiniLM-L6-v2.onnx
cp all-MiniLM-L6-v2.onnx ~/.local/share/plaesy/models/

# 3. Download vocab (for WordPiece)
curl -O https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/vocab.txt
cp vocab.txt ~/.local/share/plaesy/models/

# 4. Configure
memory config  # verify settings

# 5. Index
memory index C:/_
```

### Environment Variables

```bash
export PLAESY_NOTES_DIR=C:/_
export PLAESY_MODEL_DIR=~/.local/share/plaesy/models
export MEMORY_MODEL_NAME=all-MiniLM-L6-v2  # or multilingual-e5-small
```

### Config File (TOML)

```toml
[memory]
notes_dir = "C:/_"
db_path = "%LOCALAPPDATA%/Plaesy/memory.db"
model_dir = "%LOCALAPPDATA%/plaesy/models"

[embeddings]
model_name = "all-MiniLM-L6-v2"  # or multilingual-e5-small
enable_vector_search = true
embedding_cache_size = 1000  # LRU cache

[search]
chunk_size = 512
enable_vector_search = true

[ranker]
ranker_type = "weighted"
semantic_weight = 0.7
keyword_weight = 0.3
stale_penalty = 0.85
```

---

## Monitoring and Diagnostics

### Check Embedding Status

```bash
memory config
```

Output includes:

- Model loaded: yes/no
- Model name: all-MiniLM-L6-v2
- Model path: /home/user/.local/share/plaesy/models/all-MiniLM-L6-v2.onnx
- Vector dimension: 384
- Vector search enabled: true

### Metrics

```bash
memory stats
```

Includes:

- Chunks indexed (total and with embedding)
- Embedding cache hits / misses
- Average embedding time (ms)
- HNSW index size (bytes)

### Fallback to Keyword-Only

If embeddings fail:

```bash
memory search "query" --mode keyword  # explicit keyword-only
# or
memory search "query"  # hybrid, but semantic leg fails silently
```

Check metrics to see:

- `embedding_errors` (count of failed embeddings)
- `vector_search_enabled` (true/false)

---

## Performance Tuning

### Reduce Latency

**Option 1**: Disable semantic search (keyword-only mode)

```toml
[search]
enable_vector_search = false
```

Latency: 1.1 ms → saves ~10 ms per query, trades semantic accuracy.

**Option 2**: Use quantized models (if available)

`indonesian-embedding-small (q8)` is 8-bit quantized; ~2× faster, slightly lower accuracy.

```bash
export MEMORY_MODEL_NAME=indonesian-embedding-small
```

**Option 3**: Batch embed queries offline

Pre-compute embeddings for frequent queries, store in cache.

### Improve Accuracy

**Option 1**: Use multilingual model for non-English

```bash
export MEMORY_MODEL_NAME=multilingual-e5-small
```

**Option 2**: Add cross-encoder reranking

```toml
[ranker]
enable_cross_encoder = true
cross_encoder_weight = 0.5
```

Cost: +5 ms per hit; gain: +0.006 MRR on LongMemEval (not statistically significant at n = 94).

**Option 3**: Tune fusion weights

Calibrate on your labeled eval set:

```toml
[ranker]
semantic_weight = 0.6  # increase for paraphrase-heavy queries
keyword_weight = 0.4
```

---

## Known Limitations

1. **English-default**: all-MiniLM-L6-v2 trained on English; non-ASCII scripts may degrade. Use multilingual-e5-small for international text.

2. **No accent stripping in WordPiece**: `café` and `cafe` are different tokens. SentencePiece handles accents better.

3. **Tokenization mismatch**: If you switch models mid-index, embeddings become stale (vectors from old model, queries embedded with new model). Solution: `memory index --reindex-all`.

4. **ONNX RC version**: Pinned to 2.0.0-rc.14 pending stable release. May have minor version incompatibilities.

5. **Single ONNX session per process**: No concurrent embedding calls. Sequential queries are fine; high-throughput applications may hit the 5–11 ms per-query ceiling.

---

## Benchmarks and References

- **LongMemEval-S**: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (470-question benchmark, project run on 94 sampled questions)
- **ONNX Runtime**: https://onnx.ai/ (standard format for efficient inference)
- **all-MiniLM-L6-v2**: https://huggingface.co/Xenova/all-MiniLM-L6-v2 (Hugging Face Hub)
- **multilingual-e5-small**: https://huggingface.co/Xenova/multilingual-e5-small (multilingual alternative)
- **usearch**: https://github.com/unum-cloud/usearch (HNSW implementation used in plaesy-memory)
- **Microsoft ONNX Production Guide**: https://techcommunity.microsoft.com/blog/azuredevcommunityblog/-/4400142 (ONNX embeddings in production)

---

## Next Steps

1. **Indonesian evaluation**: Run full LongMemEval with multilingual-e5-small; compare to MiniLM baseline
2. **Quantization**: Test `indonesian-embedding-small (q8)` on full eval set
3. **Custom models**: If you have labeled in-domain eval data, consider fine-tuning a model for your domain
4. **Concurrent embedding**: Design multi-threaded embedding pipeline if throughput is a bottleneck
