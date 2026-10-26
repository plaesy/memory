# Embedder decision: all-MiniLM-L6-v2 vs multilingual-e5-small

Date: 2026-10-26. Owner scope: compare the two models only; the Indonesian-specific models are the owner's call.

decision: keep all-MiniLM-L6-v2 as the default. multilingual-e5-small is recommended for Indonesian notes and is selected with `MEMORY_MODEL_NAME=multilingual-e5-small`. Switching the default needs owner confirmation and a full re-index.

## Why this decision

The rule in the spec: the default changes only if a model wins on both English and Indonesian beyond noise. e5 wins clearly on Indonesian and does not win on English, so the default stays.

## English: LongMemEval-S, every 5th question (94 questions), no reranker

| Mode | MiniLM R@1 | e5 R@1 | MiniLM R@5 | e5 R@5 | MiniLM MRR | e5 MRR |
|---|---|---|---|---|---|---|
| keyword | 0.798 | 0.798 | 0.936 | 0.936 | 0.861 | 0.861 |
| semantic | 0.840 | 0.862 | 0.957 | 0.968 | 0.889 | 0.903 |
| hybrid | 0.840 | 0.883 | 0.968 | 0.947 | 0.891 | 0.908 |

Source: `docs/eval/lme-subset/all-MiniLM-L6-v2.txt`, `docs/eval/lme-subset/multilingual-e5-small.txt`.

- One question is about 1.1 points at n = 94, and the 95% interval on R@5 is about ±3.5 points (README). The hybrid R@5 gap (0.968 vs 0.947, two questions) is inside that interval.
- e5 is higher on R@1 and MRR. Those differences are also within noise at this size.
- Verdict on English: no reliable difference.

## Indonesian: probe of 8 hand-labelled queries (`docs/eval/id-probe`)

| Model | Semantic top-1 / top-3 | Hybrid top-1 / top-3 |
|---|---|---|
| all-MiniLM-L6-v2 | 4/8 · 6/8 | 6/8 · 7/8 |
| multilingual-e5-small | 7/8 · 8/8 | 6/8 · 8/8 |

- Eight queries: one query is 12.5 points. The direction is consistent (e5 ahead on semantic), but this is a probe, not a benchmark.
- Also measured: all-indo-e5-small-v4 (semantic 7/8 · 8/8, hybrid 6/8 · 8/8) and indonesian-embedding-small q8 (semantic 7/8 · 8/8, hybrid 7/8 · 8/8). Not part of this decision, recorded for the owner.

## Cost and risk of switching

- e5 has 12 layers and MiniLM has 6. Per question e5 is roughly twice as slow in the LongMemEval harness (MiniLM about 37 s per question, e5 about 60 s with two workers).
- Model file: e5 is about 470 MB, MiniLM about 90 MB.
- Changing the model re-embeds every document. Keep one database per model (SETUP.md).

## Open items

- The full 470-question run is not done. It is queued by the owner.
- Tests that must pass before switching: `cargo test --test relevance_eval` with the chosen default.
- Prefix behaviour: `multilingual-e5-small` gets `query: `/`passage: ` automatically (name contains "e5"), which matches its training.
