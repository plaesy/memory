# Best practices

Practical guidance for using and extending this engine. Each item says why; the evidence is in
[COMPARISON.md](docs/COMPARISON.md) and the README. This replaces an earlier version that described the original design.

## Writing memory (applies to any vault; the `brain` repo implements it)

1. **Keep the words.** Store notes as the author wrote them; do not summarise on write. Retrieval over verbatim text is a strong
   baseline and summaries silently lose detail. Summarise when you *read*, with the citation attached.
2. **Separate by kind.** Events (episodic), facts (semantic), procedures (procedural), and a tiny always-loaded core behave
   differently and fail differently. One folder per kind.
3. **Never edit the past.** Episodic logs are append-only. When a fact changes, write the new note with `supersedes` and mark the
   old one `status: superseded` + `superseded_by`. History stays; search hides it by default.
4. **Say how sure and how fresh.** Every claim-bearing note carries `confidence` and `last_verified`. A stale-but-confident note is
   worse than none. Re-verify on a cadence and update the date only when you actually checked.
5. **One claim per note, stated checkably.** Short notes retrieve cleanly and can be superseded individually.
6. **Name your sources.** `source:` points at the episode, file or URL a fact came from.
7. **Reference real paths in backticks.** Hygiene can then notice when they disappear. Link related notes with `[[name]]`.
8. **Keep core small.** It is pinned into every prompt. Cap it with `limit`; push everything else to semantic.

## Reading memory (agent behaviour)

1. Search before answering from memory; do not rely on recall.
2. Prefer `FACT` hits. Treat `INFERRED`, `STALE` and low-trust hits as leads to check, never as established.
3. Cite `path:start-end` for anything you take from memory.
4. Use filters rather than longer queries: `--tier procedural` for "how do I", `--tier episodic` for "what happened".
5. Ask for superseded notes (`include_inactive`) only when reconstructing history.

## Operating the engine

- **Run hygiene in CI or a pre-commit hook** (`plaesy-memory-cli hygiene --strict`). Organised memory decays as it grows unless
  something checks it.
- **Keep the index out of git** (`*.db`, `*.hnsw*`). It is rebuildable; the notes are the truth.
- **Install the embedding model.** Keyword-only works but paraphrase recall drops (R@3 0.70 vs 1.00 on the labeled set).
- **Leave reranking off** until you have measured it on your own notes. On the LongMemEval-S subset the gain was inside noise.
- **One index per vault.** Point `MEMORY_DB_PATH` at a per-vault location; do not share a database between vaults.
- **Back up the vault, not the index.**
- **Watch for silent degradation:** `stats` shows `vector search: off (keyword-only)` if the model was not found.
- **Exclude framework and template files** you do not want recalled (`MEMORY_EXCLUDE_DIRS`, for example `.plaesy,.claude,CLAUDE.md`).
  Check the document count in `stats` after changing it.

## Extending the engine

- Route every new result path through `CitationBuilder` (see CONTRIBUTING).
- Normalise any new score to a comparable 0-1 scale before fusing; unnormalised BM25 dominates other signals.
- Filter before fusion and truncation, not after, or a restrictive filter empties the list.
- Make failure of an optional leg (semantic, rerank) non-fatal and counted in metrics.
- Prefer a rebuildable cache with a validity check (the HNSW snapshot's generation) over a second source of truth.
- Add an evaluation with the change. A saturated test set (the 20-document one is) cannot show improvement; use LongMemEval or grow the set.

## Pitfalls seen in this project

| Pitfall | What happened | Guard |
|---------|---------------|-------|
| Rank-only fusion | RRF lost to semantic-only on paraphrases | weighted fusion with absolute scores |
| Re-serialised metadata cited as source lines | frontmatter chunk did not match the file | raw block stored; test re-slices the file |
| Template syntax parsed as links | `[[LLM: ...]]` reported as broken links | plausibility rules + inline-code skipping |
| Stale-looking numbers in docs | an earlier handoff claimed 166 tests and "production ready" before anything compiled | claims need a reproducing command |
| Blocking startup | initial index ran before serving; first request timed out | background sync |
| Second ONNX environment | runtime allows one per process | shared `ort_environment()` |
| Provider "activated" but never started | `Popen` with a missing `cwd` fails on Windows (WinError 267); the log still said activated | `cwd` only when the directory exists; regression test |
| Status checks look healthy, real calls fail | `hermes memory status` does not start the provider | run one `hermes chat -v --oneshot` and read the DEBUG log |
| Stale index after the vault changed | folders renamed or moved; a sync skips unchanged hashes only, so old paths stay until removed | run `index` after restructuring; check `stats` document count |
| Bare heading as the only search hit | chunker emitted a heading run with no body | heading run takes the following block; `test_nested_headings_do_not_leave_a_bare_title_chunk` |
| Index from before a chunker change | hash-based sync does not re-chunk unchanged files | delete and rebuild the index (SETUP, "Upgrading") |
