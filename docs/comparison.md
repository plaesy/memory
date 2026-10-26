# Comparison with other agent-memory systems

Research date: 2026-10-19. Sources: vendor documentation and READMEs, the Zep paper (arXiv 2501.13956), Letta/MemGPT
documentation, Basic Memory docs, arXiv 2607.26637 (filesystem memory), arXiv 2605.03675 (MemTier), and 2026 comparison
articles. **Every benchmark figure quoted by those sources is self-reported** with differing readers, judges and retrieval
depth; some are disputed (Zep's LoCoMo 94.7% versus 75.1% in a third-party test). Scores across vendors are not comparable,
so this page compares design, not leaderboards. Our own numbers are in the README.

## Design comparison

| System | Source of truth | How memory is written | Retrieval | Offline | Result provenance |
|--------|-----------------|-----------------------|-----------|---------|-------------------|
| Mem0 | rows in a DB (opaque to the user) | LLM extracts facts on write | vector + entity | self-host | no |
| Zep / Graphiti | temporal knowledge graph | LLM extracts entities and edges | hybrid + graph, bi-temporal validity | partial | edges with validity intervals |
| Letta (MemGPT) | DB, optional git-backed files | the agent edits its own blocks | always-in-context core blocks + archival + recall search | self-host | partial |
| Cognee | graph built from data | extraction pipeline | graph + vector | yes | partial |
| MemPalace | verbatim text | store raw, no extraction | vector (Chroma) | yes | yes |
| agentmemory | logs / JSON | agent hooks | BM25 + MiniLM vectors | yes, no key | partial |
| Basic Memory | Markdown files | agent or human writes notes | FTS + vectors + graph, via MCP | yes | file-level |
| COG | Markdown files | agent skills | agent reads files; no index | yes | file-level |
| **plaesy** | Markdown files (index is rebuildable) | human or agent writes notes; the engine is read-only | FTS5 + HNSW weighted fusion, optional cross-encoder | yes | file, line range, git revision, index time, source trust |

## Practices the research converges on, and where we stand

| Practice | Evidence | plaesy |
|----------|----------|--------|
| Keep verbatim text; index is derived | MemPalace raw text reaches 96.6% R@5 with no extraction (self-reported); Basic Memory/COG treat files as truth | yes |
| Hybrid lexical + vector beats either alone | agentmemory BM25 86.2% vs hybrid 95.2% R@5 (self-reported); our subset: 93.6 / 95.7 / 96.8 | yes, within noise at n = 94 |
| Normalise scores before fusing | MemTier: unnormalised BM25 dominated other signals | yes (`s/(s+8)`) |
| Filter by metadata before fusion | Oracle, SurrealDB guidance | tier, path, trust, status |
| Separate memory by kind | working / episodic / semantic / procedural | tiers + folder convention in `brain` |
| Facts have time and confidence | Zep validity intervals; COG `memory-hygiene` | `last_verified`, `confidence`, `status`, staleness penalty, hygiene audit; **no** valid-from/valid-to |
| Small always-in-context core | Letta blocks with limits and `read_only` | Hermes provider pins `tier: core` notes with `limit`; other clients get search only |
| Organisation needs upkeep | arXiv 2607.26637: structure erodes as memory grows unless actively maintained | `hygiene --strict` as a gate |
| Provenance on every result | rare among competitors | yes (EV-01) |
| Open, reproducible evaluation | vendors dispute each other's numbers | `make lme`; subset only so far |

## Gaps, and what is done about each

| Gap | Status (2026-10-20) |
|-----|---------------------|
| No graph or entity layer | **Partly closed.** `search_expanded` (`expand_links`, `--expand-links`) adds the best passage of notes one wikilink away from the top hits. No entity extraction, no multi-hop, no relation types. |
| No temporal validity model | **Partly closed.** Sources can declare `valid_from` / `valid_to` (inclusive); `as_of` (`--as-of`) keeps the sources valid on a date. Undated sources always pass. Validity is per source, not per fact, and there are no intervals between facts. |
| No automatic extraction or consolidation | **Not planned.** Extraction needs an LLM in the write path, which breaks the offline rule (NN-01). Writing stays a human or agent task. |
| Single local user | **By design.** No multi-tenant ACLs, remote server or sync; these would contradict the offline, read-only design. |
| Evaluation is thin | **Open.** 94 LongMemEval questions on one machine. LoCoMo not run; no latency under concurrent load. Needs a measurement, not a feature. |
| Weak on non-English queries | **Partly closed.** `multilingual-e5-small` is supported (`MEMORY_MODEL_NAME`) and did better on an 8-query Indonesian probe (semantic top-1 7/8 vs 4/8). Not the default: its English-side quality is unmeasured. |
| Preference-style questions | **Open.** R@5 0.83 on n = 6: too few questions to conclude anything. |

Nothing here makes plaesy the best system. A gap is closed when the feature exists and has a test, not when the page says so.
Whether it beats a competitor on a given vault is an empirical question, and no head-to-head run has been done yet.

## Where it is stronger

- Every result cites file, line range, revision and time, and the text is a verbatim slice of the source.
- Fully offline after a one-time model download; no API key, no LLM in the write path.
- Memory a human can read, grep, diff and review in git.
- Source trust and staleness are visible at the point of use, and an audit can gate CI.
- 0.37 s cold start; sub-millisecond hybrid search on small corpora.
