# Assessment: is plaesy-memory the best option?

Date: 2026-10-26. Scope: offline agent memory over a Markdown second brain, exposed over MCP and Hermes.

## Verdict

**Not the best overall. It is the best of this set on provenance and offline guarantees, and behind on evaluation breadth, default multilingual support, governance of secrets, and MCP feature coverage.**

Which "best" depends on the goal. For an owner who needs to cite and verify what an agent reads from their own notes, plaesy is the strongest choice in the research set (see [COMPARISON](COMPARISON.md)). For a general agent that must answer broad questions over many sources, QMD or a hosted memory service is stronger on raw retrieval, at the cost of an LLM in the loop.

## Scorecard against practice

| Area | Practice | plaesy | Status |
|---|---|---|---|
| Provenance | Every result cites source, line range, revision, time | file, lines, revision, index time, FACT/INFERRED | Met, best in set |
| Offline | No network or API key at runtime | yes, after one-time model download | Met |
| Read-only | Agents should not write to the source of truth | read-only MCP tools | Met |
| Tool annotations | `readOnlyHint`, `openWorldHint` declared | all four tools declare `readOnlyHint: true`, `openWorldHint: false` | Met |
| Output schema | Declared and conformant when present (MCP 2025-06-18) | `memory_search` now declares `outputSchema`; a test checks a real result against it | Met (added 2026-10-26) |
| Server instructions | Tell the model how to use the tools | `instructions` in `initialize` | Met |
| Errors | Tool errors as `isError: true`, protocol errors as JSON-RPC | yes | Met |
| Protocol versions | Negotiate, do not reject newer clients | yes, falls back to latest supported | Met |
| Pagination | Required when lists can grow | `tools/list` is a fixed list of four; `memory_search` limit ≤ 50 | Not needed |
| Resources and prompts | Optional | `resources/list` answered; no resources exposed | Partial, by design |
| Secrets | Keep credentials out of memory (JetBrains 2026 survey guidance) | No secret scan. Anything in the vault is indexed and can be returned to a model. | **Gap** |
| Staleness | Mark unverified and obsolete facts | `last_verified`, STALE penalty, `status`, `valid_from`/`valid_to` | Met |
| Lifecycle | Write, update, forget are measured (MemOps, arXiv 2607.12893) | not in scope: read-only by design | Not applicable, by design |
| Evaluation | Retrieval recall, end-to-end QA accuracy, and token cost reported separately | Recall@k on LongMemEval-S, 94 questions only; no QA accuracy; no token cost | **Gap** |
| Reproducibility | Others can rerun the numbers | `make lme`; full 470-question run not done | **Gap** |
| Multilingual default | Non-English notes retrieve well | MiniLM default (English-only); e5 measured better on Indonesian but not default | **Gap** |
| Maintenance | Hygiene is run regularly, not once | `memory hygiene --strict` exists; no schedule | Partial |

## What the research says, and how it applies

- **MCP tool annotations are hints, not guarantees.** Clients must treat them as untrusted unless the server is trusted (MCP 2025-06-18). plaesy's annotations are accurate, but the real safety comes from the tools being read-only. Source: [MCP tools specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools).
- **Provenance is a recognised direction.** Recent work frames memory as needing provenance and poisoning resistance (for example Eywa, arXiv 2605.30771). Titles only were read; the details are unverified. plaesy already does provenance; poisoning resistance is not addressed.
- **Secrets do not belong in agent memory.** The JetBrains 2026 guidance says so directly. plaesy indexes everything under the vault, so a pasted API key in a note is returned to the model. This is the largest practical gap. Source: [JetBrains, long-term memory in AI agents](https://www.jetbrains.com/pages/ai-agents/building/long-term-memory/).
- **LongMemEval and LoCoMo numbers are not comparable across vendors.** Vendor numbers cluster in the 90s, but they measure different things (Recall@k versus LLM-judged accuracy), use different harnesses, and LoCoMo has disputed answer-key errors (one audit reports 6.4% wrong). Source: [Mem0, AI memory benchmarks](https://mem0.ai/library/agent-memory/ai-memory-benchmarks-in-2026) (vendor, unverified); [LoCoMo audit](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer) (competitor, unverified).
- **Retrieval is measured far more than writing and forgetting.** Recommended reading: pair retrieval scores with token cost and lifecycle behaviour. plaesy reports only retrieval.
- **Second-brain systems fail on maintenance.** Practitioner reports say capture without revisiting produces an archive. plaesy's hygiene audit addresses part of this; scheduling and a review routine do not exist yet. Source: [Nexasphere, 2026 guide](https://nexasphere.io/blog/ai-personal-knowledge-management-system-2026) (vendor opinion, unverified).

## Improvements, in order of value

1. **Secret scan in `hygiene`.** Flag likely credentials (API key patterns, private key headers, `password=` lines) and report file and line only, never the value. Keep it read-only. This closes the largest gap.
2. **Evaluate end-to-end.** Add QA accuracy and token cost next to Recall@k on the same 94 questions, and run the full 470 set. Report all three.
3. **Decide the default embedder.** Measured: e5 wins on Indonesian and ties on English. Owner decision pending; see [embedder-decision](eval/embedder-decision.md).
4. **Scheduled hygiene.** Document a cadence (see [OPERATIONS](OPERATIONS.md)).
5. **Poisoning check.** Flag notes whose `source` or `last_verified` is missing when the tier makes a factual claim. Partly covered by `memory_hygiene` already.

## Not planned

- LLM in the read path (see [competitor-improvements-scope](../.plaesy/decisions/competitor-improvements-scope.md)).
- Write tools over MCP.
- Automatic capture of conversations.

## Caveats

- Competitor claims come from READMEs and search summaries. Items marked "unverified" were not read in full.
- The probe for Indonesian is eight queries. The English comparison is 94 questions; one question is about 1.1 points.
