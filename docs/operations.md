# Operations: running the second brain day to day

Examples use `<vault>` for the folder that holds your notes. Replace it with your own path. Commands are shown for Windows PowerShell; the CLI is `memory`.

## 1. What is where

| Item | Location | Set by |
|---|---|---|
| Vault (your notes, source of truth) | `<vault>`, any folder you choose | `MEMORY_WATCH_DIR` |
| Index database | `%LOCALAPPDATA%\plaesy\memory.db` (+ `.hnsw` snapshot) | default; `MEMORY_DB_PATH` overrides |
| Embedding model | `%LOCALAPPDATA%\plaesy\models\<name>\` | default; `MEMORY_MODEL_DIR` overrides |
| Datasets (evaluation only) | `%LOCALAPPDATA%\plaesy\datasets\` | convention |
| Exclusions | folders or file names that are never indexed | `MEMORY_EXCLUDE_DIRS` |

The index is a cache: deleting `memory.db` and re-indexing rebuilds it from the notes. The notes are never written by plaesy.

## 2. Writing notes so they can be trusted

The engine reads optional frontmatter. Notes without it still work, but cannot declare themselves.

```yaml
---
tier: semantic            # core | episodic | semantic | procedural | archive (inferred from folder if omitted)
confidence: high         # high | medium | low: how sure you are of the claims
last_verified: 2026-10-26
status: active           # superseded | deprecated | archived are hidden unless asked for
valid_from: 2026-01-01   # optional: the period the facts hold
valid_to:                # optional
---
```

Rules that match the engine's behaviour:

- Keep credentials out of notes. The index stores the text, and `memory_search` can return it to a model. See the secret-scan item in [ASSESSMENT](ASSESSMENT.md).
- A note older than 180 days (`last_verified`) is flagged STALE and its score is multiplied by 0.85. It is down-weighted, not hidden.
- Use `[[wikilinks]]` for relations. They power `memory backlinks` and dangling-link checks.
- Folder names are tiers by default: put episodic logs in an `episodic` folder, not in `semantic`.

## 3. Index and search

```powershell
memory index "<vault>"                    # first run, or after large restructures
memory search "how do we deploy"          # hybrid by default
memory search "deploy" --mode keyword    # exact terms
memory search "deploy" --tier procedural --min-trust high
memory links runbook                      # who links to it
memory stats                              # documents, chunks, vector search on/off
```

`memory index` picks up changes on the next run. The watch mode in the CLI keeps the index current while the machine is on.

## 4. Changing the embedding model

The index records the model it was built with. Changing `MEMORY_MODEL_NAME` (for example to `multilingual-e5-small`) re-embeds every document on the next sync. Expect about an hour for a large vault with e5. Keyword search keeps working during the rebuild; semantic results are incomplete until it finishes.

Do not mix models in one database. To compare, use a separate `MEMORY_DB_PATH` per model.

## 5. Connecting an AI

**Claude Code (MCP):**

```powershell
claude mcp add plaesy-memory -e MEMORY_WATCH_DIR=<vault> -- "<path>\memory.exe" serve --mcp
```

Add `-e MEMORY_MODEL_NAME=multilingual-e5-small` only after the model switch in section 4. The server is read-only: four tools (`memory_search`, `memory_backlinks`, `memory_hygiene`, `memory_stats`), each with `readOnlyHint: true`. `memory_search` declares an output schema. Details: [MCP](MCP.md).

**Hermes:** run `make install-hermes-plugin` after any change to the shim, then `hermes config set memory.provider plaesy-memory`. The provider pins `tier: core` notes into the prompt and recalls per turn. The shim does not set `MEMORY_DB_PATH`, so Hermes and the CLI share the default database.

## 6. Routine maintenance

| When | Command | What to look at |
|---|---|---|
| Weekly | `memory hygiene "<vault>" --strict` | STALE notes, missing `last_verified`, broken links, backticked paths that no longer exist |
| Monthly | review `confidence: low` and STALE notes | update, demote to `archive`, or delete |
| After restructuring the vault | `memory index "<vault>"` on a fresh database | the index holds the old paths otherwise |
| After a model change | `memory stats` shows the new model | see section 4 |

`hygiene --strict` exits with a non-zero code when it finds problems, so it can gate a pre-commit hook or a scheduled task.

### Scheduled jobs

`memory schedule once` runs the jobs that are due and exits. `memory schedule run` keeps running until Ctrl-C.

- Jobs are set in the config under `[[schedule.jobs]]`: `name`, `every` (`30m`, `6h`, `1d`), `action` (`index`, `hygiene` or `missing`). With no jobs configured, `index` runs every hour.
- `hygiene` and `missing` do not run unless listed. A vault with known broken links would otherwise add a line to the log every hour.
- Each run appends one line per job to `<root>\Plaesy\schedule.log`. A failing job is logged and the rest still run.
- Last-run times are kept in the database (`schedule_runs`). A second `once` within the interval does nothing.
- Windows: `scripts\install.ps1 -Schedule [-ScheduleEvery 60]` registers the task `PlaesyMemorySchedule` for the current user. It runs only while that user is logged on. Re-running the installer replaces the task. Removing it is manual: `schtasks /Delete /TN PlaesyMemorySchedule /F`.

## 7. Backup and recovery

- The notes are the backup. Keep the vault in git or in a sync folder with history.
- The index is rebuildable. Back it up only if re-indexing a large vault is slow for you.
- Models are a cache. `scripts/install-model.sh` downloads them again.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `stats` shows `documents: 0` | database is new or points to the wrong file | check `MEMORY_DB_PATH`, run `memory index` |
| `vector search: off` | model not found in the models folder | run `scripts/install-model.sh`, or set `PLAESY_MODEL_DIR` |
| Results cite old paths | notes moved after indexing | `memory index` again |
| Hermes writes to its own `state.db` | shim installed before the change | `make install-hermes-plugin` |
| Search is slow on first run | model load and first embedding pass | normal; later runs use the embedding cache |

## 9. Checklist for a new machine

1. Build: `cargo build --release` with the MSVC toolchain.
2. Models: `bash scripts/install-model.sh` (add `--rerank` for the optional cross-encoder).
3. Index: `memory index "<vault>"`.
4. Connect: MCP or Hermes, as in section 5.
5. Check: `memory stats` shows `vector search: on` and the expected document count.
