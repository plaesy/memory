# Projects, Decisions, Contexts — the cross-agent memory graph

This is how every agent (any session, any chat, any tool) reads and writes the
*same* projects, decisions, contexts, and knowledge, instead of each agent
re-learning things the user already explained, and without the character-limit
problem of stuffing everything into one `MEMORY.md`.

No new database, no migration: a note becomes part of the graph purely by
declaring `type` in its frontmatter. Notes without `type` are unaffected and
still searchable as before — this is a strict opt-in addition on top of the
existing storage.

## Frontmatter contract

```yaml
---
type: project            # project | decision | context | knowledge | capture | preference
id: studio-a              # stable id, referenced by `project` / `scope` / `supersedes`
status: active            # see "Status values" below — per type
project: studio-a         # decision/context: the project it mainly belongs to
scope: [studio-a, studio-b]   # optional: also relevant to these projects, without being "owned" by them
supersedes: [old-decision-id] # decision only: ids of decisions this one replaces
title: Human readable title
---
```

- `id` is required once `type` is set. A note with `type` but no `id` is
  **skipped from every listing** (never crashes a command) and is reported by
  `memory hygiene` as `malformed_graph_note`.
- `project` is singular ownership. `scope` is a list for "also relevant to,
  without being owned by" — a decision reached via `scope` only (not
  `project`) is shown tagged `(shared)` when listed under that project.
- `supersedes` only applies to decisions and enables a reverse
  `superseded by: <title>` lookup on the decision it replaces.

### Status values

| type     | values                                   | hidden by default            |
|----------|-------------------------------------------|-------------------------------|
| project  | active, paused, completed, archived       | completed, archived          |
| decision | proposed, active, superseded, revoked     | superseded, revoked          |
| context  | active, archived                          | archived                     |

Missing `status` defaults to `active`. Pass `--include-inactive` or an
explicit `--status` to see hidden ones; `--status` always wins and shows
exactly what you asked for.

## Commands

```
memory projects [<id>] [--include-inactive] [--format text|json]
memory projects <id|*> decisions [--include-inactive] [--status ...] [-n N]
memory projects <id|*> contexts  [--include-inactive] [--status ...] [-n N]
memory decisions [--project <id>]... [--status ...] [--include-inactive] [-n N]
memory contexts  [--project <id>]... [--status ...] [--include-inactive] [-n N]
memory hygiene <vault>   # also reports broken_project_reference, broken_scope_reference,
                          # conflicting_supersession, and malformed_graph_note
```

### Id resolution

`<id>` accepts an exact id, a unique id prefix (git-style — e.g. `8f3a` for
`8f3a9c21-...`), or a case-insensitive title match.

- Ambiguous prefix → error listing every candidate, exit 1.
- No match → error with Levenshtein-based fuzzy title suggestions, exit 1.
- `*` means "every project", but **only directly before a subcommand**
  (`projects * decisions`). Used alone (`projects *`) it is an explicit
  error — never silently treated as a literal id, so scripts fail loudly
  instead of silently returning nothing.

### Filters vs. direct targets (important distinction)

- `projects <id>` (direct target): an unknown id is an **error**, exit 1.
- `--project <id>` (filter, on `decisions`/`contexts`): an unknown id
  **filters to an empty result**, exit 0, with a `note:` on stderr. This
  matches the existing `--tier` filter behavior and keeps scripts from
  breaking when a project is renamed or removed.
- Multiple `--project` flags are a union, not an intersection — consistent
  with `--tier`.

### Data integrity is availability-first

Two decisions both claiming to supersede the same decision, or a `supersedes`
id pointing nowhere, do **not** block the listing — both are shown with a
`⚠ conflicting supersession — run 'memory hygiene --strict'` warning instead.
Corruption is surfaced, never hidden, but a dirty vault still answers queries.
Run `memory hygiene` for the full detail and exact file paths.

## Why this makes agents share memory

Any agent (Hermes, Claude Code, an MCP client, a cron job) that already
indexes this vault automatically sees `type: decision`/`type: project`
notes the moment they're written by *any other* agent — no separate sync
step, no re-teaching, no per-agent memory file, no character budget. The
`graph` module (`src/memory/graph/mod.rs`) builds this view by reading the
same `documents.metadata` JSON column the search index already populates,
so it requires no schema migration and stays in sync with whatever is on
disk at query time.
