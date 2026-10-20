# Changelog

Format: [Keep a Changelog](https://keepachangelog.com). Versions follow SemVer once 0.1.0 is tagged; nothing is tagged yet.

## [Unreleased]

### Added
- Multilingual embedding model support: `multilingual-e5-small` via `MEMORY_MODEL_NAME` (Hugging Face `tokenizer.json`
  through the `tokenizers` crate; E5 `query: ` / `passage: ` prefixes applied automatically). Default unchanged.
- The index records the embedding model. Switching models re-embeds the vault on the next sync and logs a warning; a
  database built before this was recorded is treated as `all-MiniLM-L6-v2`.
- Probe and measurement scripts: `scripts/id_probe.sh`, `scripts/latency_probe.py`, data in `docs/eval/id-probe/`.
- `expand_links` in the Hermes provider tool and in MCP `memory_search`; `as_of` in the Hermes provider tool.
- Temporal validity: sources declare `valid_from` / `valid_to` in frontmatter; `as_of` (`--as-of`, `as_of` in the plugin
  protocol and MCP) keeps the sources whose period contains that date. Undated sources always pass.
- One-hop link expansion: `search_expanded` (`--expand-links`, `expand_links` in the plugin protocol) adds the best passage
  of notes linked to or from the top hits, after the direct hits, scored 0.8× with their own evidence label.
- `MEMORY_EXCLUDE_DIRS` / `[search] exclude_dirs`: folder runs or file names kept out of the index, matched anywhere in the tree.
  Applies to directory sync, watcher events and coverage. Excluded documents are removed on the next sync.
- Hermes provider regression test for a missing workspace (`hermes-plugin/tests/test_provider_cwd.py`).

### Fixed
- Hermes provider did not start on Windows when Hermes passed no workspace: `Popen` used a missing directory as `cwd`
  (WinError 267). `cwd` is now used only when the directory exists.
- `coverage_percent` was always 0.0 in `stats` outside tests. It is now computed from the watch root.
- A heading directly above another heading (or above a list, code block or quote) was indexed as a chunk of its own, so a
  search hit showed only the title. The title now joins the block that follows it.

### Changed
- Chunking: a heading run also takes following lists, code blocks and quotes (within `chunk_size`), not only paragraphs.
  Existing indexes keep the old chunks until rebuilt (see docs/SETUP.md, "Upgrading an existing index").
