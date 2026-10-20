# MCP server

`plaesy-memory-plugin --mcp` speaks the [Model Context Protocol](https://modelcontextprotocol.io) over stdio
(newline-delimited JSON-RPC 2.0; protocol versions 2025-06-18, 2025-03-26, 2024-11-05). It answers immediately and
syncs/watches the notes directory in the background, exactly like the Hermes plugin mode. All tools are read-only.

## Tools

| Tool | Arguments | Returns |
|------|-----------|---------|
| `memory_search` | `query`, `limit` (1-50, default 5), `mode` (hybrid/semantic/keyword), `tier`, `path_prefix`, `min_trust`, `include_inactive`, `as_of` (YYYY-MM-DD) | cited passages: `path:lines [FACT\|INFERRED] indexed <ts> @ <rev> (tier, trust, verified, STALE)` followed by the verbatim text; `structuredContent.hits` has the same data as JSON |
| `memory_backlinks` | `path`, `direction` (incoming/outgoing) | `[[wikilinks]]` with source file:line and resolved target |
| `memory_hygiene` | - | stale notes, missing `last_verified`/`confidence`, broken links, dead path references |
| `memory_stats` | - | document/chunk counts, sizes, whether semantic search is available |

Tool failures (bad arguments, engine errors) are returned as `isError: true` results so the model can see and correct
them; protocol errors (unknown method, malformed JSON) use JSON-RPC error codes.

## Client configuration

Environment: `MEMORY_WATCH_DIR` (notes directory), `MEMORY_DB_PATH` (index location), optionally `MEMORY_EXCLUDE_DIRS`
(comma-separated folders or file names to keep out of the index, see [SETUP.md](SETUP.md)) and `PLAESY_MODEL_DIR`.

Claude Code:

```bash
claude mcp add plaesy-memory \
  -e MEMORY_WATCH_DIR=/path/to/notes -e MEMORY_DB_PATH=/path/to/index/state.db \
  -- /path/to/plaesy-memory-plugin --mcp
```

Claude Desktop / Cursor (`mcpServers` JSON):

```json
{
  "mcpServers": {
    "plaesy-memory": {
      "command": "C:/path/to/plaesy-memory-plugin.exe",
      "args": ["--mcp"],
      "env": { "MEMORY_WATCH_DIR": "C:/path/to/notes", "MEMORY_DB_PATH": "C:/path/to/index/state.db" }
    }
  }
}
```

Hermes (`config.yaml`, `mcp_servers`):

```yaml
mcp_servers:
  plaesy-memory:
    command: /path/to/plaesy-memory-plugin
    args: ["--mcp"]
    env: { MEMORY_WATCH_DIR: /path/to/notes, MEMORY_DB_PATH: /path/to/index/state.db }
```

Hermes users can alternatively install the provider (`make install-hermes-plugin`), which additionally pins
`tier: core` notes into the system prompt and prefetches per-turn recall; MCP has no equivalent of either.

## Smoke test

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"memory_search","arguments":{"query":"deploy","mode":"keyword"}}}' \
  | MEMORY_WATCH_DIR=./notes MEMORY_DB_PATH=/tmp/s.db plaesy-memory-plugin --mcp
```

Searches issued before the first background sync finishes only see what is indexed so far.
