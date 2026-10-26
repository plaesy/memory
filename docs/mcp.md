# MCP server

`memory serve --mcp` speaks the [Model Context Protocol](https://modelcontextprotocol.io) over **two transports**:

- **stdio** (default): newline-delimited JSON-RPC 2.0 (protocol versions 2025-06-18, 2025-03-26, 2024-11-05)
- **http**: Streamable HTTP (POST /mcp for requests, SSE for notifications)

Both transports sync/watch the notes directory in the background, exactly like the Hermes plugin mode. Queries and reports are read-only.

## Transports

### `stdio` (default)

```bash
memory serve --mcp
```

Used by the Hermes plugin and any MCP client supporting stdio JSON-RPC 2.0.

### `http` (Streamable)

```bash
memory serve --mcp --transport http --port 8080
```

Used by n8n, Dify, Kilo Code, Cursor, Claude Code, and other HTTP-based MCP clients.

## Client configuration

### Claude Code / Claude Desktop

```bash
# HTTP transport (recommended)
claude mcp add plaesy-memory \
  -- /path/to/memory serve --mcp --transport http --port 8080

# Stdio alternative (single command, no HTTP server)
claude mcp add plaesy-memory \
  -e MEMORY_WATCH_DIR=/path/to/notes -e MEMORY_DB_PATH=/path/to/index/state.db \
  -- /path/to/memory serve --mcp
```

Then run: `claude mcp add plaesy-memory`

### Kilo Code

```bash
# HTTP transport
kilo mcp add plaesy-memory \
  -- /path/to/memory serve --mcp --transport http --port 8080

# Stdio alternative
kilo mcp add plaesy-memory \
  -e MEMORY_WATCH_DIR=C:/path/to/notes -e MEMORY_DB_PATH=C:/path/to/index/state.db \
  -- /path/to/memory serve --mcp
```

### n8n

**Option A - MCP Server Trigger (n8n hosts the MCP server):**

1. Start the memory server: `memory serve --mcp --transport http --port 8080`
2. In n8n, add "MCP Server Trigger" node
3. Available tools: `memory_search`, `memory_stats`, `memory_links`

**Option B - MCP Client (n8n calls remote MCP server):**

- Set "Server URL" to: `http://127.0.0.1:8080/mcp`
- Available tools: `memory_search`, `memory_stats`, `memory_links`

### Dify

In Dify: Knowledge / MCP Server / Add MCP Server → select "Streamable HTTP".
Server URL: `http://127.0.0.1:8080/mcp`
Available tools: `memory_search`, `memory_stats`, `memory_links`

### Claude Code (quick reference)

Run `memory config mcp --app claude-code` to see the exact JSON snippet.

### CrewAI

```bash
pip install crewai-tools[mcp]

from crewai_tools import MCPServerAdapter
with MCPServerAdapter({"command": "memory", "args": ["serve", "--mcp"]}) as mcp:
    agent = Agent(role="Researcher", tools=mcp.tools)
    ...
```

### Paperclip / Hindsight

```bash
# Configure in paperclip.config.ts
import { HindsightPaperclip } from "@vectorize-io/hindsight-paperclip";

export default {
  plugins: [
    new HindsightPaperclip({
      mcpCommand: "memory",
      mcpArgs: ["serve", "--mcp"],
    }),
  ],
}
```

### Hermes (external provider)

Start the MCP server (stdio or HTTP), then add Plaesy as an external provider in Hermes settings,
pointing at the stdio command or HTTP URL.

### Print all snippets

```bash
memory config mcp --list
```

### Print one app's snippet

```bash
memory config mcp --app <name>
# Available: claude-code, crewai, n8n, dify, kilocode, paperclip
```

Searches issued before the first background sync finishes only see what is indexed so far.
