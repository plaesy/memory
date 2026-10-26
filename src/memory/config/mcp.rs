use serde_json::{json, Value};
use std::collections::HashMap;

pub async fn run(
    app: Option<String>,
    list: bool,
    binary: String,
    notes_dir: String,
    http_url: Option<String>,
) -> crate::memory::Result<()> {
    let snippets = all_snippets(&binary, &notes_dir, &http_url);

    if list {
        for (name, snippet) in &snippets {
            println!("=== {} ===", name);
            println!("{}", snippet);
            println!();
        }
        return Ok(());
    }

    match app {
        Some(ref name) => {
            if let Some(snippet) = snippets.get(name.as_str()) {
                println!("{}", snippet);
            } else {
                eprintln!("error: unknown app '{name}'");
                eprintln!(
                    "available: {}",
                    snippets
                        .keys()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                std::process::exit(2);
            }
        }
        None => {
            println!("Usage: memory config mcp --app <name> [--binary BIN] [--notes_dir DIR] [--http_url URL]");
            println!();
            println!("Available apps:");
            for name in snippets.keys() {
                println!("  * {name}");
            }
            println!();
            println!("Print all snippets: memory config mcp --list");
        }
    }

    Ok(())
}

fn all_snippets(
    binary: &str,
    notes_dir: &str,
    http_url: &Option<String>,
) -> HashMap<String, String> {
    let mut m: HashMap<String, String> = HashMap::new();

    m.insert(
        "claude-code".to_string(),
        claude_code(binary, notes_dir, http_url),
    );
    m.insert(
        "kilocode".to_string(),
        kilocode(binary, notes_dir, http_url),
    );
    m.insert("cursor".to_string(), cursor(binary, notes_dir, http_url));
    m.insert("n8n".to_string(), n8n(binary, notes_dir, http_url));
    m.insert("dify".to_string(), dify(binary, notes_dir, http_url));
    m.insert("crewai".to_string(), crewai(binary, notes_dir, http_url));
    m.insert(
        "paperclip".to_string(),
        paperclip(binary, notes_dir, http_url),
    );
    m.insert("hermes".to_string(), hermes(binary, notes_dir, http_url));

    m
}

fn stdio_config(binary: &str, notes_dir: &str, http_url: &Option<String>) -> Value {
    let args = match http_url {
        Some(url) => vec![
            binary.to_string(),
            "serve".to_string(),
            "--mcp".to_string(),
            "--transport".to_string(),
            "http".to_string(),
            "--port".to_string(),
            url.trim_start_matches("http://")
                .trim_start_matches("https://")
                .to_string(),
        ],
        None => vec![binary.to_string(), "serve".to_string(), "--mcp".to_string()],
    };

    json!({
        "type": "stdio",
        "command": args[0],
        "args": &args[1..],
        "env": {},
        "description": format!("Plaesy Memory MCP (stdio), notes_dir={notes_dir}")
    })
}

fn http_config(http_url: &Option<String>) -> Value {
    match http_url {
        Some(url) => json!({
            "type": "streamable_http",
            "url": url,
            "headers": {}
        }),
        None => json!({
            "type": "streamable_http",
            "url": "http://127.0.0.1:8080/mcp",
            "headers": {}
        }),
    }
}

fn claude_code(binary: &str, notes_dir: &str, http_url: &Option<String>) -> String {
    let cfg = http_config(http_url);
    let stdio_cfg = stdio_config(binary, notes_dir, http_url);
    format!(
        r#"# Claude Code / Claude Desktop — Plaesy Memory MCP
# Add to: ~/.claude.json or ~/.config/claude/claude_desktop_config.json

{{
  "mcpServers": {{
    "plaesy-memory": {}
  }}
}}

# Stdio alternative (single command, no HTTP server):
{{
  "mcpServers": {{
    "plaesy-memory": {}
  }}
}}

# Then run: claude mcp add plaesy-memory
"#,
        serde_json::to_string_pretty(&cfg).unwrap(),
        serde_json::to_string_pretty(&stdio_cfg).unwrap()
    )
}

fn kilocode(binary: &str, notes_dir: &str, http_url: &Option<String>) -> String {
    let cfg = http_config(http_url);
    let stdio_cfg = stdio_config(binary, notes_dir, http_url);
    format!(
        r#"# Kilo Code / Kilo MCP — Plaesy Memory MCP
# Add to kilo.json / kilo-config.json:

{{
  "mcpServers": {{
    "plaesy-memory": {}
  }}
}}

# Stdio alternative:
{{
  "mcpServers": {{
    "plaesy-memory": {}
  }}
}}

# Then run: kilo mcp add plaesy-memory
"#,
        serde_json::to_string_pretty(&cfg).unwrap(),
        serde_json::to_string_pretty(&stdio_cfg).unwrap()
    )
}

fn cursor(binary: &str, notes_dir: &str, http_url: &Option<String>) -> String {
    let cfg = http_config(http_url);
    let stdio_cfg = stdio_config(binary, notes_dir, http_url);
    format!(
        r#"# Cursor — Plaesy Memory MCP
# Add to: ~/.cursor/mcp.json

{{
  "mcpServers": {{
    "plaesy-memory": {}
  }}
}}

# Stdio alternative:
{{
  "mcpServers": {{
    "plaesy-memory": {}
  }}
}}

# Then run: cursor mcp add plaesy-memory
"#,
        serde_json::to_string_pretty(&cfg).unwrap(),
        serde_json::to_string_pretty(&stdio_cfg).unwrap()
    )
}

fn n8n(binary: &str, _notes_dir: &str, http_url: &Option<String>) -> String {
    format!(
        r#"# n8n — Plaesy Memory MCP
# Option A: MCP Server Trigger node (n8n is the MCP server host)
# In n8n, add "MCP Server Trigger" and configure it to run:
#   {binary} serve --mcp --transport http --port 8080
# Then tools "search", "stats", "links" are available in workflows.

# Option B: MCP Client node (n8n calls remote MCP server)
# Set "Server URL" to:
#   {url}
# Available tools: memory_search, memory_stats, memory_links

# Start the server first:
#   {binary} serve --mcp --transport http --port 8080
"#,
        binary = binary,
        url = http_url.as_deref().unwrap_or("http://127.0.0.1:8080/mcp")
    )
}

fn dify(binary: &str, _notes_dir: &str, http_url: &Option<String>) -> String {
    format!(
        r#"# Dify — Plaesy Memory MCP
# In Dify: Knowledge / MCP Server / Add MCP Server
# Select "Streamable HTTP" transport.
# Server URL:
#   {url}
# Available tools: memory_search, memory_stats, memory_links
#
# Start the server first:
#   {binary} serve --mcp --transport http --port 8080
"#,
        binary = binary,
        url = http_url.as_deref().unwrap_or("http://127.0.0.1:8080/mcp")
    )
}

fn crewai(binary: &str, _notes_dir: &str, _http_url: &Option<String>) -> String {
    format!(
        r#"# CrewAI — Plaesy Memory MCP
# pip install crewai-tools[mcp]

from crewai_tools import MCPServerAdapter
from crewai import Agent, Task, Crew

# Stdio transport:
with MCPServerAdapter({{ command: "{binary}", args: ["serve", "--mcp"] }}) as mcp:
    agent = Agent(role="Researcher", tools=mcp.tools)
    ...
"#,
        binary = binary
    )
}

fn paperclip(binary: &str, _notes_dir: &str, _http_url: &Option<String>) -> String {
    format!(
        r#"# Paperclip / Hindsight — Plaesy Memory MCP
# Paperclip uses @vectorize-io/hindsight-paperclip plugin.
# Configure in paperclip.config.ts:

import {{ HindsightPaperclip }} from "@vectorize-io/hindsight-paperclip";

export default {{
  plugins: [
    new HindsightPaperclip({{
      mcpCommand: "{binary}",
      mcpArgs: ["serve", "--mcp"],
    }}),
  ],
}};
"#,
        binary = binary
    )
}

fn hermes(binary: &str, _notes_dir: &str, _http_url: &Option<String>) -> String {
    format!(
        r#"# Hermes — Plaesy Memory MCP (external provider)
# Hermes connects to external memory providers via MCP.
# Start the MCP server (stdio):
#   {binary} serve --mcp
#
# Or over HTTP Streamable:
#   {binary} serve --mcp --transport http --port 8080
#
# Then add Plaesy as an external provider in Hermes settings,
# pointing at the stdio command or the HTTP URL above.
"#,
        binary = binary
    )
}
