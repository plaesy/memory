"""Plaesy Memory — Hermes MemoryProvider backed by the ``plaesy-memory-plugin`` Rust binary.

The binary speaks line-delimited JSON on stdin/stdout (see ``src/memory/service.rs``). It indexes
``<workspace>/.plaesy`` (or ``PLAESY_WATCH_DIR``), keeps the index current with a file watcher and
answers hybrid searches. Every hit carries a citation (file, line range, git revision, timestamp)
and an evidence label (FACT | INFERRED), so recalled text is never presented without a source.

Binary lookup: ``$PLAESY_MEMORY_BIN``, ``<this dir>/bin/``, then ``PATH``.
Storage: ``$HERMES_HOME/plaesy-memory/state.db`` (profile-scoped).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_BIN_NAME = "plaesy-memory-plugin" + (".exe" if sys.platform == "win32" else "")
_REQUEST_TIMEOUT = 30.0
_PREFETCH_LIMIT = 5

SEARCH_SCHEMA = {
    "name": "plaesy_memory_search",
    "description": (
        "Search the project's .plaesy/ knowledge (specs, decisions, instructions, memory) with hybrid "
        "semantic + keyword retrieval. Each result includes its source file, line range and an evidence "
        "label: FACT (lexical match in the source text) or INFERRED (semantic similarity only)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "limit": {"type": "integer", "description": "Max results (default 5, max 50).", "default": 5},
            "mode": {
                "type": "string",
                "enum": ["hybrid", "semantic", "keyword"],
                "description": "hybrid (default), semantic only, or keyword only.",
            },
            "tier": {
                "type": "string",
                "enum": ["core", "episodic", "semantic", "procedural", "archive"],
                "description": "Only notes of this memory tier.",
            },
            "min_trust": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Only notes whose declared confidence is at least this.",
            },
            "include_inactive": {
                "type": "boolean",
                "description": "Also return superseded/deprecated notes (default false).",
            },
            "as_of": {
                "type": "string",
                "description": "Only notes whose valid_from..valid_to period contains this date (YYYY-MM-DD). Undated notes always pass.",
            },
            "expand_links": {
                "type": "boolean",
                "description": "Also add the best passage of notes linked to or from the top hits (default false).",
            },
        },
        "required": ["query"],
    },
}

BACKLINKS_SCHEMA = {
    "name": "plaesy_memory_backlinks",
    "description": "Notes that link to a note via [[wikilinks]] (or, with direction=outgoing, the notes it links to).",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Note path, path without extension, or bare file name."},
            "direction": {"type": "string", "enum": ["incoming", "outgoing"], "default": "incoming"},
        },
        "required": ["path"],
    },
}

HYGIENE_SCHEMA = {
    "name": "plaesy_memory_hygiene",
    "description": "Audit the memory: stale notes, missing last_verified/confidence, broken wikilinks, dead file-path references.",
    "parameters": {"type": "object", "properties": {}},
}

STATS_SCHEMA = {
    "name": "plaesy_memory_stats",
    "description": "Index statistics for the Plaesy memory (documents, chunks, sizes).",
    "parameters": {"type": "object", "properties": {}},
}


def _find_binary() -> Optional[str]:
    env = os.environ.get("PLAESY_MEMORY_BIN")
    if env and Path(env).is_file():
        return env
    local = Path(__file__).parent / "bin" / _BIN_NAME
    if local.is_file():
        return str(local)
    return shutil.which(_BIN_NAME)


class _Client:
    """Thread-safe request/response client for the line-delimited JSON protocol."""

    def __init__(self, binary: str, env: dict[str, str], cwd: Optional[str]) -> None:
        self._proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._next_id = 0

    def alive(self) -> bool:
        return self._proc.poll() is None

    def call(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Send one request and return the ``result`` dict; raise RuntimeError on any failure."""
        with self._lock:
            if not self.alive():
                raise RuntimeError("plaesy-memory-plugin is not running")
            self._next_id += 1
            req_id = self._next_id
            payload = json.dumps({"id": req_id, "method": method, "params": params or {}})
            result: dict[str, Any] = {}

            def _roundtrip() -> None:
                try:
                    assert self._proc.stdin is not None and self._proc.stdout is not None
                    self._proc.stdin.write(payload + "\n")
                    self._proc.stdin.flush()
                    result["line"] = self._proc.stdout.readline()
                except Exception as exc:  # noqa: BLE001 - surfaced below
                    result["error"] = exc

            worker = threading.Thread(target=_roundtrip, daemon=True)
            worker.start()
            worker.join(_REQUEST_TIMEOUT)
            if worker.is_alive():
                self.close()  # a hung binary must not wedge the agent; restart is the caller's call
                raise RuntimeError(f"{method} timed out after {_REQUEST_TIMEOUT:.0f}s")
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        line = result.get("line", "")
        if not line:
            raise RuntimeError("plaesy-memory-plugin closed the pipe")
        resp = json.loads(line)
        if resp.get("id") != req_id:
            raise RuntimeError(f"response id mismatch: {resp.get('id')} != {req_id}")
        if not resp.get("ok"):
            err = resp.get("error") or {}
            raise RuntimeError(f"{err.get('kind', 'error')}: {err.get('message', 'unknown error')}")
        return resp.get("result") or {}

    def close(self) -> None:
        proc = self._proc
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


_CORE_DIR_NAMES = {"core", "working", "pinned"}
_INACTIVE = {"superseded", "deprecated", "archived", "obsolete"}
_DEFAULT_BLOCK_LIMIT = 2000
_DEFAULT_TOTAL_LIMIT = 6000


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (flat key: value frontmatter, body). Tolerant: no YAML dependency."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key.startswith((" ", "\t", "#")):
            meta[key.strip().lower()] = value.strip().strip("\"'")
    body = text[end + 4 :].lstrip("\n")
    return meta, body


def _is_core(rel: Path, meta: dict[str, str]) -> bool:
    if meta.get("tier", "").lower() == "core":
        return True
    if "tier" in meta:  # an explicit tier wins over the path
        return False
    for part in rel.parts[:-1]:
        name = part.lower().lstrip("0123456789-_")
        if name in _CORE_DIR_NAMES:
            return True
    return False


def load_core_blocks(watch_dir: Path, total_limit: int = _DEFAULT_TOTAL_LIMIT) -> str:
    """Pinned notes (tier: core) rendered for the system prompt.

    Core notes are the small set of facts that must always be in context. Each note is capped
    by its frontmatter ``limit`` (characters, default 2000) and the whole block by
    ``total_limit``; anything cut is marked so the model knows it saw a truncated note.
    Superseded/deprecated notes are skipped.
    """
    if not watch_dir.is_dir():
        return ""
    blocks: list[str] = []
    used = 0
    for path in sorted(watch_dir.rglob("*.md")):
        rel = path.relative_to(watch_dir)
        try:
            meta, body = _split_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if not _is_core(rel, meta) or meta.get("status", "").lower() in _INACTIVE:
            continue
        body = body.strip()
        if not body:
            continue
        try:
            limit = max(200, int(meta.get("limit", _DEFAULT_BLOCK_LIMIT)))
        except ValueError:
            limit = _DEFAULT_BLOCK_LIMIT
        room = total_limit - used
        if room < 200:
            blocks.append("[further core notes omitted: core memory budget reached]")
            break
        cap = min(limit, room)
        if len(body) > cap:
            body = body[:cap].rstrip() + " …[truncated]"
        label = rel.as_posix() + (" (read-only)" if meta.get("read_only", "").lower() == "true" else "")
        blocks.append(f"## {label}\n{body}")
        used += len(body)
    if not blocks:
        return ""
    return "<core-memory>\n" + "\n\n".join(blocks) + "\n</core-memory>"


def _source_note(src: dict[str, Any]) -> str:
    """What the source says about itself (tier, trust, last verification, STALE)."""
    parts = []
    if src.get("tier"):
        parts.append(f"tier {src['tier']}")
    if src.get("trust"):
        parts.append(f"trust {src['trust']}")
    if src.get("last_verified"):
        parts.append(f"verified {src['last_verified']}")
    if src.get("stale"):
        parts.append("STALE - verify before relying on it")
    return f" ({', '.join(parts)})" if parts else ""


def format_hits(hits: list[dict[str, Any]], max_chars: int = 1200) -> str:
    """Render hits as cited text. Every entry names its source (EV-01) and evidence label (EV-02)."""
    blocks = []
    for i, hit in enumerate(hits, 1):
        c = hit.get("citation", {})
        lo, hi = c.get("line_range", [0, 0])
        rev = f" @ {c['revision'][:10]}" if c.get("revision") else ""
        text = hit.get("text", "").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " …[truncated]"
        blocks.append(
            f"[{i}] {c.get('file_path', '?')}:{lo}-{hi}{rev} [{hit.get('confidence', '?')}] "
            f"score={hit.get('score', 0):.2f}{_source_note(c.get('source') or {})}\n{text}"
        )
    return "\n\n".join(blocks)


class PlaesyMemoryProvider(MemoryProvider):
    """Memory provider exposing the Rust index as a search tool plus cached per-turn recall."""

    def __init__(self) -> None:
        self._client: Optional[_Client] = None
        self._binary = _find_binary()
        self._lock = threading.Lock()
        self._prefetched: dict[str, str] = {}
        self._read_only = False
        self._watch_dir: Optional[Path] = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def name(self) -> str:
        return "plaesy-memory"

    def is_available(self) -> bool:
        return self._binary is not None

    def unavailable_reason(self) -> str:
        return (
            f"{_BIN_NAME} not found. Build it (cargo build --release), then copy it to "
            f"<plugin dir>/bin/, put it on PATH, or set PLAESY_MEMORY_BIN."
        )

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        if not self._binary:
            raise RuntimeError(self.unavailable_reason())
        home = Path(kwargs.get("hermes_home") or os.environ.get("HERMES_HOME", "."))
        workspace = Path(kwargs.get("agent_workspace") or os.getcwd())
        data_dir = home / "plaesy-memory"
        data_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("MEMORY_DB_PATH", str(data_dir / "state.db"))
        env.setdefault("MEMORY_WATCH_DIR", str(workspace / ".plaesy"))
        env.setdefault("MEMORY_LOG_LEVEL", "warn")
        self._watch_dir = Path(env["MEMORY_WATCH_DIR"])
        # Popen fails with WinError 267 when cwd is not an existing directory (Hermes may run
        # from a cwd it cannot pass to CreateProcess). Fall back to the process default.
        cwd = str(workspace) if workspace.is_dir() else None
        self._client = _Client(self._binary, env, cwd)
        self._read_only = kwargs.get("agent_context", "primary") != "primary"
        try:
            pong = self._client.call("ping")
            logger.info("plaesy-memory ready (vector_search=%s)", pong.get("vector_search"))
        except RuntimeError:
            self._client.close()
            self._client = None
            raise

    def shutdown(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # -- recall ------------------------------------------------------------
    def system_prompt_block(self) -> str:
        core = load_core_blocks(self._watch_dir) if self._watch_dir else ""
        return (core + "\n\n" if core else "") + (
            "You have a `plaesy_memory_search` tool over the project's .plaesy/ knowledge. Results cite their "
            "source (file:lines) and carry an evidence label: FACT = lexical match in the source text, "
            "INFERRED = semantic similarity only. Cite the source when you rely on a result, and never present "
            "an INFERRED result as established fact."
        )

    def _search(
        self, query: str, limit: int, mode: str = "hybrid", **filters: Any
    ) -> list[dict[str, Any]]:
        if not self._client:
            raise RuntimeError("provider not initialized")
        params: dict[str, Any] = {"query": query, "limit": limit, "mode": mode}
        params.update({k: v for k, v in filters.items() if v not in (None, "")})
        result = self._client.call("search", params)
        return result.get("hits", [])

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        def run() -> None:
            try:
                text = format_hits(self._search(query, _PREFETCH_LIMIT))
            except RuntimeError as exc:
                logger.debug("plaesy-memory prefetch failed: %s", exc)
                return
            with self._lock:
                self._prefetched[session_id] = text

        threading.Thread(target=run, daemon=True).start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._lock:
            text = self._prefetched.pop(session_id, "")
        return f"<plaesy-memory>\n{text}\n</plaesy-memory>" if text else ""

    # -- tools -------------------------------------------------------------
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [SEARCH_SCHEMA, BACKLINKS_SCHEMA, HYGIENE_SCHEMA, STATS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        try:
            if tool_name == "plaesy_memory_search":
                query = str(args.get("query", "")).strip()
                if not query:
                    return json.dumps({"error": "query is required"})
                limit = max(1, min(int(args.get("limit", 5)), 50))
                hits = self._search(
                    query,
                    limit,
                    str(args.get("mode", "hybrid")),
                    tier=args.get("tier"),
                    min_trust=args.get("min_trust"),
                    include_inactive=bool(args.get("include_inactive", False)) or None,
                    as_of=args.get("as_of"),
                    expand_links=bool(args.get("expand_links", False)) or None,
                )
                return json.dumps({"count": len(hits), "results": format_hits(hits) or "No results."})
            if tool_name == "plaesy_memory_backlinks":
                assert self._client is not None
                method = "links" if args.get("direction") == "outgoing" else "backlinks"
                return json.dumps(self._client.call(method, {"path": str(args.get("path", ""))}))
            if tool_name == "plaesy_memory_hygiene":
                assert self._client is not None
                return json.dumps(self._client.call("hygiene"))
            if tool_name == "plaesy_memory_stats":
                assert self._client is not None
                return json.dumps(self._client.call("stats"))
        except (RuntimeError, ValueError) as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps({"error": f"unknown tool {tool_name}"})


def register(ctx) -> None:
    """Register Plaesy Memory as a memory provider plugin."""
    ctx.register_memory_provider(PlaesyMemoryProvider())
