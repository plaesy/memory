"""Tests for the provider shim's pure helpers (no Hermes, no Rust binary needed).

Run:  python -I -m unittest discover -s hermes-plugin/tests -v
"""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

SHIM = Path(__file__).resolve().parents[1] / "plaesy-memory" / "__init__.py"


def _load_shim():
    # The shim imports Hermes' MemoryProvider; stub it so the helpers can be tested standalone.
    pkg = types.ModuleType("agent")
    mod = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # noqa: D401 - stand-in base class
        pass

    mod.MemoryProvider = MemoryProvider
    sys.modules.setdefault("agent", pkg)
    sys.modules["agent.memory_provider"] = mod
    spec = importlib.util.spec_from_file_location("plaesy_shim_under_test", SHIM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shim = _load_shim()


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


class CoreBlocks(unittest.TestCase):
    def test_picks_core_by_path_or_tier_and_skips_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root, "00-core/persona.md", "---\nread_only: true\n---\nBe direct.\n")
            write(root, "20-semantic/fact.md", "---\ntier: semantic\n---\nNot pinned.\n")
            write(root, "misc/pinned-by-tier.md", "---\ntier: core\n---\nPinned by tier.\n")
            write(root, "00-core/explicit-other.md", "---\ntier: semantic\n---\nTier overrides path.\n")
            write(root, "00-core/old.md", "---\nstatus: superseded\n---\nOld persona.\n")
            out = shim.load_core_blocks(root)
            self.assertIn("Be direct.", out)
            self.assertIn("00-core/persona.md (read-only)", out)
            self.assertIn("Pinned by tier.", out)
            self.assertNotIn("Not pinned.", out)
            self.assertNotIn("Tier overrides path.", out)
            self.assertNotIn("Old persona.", out)
            self.assertTrue(out.startswith("<core-memory>") and out.endswith("</core-memory>"))

    def test_block_and_total_limits_are_marked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root, "00-core/a.md", "---\nlimit: 300\n---\n" + "x" * 1000)
            write(root, "00-core/b.md", "b" * 100)
            out = shim.load_core_blocks(root, total_limit=350)
            self.assertIn("…[truncated]", out)
            self.assertNotIn("b" * 100, out)  # budget exhausted before b.md
            self.assertIn("omitted", out)

    def test_empty_or_missing_dir(self):
        self.assertEqual(shim.load_core_blocks(Path("/definitely/not/here")), "")
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(shim.load_core_blocks(Path(d)), "")

    def test_frontmatter_parser_is_tolerant(self):
        meta, body = shim._split_frontmatter("---\ntier: 'core'\n# c\nlimit: 5\n---\nhello")
        self.assertEqual(meta, {"tier": "core", "limit": "5"})
        self.assertEqual(body, "hello")
        self.assertEqual(shim._split_frontmatter("no frontmatter"), ({}, "no frontmatter"))
        self.assertEqual(shim._split_frontmatter("---\nunterminated"), ({}, "---\nunterminated"))


class Formatting(unittest.TestCase):
    def test_hits_show_source_trust_and_stale(self):
        hit = {
            "text": "Deploy on Fridays.",
            "score": 0.8,
            "confidence": "FACT",
            "citation": {
                "file_path": "20-semantic/deploy.md",
                "line_range": [3, 4],
                "revision": "576b7b5e59cfd6516",
                "source": {"tier": "semantic", "trust": "high", "last_verified": "2020-01-01", "stale": True},
            },
        }
        out = shim.format_hits([hit])
        self.assertIn("20-semantic/deploy.md:3-4 @ 576b7b5e59", out)
        self.assertIn("tier semantic, trust high, verified 2020-01-01, STALE", out)
        self.assertIn("Deploy on Fridays.", out)

    def test_hits_without_source_metadata_still_render(self):
        out = shim.format_hits([{"text": "t", "score": 0.1, "citation": {"file_path": "a.md", "line_range": [1, 1]}}])
        self.assertIn("a.md:1-1", out)
        self.assertNotIn("STALE", out)


if __name__ == "__main__":
    unittest.main()
