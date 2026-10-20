"""Regression: the provider must start even when Hermes passes a workspace that is not a directory.

Popen(cwd=<missing dir>) fails on Windows with WinError 267, which left the provider uninitialized
while Hermes still logged it as activated (2026-10-19). Needs the release binary; skipped without it.

Run:  python -I -m unittest discover -s hermes-plugin/tests -v
"""

import os
import tempfile
import unittest
from pathlib import Path

from test_core_blocks import _load_shim

REPO = Path(__file__).resolve().parents[2]
EXE = "plaesy-memory-plugin.exe" if os.name == "nt" else "plaesy-memory-plugin"
BINARY = REPO / "target" / "release" / EXE


class ProviderStartsWithMissingWorkspace(unittest.TestCase):
    def test_initialize_with_missing_workspace_connects(self):
        if not BINARY.is_file():
            self.skipTest(f"release binary not built: {BINARY}")
        saved = {k: os.environ.get(k) for k in ("PLAESY_MEMORY_BIN", "MEMORY_WATCH_DIR", "MEMORY_DB_PATH")}
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.environ["PLAESY_MEMORY_BIN"] = str(BINARY)
                os.environ["MEMORY_WATCH_DIR"] = str(Path(tmp) / "vault")  # empty: nothing to index
                os.environ["MEMORY_DB_PATH"] = str(Path(tmp) / "state.db")
                shim = _load_shim()
                provider = shim.PlaesyMemoryProvider()
                provider.initialize(
                    "regression",
                    hermes_home=tmp,
                    agent_workspace=str(Path(tmp) / "does-not-exist"),
                    agent_context="primary",
                )
                try:
                    self.assertIsNotNone(provider._client, "binary was not spawned")
                    self.assertTrue(provider._client.alive())
                finally:
                    provider.shutdown()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
