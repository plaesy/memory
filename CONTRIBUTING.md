# Contributing

## Setup

- Rust 1.88+. On Windows use the MSVC toolchain (`stable-x86_64-pc-windows-msvc`); the GNU toolchain lacks `dlltool`/`gcc` for the native dependencies.
- `make model` downloads the embedding model (and `scripts/install-model.sh --rerank` the cross-encoder). Without them the
  ONNX tests print `SKIPPED` and pass, so CI-style runs work, but they cover less.

## Before sending a change

```bash
make verify      # build + cargo test + python shim tests + clippy -D warnings + fmt --check
```

`make test-plugin` runs the Python shim tests with `$(PYTHON)` (default `python`). On Windows, where `python` may be a
Store alias, pass the interpreter explicitly: `make test-plugin PYTHON=path/to/python.exe`. The shim tests need no Hermes
installation; `test_provider_cwd` also needs the release binary and is skipped without it.

Also run `make eval` if you touched ranking, chunking or tokenisation, and say what moved.

## Rules this code base holds itself to

- **No `unwrap()`/`expect()` in library code.** Return `MemoryError` (`thiserror`). Tests may use `expect`.
- **EV-01/EV-03.** Anything that produces a search result must go through `CitationBuilder`; chunk text must stay a verbatim
  slice of the file. If you change the chunker, the line-exactness test must still pass.
- **A label describes how a result was found.** Do not mark a result `FACT` unless the keyword leg matched it.
- **Offline.** No runtime network calls. Model downloads belong in `scripts/`.
- **Measure, don't assert.** A claim about quality or speed goes in the README only with the command that reproduces it, the
  sample size, and the machine. Say when a difference is within noise.
- **Stdio is a protocol.** Never print to stdout from the service/MCP paths; logs go to stderr.
- **Schema changes** bump `SCHEMA_VERSION`, migrate on open, and add a test that opens an older database.

## Tests

Add unit tests beside the code, cross-module behaviour in `tests/`. Tests that need a model must skip with a message when it is
absent. Use `tempfile` and `:memory:` databases; never write into the repository.

## Documentation

Update `docs/api.md` for protocol/API changes, `docs/setup.md` for config keys, `CHANGELOG.md` for anything user-visible,
and `docs/architecture.md` (Decisions table) when you make a trade-off.

## Scripts

Edits with `sed`/`perl` on Rust sources have corrupted files before; prefer a real editor or a script that asserts its match count.
