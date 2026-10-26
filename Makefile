.PHONY: help build release test test-one lint fmt fmt-check bench eval lme test-plugin docs clean model install-hermes-plugin verify reload

help:
	@echo "Plaesy Memory - build targets"
	@echo ""
	@echo "  make build                 Debug build"
	@echo "  make release               Optimised build"
	@echo "  make reload                Debug build + install to local"
	@echo "  make test                  Run all tests"
	@echo "  make test-one TEST=name    Run tests matching name"
	@echo "  make eval                  Relevance evaluation (Recall@3 / MRR), prints the table"
	@echo "  make lme DATA=file.json    LongMemEval-S retrieval benchmark (STRIDE=5 for every 5th question)"
	@echo "  make test-plugin           Python tests for the Hermes provider shim"
	@echo "  make bench                 Criterion benchmarks"
	@echo "  make lint                  clippy with -D warnings"
	@echo "  make fmt / fmt-check       Format / check formatting"
	@echo "  make model                 Download the ONNX embedding model (one-time)"
	@echo "  make install-hermes-plugin Install provider + binary into the Hermes home"
	@echo "  make verify                build + test + lint + fmt-check"

build:
	cargo build

release:
	cargo build --release

test:
	cargo test --all

test-one:
	cargo test --all $(TEST)

eval:
	cargo test --test relevance_eval -- --nocapture

bench:
	cargo bench

STRIDE ?= 1
lme:
	@test -n "$(DATA)" || { echo "usage: make lme DATA=longmemeval_s_cleaned.json [STRIDE=5]"; exit 2; }
	cargo run --release --example longmemeval -- "$(DATA)" --stride $(STRIDE)

PYTHON ?= python
test-plugin:
	@test -d hermes-plugin/tests || { echo "hermes-plugin/tests missing"; exit 1; }
	$(PYTHON) -I -m unittest discover -s hermes-plugin/tests -v

lint:
	cargo clippy --all-targets -- -D warnings

fmt:
	cargo fmt

fmt-check:
	cargo fmt -- --check

# Cargo.lock is intentionally kept (this crate ships binaries).
clean:
	cargo clean

docs:
	cargo doc --no-deps --open

model:
	./scripts/install-model.sh

install-hermes-plugin: release
	./scripts/install-hermes-plugin.sh

verify: build test test-plugin lint fmt-check
	@echo "All checks passed"

reload: build
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/reload.ps1

.DEFAULT_GOAL := help

# Development

# GNU make on Windows ignores SHELL and runs recipes with cmd.exe, where `rm` does not exist.
ifeq ($(OS),Windows_NT)
RM_GIT = powershell -NoProfile -Command "if (Test-Path .git) { Remove-Item -Recurse -Force .git }"
else
RM_GIT = rm -rf .git
endif

reset:
	$(RM_GIT) && git init && git remote add origin git@github.com:plaesy/memory.git && git checkout -b main && git add -A && git commit -m "Initial Commit"

resetf: reset
	git push origin main -f
