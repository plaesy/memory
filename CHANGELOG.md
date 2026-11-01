# Changelog

Format: [Keep a Changelog](https://keepachangelog.com). Versions follow SemVer once 0.1.0 is tagged; nothing is tagged yet.

## [Unreleased]

### Added

- `memory hygiene` detects likely credentials without exposing their values. Findings include only a category and source line, and fenced code examples are ignored.

### Security

- Secret detection is read-only and heuristic; it does not replace credential rotation or a dedicated secret scanner.
