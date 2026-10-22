# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-22

First public release.

### Added

- Floating-mean generalized Lomb-Scargle periodogram applied directly to
  irregular timestamps, with no interpolation in the primary analysis.
- Permutation-bootstrap global false-alarm probabilities for every spectral
  peak.
- Interval-based anomaly detection with evidence and plain-language reasons.
- Python API, `telespec` command-line interface, and a local web dashboard
  (`telespec view`).
- Reproducible benchmarks covering frequency recovery under missing and
  jittered samples, false-alarm calibration, synthetic anomalies, and selected
  NAB series.

[Unreleased]: https://github.com/akshajbachu/telespec/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/akshajbachu/telespec/releases/tag/v0.1.0
