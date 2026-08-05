# Project Summary: Scaling Ladder Eval Observatory

> **Read this file at the start of every Claude Code session**, along with `architecture.md` and `tasks.md`, before writing any code.

## What this project is

A small, complete, end-to-end LLM evaluation system that:

1. Evaluates a **scaling ladder** of Pythia models (70M, 160M, 410M, 1B) — including a few intermediate training checkpoints — on perplexity and downstream benchmarks.
2. Implements **perplexity from scratch** (sliding-window, with bits-per-byte as the cross-model metric) and charts its relationship to downstream accuracy across scale.
3. Verifies **parity against lm-evaluation-harness** on two benchmarks, with per-example diffs and root-caused discrepancies.
4. Runs a small **prompt sensitivity experiment**: 2–3 prompt formats per benchmark, showing how reported accuracy shifts with formatting across the ladder.
5. Stores every run in **SQLite** and generates the report figures (scaling curves, PPL-vs-accuracy, prompt sensitivity) with a single CLI command.

## Primary deliverable

A short written **findings report** (`report/findings.md`) with 3–4 generated figures:

- PPL/bits-per-byte improving smoothly across the ladder while "emergent" benchmarks (MMLU, GSM8K) stay at chance and "smooth" benchmarks (LAMBADA, ARC-Easy, HellaSwag) track it — perplexity's power and its limits in one chart.
- A parity table vs. lm-evaluation-harness with every discrepancy explained.
- A prompt sensitivity table showing accuracy deltas across formats.


## Scope and standalone status

Fully self-contained: everything is implemented from scratch in this repo. One Python package, one SQLite file, one CLI (`ladderctl`). No web services, no dashboard, no training.

## Core design decisions

1. **Models: Pythia only** — 70M, 160M, 410M, 1B; final checkpoint plus 2 intermediate revisions each (~12 model×revision points). All runnable on one consumer GPU (16GB NVIDIA GeForce RTX 4090).
2. **Perplexity corpora: WikiText-103 test + a fixed C4 validation slice.** Bits-per-byte is the canonical cross-model metric; raw PPL is reported per-model only. (Paloma per-domain PPL is a stretch goal, not core scope.)
3. **Five benchmarks, chosen to span behavior and format:**
   - Smooth at small scale: LAMBADA (cloze), ARC-Easy, HellaSwag (loglik multiple-choice)
   - Flat until large scale: MMLU (subset of subjects), GSM8K (generative)
4. **Parity: lm-evaluation-harness only**, on ARC-Easy and HellaSwag. We parse its output files with an importer; we never run it from our code or tests.
5. **Prompt variants are versioned YAML files** (2–3 per MC benchmark: e.g. lettered options vs. full-option-text scoring, with/without instruction line). Editing a template means bumping the version.
6. **Every metric has a hand-computed fixture test; every dataset loader has a bundled JSONL fixture** and works from it with the network disabled. The whole test suite runs offline against a deterministic `DummyClient`.
7. **Evaluation sweep size is fixed and modest**: ~12 model points × 7 eval targets (5 benchmarks + 2 PPL corpora), capped at ~500 examples per benchmark. Predictions are cached so interrupted sweeps resume for free.

## Explicitly out of scope (future-work section of the report, not code)

- Training any models; OLMo or other suites; Paloma per-domain PPL
- A dashboard or any web service (figures are generated files)
- A third parity framework; concurrency; DB migrations
- More benchmarks, more prompt variants, few-shot sweeps beyond 0-shot vs 5-shot on one benchmark

## Success criteria (definition of done)

1. `ladderctl sweep run sweeps/main.yaml` completes the full grid, resumably.
2. `ladderctl figures` regenerates all report figures from the DB alone.
3. Parity report: ≥2 benchmarks, all diffs categorized, zero unexplained.
4. `pytest` passes with network disabled.
5. `report/findings.md` is complete.

