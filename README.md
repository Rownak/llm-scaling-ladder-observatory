# Scaling Ladder Eval Observatory

An offline-first LLM evaluation system that evaluates a ladder of Pythia checkpoints (70M–1B) on perplexity and downstream benchmarks, implements sliding-window perplexity and bits-per-byte from scratch, and treats prompt formatting as a measured experimental axis rather than an implementation detail.

## The question this project answers

*(In progress — the findings that answer it land in Sprints 3–5.)*

## What it does

- Evaluates a **scaling ladder** of Pythia models (70M, 160M, 410M, 1B), including intermediate training checkpoints, on perplexity and five downstream benchmarks.
- Implements **perplexity from scratch** — sliding-window, with bits-per-byte as the cross-model metric — and charts its relationship to downstream accuracy across scale.
- Verifies **parity against lm-evaluation-harness** on two benchmarks, with per-example diffs and root-caused discrepancies.
- Runs a **prompt-sensitivity experiment**: 2–3 prompt formats per benchmark, showing how reported accuracy shifts with formatting alone.
- Stores every run in **SQLite** and regenerates all report figures from the database with one CLI command.

## Status

**In progress.** Sprint 1 of 5 is complete; the rest are planned and not yet implemented.

| Sprint | Status | What it adds |
| --- | --- | --- |
| 1 — Walking skeleton | Complete | One model, one benchmark, end-to-end: records, clients, loaders, prompts, `loglik_mc`, accuracy, SQLite, CLI, one figure. All downstream interfaces locked. |
| 2 — Ladder + sweeps | Pending | HellaSwag/MMLU loaders, `acc_norm`, content-hash prediction cache, resumable sweep executor, scaling-curve and trajectory figures. |
| 3 — Perplexity + formats | Pending | `token_nlls`, sliding-window perplexity with ppl/bpb, cloze (LAMBADA) and generative (GSM8K) evaluators, the headline figure. |
| 4 — Prompt sensitivity | Pending | Alternative MC variants (letter vs. option-text, instruction line), 5-shot rendering, variant sweep, sensitivity dot plot. |
| 5 — Parity + report | Pending | lm-eval-harness importer, per-example diff with a fixed discrepancy taxonomy, final findings report. |

Only the Sprint-1 surface described under Quickstart exists today. `ladderctl sweep`, `parity`, and `import` are specified in [architecture.md](architecture.md) but not yet implemented.

## Quickstart

```bash
git clone <repo-url> && cd llm-scaling-ladder-observatory
python -m venv .venv && .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Fully offline demo — deterministic `DummyClient` against the bundled JSONL fixture, no network and no model download:

```bash
ladderctl run --model dummy --dataset arc_easy --variant arc_easy/mc_letter_v1 \
  --evaluator loglik_mc --split fixture --db ./demo.db

ladderctl results --db ./demo.db
ladderctl show <run_id> --db ./demo.db      # full reproducibility metadata
ladderctl figures --db ./demo.db            # writes report/figures/accuracy_per_run.png
```

Real hardware — downloads `EleutherAI/pythia-70m` and `allenai/ai2_arc` on first use, runs on CPU or CUDA:

```bash
ladderctl run --model pythia-70m --revision main --dataset arc_easy \
  --variant arc_easy/mc_letter_v1 --evaluator loglik_mc --split test \
  --limit 200 --db ./ladder.db
```

Implemented today: models `dummy`, `pythia-70m`, `pythia-160m`, `pythia-410m`, `pythia-1b`; dataset `arc_easy`; variant `arc_easy/mc_letter_v1`; evaluator `loglik_mc`; metric `acc`.

## Architecture

One package, ~10 single-file modules, one SQLite file, one CLI. Every stage communicates via Pydantic records, so any stage's output can be dumped to JSONL and inspected.

```
DatasetLoader → PromptVariant renderer → Evaluator → ModelClient (via cache)
                                             │
                                             ▼
                                    SQLite (runs, results, predictions)
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                       figures.py (charts)          parity.py (diff vs lm-eval-harness)
```

| Module | Responsibility |
| --- | --- |
| `records.py` | All Pydantic record types — the locked contract between modules |
| `client.py` | `ModelClient` ABC + registry + `HFClient` + `DummyClient` |
| `datasets.py` | `DatasetLoader` ABC + registry + loaders (HF tier / fixture tier) |
| `prompts.py` | Versioned `PromptVariant` YAML loading + pure renderer |
| `evaluators.py` | `loglik_mc` today; cloze, generative, perplexity in Sprints 3+ |
| `metrics.py` | `acc` today; `acc_norm`, ppl/bpb, answer extraction later |
| `storage.py` | SQLite schema, run/result CRUD, prediction cache. The only writer |
| `figures.py` | matplotlib figures, read-only against the DB |
| `cli.py` | `ladderctl` |
| `sweep.py`, `parity.py` | Sweep executor and harness parity — not yet implemented |

See [architecture.md](architecture.md) for record schemas, the storage schema, and the invariants checklist.

## Design decisions worth noting

- **Bits-per-byte is the canonical cross-model metric.** Raw perplexity is reported per-model only — it isn't comparable across models with different tokenizers.
- **Continuations are tokenized in context** (tokenize prompt+continuation, subtract the prompt token count) rather than independently. This is the single largest source of cross-framework accuracy mismatch, and it has a dedicated regression test.
- **Prompt variants are versioned YAML files, never inline strings.** Editing a template bumps the version and the old file stays, so a stored run always resolves to the exact prompt it used.
- **Predictions are cached under a content hash** of (model, revision, kind, prompt, continuations, gen_params), which makes interrupted sweeps resume with zero recomputation and re-runs free. *(Schema exists in Sprint 1; cache logic lands in Sprint 2.)*
- **A `RunRecord` contains everything needed to reproduce its run** — config, seed, code version, prompt variant id, timestamps.
- **The entire default test suite runs offline** against a deterministic `DummyClient` and bundled JSONL fixtures.

## Testing

```bash
pytest                  # default suite: offline, no network, no API keys
pytest -m slow          # tokenizer-boundary regression on a tiny HF model (downloads once)
```

The default suite is 59 tests and passes on a network-disabled machine — no downloads, no credentials. Anything touching a real model is gated behind the `slow` marker and deselected by default. Every metric has a hand-computed fixture test; every dataset loader has a bundled ~20-example JSONL fixture and works from it with the network off.

## Roadmap / future work

Deliberately out of scope, to keep the system small enough to finish and verify end-to-end:

- Training any models; the OLMo suite or other model families; Paloma per-domain perplexity.
- A dashboard or any web service — figures are generated files, committed to the repo.
- A third parity framework beyond lm-evaluation-harness; concurrency; DB migrations.
- More benchmarks, more prompt variants, or few-shot sweeps beyond 0-shot vs. 5-shot on one benchmark.
