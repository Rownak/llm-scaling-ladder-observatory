# Scaling Ladder Eval Observatory

An offline-first LLM evaluation system that evaluates a ladder of Pythia checkpoints (70M–1B) on perplexity and downstream benchmarks, implements sliding-window perplexity and bits-per-byte from scratch, and treats prompt formatting as a measured experimental axis rather than an implementation detail.

## The question this project answers

**Does perplexity improve smoothly with scale while benchmark accuracy stays flat?** On a Pythia ladder from 70M to 1B, yes — and the split is clean. Bits-per-byte falls monotonically on both corpora (WikiText-103 1.439 → 1.021, C4 slice 1.174 → 0.866) and HellaSwag rises with it (0.314 → 0.384), while ARC-Easy and MMLU sit flat at their 4-option chance line and LAMBADA and GSM8K sit at ~0 regardless of size. See [report/figures/headline.png](report/figures/headline.png) and [report/findings.md](report/findings.md).

Sprints 4–5 extend this with prompt sensitivity and harness parity.

## What it does

- Evaluates a **scaling ladder** of Pythia models (70M, 160M, 410M, 1B), including intermediate training checkpoints, on perplexity and five downstream benchmarks.
- Implements **perplexity from scratch** — sliding-window, with bits-per-byte as the cross-model metric — and charts its relationship to downstream accuracy across scale.
- Verifies **parity against lm-evaluation-harness** on two benchmarks, with per-example diffs and root-caused discrepancies.
- Runs a **prompt-sensitivity experiment**: 2–3 prompt formats per benchmark, showing how reported accuracy shifts with formatting alone.
- Stores every run in **SQLite** and regenerates all report figures from the database with one CLI command.

## Status

**In progress.** Sprints 1–3 of 5 are complete; the rest are planned and not yet implemented.

| Sprint | Status | What it adds |
| --- | --- | --- |
| 1 — Walking skeleton | Complete | One model, one benchmark, end-to-end: records, clients, loaders, prompts, `loglik_mc`, accuracy, SQLite, CLI, one figure. All downstream interfaces locked. |
| 2 — Ladder + sweeps | Complete | HellaSwag/MMLU loaders, `acc_norm`, content-hash prediction cache, resumable sweep executor, scaling-curve and trajectory figures. |
| 3 — Perplexity + formats | Complete | `token_nlls`, sliding-window perplexity with ppl/bpb, cloze (LAMBADA) and generative (GSM8K) evaluators, the headline figure. Full 84-run sweep executed on real hardware. |
| 4 — Prompt sensitivity | Pending | Alternative MC variants (letter vs. option-text, instruction line), 5-shot rendering, variant sweep, sensitivity dot plot. |
| 5 — Parity + report | Pending | lm-eval-harness importer, per-example diff with a fixed discrepancy taxonomy, final findings report. |

The Sprint 1–3 surface described under Quickstart exists today: single runs, resumable sweeps over all 7 eval targets, perplexity with bits-per-byte, and the full figure set including the headline plot. `sweeps/main.yaml` has been executed end-to-end against real Pythia checkpoints (84 runs, 4 sizes × 3 checkpoints × 7 targets, on a 16GB RTX 4090). `ladderctl parity` and `import` are specified in [architecture.md](architecture.md) but not yet implemented.

## Quickstart

```bash
git clone <repo-url> && cd llm-scaling-ladder-observatory
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs the CPU build of torch. **For GPU runs, reinstall torch against a CUDA index afterwards** — otherwise the full sweep runs on CPU and takes far longer:

```bash
pip install --force-reinstall torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Fully offline demo — deterministic `DummyClient` against the bundled JSONL fixture, no network and no model download:

```bash
ladderctl run --model dummy --dataset arc_easy --variant arc_easy/mc_letter_v1 \
  --evaluator loglik_mc --split fixture --db ./demo.db

ladderctl results --db ./demo.db
ladderctl show <run_id> --db ./demo.db      # full reproducibility metadata
ladderctl figures --db ./demo.db            # writes report/figures/accuracy_per_run.png
```

Single real-hardware run — downloads `EleutherAI/pythia-70m` and `allenai/ai2_arc` on first use, runs on CPU or CUDA:

```bash
ladderctl run --model pythia-70m --revision main --dataset arc_easy \
  --variant arc_easy/mc_letter_v1 --evaluator loglik_mc --split test \
  --limit 200 --db ./ladder.db
```

Implemented today: models `dummy`, `pythia-70m`, `pythia-160m`, `pythia-410m`, `pythia-1b`; datasets `arc_easy`, `hellaswag`, `mmlu`, `lambada`, `gsm8k`, `wikitext103`, `c4_slice`; evaluators `loglik_mc`, `cloze`, `generative`, `perplexity`; metrics `acc`, `acc_norm`, `ppl`, `bpb`.

All seven loaders support `--split fixture`, which reads a bundled ~20-example JSONL instead of downloading — so every command in this README can be run offline by pairing `--split fixture` with `--model dummy`.

### Sprint 2 — resumable sweeps + scaling figures

`sweeps/main.yaml` declares the full ladder grid: 4 model sizes × 3 checkpoints (`step1000`, `step64000`, `main`) × 3 MC benchmarks, capped at 500 examples per run. `ladderctl sweep run` executes it sequentially, one model checkpoint loaded at a time, and **resumes by default** — rerunning the same command against the same DB skips every run already marked `done` and picks up only what's left:

```bash
ladderctl sweep run sweeps/main.yaml --db ./ladder.db
# ^ Ctrl-C partway through is safe — rerun the exact same command to resume
# with zero recomputation on the runs that already finished.
ladderctl sweep run sweeps/main.yaml --db ./ladder.db

ladderctl results --db ./ladder.db          # one row per (model, revision, dataset) run
ladderctl figures --db ./ladder.db          # now also writes scaling_curve.png and trajectory.png
```

`ladderctl figures` writes `accuracy_per_run.png` (Sprint 1), `scaling_curve.png` (log-params vs. accuracy per benchmark, at each model's final checkpoint, with chance lines), and `trajectory.png` (accuracy vs. training step, colored by benchmark and marker-coded by model size). Sprint 3 adds two more — see below.

To try the sweep + figures pipeline fully offline (no downloads), point a spec at the `dummy` model and the bundled fixtures:

```bash
cat > /tmp/demo_sweep.yaml <<'EOF'
models:
  - model_id: dummy
    revisions: [main]
targets:
  - dataset: arc_easy
    variant: arc_easy/mc_letter_v1
    evaluator: loglik_mc
    split: fixture
limit: 20
seed: 0
EOF
ladderctl sweep run /tmp/demo_sweep.yaml --db ./demo.db
ladderctl figures --db ./demo.db
```

### Sprint 3 — perplexity, all four formats, the headline figure

`sweeps/main.yaml` now spans all 7 eval targets: 3 multiple-choice (`arc_easy`, `hellaswag`, `mmlu`), cloze (`lambada`), generative (`gsm8k`), and 2 perplexity corpora (`wikitext103`, `c4_slice`). Against the 4×3 model grid that's **84 runs**, executed end-to-end on a 16GB RTX 4090:

```bash
ladderctl sweep run sweeps/main.yaml --db ./ladder.db
ladderctl figures --db ./ladder.db --out-dir ./report/figures
```

Perplexity targets take no `--variant` — PPL scores raw documents, so there's no prompt to render:

```bash
ladderctl run --model pythia-1b --dataset wikitext103 \
  --evaluator perplexity --split test --limit 500 --db ./ladder.db
```

`ladderctl figures` now writes five PNGs, adding `trajectory_bpb.png` (bpb vs. training step) and `headline.png` — log10(params) on x, accuracy on the left axis with dashed chance lines, bpb on the right axis **inverted** so "up = better" holds on both at once.

Final-checkpoint results across the ladder:

| dataset | 70m | 160m | 410m | 1b |
| --- | --- | --- | --- | --- |
| wikitext103 (bpb ↓) | 1.439 | 1.260 | 1.090 | 1.021 |
| c4_slice (bpb ↓) | 1.174 | 1.040 | 0.916 | 0.866 |
| hellaswag (acc ↑) | 0.314 | 0.328 | 0.362 | 0.384 |
| arc_easy (acc) | 0.232 | 0.230 | 0.234 | 0.232 |
| mmlu (acc) | 0.226 | 0.222 | 0.220 | 0.224 |
| lambada (acc) | 0.002 | 0.002 | 0.002 | 0.002 |
| gsm8k (acc) | 0.004 | 0.002 | 0.016 | 0.014 |

Both bpb columns fall monotonically and HellaSwag tracks them upward; ARC-Easy and MMLU hold at chance; LAMBADA and GSM8K stay at ~0. GSM8K at ~0 is [the expected finding, not a bug](report/findings.md) — base models at this scale don't do multi-step arithmetic. Two numbers deserve scepticism rather than citation: LAMBADA's flat 0.002 is likely an exact-match strictness artifact, and GSM8K's non-monotonicity is noise (at 500 examples, one correct answer is 0.002, so those are 1–8 raw hits).

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
| `datasets.py` | `DatasetLoader` ABC + registry + 7 loaders: ARC-Easy, HellaSwag, MMLU, LAMBADA, GSM8K, WikiText-103, C4 slice (HF tier / fixture tier) |
| `prompts.py` | Versioned `PromptVariant` YAML loading + pure renderer |
| `evaluators.py` | `loglik_mc`, `cloze`, `generative`, `perplexity` (sliding-window, W=1024/stride 512) |
| `metrics.py` | `acc`, `acc_norm`, `ppl`, `bpb`, numeric answer extraction |
| `storage.py` | SQLite schema, run/result CRUD, content-hash prediction cache. The only writer |
| `sweep.py` | Sweep spec expansion + resumable sequential executor, grouped by (model, revision) |
| `figures.py` | matplotlib figures (accuracy bar, scaling curve, trajectory, bpb trajectory, headline), read-only against the DB |
| `cli.py` | `ladderctl` |
| `parity.py` | Harness parity — not yet implemented (Sprint 5) |

See [architecture.md](architecture.md) for record schemas, the storage schema, and the invariants checklist.

## Design decisions worth noting

- **Bits-per-byte is the canonical cross-model metric.** Raw perplexity is reported per-model only — it isn't comparable across models with different tokenizers.
- **Continuations are tokenized in context** (tokenize prompt+continuation, subtract the prompt token count) rather than independently. This is the single largest source of cross-framework accuracy mismatch, and it has a dedicated regression test.
- **Prompt variants are versioned YAML files, never inline strings.** Editing a template bumps the version and the old file stays, so a stored run always resolves to the exact prompt it used.
- **Predictions are cached under a content hash** of (model, revision, kind, prompt, continuations, gen_params), which makes interrupted sweeps resume with zero recomputation and re-runs free.
- **Sweep resume is config-based, not run-id-based.** Every `sweep run` invocation mints fresh run ids, so "already done" is decided by matching (model, revision, dataset, split, variant, evaluator, limit, seed) against existing `done` rows — killing a sweep and rerunning the same command is the whole resume story, no separate flag.
- **A `RunRecord` contains everything needed to reproduce its run** — config, seed, code version, prompt variant id, timestamps.
- **The entire default test suite runs offline** against a deterministic `DummyClient` and bundled JSONL fixtures.

## Testing

```bash
pytest                  # default suite: offline, no network, no API keys
pytest -m slow          # tokenizer-boundary regression on a tiny HF model (downloads once)
```

The default suite is 181 tests and passes on a network-disabled machine — no downloads, no credentials. Anything touching a real model is gated behind the `slow` marker and deselected by default. Every metric has a hand-computed fixture test; every dataset loader has a bundled ~20-example JSONL fixture and works from it with the network off; the sweep executor has a dedicated interrupt-and-resume test asserting zero recomputation via cache-hit accounting.

## Roadmap / future work

Deliberately out of scope, to keep the system small enough to finish and verify end-to-end:

- Training any models; the OLMo suite or other model families; Paloma per-domain perplexity.
- A dashboard or any web service — figures are generated files, committed to the repo.
- A third parity framework beyond lm-evaluation-harness; concurrency; DB migrations.
- More benchmarks, more prompt variants, or few-shot sweeps beyond 0-shot vs. 5-shot on one benchmark.
