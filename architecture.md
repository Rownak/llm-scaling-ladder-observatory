# Architecture: Scaling Ladder Eval Observatory

The system is deliberately small: **one package, ~10 modules, one SQLite file, one CLI.** Dataflow:

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

Every stage communicates via Pydantic records, so any stage's output can be dumped to JSONL and inspected.

---

## 1. Repository layout

```
ladder/
├── pyproject.toml
├── project_summary.md / architecture.md
├── src/ladder/
│   ├── records.py        # All Pydantic record types
│   ├── client.py         # ModelClient ABC + registry + HFClient + DummyClient
│   ├── datasets.py       # DatasetLoader ABC + registry + all loaders
│   ├── prompts.py        # PromptVariant loading + rendering
│   ├── evaluators.py     # loglik_mc, cloze, generative, perplexity
│   ├── metrics.py        # accuracy, acc_norm, ppl/bpb aggregation, answer extraction
│   ├── storage.py        # SQLite schema + run/result CRUD + prediction cache
│   ├── sweep.py          # sweep spec expansion + resumable executor
│   ├── parity.py         # lm-eval-harness importer + per-example diff + report
│   ├── figures.py        # matplotlib report figures from the DB
│   └── cli.py            # ladderctl
├── prompts/library/      # versioned *.yaml prompt variants
├── sweeps/main.yaml      # the one sweep spec for the project
├── report/               # findings.md + generated figures/
└── tests/                # offline-only; fixtures/ holds JSONL + hand-computed cases
```

Single-file modules over subpackages — at this size, subpackages are ceremony.

---

## 2. Record types (`records.py`)

The contract between all modules. Changing a field is a deliberate decision, not a casual edit.

```python
class Example(BaseModel):
    dataset: str; split: str; example_id: str   # stable ID (dataset-native or content hash)
    payload: dict                               # task-specific fields

class RenderedRequest(BaseModel):
    example_id: str
    prompt_variant_id: str | None               # None for PPL runs
    kind: Literal["loglik", "generate", "nll"]
    prompt: str
    continuations: list[str] | None             # loglik: one per option
    gen_params: dict | None

class Prediction(BaseModel):
    request_hash: str; model_id: str; revision: str
    logliks: list[float] | None                 # sum logprob per continuation
    token_nlls: list[float] | None              # for PPL windows
    generation: str | None
    n_bytes: int | None                         # UTF-8 bytes of scored text (for bpb)

class ExampleResult(BaseModel):
    run_id: str; example_id: str
    correct: bool | None                        # None for PPL
    score: float
    detail: dict                                # chosen option / extracted answer / window nlls

class RunRecord(BaseModel):
    run_id: str; model_id: str; revision: str
    dataset: str; split: str
    prompt_variant_id: str | None
    evaluator: str                              # loglik_mc | cloze | generative | perplexity
    framework: str                              # "ladder" | "lm_eval_harness"
    metrics: dict[str, float]
    n_examples: int; seed: int; code_version: str
    config: dict                                # full resolved config
    status: Literal["running", "done", "failed"]
```

**Invariant: a `RunRecord` contains everything needed to reproduce the run.**

---

## 3. Model clients (`client.py`)

```python
class ModelClient(ABC):
    def loglikelihood(self, prompt: str, continuations: list[str]) -> list[LoglikResult]: ...
    def generate(self, prompt: str, params: GenParams) -> str: ...
    def token_nlls(self, text: str, context: str = "") -> TokenNLLs: ...
```

Three methods because the three eval formats genuinely need different things: summed continuation logprobs (MC), sampling (GSM8K), per-token NLLs with maskable context (sliding-window PPL).

- **Registry:** `get_client(name, revision=...)`. No module imports a client class directly; tests of the client itself are the only exception.
- **`HFClient`** wraps `transformers` with `revision=` for checkpoint selection. Key correctness rule: **continuations are tokenized in context** (tokenize prompt+continuation, subtract prompt token count) — never independently. This is the #1 source of cross-framework mismatch and has a dedicated regression test. `token_nlls` = one forward pass, shift-by-one NLL, context positions masked. fp16 on CUDA, fp32 on CPU. One model loaded at a time; explicit `unload()`.
- **`DummyClient`** is deterministic from seed + input hash. Every evaluator, the sweep executor, storage, parity, and figures run end-to-end against it with the network disabled — this is both CI and the offline demo path.

---

## 4. Datasets (`datasets.py`)

```python
class DatasetLoader(ABC):
    name: str
    def load(self, split: str, limit: int | None = None) -> Iterator[Example]: ...
```

- Registry access, same rule as clients.
- **Every loader has a two-tier source policy: Hugging Face (cached locally) or the bundled ~20-example JSONL fixture in `tests/fixtures/`.** Tests use fixtures only; a loader without a fixture test does not merge.
- Normalized payloads per family:
  - MC (ARC-Easy, HellaSwag, MMLU-subset): `{question, choices, answer_index}`
  - Cloze (LAMBADA, OpenAI variant): `{context, target}`
  - Generative (GSM8K): `{question, answer_number}` (number extracted at load time)
  - PPL (WikiText-103 test, fixed C4 validation slice): `{text}` — windowing is the evaluator's job, not the loader's.
- MMLU uses a fixed 8-subject subset (recorded in the loader) to keep the sweep small; the C4 slice is the first N validation docs with a fixed seed, so results are reproducible.

---

## 5. Prompt variants (`prompts.py` + `prompts/library/`)

A variant is a YAML file, e.g. `arc_easy/mc_letter_v1.yaml`:

```yaml
id: arc_easy/mc_letter_v1
task_family: mc
template: |
  Question: {question}
  {lettered_choices}
  Answer:
continuation_style: letter      # letter (" A".." D") | option_text (full option strings)
num_fewshot: 0
```

Rules: variants are files, never inline strings; editing a template bumps the version (old file stays — stored runs reference it); rendering is a pure function `(Example, PromptVariant) -> RenderedRequest`. `continuation_style` is an explicit experimental axis — letter vs. option-text scoring moves MC accuracy and is half of the prompt-sensitivity study. One benchmark (ARC-Easy) additionally gets a 5-shot variant with demos drawn deterministically (fixed seed, train split).

---

## 6. Evaluators (`evaluators.py`)

All share `run(examples, variant, client, cache) -> Iterator[ExampleResult]`.

- **`loglik_mc`** — one continuation per option, argmax. Computes both `acc` and `acc_norm` (loglik / continuation byte length); their divergence across prompt styles is a finding.
- **`cloze`** (LAMBADA) — greedy generation of the target length must equal the target; target logprob also stored in `detail`.
- **`generative`** (GSM8K) — generate with stop sequences; numeric extraction in `metrics.py` (pattern first, last-number fallback), pure and fixture-tested on tricky cases (negatives, commas, decimals).
- **`perplexity`** — sliding window (default W=1024, stride W/2); each window scores only tokens not covered by a previous window, earlier tokens are context. Aggregation in `metrics.py`:
  - `ppl = exp(total_nll_nats / scored_tokens)` — per-model only, never compared across tokenizers.
  - `bpb = total_nll_bits / total_utf8_bytes` — **the canonical cross-model metric**; bytes counted on raw text.
  - Fixture test: tiny toy corpus with hand-computed NLLs via DummyClient pinned values; windowed and unwindowed totals verified by hand in a comment. Non-negotiable.

---

## 7. Storage + cache (`storage.py`)

One SQLite file, WAL mode, schema created on first open (no migration framework):

```sql
runs(run_id PK, model_id, revision, dataset, split, prompt_variant_id,
     evaluator, framework, seed, code_version, metrics_json, config_json,
     started_at, finished_at, status)
example_results(run_id FK, example_id, correct, score, detail_json)
predictions(request_hash PK, model_id, revision, payload_json)
```

- Metrics stored as JSON on the run row — simple, and `figures.py`/`parity.py` are the only readers.
- **Prediction cache:** key = `sha256(model_id | revision | kind | prompt | continuations | gen_params)` over canonical JSON. The executor consults it before every client call, so interrupted sweeps resume with zero recomputation and re-runs are free.
- Only `storage.py` writes to the DB; `figures.py` and `parity.py` are read-only.

---

## 8. Sweep (`sweep.py` + `sweeps/main.yaml`)

- The spec YAML declares axes (models×revisions, eval targets, variants, example cap) and expands deterministically into a run list.
- Executor is **sequential**, grouped by (model_id, revision) so each checkpoint is loaded once, all its runs executed, then unloaded — peak memory is one model. A failed run is marked `failed` and the sweep continues; `ladderctl sweep run` re-executes only non-`done` runs (resume is the default behavior, not a separate mode).
- Rough budget check (do this before running): ~12 model points × 7 targets × ≤500 examples ≈ 40k examples total; the 1B model is the long pole. If the first full sweep exceeds an overnight run, cut example caps, not benchmarks.

---

## 9. Parity (`parity.py`)

- **Reference:** lm-evaluation-harness on ARC-Easy and HellaSwag, run manually (documented command in the report); its per-example output files are ingested by an importer into normal `RunRecord`/`ExampleResult` rows with `framework="lm_eval_harness"`.
- Diff joins on `example_id`; every disagreement gets a category from a fixed enum — `PROMPT_FORMAT`, `ANSWER_EXTRACTION`, `NORMALIZATION`, `TOKENIZATION_BOUNDARY`, `DATA_MISMATCH`, `UNEXPLAINED` — plus a free-text root cause. Output is a markdown report; any `UNEXPLAINED` entries fail the parity check.
- Fixture test: two synthetic runs with planted disagreements of each category, verified end-to-end offline.

---

## 10. Figures + report (`figures.py`, `report/`)

`ladderctl figures` reads only the DB and writes matplotlib PNGs to `report/figures/`:

1. **Scaling curves** — log-params vs. accuracy for all five benchmarks, with bpb on a twin axis: smooth benchmarks track bpb, MMLU/GSM8K sit at chance. The headline figure.
2. **Training trajectory** — metric vs. checkpoint step for the intermediate revisions.
3. **Prompt sensitivity** — accuracy per variant per model (dot plot).
4. **Parity summary** — agreement rate + discrepancy category counts.

`report/findings.md` is written by hand, embeds these figures, and includes a future-work section (Paloma per-domain PPL, OLMo suite, dashboard, third framework) — the interview answer to "what would you do next."

---

## 11. CLI (`cli.py` → `ladderctl`)

```
ladderctl run      --model pythia-160m --revision step143000 --dataset arc_easy \
                   --variant arc_easy/mc_letter_v1 --evaluator loglik_mc [--limit N]
ladderctl sweep run sweeps/main.yaml          # resumable by default
ladderctl results  [--dataset ...] [--model ...]
ladderctl show     <run_id>
ladderctl parity   <run_id_a> <run_id_b> [--report out.md]
ladderctl import   --framework lm_eval_harness <path>
ladderctl figures
```

All commands take `--db` (default `./ladder.db`). Nonzero exit on failure.

---

## 12. Testing strategy

- **The whole suite runs offline**: DummyClient + JSONL fixtures, no network, no downloads.
- Minimums: hand-computed fixture test per metric; fixture test per loader (schema + stable IDs); HFClient tokenizer-boundary regression test (marked `slow`, uses a tiny model); sweep interrupt-and-resume test (assert zero recomputation via cache hits); parity planted-disagreement test.
- `slow`/`integration` markers gate anything touching real models; default `pytest` passes on a network-disabled machine.

---

## 13. Invariants checklist (review before every merge)

1. Registry access only — no direct client/loader class imports.
2. Every metric has a hand-computed fixture test; every loader has a JSONL fixture + test.
3. Cross-model comparisons use bpb, never raw PPL.
4. Prompt templates are versioned files; edits bump the version.
5. `RunRecord` is sufficient to reproduce its run.
6. `figures.py` and `parity.py` read the DB only.
7. Default test suite passes offline.
8. Continuation logliks use in-context tokenization.
