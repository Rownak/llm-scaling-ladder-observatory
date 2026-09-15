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
    started_at: str | None = None               # ISO 8601; set when the run begins
    finished_at: str | None = None              # ISO 8601; set on "done"/"failed"
```

**Invariant: a `RunRecord` contains everything needed to reproduce the run.**

**Phase 1.4 addition:** `started_at`/`finished_at` live on `RunRecord` itself, not as DB-only columns — they're reproducibility/provenance metadata, the same category as `seed`/`code_version`, so `storage.py` persists them verbatim instead of inferring them SQL-side from `status`. Both default to `None`; whichever caller tracks wall-clock timing (CLI `run` command, later the sweep executor) sets them on the record before calling `storage.save_run`.

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
- **model_id → HF repo mapping (Phase 1.2 decision):** the registry hardcodes the Pythia ladder's `model_id` (`"pythia-70m"`, `"pythia-160m"`, `"pythia-410m"`, `"pythia-1b"`) to its full HF Hub repo (`"EleutherAI/pythia-70m"`, etc.) in a module-level dict in `client.py`, via a small factory closure per entry. Callers (CLI, sweep YAML, `RunRecord.model_id`) always use the short `model_id`; only `client.py` knows the HF repo strings. Adding a model to the ladder means adding one dict entry, not touching call sites.
- **Supporting types** `LoglikResult` (`loglik: float`, `n_tokens: int`, `is_greedy_match: bool = False`), `GenParams` (`max_new_tokens`, `stop`, `temperature`), and `TokenNLLs` (`nlls: list[float]`, `n_bytes: int`) live in `client.py`, not `records.py` — they're client-internal shapes, not part of the storage/reproducibility contract. `Prediction` (in `records.py`) is what persists a client call's result.
  - **`is_greedy_match` (pre-Sprint-5, LAMBADA fix, §6):** whether every token of the scored continuation equals the model's argmax at its position, teacher-forced — computed in the same forward pass `loglik` already needs, no extra client cost. `HFClient` computes it for real from `log_probs.argmax()` at each predicted position; `DummyClient` derives a deterministic synthetic boolean from the same `(seed, model_id, revision, prompt, cont)` key `_rng_value` already mixes, since there are no real logits to check argmax against. Defaults to `False` so every pre-existing `LoglikResult(...)` construction (with no logits to check, e.g. hand-built test fixtures) still validates.

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
  - MC, question-style (ARC-Easy, MMLU-subset): `{question, choices, answer_index}`
  - MC, context-completion style (HellaSwag): `{context, choices, answer_index}` — HellaSwag's native item is a sentence to complete (`ctx` + `endings`), not a question, so it gets its own payload shape rather than forcing `ctx` into a `question` field. (Phase 2.1 decision; deviates from the original single-shape note in favor of matching the source data and `sprints/sprint2.md`'s literal spec.) `prompts.render` dispatches on which key (`question` vs `context`) is present in the payload to pick the template field.
  - Cloze (LAMBADA, OpenAI variant): `{context, target}`
  - Generative (GSM8K): `{question, answer_number}` (number extracted at load time)
  - PPL (WikiText-103 test, fixed C4 validation slice): `{text}` — windowing is the evaluator's job, not the loader's.
- MMLU uses a fixed 8-subject subset, recorded as a module-level constant (`MMLU_SUBJECTS`) in `datasets.py`, of `cais/mmlu` subject configs; the loader iterates subjects in that fixed order and pools their rows into one stream, tagging each `Example.payload` with `subject` for later breakdown. The C4 slice is the first N validation docs with a fixed seed, so results are reproducible.
- **Phase 3.3 implementation** (`WikiTextLoader`, `C4SliceLoader` in `datasets.py`):
  - `WikiTextLoader` wraps `Salesforce/wikitext` (config `wikitext-103-raw-v1`), split `test`. WikiText's raw rows include blank lines and `" = Section Heading = "` marker rows interleaved with actual prose; both are skipped (no `text`, or `text` wrapped in `=`) since neither is scorable and an empty `text` would otherwise become a zero-token PPL document. `example_id` is `wikitext103-<row index>`, assigned after filtering so it stays a dense sequence of only the kept rows.
  - `C4SliceLoader` wraps `allenai/c4` (config `en`), loaded with `streaming=True` and `.shuffle(seed=_C4_SLICE_SEED, buffer_size=10_000)` — streaming because C4 is far too large to download in full for a 200-doc slice. `_C4_SLICE_N` (200) and `_C4_SLICE_SEED` (0) are the two module-level constants that fully determine "first N docs" of the resulting deterministic shuffle order; `example_id` is `c4_slice-<position in the shuffled stream>`. Changing either constant changes the slice, same governance as bumping a prompt variant version (§5) — a deliberate decision, not a casual edit.
  - Both loaders' `split="fixture"` tier reads a bundled 5-document JSONL (`tests/fixtures/wikitext.jsonl`, `tests/fixtures/c4_slice.jsonl`) — smaller than the ~20-example convention used by the MC loaders, since PPL fixture documents are multi-sentence paragraphs (enough for `evaluators.perplexity`'s sliding window to exercise multiple windows per document in tests) rather than single MC items.
- **Phase 3.4 implementation** (`LambadaLoader` in `datasets.py`): wraps `EleutherAI/lambada_openai` (the Hub dataset that already applies LAMBADA's "OpenAI variant" detokenization/cleaning — `datasets.py` does not redo that preprocessing itself). `{context, target}` is produced by `text.rpartition(" ")` on each row's already-cleaned passage: `target` is the final whitespace-delimited word, `context` everything before it. `example_id` is `lambada-<row index>` (the dataset carries no native row ID, same situation as MMLU's content-hash IDs, but LAMBADA rows have no stable content key shorter than the whole passage, so index-based IDs were used instead — acceptable since, unlike MMLU's pooled multi-subject stream, LAMBADA's row order is a single fixed HF split with no per-loader reordering that could shift indices between runs).
- **Phase 3.5 implementation** (`Gsm8kLoader` in `datasets.py`): wraps `openai/gsm8k` (config `main`). GSM8K's raw `answer` field is a full chain-of-thought solution ending in a literal `"#### <number>"` line; the loader extracts just that trailing number via `answer_text.rpartition("####")`, stripping thousands-separator commas, so `answer_number` is a clean `float` at load time — evaluators never parse gold-answer text, only `metrics.extract_answer_number` parses the *model's* generation (§6). `example_id` is `gsm8k-<row index>` (no native row ID, same reasoning as LAMBADA's index-based IDs above).
- **Phase 4.1 implementation** (ARC-Easy train tier, for few-shot demos): the HF tier needs no change — `ArcEasyLoader.load` passes `split` straight through to `load_dataset`, so `load("train")` already works. The **fixture tier** is the actual gap: `_load_fixture_jsonl` treats `split` as a pass-through label and does no filtering, and every row in `tests/fixtures/arc_easy.jsonl` carries `"split":"fixture"`. A separate `tests/fixtures/arc_easy_train.jsonl` (~5 rows) is added and reached via `split == "fixture_train"` — a distinct file rather than a filter inside the shared helper, so no other loader's fixture path is touched. Without this tier the few-shot golden test and the variant mini-sweep couldn't run network-disabled, breaking the offline-suite invariant (§12).
- **ARC-Easy label alphabet is not fixed** (Phase 1.3 finding): `allenai/ai2_arc` rows use `choices.label` values of `"A".."D"`, `"A".."C"`, `"A".."E"`, or `"1".."4"` depending on the row, and `choices` can have 3–5 options. `answer_index` must always be computed as `row["choices"]["label"].index(row["answerKey"])` — never via a hardcoded letter→index map. The bundled fixture (`tests/fixtures/arc_easy.jsonl`) deliberately includes one example of each label pattern found in the real test split, so a loader regression that assumes `"A".."D"` fails offline.

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
fewshot_split: null             # train split demos are drawn from (few-shot variants only)
fewshot_seed: null              # seed fixing which demos are drawn (few-shot variants only)
```

Rules: variants are files, never inline strings; editing a template bumps the version (old file stays — stored runs reference it); rendering is a pure function `(Example, PromptVariant) -> RenderedRequest`. `continuation_style` is an explicit experimental axis — letter vs. option-text scoring moves MC accuracy and is half of the prompt-sensitivity study. One benchmark (ARC-Easy) additionally gets a 5-shot variant with demos drawn deterministically (fixed seed, train split).

- **Phase 3.4 implementation:** `render` now also handles `task_family == "cloze"` (`lambada/cloze_v1.yaml` — `template: "{context}"`, `continuation_style: null`). It's a pass-through template: LAMBADA's payload is already exactly the text to condition generation on, so the template file exists for provenance (recording which variant id scored a run, same as every other variant) rather than to reformat anything. The returned `RenderedRequest` has `kind="generate"` and `continuations=None` — cloze doesn't score fixed options, `evaluators.cloze` generates instead.
- **Phase 4.1 implementation:** `PromptVariant` gains two more optional fields, `fewshot_split: str | None = None` and `fewshot_seed: int | None = None`, used only by variants with `num_fewshot > 0` (`arc_easy/mc_letter_5shot_v1.yaml`). Both default to `None`, so every pre-Sprint-4 variant file validates unchanged. `num_fewshot` already existed (Sprint 1); only these two are new.
  - **Few-shot determinism is variant-owned, not run-level — deliberate.** The run-level `seed` (§2, `sweeps/main.yaml`) is *recorded metadata*: its only functional consumer anywhere in the system is `DummyClient`'s determinism hash (§3), and `render(example, variant)` receives no seed at all. Routing demo selection through it would mean threading a third parameter through all four evaluators and reclassifying `seed` from "a thing we write down" into "a thing that changes rendered output."
  - Three reasons the variant owns it instead. **(1) Cache key integrity:** the prediction cache keys on `(model_id, revision, kind, prompt, continuations, gen_params)` (§7) — `seed` is deliberately absent. Variant-owned seed keeps the chain `variant id → prompt → cache key` single-hop, with no hidden input that changes a prompt without changing the variant id. **(2) Provenance:** a run stores `prompt_variant_id` as a single string (§2); if the seed lives there, that string alone reconstructs the exact demos forever, and anything reading only the variant id (`ladderctl show`, §9's parity importer, §10's sensitivity plot) stays correct. Run-level seed would make reconstruction require the composite `(prompt_variant_id, seed)`. **(3) Precedent:** this is the same call already made twice — Phase 3.5 put `max_new_tokens`/`stop` on the variant because decoding config "is a fixed property of the prompt/answer format," and `C4SliceLoader` (§4) pinned `_C4_SLICE_SEED` as a module constant governed like a version bump. Few-shot demos are *more* prompt-defining than either.
  - Changing `fewshot_seed` or `fewshot_split` on an existing variant file is a **version bump**, same governance as editing a template — the old file stays, so stored runs keep resolving to the demos they actually used.
  - `render` stays pure: demo *selection* is the caller's job — `evaluators._select_fewshot_demos(variant)` (§6) draws `variant.num_fewshot` examples via `random.Random(variant.fewshot_seed)` from `get_loader(dataset).load(variant.fewshot_split)`, once per run (not per example, so every example shares the same demo set), and `loglik_mc` passes that fixed list into every `render(example, variant, demos=...)` call. `render` itself never constructs an RNG or touches a `DatasetLoader` — its new `demos` parameter prepends each demo's rendered template block plus its gold continuation (new helpers `_render_mc_block`/`_mc_continuation` in `prompts.py`, shared with the non-demo render path so a demo's answer encoding always matches how the eval example itself would be scored).
- **Phase 3.5 implementation:** `PromptVariant` gains two optional fields, `max_new_tokens: int | None` and `stop: list[str] | None`, used only by `task_family == "generative"` (`gsm8k/gen_v1.yaml`). Unlike cloze's per-example generation budget (derived from target length inside `evaluators.cloze`, §6), a generative variant's decoding config is a fixed property of the prompt/answer format — how much room a chain-of-thought answer needs, what token ends it — so it lives on the variant file itself, not recomputed per example. `render`'s `task_family == "generative"` branch formats `{question}` into the template and populates `RenderedRequest.gen_params = {"max_new_tokens": variant.max_new_tokens, "stop": variant.stop, "temperature": 0.0}` — always greedy (`temperature` is not a variant-level knob; every generative evaluator run in this project decodes greedily).

---

## 6. Evaluators (`evaluators.py`)

All share `run(examples, variant, client, cache) -> Iterator[ExampleResult]`.

- **`loglik_mc`** — one continuation per option, argmax. Computes both `acc` and `acc_norm` (loglik / continuation byte length); their divergence across prompt styles is a finding.
- **`cloze`** (LAMBADA) — primary metric is teacher-forced greedy target-word accuracy (`LoglikResult.is_greedy_match`), not free-form generation matching; target NLL/perplexity and a punctuation-normalized generation-match diagnostic also stored.
  - **Phase 3.4 implementation:** `cloze(run_id, examples, variant, client, cache, model_id, revision)` renders via `prompts.render` (§5, `kind="generate"`), then calls `client.generate` with `GenParams(max_new_tokens=4 * n_target_words, stop=["\n"], temperature=0.0)` — greedy, and a token budget scaled to the target's word count (`_CLOZE_TOKENS_PER_TARGET_WORD = 4` headroom per word) rather than a fixed cap, since LAMBADA targets are usually one word but the payload shape doesn't guarantee it. The target's own loglik is scored via a second call, `client.loglikelihood(prompt, [" " + target])` (reusing `loglik_mc`'s `_scored_continuations` helper and its leading-space continuation convention). Both the generate and loglik calls go through `PredictionCache` under different `kind`s (`"generate"` vs `"loglik"`), so they never collide despite reading the same example.
  - **Pre-Sprint-5 fix — primary metric switched from generation string-match to greedy target-word match.** `report/findings.md`'s Sprint 3 section flagged LAMBADA's flat 0.002 accuracy as suspicious; checking it against the already-collected `ladder.db` (`detail["target_logprob"]`, stored unconditionally since Phase 3.4) confirmed the suspicion — target log-likelihood improved substantially with scale (70m→1b) while exact-match accuracy stayed pinned at 0.002, and spot-checked generations showed the model reproducing the target correctly but with trailing content (`" Queen."` vs. target `"Queen"`) that greedy decoding has no reason to stop before, since `GenParams.stop=["\n"]` only catches a newline. That's a measurement bug, not a genuine capability floor: the old evaluator penalized correct predictions for a decoding artifact unrelated to the model's actual knowledge of the target word.
    - **Fix:** `correct`/`score` are now driven by `LoglikResult.is_greedy_match` (see §3's `ModelClient` note below) — whether the target's own tokens are exactly what argmax-at-every-position (teacher-forced greedy decoding) would produce, conditioned on the prompt. This matches standard LAMBADA scoring (lm-evaluation-harness) and needs no generation step to determine correctness at all; `generate` is still called (same token budget as before), but its output is no longer load-bearing for `correct`.
    - **Secondary metrics** (`metrics.cloze_metrics`, mirroring `perplexity_metrics`'s pattern): `target_nll_mean` (`-loglik`, averaged) and `target_ppl_mean` (`exp(nll / n_tokens)`, averaged per-example then across the run) — both free, derived from the same `loglikelihood` call the primary metric already makes.
    - **Diagnostic, explicitly non-standard:** `nonstandard_generated_word_acc` — the old approach, but punctuation-stripped (first whitespace-delimited word of `generation`, `str.strip(string.punctuation)`, compared to `target`) — kept only so the primary metric and a punctuation-tolerant version of the old approach can be sanity-checked against each other, never cited as a real metric (hence the name).
- **`generative`** (GSM8K) — generate with stop sequences; numeric extraction in `metrics.py` (pattern first, last-number fallback), pure and fixture-tested on tricky cases (negatives, commas, decimals).
  - **Phase 3.5 implementation:** `generative(run_id, examples, variant, client, cache, model_id, revision)` renders via `prompts.render` (§5, `kind="generate"`, `gen_params` sourced from the variant's own `max_new_tokens`/`stop` rather than computed per example — the opposite of `cloze`'s per-target-length budget, since GSM8K answers are open-ended chain-of-thought text with no natural target-length signal to size a budget from). Generation goes through `_cached_generate` (the same helper `cloze` uses, `kind="generate"`). The raw generation is passed to `metrics.extract_answer_number`; `correct` is an exact `float` match between the extracted number and `example.payload["answer_number"]` — a `None` extraction (no number found at all) is always incorrect, never raises. `detail` carries the raw generation, the extracted number, and the gold answer, so a failed extraction is visible in `ladderctl show` rather than silently folded into "wrong."
  - `metrics.extract_answer_number(text) -> float | None` is pattern-first, last-number fallback: it first checks, in order, an explicit `"Final answer: N"` cue (case-insensitive) and a GSM8K-native `"#### N"` cue — the *last* match of whichever cue is found first wins, so a generation that flags its answer explicitly is scored on that flagged number even if earlier reasoning mentions other numbers. With no cue present at all, it falls back to the last number appearing anywhere in the text. Comma thousands-separators are stripped before parsing; negative numbers and decimals are both recognized via one shared regex. Pure function, no evaluator/client dependency — fixture-tested directly in `test_metrics.py` on negatives, comma-grouped numbers, decimals, numbers appearing mid-reasoning with no cue, and no-number text (returns `None`, not an exception or `0`).
- **`perplexity`** — sliding window (default W=1024, stride W/2); each window scores only tokens not covered by a previous window, earlier tokens are context. Aggregation in `metrics.py`:
  - `ppl = exp(total_nll_nats / scored_tokens)` — per-model only, never compared across tokenizers.
  - `bpb = total_nll_bits / total_utf8_bytes` — **the canonical cross-model metric**; bytes counted on raw text.
  - Fixture test: tiny toy corpus with hand-computed NLLs via DummyClient pinned values; windowed and unwindowed totals verified by hand in a comment. Non-negotiable.
  - **Phase 3.2 implementation:** `perplexity(run_id, examples, client, cache, model_id, revision, window=1024, stride=None)` — no `variant` parameter, unlike the other three evaluators. PPL runs have `prompt_variant_id=None` (§2, §5); there is no template to render, so threading a `PromptVariant` through would be a dead parameter. `stride` defaults to `window // 2` when omitted.
    - Window boundaries are cut on whitespace-split words, not tokenizer output: `ModelClient.token_nlls` (§3) takes raw `text`/`context` strings, not token ids, so the evaluator has no token-level view to slide a window over. Word boundaries are an approximation of token boundaries but are what `DummyClient.token_nlls` already uses internally (Phase 3.1), so evaluator-level windowing and client-level tokenization agree in the offline/test path; a real `HFClient` window's token count will vary slightly from `window` words, which is fine since W is a budget knob, not a precision requirement.
    - Per window: the words already covered by a previous window are passed as `context` (joined with spaces), the rest as `text`; `client.token_nlls` is called once per window through `cache` (kind="nll", `gen_params` slot repurposed to carry `{"context": ...}` since PPL requests have no generation params but do need `context` as extra key material distinguishing otherwise-identical `text`).
    - Yields **one `ExampleResult` per document** (not per window): `correct=None`, `score` is that document's own bpb, `detail = {"window_nlls": [...], "n_bytes": ...}` — the concatenation of every window's per-token NLLs (nats) for that document, already deduplicated by construction (each word appears in exactly one window's scored `text`). `metrics.perplexity_metrics` sums `window_nlls`/`n_bytes` across every `ExampleResult` in a run to get the run-level `ppl`/`bpb` — summing raw nats/bytes first and dividing once, not averaging each document's own ppl/bpb, so a run's aggregate is scored-token-weighted rather than document-weighted.

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
- **Prediction cache:** key = `sha256(model_id | revision | kind | prompt | continuations | gen_params)` over canonical JSON (`storage.prediction_cache_key`). The executor consults it before every client call, so interrupted sweeps resume with zero recomputation and re-runs are free.
  - **Phase 2.3 implementation:** `storage.PredictionCache` wraps a connection and exposes `get`/`put`, counting hits/misses on the instance so a run's `config` can record `cache_hits`/`cache_misses` (evaluators don't touch the DB directly). `evaluators.loglik_mc` caches **per continuation**, not per example — each option gets its own single-continuation cache key — so a prompt-variant edit touching only one distractor still reuses the cached score for the untouched options.
  - `model_id`/`revision` are passed into evaluators as explicit parameters (from the CLI's `--model`/`--revision`), not read off the `ModelClient` instance: `HFClient.model_id` is the internal HF Hub repo string (§3), not the short registry id the cache key (and every other `model_id` in the system) uses.
- `Prediction.is_greedy_matches: list[bool] | None = None` (pre-Sprint-5, §6) is parallel to `logliks`, round-tripped through `payload_json` (`storage.get_prediction`/`save_prediction`). **A row cached before this field existed has no such key in its JSON at all** — `get_prediction` uses `payload.get("is_greedy_matches")`, not `payload[...]`, so those rows come back with `is_greedy_matches=None` ("unknown"), which `evaluators._scored_continuations` then treats as `is_greedy_match=False` on that cache hit. **Practical consequence:** every LAMBADA `loglik` prediction already cached in `ladder.db` before this fix predates the field, so a `sweep run`/`ladderctl run` against the existing DB will cache-hit on those rows and silently score every LAMBADA example `correct=False` — reusing the fix requires either clearing the cached LAMBADA `loglik` predictions first, or accepting a full LAMBADA re-run (cheap — one forward pass per example, no generation needed for the primary metric).
- Only `storage.py` writes to the DB; `figures.py` and `parity.py` are read-only.

---

## 8. Sweep (`sweep.py` + `sweeps/main.yaml`)

- The spec YAML declares axes (models×revisions, eval targets, variants, example cap) and expands deterministically into a run list.
- Executor is **sequential**, grouped by (model_id, revision) so each checkpoint is loaded once, all its runs executed, then unloaded — peak memory is one model. A failed run is marked `failed` and the sweep continues; `ladderctl sweep run` re-executes only non-`done` runs (resume is the default behavior, not a separate mode).
- Rough budget check (do this before running): ~12 model points × 7 targets × ≤500 examples ≈ 40k examples total; the 1B model is the long pole. If the first full sweep exceeds an overnight run, cut example caps, not benchmarks.
- **Phase 2.4 implementation:**
  - `SweepSpec` (`models: list[SweepModel]`, `targets: list[SweepTarget]`, `limit`, `seed`) is the YAML schema. `expand_sweep` walks models (outer) × that model's revisions × targets (inner), in file order — this nesting is what makes the executor's (model_id, revision) grouping *contiguous by construction*, so `run_sweep` never needs to sort or re-group.
  - **Resume matching is config-based, not `run_id`-based**: since every execution mints a fresh `run_id`, "already done" is decided by `_sweep_run_key` — the tuple `(model_id, revision, dataset, split, prompt_variant_id, evaluator, config["limit"], seed)` — checked against every `status == "done"` row already in the DB before a `SweepRun` executes. A run matching an existing `done` row is skipped with zero client calls and zero new rows; this is orthogonal to (and layered on top of) the per-request `PredictionCache`, which still applies within any run that *does* execute.
  - `run_sweep(conn, spec, evaluators=None)` takes an optional evaluator-name→function map (defaults to the real registry, `{"loglik_mc": loglik_mc}`) purely so tests can substitute without touching the module import graph; production callers (the CLI) never pass it.
  - The executor reuses `cli.run`'s single-run pipeline logic (load → render/score → aggregate → persist) via a private `_execute_one`, rather than the CLI command calling into `sweep.py`'s runner — `ladderctl sweep run sweeps/main.yaml` is a thin wrapper that loads the spec, calls `run_sweep`, and prints a per-run summary line + a nonzero exit if any run failed.
- **Phase 3.6 implementation** (`sweeps/main.yaml` v2, all 4 evaluators wired into `sweep.py`/`cli.py`):
  - `SweepTarget.variant` is now `str | None = None` — `perplexity` targets carry no variant at all, since `evaluators.perplexity`'s function signature omits `variant` entirely (it scores raw documents, no prompt template to render). `_NO_VARIANT_EVALUATORS = {"perplexity"}` (mirrored in both `sweep.py` and `cli.py`) is the single flag `_execute_one`/`cli.run` branch on to decide whether to call `load_variant` + pass `prompt_variant` positionally, or call the evaluator with one fewer argument. This is a deliberate asymmetry in the evaluator call signature (documented here, not papered over) rather than forcing `perplexity` to accept an unused `variant` param just for uniformity.
  - `_default_evaluators()` (`sweep.py`) and `_EVALUATORS` (`cli.py`) both now map all four names — `loglik_mc`, `perplexity`, `cloze`, `generative` — to their `evaluators.py` functions; previously only `loglik_mc` was wired (Sprint 1), with `perplexity`/`cloze`/`generative` implemented but unreachable from the CLI/sweep executor until this phase.
  - Metrics computation branches on evaluator name in both `_execute_one` and `cli.run`: `perplexity` → `metrics.perplexity_metrics(results)` (`{ppl, bpb, n_scored_tokens, n_bytes}`); every other evaluator → `{"acc": acc(results)}`, with `acc_norm` added only for `loglik_mc` (unchanged from Sprint 1/2). `RunRecord.metrics` is `dict[str, float]` regardless of evaluator, so no schema change was needed here — only which keys get populated.
  - `sweeps/main.yaml` now declares 7 targets (3 MC unchanged from Sprint 2, plus `lambada`/`cloze`, `gsm8k`/`generative`, `wikitext103`/`perplexity`, `c4_slice`/`perplexity`) × the unchanged 4-model × 3-revision grid — ~12 model points × 7 targets × ≤500 examples/target ≈ 42k examples total (C4's own `_C4_SLICE_N=200` loader constant additionally caps that target below the 500 spec cap).

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

- **Phase 2.5 implementation** (`scaling_curve_chart`, `trajectory_chart` in `figures.py`; MC benchmarks only — bpb's twin axis waits for Sprint 3's perplexity evaluator):
  - `PYTHIA_PARAM_COUNTS` (module-level dict, `figures.py`) is the single source of truth for param counts on the x-axis — deliberately *not* added to `client.py`'s model registry, since the registry's job is resolving `model_id` to an HF repo (§3), an orthogonal concern from plotting.
  - `_revision_to_step` parses `"step<N>"` labels via regex and maps the literal string `"main"` to a hardcoded `_FINAL_STEP = 143_000` (Pythia's published final checkpoint step) — both figures skip (not error on) any run whose revision doesn't match either form, so a future non-Pythia model_id in the DB degrades gracefully instead of crashing figure generation.
  - `scaling_curve_chart` only plots runs at the final checkpoint (`_revision_to_step(r.revision) == _FINAL_STEP`) — one point per (model, dataset) at the ladder's endpoint, matching "log-params vs. accuracy for all benchmarks" from architecture.md's original design. `trajectory_chart` is the complementary view: plots every parseable revision, one line per (model, dataset) pair, x-axis is training step.
  - `_CHANCE_RATE` is a small fixed dict keyed by dataset (`arc_easy`/`hellaswag`/`mmlu` → 0.25, all 4-choice MC in the Sprint-2 grid); `scaling_curve_chart` draws one dashed chance line per benchmark that has an entry, color-matched to that benchmark's line. Not generalized beyond MC yet — PPL/GSM8K have no "chance" concept in the same sense.
  - Both functions accept `metric: str = "acc"` so `acc_norm` can be plotted by passing `metric="acc_norm"`; the CLI's `figures` command currently only calls the `acc` variant (Phase 2.2's acc/acc_norm divergence-as-a-finding is future report material, not wired into the CLI's default figure set yet).
- **Phase 4.4 implementation** (`prompt_sensitivity_chart` in `figures.py`, sprints/sprint4.md): needed **no new storage path**, as predicted. `acc_norm` is already a first-class key in `RunRecord.metrics` — written for every `loglik_mc` run by both `cli.run` and `sweep._execute_one` (§8, Phase 3.6) — not buried in `detail_json`. The dot plot's new axis is *variant*, read off `RunRecord.prompt_variant_id` (§2), already stored on every non-PPL run.
  - One panel per MC benchmark with ≥2 distinct variants among its plottable runs (final checkpoint, known Pythia `model_id`, `evaluator == "loglik_mc"`); a benchmark swept under only one variant isn't a sensitivity comparison and gets no panel — mirrors `scaling_curve_chart`'s "skip, don't error" filtering.
  - Reuses the module's existing color/marker split: color = benchmark (`_color_for_dataset`, same fixed table as every other figure), marker shape = variant (`_marker_for_variant`, a new fixed table mirroring `_marker_for_model`'s pattern — known variant ids get a stable marker, unknowns fall back to a small cycle).
  - The option-text variant's `acc_norm` is drawn as a second, hollow-marker dashed line in the same color — matched by variant id ending in `mc_option_text_v1` (`_ACC_NORM_VARIANT_SUFFIX`) rather than a hardcoded list, so it also picks up the not-yet-authored `mmlu/mc_option_text_v1`-style ids for free. `acc_norm` is skipped for the letter variants since normalizing by continuation byte length is a no-op when every continuation is a single letter (`" A".." D"`, all equal length) — plotting it there would just duplicate the `acc` line.
  - `ladderctl figures` now writes a sixth file, `prompt_sensitivity.png`, unconditionally alongside the other five (empty/filtered-out input still renders a valid, empty chart).
  - `ladderctl figures` now writes three files unconditionally: `accuracy_per_run.png` (Sprint 1's bar chart, kept for now), `scaling_curve.png`, `trajectory.png`.
  - **Color/marker encoding is fixed, not a default cycler.** `trajectory_chart` plots up to 4 models × 3+ benchmarks (12+ lines); matplotlib's default color cycle only has 10 entries and starts silently reusing colors past that, making lines indistinguishable. Line color is assigned by benchmark (`_color_for_dataset`, fixed `_DATASET_COLORS` for the known Sprint-2 benchmarks, a small fallback cycle for anything else) and marker shape by model size (`_marker_for_model`, fixed `_MODEL_MARKERS` ordered smallest-to-largest Pythia size, same fallback pattern). `scaling_curve_chart` reuses `_color_for_dataset` too, so a benchmark's color is consistent across both figures. `trajectory_chart` renders two separate legends (`ax.add_artist` to keep both), one for the color→benchmark mapping and one for the marker→model mapping, rather than one combined "model/dataset" legend entry per line.
- **Phase 3.6 implementation** (`headline_figure` in `figures.py` — the headline figure described in item 1 above, now actually implemented):
  - Left axis: `acc` for the 5 accuracy-style datasets (`_HEADLINE_ACC_DATASETS = [arc_easy, hellaswag, mmlu, lambada, gsm8k]`), final checkpoint only, one dashed chance line per benchmark via the now-extended `_CHANCE_RATE` (`lambada`/`gsm8k` added at `0.0` — free-form generation/exact-match tasks have no multiple-choice guessing floor, unlike the 4-option MC benchmarks at `0.25`). Right axis (`ax2 = ax1.twinx()`): `bpb` for any run whose dataset isn't in `_HEADLINE_ACC_DATASETS` (i.e. the PPL corpora, `wikitext103`/`c4_slice`), same x-axis, same final-checkpoint filter.
  - **The right axis is inverted** (`ax2.invert_yaxis()`) so "up = better" holds simultaneously on both axes — bpb is a loss-like metric (lower is better) while accuracy is a score-like metric (higher is better); without inverting, a viewer would see the two lines move in visually opposite directions even when the underlying story (both improving with scale) is the same. This is the one twin-axis chart in the project where an inversion is load-bearing for readability, so it's called out explicitly in the function's own docstring, not just here.
  - `_color_for_dataset` extended with fixed entries for `lambada` (`tab:red`) and `gsm8k` (`tab:purple`) — previously these fell through to the fallback cycle, but the headline figure is the one place they now always appear alongside the three Sprint-2 MC benchmarks, so giving them a stable identity matters the same way it did for `arc_easy`/`hellaswag`/`mmlu`. PPL corpora (`wikitext103`/`c4_slice`) are left on the fallback cycle since they aren't otherwise fixed-colored elsewhere in the codebase.
  - `ladderctl figures` now writes two more files alongside the three Sprint 1/2 figures — `trajectory_bpb.png` (`trajectory_chart(runs, out_path, metric="bpb")`, reusing the function's existing generic `metric` parameter unchanged — no new code needed there beyond removing the hardcoded `ax.set_ylim(0, 1)` for non-accuracy metrics, since bpb has no fixed range) and `headline.png` — all unconditionally, same pattern as the others (empty/filtered-out input renders an empty-but-valid chart rather than erroring).

---

## 11. CLI (`cli.py` → `ladderctl`)

```
ladderctl run      --model pythia-160m --revision step143000 --dataset arc_easy \
                   --variant arc_easy/mc_letter_v1 --evaluator loglik_mc [--limit N]
ladderctl run      --model pythia-160m --dataset wikitext103 --evaluator perplexity  # no --variant
ladderctl sweep run sweeps/main.yaml [--db ./ladder.db]   # resumable by default
ladderctl results  [--dataset ...] [--model ...]
ladderctl show     <run_id>
ladderctl parity   <run_id_a> <run_id_b> [--report out.md]
ladderctl import   --framework lm_eval_harness <path>
ladderctl figures
```

All commands take `--db` (default `./ladder.db`). Nonzero exit on failure.
`--evaluator` accepts all four registered evaluators (`loglik_mc`, `perplexity`, `cloze`, `generative`, §8 Phase 3.6); `--variant` is optional (`None` default) and must be omitted for `--evaluator perplexity`, required for the other three.

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
