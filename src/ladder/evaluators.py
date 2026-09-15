"""Evaluators: render -> client call -> score -> `ExampleResult`.

Sprint 1 implements `loglik_mc`; `perplexity`, `cloze`, and `generative`
arrive in Sprint 3 Phases 3.2/3.4/3.5 (architecture.md §6).
"""

import math
import random
import string
from collections.abc import Iterator

from ladder.client import GenParams, LoglikResult, ModelClient
from ladder.datasets import get_loader
from ladder.metrics import extract_answer_number
from ladder.prompts import PromptVariant, render
from ladder.records import Example, ExampleResult, Prediction
from ladder.storage import PredictionCache

_DEFAULT_WINDOW = 1024
_NATS_PER_BIT = math.log(2)  # converts nats to bits for bpb
_CLOZE_TOKENS_PER_TARGET_WORD = 4  # generation budget headroom per target word


def loglik_mc(
    run_id: str,
    examples: Iterator[Example],
    variant: PromptVariant,
    client: ModelClient,
    cache: PredictionCache,
    model_id: str,
    revision: str,
) -> Iterator[ExampleResult]:
    """Score multiple-choice examples by argmax over continuation log-likelihoods.

    For each example: render its `RenderedRequest` via `variant` (plus
    few-shot demos, once per run, if `variant.num_fewshot > 0`), then for
    each continuation consult `cache` before calling `client.loglikelihood`
    — a cache hit skips the client call entirely (architecture.md §7). Picks
    the argmax (by raw summed loglik) as the model's chosen option. Both
    `acc` (raw loglik) and `acc_norm` (loglik normalized by continuation byte
    length) are computed per-example so their divergence across prompt
    styles can be studied later — `metrics.acc`/`metrics.acc_norm` aggregate
    at the run level.

    Few-shot demo selection lives here, not in `prompts.render` (architecture.md
    §5): demos are drawn once per run — `random.Random(variant.fewshot_seed)`
    picks `variant.num_fewshot` examples from `variant.fewshot_split` — and
    the same fixed list is passed into every example's `render` call, keeping
    `render` a pure function of its (example, variant, demos) arguments.

    Args:
        run_id: ID of the `RunRecord` these results belong to.
        examples: `Example`s to evaluate, in order.
        variant: `PromptVariant` used to render each example into a request.
        client: `ModelClient` used to score continuations on a cache miss.
        cache: `PredictionCache` consulted before every client call.
        model_id: Registry model_id used as part of the cache key. Passed in
            explicitly rather than read off `client` — `HFClient.model_id` is
            the internal HF Hub repo string (architecture.md §3), not the
            short registry id every other cache key / `RunRecord` uses.
        revision: Model checkpoint/revision used as part of the cache key.

    Yields:
        One `ExampleResult` per input example, with `detail` containing the
        chosen option index under both raw and length-normalized scoring.
    """
    demos = _select_fewshot_demos(variant)

    for example in examples:
        request = render(example, variant, demos=demos)
        assert request.continuations is not None  # mc always renders continuations

        scored = _scored_continuations(
            client, cache, model_id, revision, request.prompt, request.continuations
        )

        raw_scores = [s.loglik for s in scored]
        norm_scores = [
            s.loglik / len(cont.encode("utf-8"))
            for s, cont in zip(scored, request.continuations)
        ]

        answer_index = example.payload["answer_index"]
        chosen_raw = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
        chosen_norm = max(range(len(norm_scores)), key=lambda i: norm_scores[i])

        yield ExampleResult(
            run_id=run_id,
            example_id=example.example_id,
            correct=(chosen_raw == answer_index),
            score=1.0 if chosen_raw == answer_index else 0.0,
            detail={
                "answer_index": answer_index,
                "chosen_index": chosen_raw,
                "chosen_index_norm": chosen_norm,
                "correct_norm": chosen_norm == answer_index,
                "logliks": raw_scores,
                "logliks_norm": norm_scores,
            },
        )


def _select_fewshot_demos(variant: PromptVariant) -> list[Example] | None:
    """Draw `variant.num_fewshot` demo examples deterministically, or None for zero-shot.

    Demo selection is variant-owned (architecture.md §5): a fixed
    `random.Random(variant.fewshot_seed)` draw from `variant.fewshot_split`
    means `prompt_variant_id` alone is sufficient to reconstruct which demos
    a run used, without recording a run-level seed dependency. Called once
    per run (not per example) so every example in the run shares the same
    demo set.

    Args:
        variant: The `PromptVariant` to draw demos for. No-op (returns None)
            unless `num_fewshot > 0`.

    Returns:
        `variant.num_fewshot` examples sampled without replacement from
        `variant.fewshot_split` of `variant`'s own dataset (inferred from
        `variant.id`'s leading path segment, e.g. "arc_easy/..." ->
        "arc_easy"), in a fixed order determined by `variant.fewshot_seed`.
        None if `variant.num_fewshot == 0`.
    """
    if variant.num_fewshot == 0:
        return None

    dataset_name = variant.id.split("/")[0]
    loader = get_loader(dataset_name)
    pool = list(loader.load(variant.fewshot_split))
    rng = random.Random(variant.fewshot_seed)
    return rng.sample(pool, variant.num_fewshot)


def _scored_continuations(
    client: ModelClient,
    cache: PredictionCache,
    model_id: str,
    revision: str,
    prompt: str,
    continuations: list[str],
) -> list[LoglikResult]:
    """Score each continuation, consulting `cache` before any `client.loglikelihood` call.

    Each continuation is cached independently (its own cache key, over the
    single-element continuations list) so a partially-cached option set still
    saves calls — this matters once a variant changes only one distractor.

    Args:
        client: `ModelClient` used to score continuations on a cache miss.
        cache: `PredictionCache` consulted/updated per continuation.
        model_id: Registry model_id, part of the cache key.
        revision: Model checkpoint/revision, part of the cache key.
        prompt: The rendered prompt text all continuations are conditioned on.
        continuations: Candidate continuation strings to score, in order.

    Returns:
        One `LoglikResult` per continuation, in the same order, sourced from
        the cache where possible and the client otherwise. `n_tokens` is not
        part of `Prediction`'s cached schema (nothing downstream consumes it
        — `acc_norm` normalizes by UTF-8 byte length of the continuation
        string, not token count), so a cache hit reconstructs it as 0 rather
        than overloading `Prediction.n_bytes`, whose real meaning (UTF-8
        bytes of scored text) is load-bearing for bpb once PPL caching lands.
        `is_greedy_match` round-trips through `Prediction.is_greedy_matches`;
        a cache hit from before that field existed (`is_greedy_matches is
        None`) reconstructs it as `False` — same "unknown treated as
        not-yet-computed" fallback `n_tokens=0` already uses, since callers
        needing a real value should force a fresh score rather than trust a
        pre-existing cache entry's absent field.
    """
    results: list[LoglikResult] = []
    for cont in continuations:
        cached = cache.get(model_id, revision, "loglik", prompt, [cont], None)
        if cached is not None:
            is_greedy_match = cached.is_greedy_matches[0] if cached.is_greedy_matches else False
            results.append(LoglikResult(loglik=cached.logliks[0], n_tokens=0, is_greedy_match=is_greedy_match))
            continue

        (scored,) = client.loglikelihood(prompt, [cont])
        cache.put(
            model_id,
            revision,
            "loglik",
            prompt,
            [cont],
            None,
            Prediction(
                request_hash="",
                model_id=model_id,
                revision=revision,
                logliks=[scored.loglik],
                token_nlls=None,
                generation=None,
                n_bytes=None,
                is_greedy_matches=[scored.is_greedy_match],
            ),
        )
        results.append(scored)
    return results


def perplexity(
    run_id: str,
    examples: Iterator[Example],
    client: ModelClient,
    cache: PredictionCache,
    model_id: str,
    revision: str,
    window: int = _DEFAULT_WINDOW,
    stride: int | None = None,
) -> Iterator[ExampleResult]:
    """Score PPL documents with sliding-window `token_nlls`, no token scored twice.

    Each `Example.payload["text"]` (architecture.md §4) is split into words —
    `ModelClient.token_nlls` takes raw text, not pre-tokenized ids, so word
    boundaries (matching `DummyClient.token_nlls`'s own whitespace-splitting
    convention, and close enough for a real tokenizer since sliding-window
    boundaries only need to be *somewhere* reasonable, not token-exact) are
    what the evaluator uses to carve out windows. A window of `window` words
    is scored per call; only the words not already covered by a previous
    window are passed as `text`, the rest of the window (and nothing before
    it) as `context` — so context grows the scored evidence available to the
    model without ever being double-counted in the aggregate. `stride`
    defaults to `window // 2` per architecture.md §6.

    Each client call goes through `cache` first (kind="nll"), keyed on the
    window's context+text exactly like every other request type.

    Args:
        run_id: ID of the `RunRecord` these results belong to.
        examples: PPL `Example`s to evaluate, each with `payload["text"]`.
        client: `ModelClient` used to compute per-token NLLs on a cache miss.
        cache: `PredictionCache` consulted before every client call.
        model_id: Registry model_id, part of the cache key (see `loglik_mc`).
        revision: Model checkpoint/revision, part of the cache key.
        window: Number of words scored+contexted per sliding window.
        stride: Number of words advanced between windows; defaults to
            `window // 2` (50% overlap, architecture.md §6).

    Yields:
        One `ExampleResult` per document, `correct=None` (no notion of
        correctness for PPL), `score` set to the document's own bpb, and
        `detail["window_nlls"]` / `detail["n_bytes"]` holding the raw
        per-window NLLs and byte count `metrics.perplexity_metrics` aggregates
        across documents.
    """
    if stride is None:
        stride = window // 2

    for example in examples:
        text = example.payload["text"]
        words = text.split()
        n_bytes = len(text.encode("utf-8"))

        window_nlls: list[float] = []
        covered = 0  # words already scored by a previous window
        start = 0
        while start == 0 or start < len(words):
            window_words = words[start : start + window]
            if not window_words:
                break

            n_context = max(0, covered - start)
            context_text = " ".join(window_words[:n_context])
            new_text = " ".join(window_words[n_context:])

            if new_text:
                nll_result = cache.get(model_id, revision, "nll", new_text, None, {"context": context_text})
                if nll_result is not None:
                    nlls = nll_result.token_nlls
                else:
                    scored = client.token_nlls(new_text, context=context_text)
                    nlls = scored.nlls
                    cache.put(
                        model_id,
                        revision,
                        "nll",
                        new_text,
                        None,
                        {"context": context_text},
                        Prediction(
                            request_hash="",
                            model_id=model_id,
                            revision=revision,
                            logliks=None,
                            token_nlls=nlls,
                            generation=None,
                            n_bytes=len(new_text.encode("utf-8")),
                        ),
                    )
                window_nlls.extend(nlls)

            covered = start + len(window_words)
            if start + len(window_words) >= len(words):
                break
            start += stride

        total_nll_nats = sum(window_nlls)
        bpb = (total_nll_nats / _NATS_PER_BIT) / n_bytes if n_bytes > 0 else 0.0

        yield ExampleResult(
            run_id=run_id,
            example_id=example.example_id,
            correct=None,
            score=bpb,
            detail={"window_nlls": window_nlls, "n_bytes": n_bytes},
        )


def cloze(
    run_id: str,
    examples: Iterator[Example],
    variant: PromptVariant,
    client: ModelClient,
    cache: PredictionCache,
    model_id: str,
    revision: str,
) -> Iterator[ExampleResult]:
    """Score LAMBADA-style cloze examples: teacher-forced greedy target-word accuracy.

    For each example: render its `RenderedRequest` via `variant` (a pass-
    through cloze template, §5), then score the gold `target` in context via
    `client.loglikelihood`. **Primary metric** — `correct`/`score` — is
    `LoglikResult.is_greedy_match`: whether every token of `target` equals
    the model's argmax at its position, teacher-forced. This is the standard
    LAMBADA protocol (matches lm-evaluation-harness) and is immune to the
    measurement bug free-form generation has: greedy decoding has no reason
    to stop exactly at the target word boundary, so it can pick up trailing
    punctuation/tokens (" Queen." vs. target "Queen") and get marked wrong
    for a word it evidently knew (see `report/findings.md`'s LAMBADA
    writeup). `correct` no longer depends on `client.generate` at all.

    A free-form `generation` is still produced (same token budget as before,
    `_CLOZE_TOKENS_PER_TARGET_WORD` per target word) and stored in `detail`
    for two purposes only — never for scoring: (1) **secondary metrics**
    `target_nll`/`target_ppl`, both derived from the same `loglikelihood`
    call already made for the primary metric, no extra client cost; (2) a
    **diagnostic, explicitly non-standard** `nonstandard_generated_word_acc`
    — punctuation-stripped first-word match between `generation` and
    `target` — kept only to sanity-check that the primary metric and the old
    generation-based approach agree once punctuation is normalized away, not
    as a metric anyone should cite.

    Both the generation and the target-loglik calls go through `cache` first
    (architecture.md §7) — a "generate" cache entry is keyed on `(prompt,
    gen_params)`, a "loglik" entry on `(prompt, [target])`, so they never
    collide even though both read the same underlying example.

    Args:
        run_id: ID of the `RunRecord` these results belong to.
        examples: `Example`s to evaluate, each with `payload["context"]`/`["target"]`.
        variant: `PromptVariant` used to render each example into a request.
        client: `ModelClient` used to generate/score on a cache miss.
        cache: `PredictionCache` consulted before every client call.
        model_id: Registry model_id, part of the cache key (see `loglik_mc`).
        revision: Model checkpoint/revision, part of the cache key.

    Yields:
        One `ExampleResult` per input example; `correct`/`score` reflect the
        primary greedy-match metric. `detail` carries the target, the raw
        generation, `target_nll`, `target_ppl`, and the diagnostic
        `nonstandard_generated_word_acc`.
    """
    for example in examples:
        request = render(example, variant)
        target = example.payload["target"]

        max_new_tokens = _CLOZE_TOKENS_PER_TARGET_WORD * max(1, len(target.split()))
        gen_params = GenParams(max_new_tokens=max_new_tokens, stop=["\n"], temperature=0.0)
        generation = _cached_generate(client, cache, model_id, revision, request.prompt, gen_params)

        (target_scored,) = _scored_continuations(
            client, cache, model_id, revision, request.prompt, [f" {target}"]
        )

        is_correct = target_scored.is_greedy_match
        target_nll = -target_scored.loglik
        target_ppl = math.exp(target_nll / target_scored.n_tokens) if target_scored.n_tokens > 0 else float("inf")

        generated_word = generation.strip().split()[0] if generation.strip() else ""
        generated_word = generated_word.strip(string.punctuation)
        nonstandard_match = generated_word == target

        yield ExampleResult(
            run_id=run_id,
            example_id=example.example_id,
            correct=is_correct,
            score=1.0 if is_correct else 0.0,
            detail={
                "target": target,
                "generation": generation,
                "target_logprob": target_scored.loglik,
                "target_nll": target_nll,
                "target_ppl": target_ppl,
                "nonstandard_generated_word_acc": nonstandard_match,
            },
        )


def _cached_generate(
    client: ModelClient,
    cache: PredictionCache,
    model_id: str,
    revision: str,
    prompt: str,
    gen_params: GenParams,
) -> str:
    """Generate from `prompt`, consulting `cache` before any `client.generate` call.

    Args:
        client: `ModelClient` used to generate on a cache miss.
        cache: `PredictionCache` consulted/updated for this request.
        model_id: Registry model_id, part of the cache key.
        revision: Model checkpoint/revision, part of the cache key.
        prompt: The rendered prompt text to generate from.
        gen_params: Generation parameters, part of the cache key.

    Returns:
        The generated text, sourced from the cache where possible and the
        client otherwise.
    """
    gen_params_dict = gen_params.model_dump()
    cached = cache.get(model_id, revision, "generate", prompt, None, gen_params_dict)
    if cached is not None:
        return cached.generation

    generation = client.generate(prompt, gen_params)
    cache.put(
        model_id,
        revision,
        "generate",
        prompt,
        None,
        gen_params_dict,
        Prediction(
            request_hash="",
            model_id=model_id,
            revision=revision,
            logliks=None,
            token_nlls=None,
            generation=generation,
            n_bytes=None,
        ),
    )
    return generation


def generative(
    run_id: str,
    examples: Iterator[Example],
    variant: PromptVariant,
    client: ModelClient,
    cache: PredictionCache,
    model_id: str,
    revision: str,
) -> Iterator[ExampleResult]:
    """Score generative examples (GSM8K): generate -> extract a number -> exact match.

    For each example: render its `RenderedRequest` via `variant` — unlike
    `cloze`, the generation budget/stop sequences come from the variant's own
    `max_new_tokens`/`stop` (a property of the prompt/answer format, not the
    per-example target, since GSM8K answers are full chain-of-thought
    solutions rather than a single known-length word), carried on
    `request.gen_params` by `prompts.render`. Generates greedily, then runs
    `metrics.extract_answer_number` (pattern-first, last-number fallback) on
    the raw generation. `correct` is an exact match between the extracted
    number and `example.payload["answer_number"]` — `None` extraction (no
    number anywhere in the generation) is always incorrect, never raises.

    Args:
        run_id: ID of the `RunRecord` these results belong to.
        examples: `Example`s to evaluate, each with `payload["question"]`/`["answer_number"]`.
        variant: `PromptVariant` used to render each example (supplies
            `max_new_tokens`/`stop` via `gen_params`).
        client: `ModelClient` used to generate on a cache miss.
        cache: `PredictionCache` consulted before every client call.
        model_id: Registry model_id, part of the cache key (see `loglik_mc`).
        revision: Model checkpoint/revision, part of the cache key.

    Yields:
        One `ExampleResult` per input example; `detail` carries the raw
        generation, the extracted number, and the gold answer.
    """
    for example in examples:
        request = render(example, variant)
        assert request.gen_params is not None  # generative always renders gen_params

        gen_params = GenParams(**request.gen_params)
        generation = _cached_generate(client, cache, model_id, revision, request.prompt, gen_params)
        extracted = extract_answer_number(generation)

        answer_number = example.payload["answer_number"]
        is_correct = extracted is not None and extracted == answer_number

        yield ExampleResult(
            run_id=run_id,
            example_id=example.example_id,
            correct=is_correct,
            score=1.0 if is_correct else 0.0,
            detail={
                "answer_number": answer_number,
                "generation": generation,
                "extracted_number": extracted,
            },
        )
