"""Evaluators: render -> client call -> score -> `ExampleResult`.

Sprint 1 implements `loglik_mc`; `perplexity` arrives in Sprint 3 Phase 3.2;
`cloze` and `generative` arrive later in Sprint 3 (architecture.md §6).
"""

import math
from collections.abc import Iterator

from ladder.client import LoglikResult, ModelClient
from ladder.prompts import PromptVariant, render
from ladder.records import Example, ExampleResult, Prediction
from ladder.storage import PredictionCache

_DEFAULT_WINDOW = 1024
_NATS_PER_BIT = math.log(2)  # converts nats to bits for bpb


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

    For each example: render its `RenderedRequest` via `variant`, then for
    each continuation consult `cache` before calling `client.loglikelihood`
    — a cache hit skips the client call entirely (architecture.md §7). Picks
    the argmax (by raw summed loglik) as the model's chosen option. Both
    `acc` (raw loglik) and `acc_norm` (loglik normalized by continuation byte
    length) are computed per-example so their divergence across prompt
    styles can be studied later — `metrics.acc`/`metrics.acc_norm` aggregate
    at the run level.

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
    for example in examples:
        request = render(example, variant)
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
    """
    results: list[LoglikResult] = []
    for cont in continuations:
        cached = cache.get(model_id, revision, "loglik", prompt, [cont], None)
        if cached is not None:
            results.append(LoglikResult(loglik=cached.logliks[0], n_tokens=0))
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
