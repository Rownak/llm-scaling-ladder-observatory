"""Evaluators: render -> client call -> score -> `ExampleResult`.

Sprint 1 implements `loglik_mc` only; `cloze`, `generative`, and `perplexity`
arrive with their datasets/clients in later sprints (architecture.md §6).
"""

from collections.abc import Iterator

from ladder.client import LoglikResult, ModelClient
from ladder.prompts import PromptVariant, render
from ladder.records import Example, ExampleResult, Prediction
from ladder.storage import PredictionCache


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
