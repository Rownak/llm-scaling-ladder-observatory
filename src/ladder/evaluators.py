"""Evaluators: render -> client call -> score -> `ExampleResult`.

Sprint 1 implements `loglik_mc` only; `cloze`, `generative`, and `perplexity`
arrive with their datasets/clients in later sprints (architecture.md §6).
"""

from collections.abc import Iterator

from ladder.client import ModelClient
from ladder.prompts import PromptVariant, render
from ladder.records import Example, ExampleResult


def loglik_mc(
    run_id: str,
    examples: Iterator[Example],
    variant: PromptVariant,
    client: ModelClient,
) -> Iterator[ExampleResult]:
    """Score multiple-choice examples by argmax over continuation log-likelihoods.

    For each example: render its `RenderedRequest` via `variant`, score every
    continuation with `client.loglikelihood`, and pick the argmax (by raw
    summed loglik) as the model's chosen option. Both `acc` (raw loglik) and
    `acc_norm` (loglik normalized by continuation byte length) are computed
    per-example so their divergence across prompt styles can be studied later
    — `metrics.acc` aggregates whichever is selected at the run level.

    Args:
        run_id: ID of the `RunRecord` these results belong to.
        examples: `Example`s to evaluate, in order.
        variant: `PromptVariant` used to render each example into a request.
        client: `ModelClient` used to score continuations.

    Yields:
        One `ExampleResult` per input example, with `detail` containing the
        chosen option index under both raw and length-normalized scoring.
    """
    for example in examples:
        request = render(example, variant)
        assert request.continuations is not None  # mc always renders continuations

        scored = client.loglikelihood(request.prompt, request.continuations)

        raw_scores = [r.loglik for r in scored]
        norm_scores = [
            r.loglik / len(cont.encode("utf-8"))
            for r, cont in zip(scored, request.continuations)
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
