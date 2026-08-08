"""Metric aggregation functions.

Every metric here has a hand-computed fixture test (architecture.md §12).
Evaluators call these to turn per-example `ExampleResult`s into the
aggregate `metrics` dict stored on a `RunRecord`.
"""

import math

from ladder.records import ExampleResult

_NATS_PER_BIT = math.log(2)  # converts nats to bits for bpb


def acc(results: list[ExampleResult]) -> float:
    """Compute accuracy as the fraction of results marked correct.

    Args:
        results: `ExampleResult`s to aggregate. Each must have `correct` set
            to a bool (not None) — non-PPL evaluators only.

    Returns:
        Number of correct results divided by total results, as a float in
        [0, 1]. Returns 0.0 for an empty list.
    """
    if not results:
        return 0.0
    n_correct = sum(1 for r in results if r.correct)
    return n_correct / len(results)


def acc_norm(results: list[ExampleResult]) -> float:
    """Compute byte-length-normalized accuracy from `loglik_mc` results.

    Reads `detail["correct_norm"]` — whether the argmax over
    byte-length-normalized logliks (`loglik / len(continuation.encode("utf-8"))`)
    matches the answer index — as written by `evaluators.loglik_mc`. Diverges
    from raw `acc` when a longer option accumulates more total logprob but a
    shorter one wins once normalized, or vice versa (architecture.md §6).

    Args:
        results: `ExampleResult`s from `loglik_mc`. Each must carry
            `detail["correct_norm"]` (bool).

    Returns:
        Number of `correct_norm` results divided by total results, as a
        float in [0, 1]. Returns 0.0 for an empty list.
    """
    if not results:
        return 0.0
    n_correct = sum(1 for r in results if r.detail["correct_norm"])
    return n_correct / len(results)


def perplexity_metrics(results: list[ExampleResult]) -> dict[str, float]:
    """Aggregate per-document PPL results into run-level `ppl`/`bpb` (architecture.md §6).

    Sums `detail["window_nlls"]` (nats) and `detail["n_bytes"]` across every
    document — as written by `evaluators.perplexity`, one `ExampleResult` per
    document with its own already-deduplicated sliding-window NLLs — then:

    - `ppl = exp(total_nll_nats / n_scored_tokens)`: per-model only, never
      compared across tokenizers (different vocabularies score different
      numbers of tokens for the same text, so raw PPL isn't comparable
      cross-model).
    - `bpb = total_nll_bits / total_utf8_bytes`: the canonical cross-model
      metric, since byte count is tokenizer-independent.

    Args:
        results: `ExampleResult`s from `evaluators.perplexity`. Each must
            carry `detail["window_nlls"]` (list[float], nats) and
            `detail["n_bytes"]` (int).

    Returns:
        `{"ppl": ..., "bpb": ..., "n_scored_tokens": ..., "n_bytes": ...}`.
        `ppl` is `float("inf")` and `bpb` is 0.0 if there are zero scored
        tokens (e.g. an empty result list) — there is no well-defined
        per-token average of an empty sum.
    """
    total_nll_nats = 0.0
    n_scored_tokens = 0
    n_bytes = 0
    for r in results:
        total_nll_nats += sum(r.detail["window_nlls"])
        n_scored_tokens += len(r.detail["window_nlls"])
        n_bytes += r.detail["n_bytes"]

    ppl = math.exp(total_nll_nats / n_scored_tokens) if n_scored_tokens > 0 else float("inf")
    bpb = (total_nll_nats / _NATS_PER_BIT) / n_bytes if n_bytes > 0 else 0.0

    return {
        "ppl": ppl,
        "bpb": bpb,
        "n_scored_tokens": float(n_scored_tokens),
        "n_bytes": float(n_bytes),
    }
