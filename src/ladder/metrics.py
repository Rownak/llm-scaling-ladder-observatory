"""Metric aggregation functions.

Every metric here has a hand-computed fixture test (architecture.md §12).
Evaluators call these to turn per-example `ExampleResult`s into the
aggregate `metrics` dict stored on a `RunRecord`.
"""

from ladder.records import ExampleResult


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
