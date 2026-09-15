"""Metric aggregation functions.

Every metric here has a hand-computed fixture test (architecture.md §12).
Evaluators call these to turn per-example `ExampleResult`s into the
aggregate `metrics` dict stored on a `RunRecord`.
"""

import math
import re

from ladder.records import ExampleResult

_NATS_PER_BIT = math.log(2)  # converts nats to bits for bpb

# Matches a signed, comma-grouped, optionally-decimal number, e.g. "-1,234.5".
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Cue phrases GSM8K-style generations use to flag the final answer explicitly,
# checked in order — the first one present wins (architecture.md §6).
_ANSWER_CUE_PATTERNS = [
    re.compile(r"Final answer:\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)"),
]


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


def cloze_metrics(results: list[ExampleResult]) -> dict[str, float]:
    """Aggregate `evaluators.cloze` results into run-level secondary/diagnostic metrics.

    `acc` (the primary greedy-match metric, from `ExampleResult.correct`) is
    computed by the existing `acc()` function, same as every other
    exact-match evaluator — this function supplies the metrics that live
    only in `detail` (architecture.md §6's LAMBADA measurement-bug writeup).

    Args:
        results: `ExampleResult`s from `evaluators.cloze`. Each must carry
            `detail["target_nll"]`, `detail["target_ppl"]`, and
            `detail["nonstandard_generated_word_acc"]`.

    Returns:
        `{"target_nll_mean": ..., "target_ppl_mean": ...,
        "nonstandard_generated_word_acc": ...}` — mean target NLL (nats) and
        mean per-example target perplexity across all results (both 0.0 for
        an empty list), and the fraction of results whose punctuation-
        normalized generated word matched the target (the diagnostic
        metric — not a substitute for `acc`).
    """
    if not results:
        return {"target_nll_mean": 0.0, "target_ppl_mean": 0.0, "nonstandard_generated_word_acc": 0.0}

    n = len(results)
    target_nll_mean = sum(r.detail["target_nll"] for r in results) / n
    target_ppl_mean = sum(r.detail["target_ppl"] for r in results) / n
    nonstandard_acc = sum(1 for r in results if r.detail["nonstandard_generated_word_acc"]) / n

    return {
        "target_nll_mean": target_nll_mean,
        "target_ppl_mean": target_ppl_mean,
        "nonstandard_generated_word_acc": nonstandard_acc,
    }


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


def extract_answer_number(text: str) -> float | None:
    """Extract the model's final numeric answer from a generative response (architecture.md §6).

    Pattern-first, last-number fallback:

    1. If `text` contains an explicit answer cue (`"Final answer: N"`, case-
       insensitive, or a GSM8K-style `"#### N"` line), the number following
       the *last* such cue wins — a generation that reasons its way through
       several numbers but explicitly flags its answer should be scored on
       that flagged number, not whatever number happens to appear last in
       free text.
    2. Otherwise, falls back to the last number appearing anywhere in
       `text` — the closest a free-form chain-of-thought response gets to
       "the final answer" without an explicit cue.

    Comma thousands-separators are stripped before parsing (`"1,234"` ->
    `1234.0`); negative numbers and decimals are both recognized.

    Args:
        text: Raw model generation to extract a numeric answer from.

    Returns:
        The extracted number as a float, or None if `text` contains no
        number at all.
    """
    for pattern in _ANSWER_CUE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return float(matches[-1].replace(",", ""))

    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))
