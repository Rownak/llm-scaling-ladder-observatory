"""Hand-computed fixture tests for metric aggregation (architecture.md §6, §12)."""

import math

from ladder.metrics import acc, acc_norm, perplexity_metrics
from ladder.records import ExampleResult


def _result(correct: bool) -> ExampleResult:
    return ExampleResult(
        run_id="run-1", example_id="ex", correct=correct, score=1.0 if correct else 0.0, detail={}
    )


def test_acc_hand_computed_three_of_five():
    # 3 correct, 2 incorrect -> 3/5 = 0.6 (worked by hand).
    results = [_result(True), _result(True), _result(True), _result(False), _result(False)]
    assert acc(results) == 0.6


def test_acc_all_correct():
    results = [_result(True) for _ in range(4)]
    assert acc(results) == 1.0


def test_acc_all_incorrect():
    results = [_result(False) for _ in range(4)]
    assert acc(results) == 0.0


def test_acc_empty_list():
    assert acc([]) == 0.0


def _mc_result(*, correct: bool, correct_norm: bool) -> ExampleResult:
    return ExampleResult(
        run_id="run-1",
        example_id="ex",
        correct=correct,
        score=1.0 if correct else 0.0,
        detail={"correct_norm": correct_norm},
    )


def test_acc_norm_hand_computed_disagrees_with_acc():
    # Long-option bias case, worked by hand.
    #
    # Example A: correct answer is a long option. Raw summed loglik is more
    # negative for longer strings even when the model favors that option per
    # token, so an unnormalized argmax picks a short *wrong* distractor
    # instead (correct=False). Once normalized by continuation byte length,
    # the long correct option wins (correct_norm=True):
    #   raw:  wrong short option:  -2.0            (argmax: wrong, since -2.0 > -8.0)
    #         right long option:   -8.0 / 20 bytes
    #   norm: wrong short option:  -2.0 / 4 bytes  = -0.50
    #         right long option:   -8.0 / 20 bytes = -0.40   (argmax: right, -0.40 > -0.50)
    #
    # Example B: both raw and normalized argmax agree and are correct.
    #
    # acc:      only B correct      -> 1/2 = 0.5
    # acc_norm: both A and B correct -> 2/2 = 1.0
    results = [
        _mc_result(correct=False, correct_norm=True),  # A
        _mc_result(correct=True, correct_norm=True),  # B
    ]

    assert acc(results) == 0.5
    assert acc_norm(results) == 1.0
    assert acc(results) != acc_norm(results)


def test_acc_norm_all_correct():
    results = [_mc_result(correct=True, correct_norm=True) for _ in range(4)]
    assert acc_norm(results) == 1.0


def test_acc_norm_all_incorrect():
    results = [_mc_result(correct=False, correct_norm=False) for _ in range(4)]
    assert acc_norm(results) == 0.0


def test_acc_norm_empty_list():
    assert acc_norm([]) == 0.0


def _ppl_result(window_nlls: list[float], n_bytes: int, example_id: str = "doc") -> ExampleResult:
    return ExampleResult(
        run_id="run-1",
        example_id=example_id,
        correct=None,
        score=0.0,
        detail={"window_nlls": window_nlls, "n_bytes": n_bytes},
    )


def test_perplexity_metrics_hand_computed_single_document():
    # Toy doc: 4 pinned NLLs (nats), 10 UTF-8 bytes — values chosen by hand,
    # not from a real client, to isolate the aggregation formula itself.
    #   window_nlls = [1.0, 2.0, 0.5, 0.5]
    #   total_nll_nats = 4.0
    #   n_scored_tokens = 4
    #   ppl = exp(4.0 / 4) = exp(1.0) = 2.718281828459045
    #   bpb = (4.0 / ln(2)) / 10 = (4.0 / 0.6931471805599453) / 10 = 0.5770780163555853
    results = [_ppl_result([1.0, 2.0, 0.5, 0.5], n_bytes=10)]

    metrics = perplexity_metrics(results)

    assert metrics["n_scored_tokens"] == 4.0
    assert metrics["n_bytes"] == 10.0
    assert math.isclose(metrics["ppl"], 2.718281828459045, rel_tol=1e-9)
    assert math.isclose(metrics["bpb"], 0.5770780163555853, rel_tol=1e-9)


def test_perplexity_metrics_hand_computed_multi_document_sums_across_docs():
    # Two documents; aggregation sums nlls/tokens/bytes across both before
    # dividing (not a per-document average of per-document ppl/bpb).
    #   doc A: window_nlls=[1.0, 1.0], n_bytes=5   -> nll=2.0, tokens=2
    #   doc B: window_nlls=[3.0],       n_bytes=3   -> nll=3.0, tokens=1
    #   total_nll_nats = 5.0, n_scored_tokens = 3, n_bytes = 8
    #   ppl = exp(5.0 / 3) = exp(1.666666...) = 5.29449005047003
    #   bpb = (5.0 / ln(2)) / 8 = (5.0 / 0.6931471805599453) / 8 = 0.9016844005556022
    results = [
        _ppl_result([1.0, 1.0], n_bytes=5, example_id="doc-a"),
        _ppl_result([3.0], n_bytes=3, example_id="doc-b"),
    ]

    metrics = perplexity_metrics(results)

    assert metrics["n_scored_tokens"] == 3.0
    assert metrics["n_bytes"] == 8.0
    assert math.isclose(metrics["ppl"], 5.29449005047003, rel_tol=1e-9)
    assert math.isclose(metrics["bpb"], 0.9016844005556022, rel_tol=1e-9)


def test_perplexity_metrics_empty_results():
    metrics = perplexity_metrics([])
    assert metrics["n_scored_tokens"] == 0.0
    assert metrics["n_bytes"] == 0.0
    assert metrics["ppl"] == float("inf")
    assert metrics["bpb"] == 0.0
