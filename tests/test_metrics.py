"""Hand-computed fixture tests for metric aggregation (architecture.md §6, §12)."""

import math

from ladder.metrics import acc, acc_norm, cloze_metrics, extract_answer_number, perplexity_metrics
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


def _cloze_result(
    *, target_nll: float, target_ppl: float, nonstandard_match: bool, example_id: str = "ex"
) -> ExampleResult:
    return ExampleResult(
        run_id="run-1",
        example_id=example_id,
        correct=None,  # cloze_metrics doesn't read `correct` (acc() already covers the primary metric)
        score=0.0,
        detail={
            "target_nll": target_nll,
            "target_ppl": target_ppl,
            "nonstandard_generated_word_acc": nonstandard_match,
        },
    )


def test_cloze_metrics_hand_computed():
    # Two examples, worked by hand:
    #   ex0: target_nll=2.0, target_ppl=exp(2.0)=7.389, nonstandard match=True
    #   ex1: target_nll=4.0, target_ppl=exp(4.0)=54.598, nonstandard match=False
    # target_nll_mean = (2.0 + 4.0) / 2 = 3.0
    # target_ppl_mean = (7.389 + 54.598) / 2 = 30.994 (approx)
    # nonstandard_generated_word_acc = 1/2 = 0.5
    results = [
        _cloze_result(target_nll=2.0, target_ppl=math.exp(2.0), nonstandard_match=True, example_id="ex0"),
        _cloze_result(target_nll=4.0, target_ppl=math.exp(4.0), nonstandard_match=False, example_id="ex1"),
    ]

    metrics = cloze_metrics(results)

    assert metrics["target_nll_mean"] == 3.0
    assert math.isclose(metrics["target_ppl_mean"], (math.exp(2.0) + math.exp(4.0)) / 2)
    assert metrics["nonstandard_generated_word_acc"] == 0.5


def test_cloze_metrics_empty_list():
    metrics = cloze_metrics([])
    assert metrics == {"target_nll_mean": 0.0, "target_ppl_mean": 0.0, "nonstandard_generated_word_acc": 0.0}


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


# --- extract_answer_number (Sprint 3, Phase 3.5) ----------------------------
#
# Extraction fixture set — pattern-first ("Final answer:" / "#### N"), then
# last-number fallback. Each case's expected value is spelled out below.


def test_extract_answer_number_final_answer_cue():
    # "Final answer:" cue present -> take the number right after it, ignoring
    # earlier numbers in the reasoning. Expected: 8.0
    text = "She had 3 apples then bought 5 more. Final answer: 8"
    assert extract_answer_number(text) == 8.0


def test_extract_answer_number_hashes_cue():
    # GSM8K-native "#### N" cue, no "Final answer:" present. Expected: 1234.0
    # (comma thousands-separator stripped).
    text = "Total receipts across the year sum to #### 1,234"
    assert extract_answer_number(text) == 1234.0


def test_extract_answer_number_final_answer_cue_wins_over_hashes():
    # Both cues present, on different lines -> "Final answer:" (checked
    # first) wins over the earlier "#### 5". Expected: 8.0, not 5.0.
    text = "#### 5\nFinal answer: 8"
    assert extract_answer_number(text) == 8.0


def test_extract_answer_number_negative_number():
    # No cue phrase; last number in free text is negative. Expected: -50.0
    text = "He lost $-50 on the trade."
    assert extract_answer_number(text) == -50.0


def test_extract_answer_number_comma_grouped_no_cue():
    # No cue phrase; comma thousands-separator must be stripped before
    # parsing. Expected: 12345.0
    text = "The total sales were 12,345 units this quarter."
    assert extract_answer_number(text) == 12345.0


def test_extract_answer_number_decimal_no_cue():
    # No cue phrase; a decimal number. Expected: 19.99
    text = "The price is $19.99 after the discount was applied."
    assert extract_answer_number(text) == 19.99


def test_extract_answer_number_mid_reasoning_last_number_fallback():
    # No cue phrase; several numbers appear during reasoning, no explicit
    # flag for which is final -> falls back to the last one. Expected: 21.0
    text = "Step 1: 12 - 5 = 7. Step 2: 7 * 3 = 21. That's the result."
    assert extract_answer_number(text) == 21.0


def test_extract_answer_number_no_number_returns_none():
    # No number anywhere in the text -> None, not an exception or 0.
    text = "I'm not sure how to solve this one."
    assert extract_answer_number(text) is None


def test_extract_answer_number_negative_decimal_with_cue():
    # "Final Answer:" cue, case-insensitive, negative decimal. Expected: -12.5
    text = "After accounting for the loss, Final Answer: -12.5"
    assert extract_answer_number(text) == -12.5
