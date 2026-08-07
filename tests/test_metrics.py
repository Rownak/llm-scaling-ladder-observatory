"""Hand-computed fixture tests for metric aggregation (architecture.md §6, §12)."""

from ladder.metrics import acc, acc_norm
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
