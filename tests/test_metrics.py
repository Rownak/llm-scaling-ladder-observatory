"""Hand-computed fixture tests for metric aggregation (architecture.md §6, §12)."""

from ladder.metrics import acc
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
