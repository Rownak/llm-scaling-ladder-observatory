"""accuracy_bar_chart: renders from a list of RunRecords only (architecture.md §10)."""

from ladder.figures import accuracy_bar_chart
from ladder.records import RunRecord


def _run(run_id, acc=None):
    return RunRecord(
        run_id=run_id,
        model_id="dummy",
        revision="main",
        dataset="arc_easy",
        split="fixture",
        prompt_variant_id="arc_easy/mc_letter_v1",
        evaluator="loglik_mc",
        framework="ladder",
        metrics={} if acc is None else {"acc": acc},
        n_examples=20,
        seed=0,
        code_version="0.1.0",
        config={},
        status="done",
    )


def test_accuracy_bar_chart_writes_png(tmp_path):
    out_path = tmp_path / "acc.png"
    runs = [_run("run-1", acc=0.3), _run("run-2", acc=0.5)]

    result = accuracy_bar_chart(runs, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_accuracy_bar_chart_skips_runs_without_acc_metric(tmp_path):
    out_path = tmp_path / "acc.png"
    runs = [_run("run-1", acc=0.3), _run("run-2", acc=None)]

    # Must not raise despite one run lacking an "acc" key.
    accuracy_bar_chart(runs, out_path)

    assert out_path.exists()
