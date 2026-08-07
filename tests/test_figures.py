"""Figures render from a list of RunRecords only (architecture.md §10)."""

from ladder.figures import (
    PYTHIA_PARAM_COUNTS,
    _revision_to_step,
    accuracy_bar_chart,
    scaling_curve_chart,
    trajectory_chart,
)
from ladder.records import RunRecord


def _run(
    run_id,
    acc=None,
    acc_norm=None,
    model_id="dummy",
    revision="main",
    dataset="arc_easy",
    status="done",
):
    metrics = {}
    if acc is not None:
        metrics["acc"] = acc
    if acc_norm is not None:
        metrics["acc_norm"] = acc_norm
    return RunRecord(
        run_id=run_id,
        model_id=model_id,
        revision=revision,
        dataset=dataset,
        split="fixture",
        prompt_variant_id=f"{dataset}/mc_letter_v1",
        evaluator="loglik_mc",
        framework="ladder",
        metrics=metrics,
        n_examples=20,
        seed=0,
        code_version="0.1.0",
        config={},
        status=status,
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


def test_revision_to_step_parses_step_labels_and_main():
    assert _revision_to_step("step1000") == 1000
    assert _revision_to_step("step64000") == 64000
    assert _revision_to_step("main") == 143_000


def test_revision_to_step_returns_none_for_unrecognized_labels():
    assert _revision_to_step("not-a-revision") is None
    assert _revision_to_step("") is None


def test_pythia_param_counts_covers_the_full_ladder():
    assert set(PYTHIA_PARAM_COUNTS) == {"pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b"}
    # Strictly increasing with model size, as every scaling-curve x-axis assumes.
    ordered = [PYTHIA_PARAM_COUNTS[m] for m in ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b"]]
    assert ordered == sorted(ordered)
    assert len(set(ordered)) == len(ordered)


def test_scaling_curve_chart_writes_png(tmp_path):
    out_path = tmp_path / "scaling.png"
    runs = [
        _run("r1", acc=0.3, model_id="pythia-70m", revision="main", dataset="arc_easy"),
        _run("r2", acc=0.4, model_id="pythia-160m", revision="main", dataset="arc_easy"),
        _run("r3", acc=0.35, model_id="pythia-70m", revision="main", dataset="hellaswag"),
    ]

    result = scaling_curve_chart(runs, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_scaling_curve_chart_skips_non_final_and_unknown_model_and_missing_metric(tmp_path):
    out_path = tmp_path / "scaling.png"
    runs = [
        _run("r1", acc=0.3, model_id="pythia-70m", revision="step1000", dataset="arc_easy"),  # not final
        _run("r2", acc=0.4, model_id="not-a-pythia-model", revision="main", dataset="arc_easy"),  # unknown model
        _run("r3", acc=None, model_id="pythia-70m", revision="main", dataset="arc_easy"),  # no acc metric
        _run("r4", acc=0.5, model_id="pythia-70m", revision="main", dataset="arc_easy", status="failed"),
    ]

    # Must not raise despite every run being filtered out; produces an (empty) chart.
    scaling_curve_chart(runs, out_path)

    assert out_path.exists()


def test_scaling_curve_chart_plots_acc_norm_when_requested(tmp_path):
    out_path = tmp_path / "scaling_norm.png"
    runs = [
        _run("r1", acc=0.3, acc_norm=0.28, model_id="pythia-70m", revision="main", dataset="arc_easy"),
    ]

    scaling_curve_chart(runs, out_path, metric="acc_norm")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_trajectory_chart_writes_png(tmp_path):
    out_path = tmp_path / "trajectory.png"
    runs = [
        _run("r1", acc=0.26, model_id="pythia-70m", revision="step1000", dataset="arc_easy"),
        _run("r2", acc=0.30, model_id="pythia-70m", revision="step64000", dataset="arc_easy"),
        _run("r3", acc=0.32, model_id="pythia-70m", revision="main", dataset="arc_easy"),
    ]

    result = trajectory_chart(runs, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_trajectory_chart_skips_unparseable_revisions_and_missing_metric(tmp_path):
    out_path = tmp_path / "trajectory.png"
    runs = [
        _run("r1", acc=0.26, model_id="pythia-70m", revision="not-a-checkpoint", dataset="arc_easy"),
        _run("r2", acc=None, model_id="pythia-70m", revision="step1000", dataset="arc_easy"),
    ]

    # Must not raise despite every run being filtered out.
    trajectory_chart(runs, out_path)

    assert out_path.exists()
