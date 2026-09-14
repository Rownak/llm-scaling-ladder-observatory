"""Figures render from a list of RunRecords only (architecture.md §10)."""

from ladder.figures import (
    PYTHIA_PARAM_COUNTS,
    _color_for_dataset,
    _marker_for_model,
    _revision_to_step,
    accuracy_bar_chart,
    headline_figure,
    scaling_curve_chart,
    trajectory_chart,
)
from ladder.records import RunRecord


def _run(
    run_id,
    acc=None,
    acc_norm=None,
    bpb=None,
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
    if bpb is not None:
        metrics["bpb"] = bpb
    evaluator = "perplexity" if bpb is not None and acc is None else "loglik_mc"
    prompt_variant_id = None if evaluator == "perplexity" else f"{dataset}/mc_letter_v1"
    return RunRecord(
        run_id=run_id,
        model_id=model_id,
        revision=revision,
        dataset=dataset,
        split="fixture",
        prompt_variant_id=prompt_variant_id,
        evaluator=evaluator,
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


def test_trajectory_chart_renders_full_ladder_without_raising(tmp_path):
    # 4 models x 3 benchmarks x 3 revisions = 36 lines' worth of points — the
    # scenario that overflowed matplotlib's default 10-color cycle before
    # color-by-benchmark/marker-by-model was introduced.
    out_path = tmp_path / "trajectory.png"
    runs = [
        _run(f"r-{model_id}-{dataset}-{revision}", acc=0.3, model_id=model_id, revision=revision, dataset=dataset)
        for model_id in PYTHIA_PARAM_COUNTS
        for dataset in ["arc_easy", "hellaswag", "mmlu"]
        for revision in ["step1000", "step64000", "main"]
    ]

    trajectory_chart(runs, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_color_for_dataset_is_fixed_for_known_benchmarks():
    seen = {}
    assert _color_for_dataset("arc_easy", seen) == "tab:blue"
    assert _color_for_dataset("hellaswag", seen) == "tab:orange"
    assert _color_for_dataset("mmlu", seen) == "tab:green"
    assert _color_for_dataset("lambada", seen) == "tab:red"
    assert _color_for_dataset("gsm8k", seen) == "tab:purple"
    assert seen == {}  # known benchmarks never touch the fallback cache


def test_color_for_dataset_assigns_stable_fallback_colors():
    seen = {}
    first = _color_for_dataset("some_unknown_dataset", seen)
    second = _color_for_dataset("some_unknown_dataset", seen)
    other = _color_for_dataset("another_unknown_dataset", seen)
    assert first == second
    assert other != first


def test_marker_for_model_is_fixed_and_distinct_for_the_ladder():
    seen = {}
    markers = {_marker_for_model(m, seen) for m in PYTHIA_PARAM_COUNTS}
    assert len(markers) == len(PYTHIA_PARAM_COUNTS)  # every model gets its own marker
    assert seen == {}  # known Pythia sizes never touch the fallback cache


def test_marker_for_model_assigns_stable_fallback_markers():
    seen = {}
    first = _marker_for_model("some-other-model", seen)
    second = _marker_for_model("some-other-model", seen)
    assert first == second


def test_trajectory_chart_plots_bpb_without_the_zero_one_ylim(tmp_path):
    out_path = tmp_path / "trajectory_bpb.png"
    runs = [
        _run("r1", bpb=1.2, model_id="pythia-70m", revision="step1000", dataset="wikitext103"),
        _run("r2", bpb=0.9, model_id="pythia-70m", revision="step64000", dataset="wikitext103"),
        _run("r3", bpb=0.6, model_id="pythia-70m", revision="main", dataset="wikitext103"),
    ]

    result = trajectory_chart(runs, out_path, metric="bpb")

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


# --- headline_figure (Sprint 3, Phase 3.6) -----------------------------------


def test_headline_figure_writes_png(tmp_path):
    out_path = tmp_path / "headline.png"
    runs = [
        _run("r1", acc=0.3, model_id="pythia-70m", revision="main", dataset="arc_easy"),
        _run("r2", acc=0.4, model_id="pythia-160m", revision="main", dataset="arc_easy"),
        _run("r3", acc=0.26, model_id="pythia-70m", revision="main", dataset="gsm8k"),
        _run("r4", bpb=0.9, model_id="pythia-70m", revision="main", dataset="wikitext103"),
        _run("r5", bpb=0.7, model_id="pythia-160m", revision="main", dataset="wikitext103"),
    ]

    result = headline_figure(runs, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_headline_figure_skips_non_final_and_unknown_model_and_non_headline_dataset(tmp_path):
    out_path = tmp_path / "headline.png"
    runs = [
        _run("r1", acc=0.3, model_id="pythia-70m", revision="step1000", dataset="arc_easy"),  # not final
        _run("r2", acc=0.4, model_id="not-a-pythia-model", revision="main", dataset="arc_easy"),  # unknown model
        _run("r3", acc=0.5, model_id="pythia-70m", revision="main", dataset="arc_easy", status="failed"),
    ]

    # Must not raise despite every run being filtered out; produces an (empty) chart.
    headline_figure(runs, out_path)

    assert out_path.exists()


def test_headline_figure_renders_empty_when_no_runs(tmp_path):
    out_path = tmp_path / "headline.png"

    headline_figure([], out_path)

    assert out_path.exists()


def test_headline_figure_full_seven_target_ladder_without_raising(tmp_path):
    # 4 models x (5 acc datasets + 2 bpb datasets), final checkpoint only —
    # the shape `sweeps/main.yaml`'s full 7-target grid produces.
    out_path = tmp_path / "headline.png"
    acc_datasets = ["arc_easy", "hellaswag", "mmlu", "lambada", "gsm8k"]
    bpb_datasets = ["wikitext103", "c4_slice"]
    runs = [
        _run(f"r-{model_id}-{dataset}", acc=0.3, model_id=model_id, revision="main", dataset=dataset)
        for model_id in PYTHIA_PARAM_COUNTS
        for dataset in acc_datasets
    ] + [
        _run(f"r-{model_id}-{dataset}", bpb=0.8, model_id=model_id, revision="main", dataset=dataset)
        for model_id in PYTHIA_PARAM_COUNTS
        for dataset in bpb_datasets
    ]

    headline_figure(runs, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
