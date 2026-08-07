"""End-to-end integration test: `run` -> DB contents -> `figures` -> PNG.

Offline throughout (DummyClient + the ARC-Easy fixture), exercised via
`typer.testing.CliRunner` against the real `ladderctl` Typer app.
"""

from typer.testing import CliRunner

from ladder.cli import app
from ladder.client import DummyClient, register
from ladder.storage import connect, get_example_results, list_runs

runner = CliRunner()

# A second DummyClient registration so the mini-sweep test (below) can exercise
# a real 2-model axis without touching HFClient/real downloads.
register("dummy2", lambda revision="main", **kwargs: DummyClient(revision=revision, model_id="dummy2", **kwargs))


def test_run_then_figures_end_to_end(tmp_path):
    db_path = tmp_path / "ladder.db"
    fig_dir = tmp_path / "figures"

    run_result = runner.invoke(
        app,
        [
            "run",
            "--model", "dummy",
            "--revision", "main",
            "--dataset", "arc_easy",
            "--variant", "arc_easy/mc_letter_v1",
            "--evaluator", "loglik_mc",
            "--split", "fixture",
            "--db", str(db_path),
        ],
    )
    assert run_result.exit_code == 0, run_result.output
    assert "run_id:" in run_result.output
    assert "metrics:" in run_result.output

    # Assert DB contents directly (architecture.md §7's storage contract).
    conn = connect(db_path)
    runs = list_runs(conn)
    assert len(runs) == 1
    run_record = runs[0]
    assert run_record.status == "done"
    assert run_record.model_id == "dummy"
    assert run_record.dataset == "arc_easy"
    assert run_record.n_examples == 20
    assert "acc" in run_record.metrics
    assert "acc_norm" in run_record.metrics
    # First run against a fresh DB: every continuation is a cache miss, none are hits.
    assert run_record.config["cache_misses"] > 0
    assert run_record.config["cache_hits"] == 0
    assert run_record.started_at is not None
    assert run_record.finished_at is not None

    example_results = get_example_results(conn, run_record.run_id)
    assert len(example_results) == 20

    figures_result = runner.invoke(
        app, ["figures", "--db", str(db_path), "--out-dir", str(fig_dir)]
    )
    assert figures_result.exit_code == 0, figures_result.output

    png_path = fig_dir / "accuracy_per_run.png"
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_run_reusing_same_db_hits_prediction_cache(tmp_path):
    db_path = tmp_path / "ladder.db"
    args = [
        "run",
        "--model", "dummy",
        "--dataset", "arc_easy",
        "--variant", "arc_easy/mc_letter_v1",
        "--evaluator", "loglik_mc",
        "--split", "fixture",
        "--db", str(db_path),
    ]

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output

    conn = connect(db_path)
    runs = list_runs(conn)
    assert len(runs) == 2  # two distinct run_ids, same underlying requests

    second_run = runs[1]
    # Same model/dataset/variant/split as the first run -> every request was
    # already cached from run 1, so run 2 makes zero new client calls.
    assert second_run.config["cache_hits"] > 0
    assert second_run.config["cache_misses"] == 0
    assert second_run.metrics == runs[0].metrics


def test_results_and_show_after_run(tmp_path):
    db_path = tmp_path / "ladder.db"

    runner.invoke(
        app,
        [
            "run",
            "--model", "dummy",
            "--dataset", "arc_easy",
            "--variant", "arc_easy/mc_letter_v1",
            "--evaluator", "loglik_mc",
            "--split", "fixture",
            "--db", str(db_path),
        ],
    )

    results_result = runner.invoke(app, ["results", "--db", str(db_path)])
    assert results_result.exit_code == 0
    assert "dummy" in results_result.output
    assert "arc_easy" in results_result.output

    conn = connect(db_path)
    run_id = list_runs(conn)[0].run_id

    show_result = runner.invoke(app, ["show", run_id, "--db", str(db_path)])
    assert show_result.exit_code == 0
    assert run_id in show_result.output
    assert "seed:" in show_result.output
    assert "code_version:" in show_result.output
    assert "per-example results (20)" in show_result.output


def test_run_unknown_evaluator_exits_nonzero(tmp_path):
    db_path = tmp_path / "ladder.db"
    result = runner.invoke(
        app,
        [
            "run",
            "--model", "dummy",
            "--dataset", "arc_easy",
            "--variant", "arc_easy/mc_letter_v1",
            "--evaluator", "not_a_real_evaluator",
            "--split", "fixture",
            "--db", str(db_path),
        ],
    )
    assert result.exit_code != 0


def test_run_unknown_model_marks_run_failed_and_exits_nonzero(tmp_path):
    db_path = tmp_path / "ladder.db"
    result = runner.invoke(
        app,
        [
            "run",
            "--model", "not-a-real-model",
            "--dataset", "arc_easy",
            "--variant", "arc_easy/mc_letter_v1",
            "--evaluator", "loglik_mc",
            "--split", "fixture",
            "--db", str(db_path),
        ],
    )
    assert result.exit_code != 0

    conn = connect(db_path)
    runs = list_runs(conn)
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].finished_at is not None


def test_show_missing_run_exits_nonzero(tmp_path):
    db_path = tmp_path / "ladder.db"
    result = runner.invoke(app, ["show", "no-such-run", "--db", str(db_path)])
    assert result.exit_code != 0


def test_results_empty_db_exits_nonzero(tmp_path):
    db_path = tmp_path / "ladder.db"
    result = runner.invoke(app, ["results", "--db", str(db_path)])
    assert result.exit_code != 0


def test_figures_empty_db_exits_nonzero(tmp_path):
    db_path = tmp_path / "ladder.db"
    result = runner.invoke(app, ["figures", "--db", str(db_path), "--out-dir", str(tmp_path / "figs")])
    assert result.exit_code != 0


def test_sweep_run_then_resume_via_cli(tmp_path):
    db_path = tmp_path / "ladder.db"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        """
models:
  - model_id: dummy
    revisions: [main]
targets:
  - dataset: arc_easy
    variant: arc_easy/mc_letter_v1
    evaluator: loglik_mc
    split: fixture
limit: 5
seed: 0
""",
        encoding="utf-8",
    )

    first = runner.invoke(app, ["sweep", "run", str(spec_path), "--db", str(db_path)])
    assert first.exit_code == 0, first.output
    assert "Executed 1 run(s); 0 failed." in first.output

    conn = connect(db_path)
    runs = list_runs(conn)
    assert len(runs) == 1
    assert runs[0].status == "done"

    # Rerunning the identical spec against the same DB resumes: the one run
    # is already `done`, so nothing new executes and no duplicate row appears.
    second = runner.invoke(app, ["sweep", "run", str(spec_path), "--db", str(db_path)])
    assert second.exit_code == 0, second.output
    assert "Executed 0 run(s); 0 failed." in second.output

    runs = list_runs(conn)
    assert len(runs) == 1


def test_mini_sweep_then_figures_render_from_db(tmp_path):
    """Sprint 2 Phase 2.5 integration test: 2 models x 2 revisions x 2 datasets, cap 10.

    DummyClient stands in for both model axes ("dummy"/"dummy2" are both
    registered, architecture.md §3) since the offline suite never touches
    real Pythia checkpoints; the point is proving `sweep run` -> `figures`
    renders scaling + trajectory PNGs from the DB alone, not real scaling
    behavior.
    """
    db_path = tmp_path / "ladder.db"
    fig_dir = tmp_path / "figures"
    spec_path = tmp_path / "mini_sweep.yaml"
    spec_path.write_text(
        """
models:
  - model_id: dummy
    revisions: [step1000, main]
  - model_id: dummy2
    revisions: [step1000, main]
targets:
  - dataset: arc_easy
    variant: arc_easy/mc_letter_v1
    evaluator: loglik_mc
    split: fixture
  - dataset: hellaswag
    variant: hellaswag/mc_context_v1
    evaluator: loglik_mc
    split: fixture
limit: 10
seed: 0
""",
        encoding="utf-8",
    )

    sweep_result = runner.invoke(app, ["sweep", "run", str(spec_path), "--db", str(db_path)])
    assert sweep_result.exit_code == 0, sweep_result.output
    assert "Executed 8 run(s); 0 failed." in sweep_result.output

    conn = connect(db_path)
    runs = list_runs(conn)
    assert len(runs) == 8
    assert all(r.status == "done" for r in runs)
    assert all(r.n_examples == 10 for r in runs)

    figures_result = runner.invoke(app, ["figures", "--db", str(db_path), "--out-dir", str(fig_dir)])
    assert figures_result.exit_code == 0, figures_result.output

    for name in ["accuracy_per_run.png", "scaling_curve.png", "trajectory.png"]:
        png_path = fig_dir / name
        assert png_path.exists(), name
        assert png_path.stat().st_size > 0, name
