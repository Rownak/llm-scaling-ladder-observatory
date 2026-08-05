"""End-to-end integration test: `run` -> DB contents -> `figures` -> PNG.

Offline throughout (DummyClient + the ARC-Easy fixture), exercised via
`typer.testing.CliRunner` against the real `ladderctl` Typer app.
"""

from typer.testing import CliRunner

from ladder.cli import app
from ladder.storage import connect, get_example_results, list_runs

runner = CliRunner()


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
