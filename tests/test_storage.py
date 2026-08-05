"""SQLite schema creation + run/result CRUD round-trip (architecture.md §7)."""

from ladder.records import ExampleResult, RunRecord
from ladder.storage import (
    connect,
    get_example_results,
    get_run,
    list_runs,
    save_example_results,
    save_run,
)


def _make_run(run_id="run-1", status="done") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        model_id="pythia-70m",
        revision="main",
        dataset="arc_easy",
        split="fixture",
        prompt_variant_id="arc_easy/mc_letter_v1",
        evaluator="loglik_mc",
        framework="ladder",
        metrics={"acc": 0.6},
        n_examples=5,
        seed=0,
        code_version="0.1.0",
        config={"limit": 5},
        status=status,
    )


def test_connect_creates_all_three_tables(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"runs", "example_results", "predictions"} <= tables


def test_connect_is_idempotent(tmp_path):
    db_path = tmp_path / "ladder.db"
    connect(db_path)
    conn = connect(db_path)  # second open must not error or wipe data
    save_run(conn, _make_run())
    assert get_run(conn, "run-1") is not None


def test_save_and_get_run_round_trip(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    run = _make_run()
    save_run(conn, run)

    fetched = get_run(conn, "run-1")
    assert fetched == run


def test_get_run_missing_returns_none(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    assert get_run(conn, "does-not-exist") is None


def test_list_runs_returns_all(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    save_run(conn, _make_run("run-1"))
    save_run(conn, _make_run("run-2"))
    runs = list_runs(conn)
    assert {r.run_id for r in runs} == {"run-1", "run-2"}


def test_save_run_replaces_existing_row(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    save_run(conn, _make_run(status="running"))
    save_run(conn, _make_run(status="done"))

    runs = list_runs(conn)
    assert len(runs) == 1
    assert runs[0].status == "done"


def test_save_and_get_run_round_trips_timestamps(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    run = _make_run()
    run.started_at = "2026-08-05T10:00:00"
    run.finished_at = "2026-08-05T10:05:00"
    save_run(conn, run)

    fetched = get_run(conn, "run-1")
    assert fetched.started_at == "2026-08-05T10:00:00"
    assert fetched.finished_at == "2026-08-05T10:05:00"


def test_save_run_defaults_timestamps_to_none(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    save_run(conn, _make_run())

    fetched = get_run(conn, "run-1")
    assert fetched.started_at is None
    assert fetched.finished_at is None


def test_save_and_get_example_results_round_trip(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    save_run(conn, _make_run())

    results = [
        ExampleResult(
            run_id="run-1", example_id="ex-1", correct=True, score=1.0, detail={"chosen_index": 0}
        ),
        ExampleResult(
            run_id="run-1", example_id="ex-2", correct=False, score=0.0, detail={"chosen_index": 2}
        ),
    ]
    save_example_results(conn, results)

    fetched = get_example_results(conn, "run-1")
    assert fetched == results


def test_get_example_results_correct_none_round_trips(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    save_run(conn, _make_run())

    result = ExampleResult(
        run_id="run-1", example_id="ex-ppl", correct=None, score=1.23, detail={"window_nlls": [0.1]}
    )
    save_example_results(conn, [result])

    fetched = get_example_results(conn, "run-1")
    assert fetched == [result]


def test_get_example_results_empty_for_unknown_run(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    assert get_example_results(conn, "no-such-run") == []
