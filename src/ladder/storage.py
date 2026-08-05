"""SQLite schema + run/result CRUD + prediction cache.

One SQLite file, WAL mode, schema created on first open (no migration
framework — architecture.md §7). Only this module writes to the DB;
`figures.py` and `parity.py` are read-only.
"""

import json
import sqlite3
from pathlib import Path

from ladder.records import ExampleResult, RunRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    prompt_variant_id TEXT,
    evaluator TEXT NOT NULL,
    framework TEXT NOT NULL,
    seed INTEGER NOT NULL,
    code_version TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    n_examples INTEGER NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS example_results (
    run_id TEXT NOT NULL,
    example_id TEXT NOT NULL,
    correct INTEGER,
    score REAL NOT NULL,
    detail_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    request_hash TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the ladder SQLite database.

    Args:
        db_path: Filesystem path to the SQLite file.

    Returns:
        A `sqlite3.Connection` in WAL mode with the full schema ensured
        (all three tables created if they did not already exist).

    Side Effects:
        Creates the file at `db_path` if it does not exist; creates any
        missing tables.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def save_run(conn: sqlite3.Connection, run: RunRecord) -> None:
    """Insert or replace a `RunRecord` row.

    Args:
        conn: Open connection from `connect`.
        run: The run to persist. `metrics` and `config` are stored as JSON;
            `started_at`/`finished_at` are stored verbatim (None -> NULL) —
            setting them as a run progresses is the caller's responsibility
            (e.g. the sweep executor), not inferred here from `status`.

    Side Effects:
        Writes one row to `runs`, replacing any existing row with the same
        `run_id` (used to move a run from "running" to "done"/"failed").
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (
            run_id, model_id, revision, dataset, split, prompt_variant_id,
            evaluator, framework, seed, code_version, metrics_json,
            config_json, n_examples, started_at, finished_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run.run_id,
            run.model_id,
            run.revision,
            run.dataset,
            run.split,
            run.prompt_variant_id,
            run.evaluator,
            run.framework,
            run.seed,
            run.code_version,
            json.dumps(run.metrics),
            json.dumps(run.config),
            run.n_examples,
            run.started_at,
            run.finished_at,
            run.status,
        ),
    )
    conn.commit()


def save_example_results(conn: sqlite3.Connection, results: list[ExampleResult]) -> None:
    """Insert `ExampleResult` rows.

    Args:
        conn: Open connection from `connect`.
        results: Results to persist. `detail` is stored as JSON; `correct`
            is stored as 0/1/NULL.

    Side Effects:
        Appends one row per result to `example_results`.
    """
    conn.executemany(
        """
        INSERT INTO example_results (run_id, example_id, correct, score, detail_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                r.run_id,
                r.example_id,
                None if r.correct is None else int(r.correct),
                r.score,
                json.dumps(r.detail),
            )
            for r in results
        ],
    )
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> RunRecord | None:
    """Fetch a single `RunRecord` by id.

    Args:
        conn: Open connection from `connect`.
        run_id: ID of the run to fetch.

    Returns:
        The `RunRecord`, or None if no run with `run_id` exists.
    """
    row = conn.execute(
        """
        SELECT run_id, model_id, revision, dataset, split, prompt_variant_id,
               evaluator, framework, seed, code_version, metrics_json,
               config_json, n_examples, status, started_at, finished_at
        FROM runs WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_run_record(row)


def list_runs(conn: sqlite3.Connection) -> list[RunRecord]:
    """Fetch all `RunRecord`s in the database.

    Args:
        conn: Open connection from `connect`.

    Returns:
        All runs, in insertion (rowid) order.
    """
    rows = conn.execute(
        """
        SELECT run_id, model_id, revision, dataset, split, prompt_variant_id,
               evaluator, framework, seed, code_version, metrics_json,
               config_json, n_examples, status, started_at, finished_at
        FROM runs ORDER BY rowid
        """
    ).fetchall()
    return [_row_to_run_record(row) for row in rows]


def get_example_results(conn: sqlite3.Connection, run_id: str) -> list[ExampleResult]:
    """Fetch all `ExampleResult`s for a run.

    Args:
        conn: Open connection from `connect`.
        run_id: ID of the run whose results to fetch.

    Returns:
        `ExampleResult`s belonging to `run_id`, in insertion order.
    """
    rows = conn.execute(
        """
        SELECT run_id, example_id, correct, score, detail_json
        FROM example_results WHERE run_id = ? ORDER BY rowid
        """,
        (run_id,),
    ).fetchall()
    return [
        ExampleResult(
            run_id=row[0],
            example_id=row[1],
            correct=None if row[2] is None else bool(row[2]),
            score=row[3],
            detail=json.loads(row[4]),
        )
        for row in rows
    ]


def _row_to_run_record(row: tuple) -> RunRecord:
    """Convert a `runs` table row tuple into a `RunRecord`.

    Args:
        row: Tuple matching the column order used by `get_run`/`list_runs`.

    Returns:
        The reconstructed `RunRecord`.
    """
    (
        run_id,
        model_id,
        revision,
        dataset,
        split,
        prompt_variant_id,
        evaluator,
        framework,
        seed,
        code_version,
        metrics_json,
        config_json,
        n_examples,
        status,
        started_at,
        finished_at,
    ) = row
    return RunRecord(
        run_id=run_id,
        model_id=model_id,
        revision=revision,
        dataset=dataset,
        split=split,
        prompt_variant_id=prompt_variant_id,
        evaluator=evaluator,
        framework=framework,
        metrics=json.loads(metrics_json),
        n_examples=n_examples,
        seed=seed,
        code_version=code_version,
        config=json.loads(config_json),
        status=status,
        started_at=started_at,
        finished_at=finished_at,
    )
