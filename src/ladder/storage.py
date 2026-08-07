"""SQLite schema + run/result CRUD + prediction cache.

One SQLite file, WAL mode, schema created on first open (no migration
framework — architecture.md §7). Only this module writes to the DB;
`figures.py` and `parity.py` are read-only.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

from ladder.records import ExampleResult, Prediction, RunRecord

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


def prediction_cache_key(
    model_id: str,
    revision: str,
    kind: str,
    prompt: str,
    continuations: list[str] | None,
    gen_params: dict | None,
) -> str:
    """Compute the content-hash cache key for one client call (architecture.md §7).

    `sha256` over the canonical JSON encoding of
    `(model_id, revision, kind, prompt, continuations, gen_params)` — same
    inputs always produce the same key, regardless of process or dict
    key-insertion order (`sort_keys=True`).

    Args:
        model_id: Registry name of the model.
        revision: Model checkpoint/revision.
        kind: Request kind ("loglik" | "generate" | "nll").
        prompt: The rendered prompt text.
        continuations: Continuation strings scored (loglik requests), or None.
        gen_params: Generation parameters (dict form), or None.

    Returns:
        A hex-encoded sha256 digest identifying this exact request.
    """
    canonical = json.dumps(
        {
            "model_id": model_id,
            "revision": revision,
            "kind": kind,
            "prompt": prompt,
            "continuations": continuations,
            "gen_params": gen_params,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_prediction(conn: sqlite3.Connection, request_hash: str) -> Prediction | None:
    """Look up a cached `Prediction` by its request hash.

    Args:
        conn: Open connection from `connect`.
        request_hash: Key produced by `prediction_cache_key`.

    Returns:
        The cached `Prediction`, or None on a cache miss.
    """
    row = conn.execute(
        "SELECT model_id, revision, payload_json FROM predictions WHERE request_hash = ?",
        (request_hash,),
    ).fetchone()
    if row is None:
        return None
    model_id, revision, payload_json = row
    payload = json.loads(payload_json)
    return Prediction(
        request_hash=request_hash,
        model_id=model_id,
        revision=revision,
        logliks=payload["logliks"],
        token_nlls=payload["token_nlls"],
        generation=payload["generation"],
        n_bytes=payload["n_bytes"],
    )


def save_prediction(conn: sqlite3.Connection, prediction: Prediction) -> None:
    """Insert or replace a cached `Prediction` row.

    Args:
        conn: Open connection from `connect`.
        prediction: The prediction to cache, keyed by `prediction.request_hash`.

    Side Effects:
        Writes one row to `predictions`, replacing any existing row with the
        same `request_hash` (a given hash is expected to always map to the
        same output, so replacement is a no-op in practice, not a semantic
        change).
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO predictions (request_hash, model_id, revision, payload_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            prediction.request_hash,
            prediction.model_id,
            prediction.revision,
            json.dumps(
                {
                    "logliks": prediction.logliks,
                    "token_nlls": prediction.token_nlls,
                    "generation": prediction.generation,
                    "n_bytes": prediction.n_bytes,
                }
            ),
        ),
    )
    conn.commit()


class PredictionCache:
    """Content-hash-keyed cache of `Prediction`s over one DB connection.

    The object evaluators consult before every `ModelClient` call
    (architecture.md §6/§7). Wraps `prediction_cache_key`/`get_prediction`/
    `save_prediction` and additionally counts hits/misses so a run's
    `config` can record cache effectiveness (Phase 2.3).

    Attributes:
        hits: Number of `get` calls that found a cached `Prediction`.
        misses: Number of `get` calls that found nothing.
    """

    def __init__(self, conn: sqlite3.Connection):
        """Bind the cache to an open connection.

        Args:
            conn: Open connection from `connect`.
        """
        self._conn = conn
        self.hits = 0
        self.misses = 0

    def get(
        self,
        model_id: str,
        revision: str,
        kind: str,
        prompt: str,
        continuations: list[str] | None,
        gen_params: dict | None,
    ) -> Prediction | None:
        """Look up a cached `Prediction` for this exact request, counting the outcome.

        Args:
            model_id: Registry name of the model.
            revision: Model checkpoint/revision.
            kind: Request kind ("loglik" | "generate" | "nll").
            prompt: The rendered prompt text.
            continuations: Continuation strings scored (loglik requests), or None.
            gen_params: Generation parameters (dict form), or None.

        Returns:
            The cached `Prediction`, or None on a miss.

        Side Effects:
            Increments `self.hits` or `self.misses`.
        """
        request_hash = prediction_cache_key(model_id, revision, kind, prompt, continuations, gen_params)
        prediction = get_prediction(self._conn, request_hash)
        if prediction is None:
            self.misses += 1
        else:
            self.hits += 1
        return prediction

    def put(
        self,
        model_id: str,
        revision: str,
        kind: str,
        prompt: str,
        continuations: list[str] | None,
        gen_params: dict | None,
        prediction: Prediction,
    ) -> None:
        """Cache `prediction` under the key for this exact request.

        Args:
            model_id: Registry name of the model.
            revision: Model checkpoint/revision.
            kind: Request kind ("loglik" | "generate" | "nll").
            prompt: The rendered prompt text.
            continuations: Continuation strings scored (loglik requests), or None.
            gen_params: Generation parameters (dict form), or None.
            prediction: The `Prediction` to store (its own `request_hash` is
                overwritten with the freshly computed key, so callers may pass
                a placeholder).

        Side Effects:
            Writes one row to `predictions` via `save_prediction`.
        """
        request_hash = prediction_cache_key(model_id, revision, kind, prompt, continuations, gen_params)
        prediction.request_hash = request_hash
        save_prediction(self._conn, prediction)


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
