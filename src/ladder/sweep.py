"""Sweep spec expansion + resumable sequential executor (architecture.md §8).

A `SweepSpec` declares axes (models×revisions, eval targets, example cap) and
expands deterministically into an ordered list of `SweepRun`s, grouped by
(model_id, revision) so each checkpoint is loaded once. `run_sweep` executes
that list against the real pipeline (loader -> variant -> evaluator -> cache
-> storage), skipping any run whose identical config already has a `done`
row in the DB — resume is the default behavior, not a separate mode.
"""

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from ladder import __version__
from ladder.client import ModelClient, get_client
from ladder.datasets import get_loader
from ladder.metrics import acc, acc_norm
from ladder.prompts import load_variant
from ladder.records import RunRecord
from ladder.storage import PredictionCache, list_runs, save_example_results, save_run

_EVALUATORS: dict[str, Callable] = {}


def _default_evaluators() -> dict[str, Callable]:
    """Build the evaluator-name -> function map, imported lazily to avoid a cycle.

    Returns:
        Mapping from evaluator name (as used in sweep targets / `RunRecord.evaluator`)
        to its `ladder.evaluators` function.
    """
    from ladder.evaluators import loglik_mc

    return {"loglik_mc": loglik_mc}


class SweepTarget(BaseModel):
    """One (dataset, variant, evaluator) evaluation target within a sweep.

    Attributes:
        dataset: Registry dataset name.
        variant: Prompt variant id (`prompts/library/` path, no .yaml suffix).
        evaluator: Evaluator name (key into the evaluator registry, e.g. "loglik_mc").
        split: Dataset split to evaluate.
    """

    dataset: str
    variant: str
    evaluator: str
    split: str = "test"


class SweepModel(BaseModel):
    """One model axis entry: a model_id and the revisions to sweep over it.

    Attributes:
        model_id: Registry model_id (e.g. "pythia-70m").
        revisions: Checkpoint/revisions to run, in the order they execute.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    revisions: list[str]


class SweepSpec(BaseModel):
    """Declarative sweep definition, loaded from a YAML file (architecture.md §8).

    Attributes:
        models: Model axis: each entry's revisions are swept in order.
        targets: Eval-target axis: (dataset, variant, evaluator, split) tuples,
            run against every model x revision.
        limit: Max examples per run (applied to every target), or None for no cap.
        seed: Random seed recorded on every run.
    """

    models: list[SweepModel]
    targets: list[SweepTarget]
    limit: int | None = None
    seed: int = 0


class SweepRun(BaseModel):
    """One fully-resolved run within an expanded sweep — the executor's unit of work.

    Attributes:
        model_id: Registry model_id.
        revision: Checkpoint/revision.
        dataset: Registry dataset name.
        variant: Prompt variant id.
        evaluator: Evaluator name.
        split: Dataset split.
        limit: Max examples for this run, or None.
        seed: Random seed for this run.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    revision: str
    dataset: str
    variant: str
    evaluator: str
    split: str
    limit: int | None
    seed: int


def load_sweep_spec(path: str | Path) -> SweepSpec:
    """Load and validate a `SweepSpec` from a YAML file.

    Args:
        path: Filesystem path to the sweep spec YAML.

    Returns:
        The parsed and validated `SweepSpec`.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SweepSpec.model_validate(data)


def expand_sweep(spec: SweepSpec) -> list[SweepRun]:
    """Deterministically expand a `SweepSpec` into an ordered list of `SweepRun`s.

    Iteration order is models (outer) x that model's revisions x targets
    (inner), matching spec file order exactly. This ordering is what makes
    the executor's (model_id, revision) grouping contiguous — each checkpoint
    is loaded once, all its runs executed, then unloaded (architecture.md §8).

    Args:
        spec: The sweep spec to expand.

    Returns:
        One `SweepRun` per (model, revision, target) combination, in
        deterministic order.
    """
    runs = []
    for model in spec.models:
        for revision in model.revisions:
            for target in spec.targets:
                runs.append(
                    SweepRun(
                        model_id=model.model_id,
                        revision=revision,
                        dataset=target.dataset,
                        variant=target.variant,
                        evaluator=target.evaluator,
                        split=target.split,
                        limit=spec.limit,
                        seed=spec.seed,
                    )
                )
    return runs


def _sweep_run_key(run: SweepRun) -> tuple:
    """Identity tuple for a `SweepRun`, used to detect already-completed runs.

    Args:
        run: The sweep run to key.

    Returns:
        A tuple of every field that determines the run's output — matching
        this against a stored `RunRecord`'s fields (plus its `config["limit"]`/
        `config["seed"]`) is how the executor recognizes "the same run" across
        process restarts, since a fresh `run_id` is minted each execution.
    """
    return (
        run.model_id,
        run.revision,
        run.dataset,
        run.split,
        run.variant,
        run.evaluator,
        run.limit,
        run.seed,
    )


def _done_run_keys(conn) -> set[tuple]:
    """Collect identity keys of every already-`done` run in the DB.

    Args:
        conn: Open connection from `storage.connect`.

    Returns:
        Set of `_sweep_run_key`-shaped tuples for all `done` runs, used by
        `run_sweep` to skip re-executing identical work on resume.
    """
    keys = set()
    for r in list_runs(conn):
        if r.status != "done":
            continue
        keys.add(
            (
                r.model_id,
                r.revision,
                r.dataset,
                r.split,
                r.prompt_variant_id,
                r.evaluator,
                r.config.get("limit"),
                r.seed,
            )
        )
    return keys


def _now_iso() -> str:
    """Current UTC time as an ISO 8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _execute_one(conn, run: SweepRun, client: ModelClient, evaluators: dict[str, Callable]) -> RunRecord:
    """Execute a single `SweepRun` against an already-loaded `client` and persist it.

    Mirrors `cli.run`'s pipeline (load examples -> render+score -> aggregate
    -> persist) but takes a pre-constructed client so the executor can reuse
    one model load across every revision's runs.

    Args:
        conn: Open DB connection from `storage.connect`.
        run: The resolved run to execute.
        client: `ModelClient` already loaded for `run.model_id`/`run.revision`.
        evaluators: Evaluator-name -> function map.

    Returns:
        The final `RunRecord`, with `status` "done" or "failed".

    Side Effects:
        Writes a "running" row, then a final "done"/"failed" row, plus one
        `example_results` row per evaluated example on success.
    """
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    config = {
        "model": run.model_id,
        "revision": run.revision,
        "dataset": run.dataset,
        "variant": run.variant,
        "evaluator": run.evaluator,
        "split": run.split,
        "limit": run.limit,
        "seed": run.seed,
    }
    run_record = RunRecord(
        run_id=run_id,
        model_id=run.model_id,
        revision=run.revision,
        dataset=run.dataset,
        split=run.split,
        prompt_variant_id=run.variant,
        evaluator=run.evaluator,
        framework="ladder",
        metrics={},
        n_examples=0,
        seed=run.seed,
        code_version=__version__,
        config=config,
        status="running",
        started_at=_now_iso(),
    )
    save_run(conn, run_record)

    try:
        evaluator_fn = evaluators[run.evaluator]
        loader = get_loader(run.dataset)
        examples = list(loader.load(run.split, limit=run.limit))
        prompt_variant = load_variant(run.variant)
        cache = PredictionCache(conn)

        results = list(
            evaluator_fn(run_id, examples, prompt_variant, client, cache, run.model_id, run.revision)
        )
        save_example_results(conn, results)

        metrics = {"acc": acc(results)}
        if run.evaluator == "loglik_mc":
            metrics["acc_norm"] = acc_norm(results)

        run_record.status = "done"
        run_record.metrics = metrics
        run_record.n_examples = len(results)
        run_record.finished_at = _now_iso()
        run_record.config = {**run_record.config, "cache_hits": cache.hits, "cache_misses": cache.misses}
        save_run(conn, run_record)
    except Exception:
        run_record.status = "failed"
        run_record.finished_at = _now_iso()
        save_run(conn, run_record)

    return run_record


def run_sweep(conn, spec: SweepSpec, evaluators: dict[str, Callable] | None = None) -> list[RunRecord]:
    """Execute a `SweepSpec`, resuming from whatever is already `done` in the DB.

    Expands `spec` (`expand_sweep`), groups the resulting runs by
    (model_id, revision) — contiguous by construction — and for each group:
    loads the client once, executes every non-`done` run in the group, then
    calls `client.unload()`. A run that raises is caught inside
    `_execute_one`, marked `failed`, and the sweep continues to the next run
    (architecture.md §8). Runs already `done` for an identical config
    (matched via `_sweep_run_key`) are skipped entirely — no client call, no
    new row — which is what makes interrupt-and-resume free.

    Args:
        conn: Open DB connection from `storage.connect`.
        spec: The sweep spec to execute.
        evaluators: Evaluator-name -> function map; defaults to the full
            registry (`loglik_mc`). Overridable for tests.

    Returns:
        `RunRecord`s actually executed this call, in execution order —
        skipped (already-`done`) runs are not included.

    Side Effects:
        Writes run/result rows to `conn` for every non-skipped run; loads and
        unloads one `ModelClient` per distinct (model_id, revision) group
        that has at least one non-skipped run.
    """
    if evaluators is None:
        evaluators = _default_evaluators()

    planned = expand_sweep(spec)
    done_keys = _done_run_keys(conn)

    executed: list[RunRecord] = []
    current_key: tuple[str, str] | None = None
    client: ModelClient | None = None

    for run in planned:
        if _sweep_run_key(run) in done_keys:
            continue

        group_key = (run.model_id, run.revision)
        if group_key != current_key:
            if client is not None:
                client.unload()
            client = get_client(run.model_id, revision=run.revision)
            current_key = group_key

        run_record = _execute_one(conn, run, client, evaluators)
        executed.append(run_record)
        if run_record.status == "done":
            done_keys.add(_sweep_run_key(run))

    if client is not None:
        client.unload()

    return executed
