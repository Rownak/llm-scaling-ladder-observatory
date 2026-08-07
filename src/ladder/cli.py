"""`ladderctl` — the one CLI for the project (architecture.md §11).

Sprint 1 wires the walking-skeleton commands: `run`, `results`, `show`,
`figures`. `sweep`, `parity`, and `import` arrive in later sprints.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer

from ladder import __version__
from ladder.client import get_client
from ladder.datasets import get_loader
from ladder.evaluators import loglik_mc
from ladder.figures import accuracy_bar_chart
from ladder.metrics import acc, acc_norm
from ladder.prompts import load_variant
from ladder.records import RunRecord
from ladder.storage import (
    PredictionCache,
    connect,
    get_example_results,
    get_run,
    list_runs,
    save_example_results,
    save_run,
)

app = typer.Typer(add_completion=False)

_EVALUATORS = {"loglik_mc": loglik_mc}


def _now_iso() -> str:
    """Current UTC time as an ISO 8601 string (second precision).

    Returns:
        A timestamp suitable for `RunRecord.started_at`/`finished_at`.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@app.command()
def run(
    model: str = typer.Option(..., help="model_id, e.g. pythia-70m"),
    revision: str = typer.Option("main", help="Model checkpoint/revision"),
    dataset: str = typer.Option(..., help="Registry dataset name, e.g. arc_easy"),
    variant: str = typer.Option(..., help="Prompt variant id, e.g. arc_easy/mc_letter_v1"),
    evaluator: str = typer.Option(..., help="Evaluator name, e.g. loglik_mc"),
    split: str = typer.Option("test", help="Dataset split (use 'fixture' for offline runs)"),
    limit: int = typer.Option(None, help="Max examples to evaluate"),
    seed: int = typer.Option(0, help="Random seed recorded on the run"),
    db: Path = typer.Option(Path("./ladder.db"), help="SQLite DB path"),
) -> None:
    """Run the full pipeline for one (model, dataset, variant, evaluator) point.

    Loads examples, evaluates them, aggregates metrics, and stores the
    `RunRecord` + `ExampleResult`s in the DB. Prints the run_id and metrics
    on success; exits nonzero and marks the run "failed" on error.

    Side Effects:
        Writes a "running" row immediately, then a final "done"/"failed" row,
        plus one `example_results` row per evaluated example.
    """
    if evaluator not in _EVALUATORS:
        typer.echo(f"Unknown evaluator: {evaluator!r}. Known: {sorted(_EVALUATORS)}", err=True)
        raise typer.Exit(code=1)

    run_id = f"run-{uuid.uuid4().hex[:12]}"
    conn = connect(db)

    config = {
        "model": model,
        "revision": revision,
        "dataset": dataset,
        "variant": variant,
        "evaluator": evaluator,
        "split": split,
        "limit": limit,
        "seed": seed,
    }
    run_record = RunRecord(
        run_id=run_id,
        model_id=model,
        revision=revision,
        dataset=dataset,
        split=split,
        prompt_variant_id=variant,
        evaluator=evaluator,
        framework="ladder",
        metrics={},
        n_examples=0,
        seed=seed,
        code_version=__version__,
        config=config,
        status="running",
        started_at=_now_iso(),
    )
    save_run(conn, run_record)

    try:
        loader = get_loader(dataset)
        examples = list(loader.load(split, limit=limit))
        prompt_variant = load_variant(variant)
        client = get_client(model, revision=revision)
        cache = PredictionCache(conn)

        evaluator_fn = _EVALUATORS[evaluator]
        results = list(evaluator_fn(run_id, examples, prompt_variant, client, cache, model, revision))
        save_example_results(conn, results)

        metrics = {"acc": acc(results)}
        if evaluator == "loglik_mc":
            metrics["acc_norm"] = acc_norm(results)

        run_record.status = "done"
        run_record.metrics = metrics
        run_record.n_examples = len(results)
        run_record.finished_at = _now_iso()
        run_record.config = {**run_record.config, "cache_hits": cache.hits, "cache_misses": cache.misses}
        save_run(conn, run_record)
    except Exception as exc:
        run_record.status = "failed"
        run_record.finished_at = _now_iso()
        save_run(conn, run_record)
        typer.echo(f"Run {run_id} failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"run_id: {run_id}")
    typer.echo(f"metrics: {run_record.metrics}")


@app.command()
def results(db: Path = typer.Option(Path("./ladder.db"), help="SQLite DB path")) -> None:
    """Print a table of all runs in the database.

    Side Effects:
        Writes to stdout; exits nonzero if the DB has no runs.
    """
    conn = connect(db)
    runs = list_runs(conn)
    if not runs:
        typer.echo("No runs found.", err=True)
        raise typer.Exit(code=1)

    header = f"{'run_id':<20} {'model_id':<14} {'dataset':<12} {'evaluator':<12} {'status':<10} metrics"
    typer.echo(header)
    for r in runs:
        typer.echo(f"{r.run_id:<20} {r.model_id:<14} {r.dataset:<12} {r.evaluator:<12} {r.status:<10} {r.metrics}")


@app.command()
def show(
    run_id: str,
    db: Path = typer.Option(Path("./ladder.db"), help="SQLite DB path"),
) -> None:
    """Print full reproducibility metadata and per-example results for one run.

    Args:
        run_id: ID of the run to display.

    Side Effects:
        Writes to stdout; exits nonzero if `run_id` is not found.
    """
    conn = connect(db)
    run_record = get_run(conn, run_id)
    if run_record is None:
        typer.echo(f"No such run: {run_id!r}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"run_id:            {run_record.run_id}")
    typer.echo(f"model_id/revision: {run_record.model_id} / {run_record.revision}")
    typer.echo(f"dataset/split:     {run_record.dataset} / {run_record.split}")
    typer.echo(f"prompt_variant_id: {run_record.prompt_variant_id}")
    typer.echo(f"evaluator:         {run_record.evaluator}")
    typer.echo(f"framework:         {run_record.framework}")
    typer.echo(f"seed:              {run_record.seed}")
    typer.echo(f"code_version:      {run_record.code_version}")
    typer.echo(f"status:            {run_record.status}")
    typer.echo(f"started_at:        {run_record.started_at}")
    typer.echo(f"finished_at:       {run_record.finished_at}")
    typer.echo(f"n_examples:        {run_record.n_examples}")
    typer.echo(f"metrics:           {run_record.metrics}")
    typer.echo(f"config:            {run_record.config}")

    example_results = get_example_results(conn, run_id)
    typer.echo(f"\nper-example results ({len(example_results)}):")
    for r in example_results:
        typer.echo(f"  {r.example_id}: correct={r.correct} score={r.score} detail={r.detail}")


@app.command()
def figures(
    db: Path = typer.Option(Path("./ladder.db"), help="SQLite DB path"),
    out_dir: Path = typer.Option(Path("./report/figures"), help="Output directory for figures"),
) -> None:
    """Regenerate all report figures from the DB alone.

    Side Effects:
        Writes PNG files to `out_dir`, creating it if needed.
    """
    conn = connect(db)
    runs = list_runs(conn)
    if not runs:
        typer.echo("No runs found; nothing to plot.", err=True)
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "accuracy_per_run.png"
    accuracy_bar_chart(runs, out_path)
    typer.echo(f"Wrote {out_path}")

