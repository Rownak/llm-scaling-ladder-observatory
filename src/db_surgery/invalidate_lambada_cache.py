"""Invalidate stale LAMBADA cache entries after the greedy-match fix.

Context: `evaluators.cloze`'s primary metric changed from generation
string-match to `LoglikResult.is_greedy_match` (architecture.md §6, pre-
Sprint-5 fix; see `execution_logs/sprint4_phase4.5.md`). `Prediction` gained
a parallel `is_greedy_matches` field, but every LAMBADA prediction already
cached in `ladder.db` predates that field. `evaluators._scored_continuations`
treats a missing field as "unknown" and falls back to `is_greedy_match=False`
on a cache hit — so re-running LAMBADA today would silently keep scoring
every example wrong via stale cache hits, not recompute.

`predictions` has no `dataset` column (it's keyed purely on a content hash
over `model_id|revision|kind|prompt|continuations|gen_params`), so which rows
belong to LAMBADA can't be read off the table directly. This script
reconstructs the exact cache keys LAMBADA's `done` runs would have produced
— by reloading each run's examples from the same loader/split/limit and
rendering them through the same variant `evaluators.cloze` uses — and deletes
only those `predictions` rows, plus the stale `runs`/`example_results` rows
for LAMBADA (so the sweep executor's run-level resume doesn't just skip them
again). Nothing outside LAMBADA's own runs/predictions is touched.

Usage:
    python -m db_surgery.invalidate_lambada_cache --db ./ladder.db --dry-run
    python -m db_surgery.invalidate_lambada_cache --db ./ladder.db
"""

import argparse
import sqlite3
from pathlib import Path

from ladder.client import GenParams
from ladder.datasets import get_loader
from ladder.evaluators import _CLOZE_TOKENS_PER_TARGET_WORD
from ladder.prompts import load_variant, render
from ladder.storage import connect, list_runs, prediction_cache_key

_DATASET = "lambada"


def _cache_keys_for_run(run) -> set[str]:
    """Recompute every `predictions` cache key `evaluators.cloze` would have written for one run.

    Reloads the run's examples from the same loader/split/limit it was
    executed with, renders each via the run's own `prompt_variant_id`, and
    reconstructs both the "loglik" key (target continuation, matching
    `_scored_continuations`) and the "generate" key (matching
    `_cached_generate`) exactly as `evaluators.cloze` builds them.

    Args:
        run: A `RunRecord` with `dataset == "lambada"` and `status == "done"`.

    Returns:
        The set of `request_hash` values this run would have populated in
        `predictions`.
    """
    loader = get_loader(run.dataset)
    variant = load_variant(run.prompt_variant_id)
    limit = run.config.get("limit")

    keys: set[str] = set()
    for example in loader.load(run.split, limit=limit):
        request = render(example, variant)
        target = example.payload["target"]

        loglik_key = prediction_cache_key(
            run.model_id, run.revision, "loglik", request.prompt, [f" {target}"], None
        )
        keys.add(loglik_key)

        max_new_tokens = _CLOZE_TOKENS_PER_TARGET_WORD * max(1, len(target.split()))
        gen_params = GenParams(max_new_tokens=max_new_tokens, stop=["\n"], temperature=0.0)
        generate_key = prediction_cache_key(
            run.model_id, run.revision, "generate", request.prompt, None, gen_params.model_dump()
        )
        keys.add(generate_key)

    return keys


def invalidate(conn: sqlite3.Connection, dry_run: bool = True) -> dict[str, int]:
    """Delete stale LAMBADA predictions and their now-invalid run/result rows.

    Args:
        conn: Open connection from `ladder.storage.connect`.
        dry_run: If True, compute and report counts without deleting anything.

    Returns:
        `{"runs": n_runs, "example_results": n_results, "predictions": n_predictions}`
        — counts of rows identified (dry run) or actually deleted.

    Side Effects:
        If `dry_run` is False: deletes matching rows from `predictions`,
        `example_results`, and `runs`, then commits.
    """
    lambada_runs = [r for r in list_runs(conn) if r.dataset == _DATASET and r.status == "done"]

    all_keys: set[str] = set()
    for run in lambada_runs:
        all_keys |= _cache_keys_for_run(run)

    run_ids = [r.run_id for r in lambada_runs]
    n_results = 0
    for run_id in run_ids:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM example_results WHERE run_id = ?", (run_id,)
        ).fetchone()
        n_results += count

    n_predictions_present = 0
    for key in all_keys:
        row = conn.execute("SELECT 1 FROM predictions WHERE request_hash = ?", (key,)).fetchone()
        if row is not None:
            n_predictions_present += 1

    if not dry_run:
        for key in all_keys:
            conn.execute("DELETE FROM predictions WHERE request_hash = ?", (key,))
        for run_id in run_ids:
            conn.execute("DELETE FROM example_results WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        conn.commit()

    return {
        "runs": len(run_ids),
        "example_results": n_results,
        "predictions": n_predictions_present,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("./ladder.db"), help="SQLite DB path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without deleting anything (default: actually delete)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"No DB found at {args.db}")

    conn = connect(args.db)
    counts = invalidate(conn, dry_run=args.dry_run)

    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"{verb}:")
    print(f"  {counts['runs']} LAMBADA run(s)")
    print(f"  {counts['example_results']} example_results row(s)")
    print(f"  {counts['predictions']} predictions row(s)")
    if args.dry_run:
        print("\nRe-run without --dry-run to actually delete.")
    else:
        print("\nDone. Re-run sweeps/main.yaml (or a LAMBADA-only spec) to recompute.")


if __name__ == "__main__":
    main()
