"""Report figures, generated from the DB alone (architecture.md §10).

Read-only with respect to the DB — never writes runs/results, only reads
them via `storage.py`'s query functions and renders matplotlib PNGs.
"""

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never an interactive window
import matplotlib.pyplot as plt

from ladder.records import RunRecord

# Published non-embedding parameter counts for the fixed Pythia ladder
# (project_summary.md's model list). Single source of truth for the x-axis
# of every scaling figure — client.py's registry only needs HF repo strings,
# not param counts, so this table lives here rather than there.
PYTHIA_PARAM_COUNTS = {
    "pythia-70m": 70_000_000,
    "pythia-160m": 160_000_000,
    "pythia-410m": 410_000_000,
    "pythia-1b": 1_000_000_000,
}

# 4-option multiple-choice chance rate for every benchmark in the Sprint-2
# sweep (arc_easy, hellaswag, mmlu all have 4 choices per item).
_CHANCE_RATE = {"arc_easy": 0.25, "hellaswag": 0.25, "mmlu": 0.25}

# Fixed dataset -> color assignment, shared by every figure that plots more
# than one benchmark. A plain matplotlib color cycler repeats after 10 lines
# and starts colliding once trajectory_chart has 4 models x 3 benchmarks (12
# lines) — assigning color by benchmark and marker by model (below) keeps
# every line visually distinct regardless of how many models are in the DB.
_DATASET_COLORS = {
    "arc_easy": "tab:blue",
    "hellaswag": "tab:orange",
    "mmlu": "tab:green",
}
_FALLBACK_COLOR_CYCLE = ["tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


def _color_for_dataset(dataset: str, seen: dict[str, str]) -> str:
    """Look up (or deterministically assign) a plot color for `dataset`.

    Args:
        dataset: Registry dataset name.
        seen: Mutable dataset -> color cache shared across one chart's calls,
            so an unrecognized dataset gets a stable color for the life of
            that figure instead of a fresh one per line.

    Returns:
        A matplotlib color string. Known benchmarks (`_DATASET_COLORS`) get
        a fixed, memorable color; anything else is assigned the next unused
        color from `_FALLBACK_COLOR_CYCLE`, in first-seen order.
    """
    if dataset in _DATASET_COLORS:
        return _DATASET_COLORS[dataset]
    if dataset not in seen:
        seen[dataset] = _FALLBACK_COLOR_CYCLE[len(seen) % len(_FALLBACK_COLOR_CYCLE)]
    return seen[dataset]


# Fixed model -> marker assignment, ordered by param count (small model =
# simple marker, large model = more complex marker) so trajectory_chart's
# legend reads as a size progression at a glance, independent of color.
_MODEL_MARKERS = {
    "pythia-70m": "o",
    "pythia-160m": "s",
    "pythia-410m": "^",
    "pythia-1b": "D",
}
_FALLBACK_MARKER_CYCLE = ["v", "P", "X", "*", "h"]


def _marker_for_model(model_id: str, seen: dict[str, str]) -> str:
    """Look up (or deterministically assign) a plot marker for `model_id`.

    Args:
        model_id: Registry model_id.
        seen: Mutable model_id -> marker cache shared across one chart's
            calls, so an unrecognized model_id gets a stable marker for the
            life of that figure instead of a fresh one per line.

    Returns:
        A matplotlib marker string. Known Pythia sizes (`_MODEL_MARKERS`)
        get a fixed marker ordered by scale; anything else is assigned the
        next unused marker from `_FALLBACK_MARKER_CYCLE`, in first-seen order.
    """
    if model_id in _MODEL_MARKERS:
        return _MODEL_MARKERS[model_id]
    if model_id not in seen:
        seen[model_id] = _FALLBACK_MARKER_CYCLE[len(seen) % len(_FALLBACK_MARKER_CYCLE)]
    return seen[model_id]

# Pythia's final checkpoint is step143000 (project_summary.md); "main" resolves
# to it on the Hub but carries no literal step number, so the trajectory
# figure's x-axis needs an explicit mapping for that one revision label.
_FINAL_STEP = 143_000


def _revision_to_step(revision: str) -> int | None:
    """Parse a Pythia revision label into a training step number.

    Args:
        revision: A revision string, e.g. "step1000" or "main".

    Returns:
        The step number, or None if `revision` isn't a recognized Pythia
        checkpoint label (callers skip such runs rather than guessing).
    """
    if revision == "main":
        return _FINAL_STEP
    match = re.fullmatch(r"step(\d+)", revision)
    return int(match.group(1)) if match else None


def accuracy_bar_chart(runs: list[RunRecord], out_path: str | Path) -> Path:
    """Render a bar chart of accuracy per run.

    Args:
        runs: `RunRecord`s to plot, in the order they should appear on the
            x-axis. Runs without an "acc" key in `metrics` are skipped.
        out_path: Destination PNG path; parent directories are not created
            here (the caller ensures `out_dir` exists).

    Returns:
        `out_path`, coerced to a `Path`, for convenience chaining.

    Side Effects:
        Writes a PNG file to `out_path`, overwriting any existing file.
    """
    out_path = Path(out_path)
    plotted = [r for r in runs if "acc" in r.metrics]

    labels = [f"{r.model_id}\n{r.dataset}" for r in plotted]
    values = [r.metrics["acc"] for r in plotted]

    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(plotted)), 4))
    ax.bar(labels, values, color="tab:blue")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy per run")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def scaling_curve_chart(
    runs: list[RunRecord], out_path: str | Path, metric: str = "acc"
) -> Path:
    """Render log-params vs. `metric`, one line per benchmark, with chance lines.

    The headline scaling figure (architecture.md §10): for each dataset seen
    in `runs`, plots `metric` (default "acc") against `log10(param count)` at
    the final checkpoint of each model in `PYTHIA_PARAM_COUNTS`, connecting
    points across the ladder. A single dashed horizontal chance line is drawn
    per benchmark that has a known chance rate (`_CHANCE_RATE`) and at least
    one plotted point, in the same color as that benchmark's line.

    Args:
        runs: `RunRecord`s to plot. Only `status == "done"` runs whose
            `model_id` is in `PYTHIA_PARAM_COUNTS`, whose `revision` resolves
            to the final checkpoint (`_revision_to_step` == `_FINAL_STEP`),
            and whose `metrics` contains `metric` are plotted — everything
            else (unknown models, intermediate checkpoints, PPL runs without
            `metric`) is silently skipped, not an error.
        out_path: Destination PNG path.
        metric: Which metric key to plot ("acc" or "acc_norm").

    Returns:
        `out_path`, coerced to a `Path`.

    Side Effects:
        Writes a PNG file to `out_path`, overwriting any existing file.
    """
    out_path = Path(out_path)

    # dataset -> list of (log10_params, metric_value), one point per model.
    series: dict[str, list[tuple[float, float]]] = {}
    for r in runs:
        if r.status != "done":
            continue
        n_params = PYTHIA_PARAM_COUNTS.get(r.model_id)
        if n_params is None:
            continue
        if _revision_to_step(r.revision) != _FINAL_STEP:
            continue
        if metric not in r.metrics:
            continue
        series.setdefault(r.dataset, []).append((math.log10(n_params), r.metrics[metric]))

    fig, ax = plt.subplots(figsize=(7, 5))
    seen_colors: dict[str, str] = {}
    for dataset in sorted(series):
        points = sorted(series[dataset])
        xs, ys = zip(*points)
        color = _color_for_dataset(dataset, seen_colors)
        ax.plot(xs, ys, marker="o", label=dataset, color=color)
        if dataset in _CHANCE_RATE:
            ax.axhline(_CHANCE_RATE[dataset], linestyle="--", linewidth=1, color=color, alpha=0.6)

    ax.set_xlabel("log10(parameters)")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(f"Scaling curve ({metric}) — final checkpoint")
    if series:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


def trajectory_chart(runs: list[RunRecord], out_path: str | Path, metric: str = "acc") -> Path:
    """Render `metric` vs. training step, one line per (model, dataset) pair.

    The training-trajectory figure (architecture.md §10): shows how each
    model's score on each benchmark moves across its own intermediate
    checkpoints, using every revision in the sweep (not just the final one).

    With up to 4 models x 3+ benchmarks in the DB, a plain color-per-line
    cycler runs out of distinguishable colors (matplotlib's default cycle is
    10) and starts reusing them. Instead, line color encodes the benchmark
    (`_color_for_dataset`) and marker shape encodes the model size
    (`_marker_for_model`, ordered small-to-large) — every line stays visually
    distinct, and the two legends below make each encoding explicit rather
    than cramming both into one "model/dataset" label per entry.

    Args:
        runs: `RunRecord`s to plot. Only `status == "done"` runs with a
            revision `_revision_to_step` can parse and `metric` present in
            `metrics` are plotted.
        out_path: Destination PNG path.
        metric: Which metric key to plot ("acc" or "acc_norm").

    Returns:
        `out_path`, coerced to a `Path`.

    Side Effects:
        Writes a PNG file to `out_path`, overwriting any existing file.
    """
    out_path = Path(out_path)

    # (model_id, dataset) -> list of (step, metric_value).
    series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for r in runs:
        if r.status != "done":
            continue
        step = _revision_to_step(r.revision)
        if step is None:
            continue
        if metric not in r.metrics:
            continue
        series.setdefault((r.model_id, r.dataset), []).append((step, r.metrics[metric]))

    fig, ax = plt.subplots(figsize=(7, 5))
    seen_colors: dict[str, str] = {}
    seen_markers: dict[str, str] = {}
    for model_id, dataset in sorted(series):
        points = sorted(series[(model_id, dataset)])
        xs, ys = zip(*points)
        ax.plot(
            xs,
            ys,
            marker=_marker_for_model(model_id, seen_markers),
            color=_color_for_dataset(dataset, seen_colors),
        )

    ax.set_xlabel("training step")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(f"Training trajectory ({metric})")

    if series:
        datasets = sorted({dataset for _, dataset in series})
        models = sorted({model_id for model_id, _ in series}, key=lambda m: _MODEL_MARKERS.get(m, m))
        color_handles = [
            plt.Line2D([], [], color=_color_for_dataset(d, seen_colors), marker="s", linestyle="", label=d)
            for d in datasets
        ]
        marker_handles = [
            plt.Line2D([], [], color="black", marker=_marker_for_model(m, seen_markers), linestyle="", label=m)
            for m in models
        ]
        benchmark_legend = ax.legend(handles=color_handles, title="benchmark", loc="upper left", fontsize="small")
        ax.add_artist(benchmark_legend)
        ax.legend(handles=marker_handles, title="model", loc="upper right", fontsize="small")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path
