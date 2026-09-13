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

# Chance rate for every accuracy-style benchmark in the full Sprint-3 sweep.
# arc_easy/hellaswag/mmlu are 4-option MC (0.25); lambada is a giant-vocabulary
# generation match, whose chance rate is ~0 (not exactly 0, but indistinguishable
# from it at the scale plotted here); gsm8k is free-form numeric generation, so
# "chance" is likewise ~0 (architecture.md §10 — GSM8K-at-chance is itself the
# expected Sprint 3 finding, not a bug).
_CHANCE_RATE = {"arc_easy": 0.25, "hellaswag": 0.25, "mmlu": 0.25, "lambada": 0.0, "gsm8k": 0.0}

# The 5 accuracy-style benchmarks plotted on the headline figure's left axis
# (architecture.md §10) — excludes the 2 PPL corpora (wikitext103, c4_slice),
# which contribute to the right (bpb) axis instead via a separate metric.
_HEADLINE_ACC_DATASETS = ["arc_easy", "hellaswag", "mmlu", "lambada", "gsm8k"]

# Fixed dataset -> color assignment, shared by every figure that plots more
# than one benchmark. A plain matplotlib color cycler repeats after 10 lines
# and starts colliding once trajectory_chart has 4 models x 3 benchmarks (12
# lines) — assigning color by benchmark and marker by model (below) keeps
# every line visually distinct regardless of how many models are in the DB.
_DATASET_COLORS = {
    "arc_easy": "tab:blue",
    "hellaswag": "tab:orange",
    "mmlu": "tab:green",
    "lambada": "tab:red",
    "gsm8k": "tab:purple",
}
_FALLBACK_COLOR_CYCLE = ["tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


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


# bpb is "smaller is better"; the headline figure plots it on a twin y-axis
# inverted (via `ax2.invert_yaxis()`) so both axes read "up = better" at a
# glance — a viewer shouldn't have to remember that one of the two lines on
# the same figure is inverted relative to the other (architecture.md §10).
def headline_figure(runs: list[RunRecord], out_path: str | Path) -> Path:
    """Render the project's headline figure: accuracy (5 benchmarks) + bpb (2 PPL corpora) vs. log-params.

    The core Sprint 3 finding (architecture.md §10, sprints/sprint3.md Phase
    3.6): left axis plots `acc` for the 5 accuracy-style benchmarks
    (`_HEADLINE_ACC_DATASETS`) against `log10(param count)`, with a dashed
    chance line per benchmark (`_CHANCE_RATE`); right axis plots `bpb` for the
    2 PPL corpora (wikitext103, c4_slice) on the same x-axis, inverted so
    "up = better" holds for both axes simultaneously — bits-per-byte falling
    as scale increases should visually track the same direction as accuracy
    rising, letting a viewer read "smooth benchmarks + bpb improve together,
    MMLU/GSM8K stay flat at chance" directly off the plot.

    Only the final checkpoint of each model in `PYTHIA_PARAM_COUNTS` is
    plotted (matching `scaling_curve_chart`) — this is the "at full training"
    scaling view; per-checkpoint training dynamics are `trajectory_chart`'s job.

    Args:
        runs: `RunRecord`s to plot. Only `status == "done"` runs whose
            `model_id` is in `PYTHIA_PARAM_COUNTS` and whose `revision`
            resolves to the final checkpoint are considered; a run
            contributes to the left axis if `"acc" in metrics` and its
            dataset is in `_HEADLINE_ACC_DATASETS`, or to the right axis if
            `"bpb" in metrics` and its dataset is a PPL corpus. Everything
            else (unknown models, intermediate checkpoints, other datasets)
            is silently skipped, not an error.
        out_path: Destination PNG path.

    Returns:
        `out_path`, coerced to a `Path`.

    Side Effects:
        Writes a PNG file to `out_path`, overwriting any existing file.
    """
    out_path = Path(out_path)

    acc_series: dict[str, list[tuple[float, float]]] = {}
    bpb_series: dict[str, list[tuple[float, float]]] = {}
    for r in runs:
        if r.status != "done":
            continue
        n_params = PYTHIA_PARAM_COUNTS.get(r.model_id)
        if n_params is None:
            continue
        if _revision_to_step(r.revision) != _FINAL_STEP:
            continue
        log_params = math.log10(n_params)

        if r.dataset in _HEADLINE_ACC_DATASETS and "acc" in r.metrics:
            acc_series.setdefault(r.dataset, []).append((log_params, r.metrics["acc"]))
        elif "bpb" in r.metrics:
            bpb_series.setdefault(r.dataset, []).append((log_params, r.metrics["bpb"]))

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax2 = ax1.twinx()
    seen_colors: dict[str, str] = {}

    acc_lines = []
    for dataset in sorted(acc_series):
        points = sorted(acc_series[dataset])
        xs, ys = zip(*points)
        color = _color_for_dataset(dataset, seen_colors)
        (line,) = ax1.plot(xs, ys, marker="o", label=dataset, color=color)
        acc_lines.append(line)
        if dataset in _CHANCE_RATE:
            ax1.axhline(_CHANCE_RATE[dataset], linestyle="--", linewidth=1, color=color, alpha=0.6)

    bpb_lines = []
    for dataset in sorted(bpb_series):
        points = sorted(bpb_series[dataset])
        xs, ys = zip(*points)
        color = _color_for_dataset(dataset, seen_colors)
        (line,) = ax2.plot(xs, ys, marker="s", linestyle="--", label=f"{dataset} (bpb)", color=color)
        bpb_lines.append(line)

    ax1.set_xlabel("log10(parameters)")
    ax1.set_ylabel("accuracy")
    ax1.set_ylim(0, 1)
    ax2.set_ylabel("bits-per-byte (bpb)")
    if bpb_series:
        ax2.invert_yaxis()  # smaller bpb = better; inverted so "up = better" matches the accuracy axis

    ax1.set_title("Headline: accuracy + bits-per-byte across the scaling ladder")
    handles = acc_lines + bpb_lines
    if handles:
        ax1.legend(handles=handles, labels=[h.get_label() for h in handles], loc="best", fontsize="small")

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
        metric: Which metric key to plot ("acc", "acc_norm", or "bpb" — the
            y-axis is capped to [0, 1] only for the two accuracy-style
            metrics, since bpb has no fixed range).

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
    if metric in ("acc", "acc_norm"):
        ax.set_ylim(0, 1)  # bpb has no fixed [0, 1] range, so the cap only applies to accuracy-style metrics
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
