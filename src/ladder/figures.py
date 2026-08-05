"""Report figures, generated from the DB alone (architecture.md §10).

Read-only with respect to the DB — never writes runs/results, only reads
them via `storage.py`'s query functions and renders matplotlib PNGs.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never an interactive window
import matplotlib.pyplot as plt

from ladder.records import RunRecord


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
