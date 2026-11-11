"""Plot the saved NEW objective-GP and cost-GP selection results.

Run from the repository root:
    py scripts/plot_gp_selection_NEW.py
    py scripts/plot_gp_selection_NEW.py --budget low

Each figure shows completed settings ordered by mean NLPD.
Orange highlights the saved selection across all budgets, even when
displaying results for one particular budget.

NLPD determines the selection; RMSE is a diagnostic. Both are minimized.
PDF/PNG figures are saved under:
    results/gp_selection_NEW/diagnostics/selection/
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("mean_nlpd", "Mean NLPD\nSelection criterion"),
    ("mean_rmse", "Mean RMSE\nDiagnostic"),
]

TARGET_LABELS = {
    "objective": "observed accuracy",
    "cost": "log1p(cost)",
}

SELECTED_COLOR = "#C65D13"
OTHER_COLOR = "#32698A"


def read_results(results_dir, stage, budget):
    """Read one budget slice without recalculating the saved scores."""
    summary_path = results_dir / stage / "summary.csv"
    selection_path = results_dir / stage / "selected.json"

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("method") != "random_without_replacement":
        raise ValueError(
            f"{selection_path}: expected the NEW random-acquisition selection."
        )
    selected = selection["setting"]

    with summary_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {
            "setting", "status", "budget_level", "mean_nlpd", "mean_rmse"
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{summary_path}: missing columns from the NEW selection format."
            )
        rows = [
            row for row in reader
            if row["budget_level"] == budget
        ]

    failed = [
        row["setting"] for row in rows
        if row["status"] != "complete"
    ]
    if failed:
        print(f"{stage}: excluding incomplete settings: {', '.join(failed)}")

    rows = [row for row in rows if row["status"] == "complete"]
    if not rows or len({row["setting"] for row in rows}) != len(rows):
        raise ValueError(
            f"{summary_path}: expected one complete row per setting for {budget}."
        )

    for row in rows:
        for key, _ in METRICS:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(
                    f"{summary_path}: non-finite {key} for {row['setting']}."
                )

    if selected not in {row["setting"] for row in rows}:
        raise ValueError(
            f"{summary_path}: the saved selection has no complete {budget} row."
        )

    # Use unrounded NLPD values, matching the selection procedure.
    rows.sort(key=lambda row: (row["mean_nlpd"], row["setting"]))
    return rows, selected


def plot_stage(stage, budget, rows, selected, output_dir):
    """Keep identical setting order in the NLPD and RMSE panels."""
    height = max(5.5, 0.32 * len(rows) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(13, height), sharey=True)

    positions = list(range(len(rows)))
    colors = [
        SELECTED_COLOR if row["setting"] == selected else OTHER_COLOR
        for row in rows
    ]

    for ax, (key, title) in zip(axes, METRICS):
        values = [row[key] for row in rows]
        low = min(0.0, min(values))
        high = max(0.0, max(values))
        span = max(high - low, 1e-6)

        for position in positions[::2]:
            ax.axhspan(
                position - 0.5, position + 0.5,
                color="#F4F5F6", zorder=0,
            )

        ax.scatter(values, positions, c=colors, s=34, zorder=3)

        for position, value in zip(positions, values):
            ax.text(
                value + 0.025 * span,
                position,
                f"{value:.4g}",
                va="center",
                fontsize=8,
            )

        # NLPD can legitimately be negative: retain its sign and full range.
        ax.set_xlim(low - 0.04 * span, high + 0.26 * span)
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_xlabel("Lower is better", fontsize=9)
        ax.grid(axis="x", color="#DFE2E5", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9)

    axes[0].set_yticks(
        positions,
        [row["setting"] for row in rows],
        fontsize=9,
    )
    axes[0].set_ylim(len(rows) - 0.5, -0.5)

    for label, row in zip(axes[0].get_yticklabels(), rows):
        if row["setting"] == selected:
            label.set_color(SELECTED_COLOR)
            label.set_fontweight("bold")

    budget_label = "all budgets" if budget == "ALL" else f"{budget} budget"
    target_label = TARGET_LABELS[stage]

    fig.suptitle(
        f"{stage.capitalize()} GP selection | {budget_label}\n"
        f"Random acquisitions without replacement | Target: {target_label}",
        fontsize=13,
    )
    fig.text(
        0.5, 0.042,
        f"Orange: saved overall winner — {selected}",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5, 0.019,
        "Saved means of per-episode scores; each prediction is scored "
        "before adding the acquired observation.",
        ha="center",
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.34, right=0.985,
        top=0.88, bottom=0.115,
        wspace=0.28,
    )

    for extension in ("pdf", "png"):
        path = output_dir / f"{stage}_{budget.lower()}.{extension}"
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.18)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--budget",
        choices=("all", "low", "medium", "high"),
        default="all",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/gp_selection_NEW"),
    )
    args = parser.parse_args()

    budget = "ALL" if args.budget == "all" else args.budget

    # Read both stages before creating any figures.
    results = {
        stage: read_results(args.results_dir, stage, budget)
        for stage in TARGET_LABELS
    }

    output_dir = args.results_dir / "diagnostics" / "selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage, (rows, selected) in results.items():
        plot_stage(stage, budget, rows, selected, output_dir)
        print(
            f"Saved {stage} selection figures for "
            f"{len(rows)} settings to {output_dir}"
        )


if __name__ == "__main__":
    main()