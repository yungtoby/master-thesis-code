"""Plot the saved Phase 2 objective-GP and cost-GP selection results.

Save this file in scripts/ and run it from the repository root:
    py scripts/plot_gp_selection.py
    py scripts/plot_gp_selection.py --budget low

Each figure shows all completed settings in mean-regret order. The highlighted
setting is the saved choice across all budgets. All three metrics are minimized.
The script reads existing results and writes PDF/PNG figures under
results/gp_selection/diagnostics/selection/.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = {
    "objective": [
        ("mean_regret", "Mean final simple regret"),
        ("mean_objective_rmse", "Mean objective RMSE"),
        ("mean_objective_nlpd", "Mean objective NLPD"),
    ],
    "cost": [
        ("mean_regret", "Mean final simple regret"),
        ("mean_cost_rmse_log1p", "Mean cost RMSE\nlog1p(cost)"),
        ("mean_cost_nlpd_log1p", "Mean cost NLPD\nlog1p(cost)"),
    ],
}
SELECTED_COLOR = "#C65D13"
OTHER_COLOR = "#32698A"


def read_results(results_dir, stage, budget):
    path = results_dir / stage / "summary.csv"
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = [row for row in csv.DictReader(source) if row["budget_level"] == budget]
    selected = json.loads((results_dir / stage / "selected.json").read_text())["setting"]
    failed = [row["setting"] for row in rows if row["status"] != "complete"]
    if failed:
        print(f"{stage}: excluding incomplete settings: {', '.join(failed)}")
    rows = [row for row in rows if row["status"] == "complete"]
    if not rows or len({row["setting"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: expected one complete row per setting for {budget}.")
    for row in rows:
        for key, _ in METRICS[stage]:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"{path}: non-finite {key} for {row['setting']}.")
    if selected not in {row["setting"] for row in rows}:
        raise ValueError(f"{path}: the saved selection has no complete {budget} row.")
    return sorted(rows, key=lambda row: (row["mean_regret"], row["setting"])), selected


def plot_stage(stage, budget, rows, selected, output_dir):
    # Keep the same setting order in all three panels so a row is easy to follow.
    height = max(5.5, 0.32 * len(rows) + 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, height), sharey=True)
    positions = list(range(len(rows)))
    colors = [SELECTED_COLOR if row["setting"] == selected else OTHER_COLOR for row in rows]
    for ax, (key, title) in zip(axes, METRICS[stage]):
        values = [row[key] for row in rows]
        low, high = min(0.0, min(values)), max(0.0, max(values))
        span = max(high - low, 1e-6)
        for position in positions[::2]:
            ax.axhspan(position - 0.5, position + 0.5, color="#F4F5F6", zorder=0)
        ax.scatter(values, positions, c=colors, s=34, zorder=3)
        for position, value in zip(positions, values):
            ax.text(value + 0.025 * span, position, f"{value:.4g}", va="center", fontsize=8)
        # NLPD may be negative. Preserve its sign and give every value room.
        ax.set_xlim(low - 0.04 * span, high + 0.24 * span)
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_xlabel("Lower is better", fontsize=9)
        ax.grid(axis="x", color="#DFE2E5", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9)
    axes[0].set_yticks(positions, [row["setting"] for row in rows], fontsize=9)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    for label, row in zip(axes[0].get_yticklabels(), rows):
        if row["setting"] == selected:
            label.set_color(SELECTED_COLOR)
            label.set_fontweight("bold")

    method = "EI" if stage == "objective" else "EI per unit cost"
    budget_label = "all budgets" if budget == "ALL" else f"{budget} budget"
    fig.suptitle(f"{stage.capitalize()} GP selection | {method} | {budget_label}", fontsize=15)
    fig.text(0.5, 0.035, f"Orange marks the saved selection: {selected}", ha="center", fontsize=10)
    fig.text(
        0.5, 0.012,
        "RMSE and NLPD summarize predictions made before each query along the evaluation trajectory.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.30, right=0.985, top=0.91, bottom=0.105, wspace=0.20)
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"{stage}_{budget.lower()}.{extension}", dpi=200,
                    bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", choices=("all", "low", "medium", "high"), default="all")
    parser.add_argument("--results-dir", type=Path, default=Path("results/gp_selection"))
    args = parser.parse_args()
    budget = "ALL" if args.budget == "all" else args.budget
    results = {stage: read_results(args.results_dir, stage, budget) for stage in METRICS}
    output_dir = args.results_dir / "diagnostics" / "selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stage, (rows, selected) in results.items():
        plot_stage(stage, budget, rows, selected, output_dir)
        print(f"Saved {stage} selection figures for {len(rows)} settings to {output_dir}")


if __name__ == "__main__":
    main()