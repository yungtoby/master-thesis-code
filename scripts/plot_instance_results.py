"""Plot matched-budget mean regret separately for every test instance.

Save in scripts/ and run from the repository root:
    py scripts/plot_instance_results_NEW.py

To use another matched-budget results folder:
    py scripts/plot_instance_results_NEW.py --results-dir results/final_testing/matched_budgets

Input:  <results-dir>/instance_specific/{low,medium,high}.csv
Output: <results-dir>/figures/regret_by_instance_{low,medium,high}.{pdf,png}

Each figure has one panel per instance. Methods are ranked by unrounded mean
final simple regret, with the best at the top. Blue indicates learned policies;
grey indicates baselines. All panels and budgets share the same x-axis scale.
The plots show the saved means, as in the reference figure. They do not refit
models, average means again, or change any CSVs. Dependencies: Matplotlib.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


RESULTS_DIR = Path("results/final_testing_NEW/matched_budgets")
BUDGETS = ("low", "medium", "high")
REGRET_COLUMN = "mean_final_simple_regret"
LABELS = {
    "no_cost": "No cost",
    "cost_mean": "Cost mean",
    "cost_uncertainty": "Cost uncertainty",
    "random": "Random",
    "ei": "EI",
    "eipu": "EI per unit cost",
    "ei_cool": "EI-cool",
}
LEARNED = {"no_cost", "cost_mean", "cost_uncertainty"}
LEARNED_COLOR = "#32698A"
BASELINE_COLOR = "#686C71"


def read_results(results_dir, budget):
    """Read one already-aggregated mean per method and instance."""
    path = results_dir / "instance_specific" / f"{budget}.csv"
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or [])
        # Older exports used instance_id; the NEW testing script uses instance.
        instance_column = "instance" if "instance" in columns else "instance_id"
        required = {instance_column, "method", "training_budget", "test_budget", REGRET_COLUMN}
        if not required.issubset(columns):
            raise ValueError(f"{path}: missing columns {sorted(required - columns)}. Use an instance_specific CSV.")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: no instance results found.")

    grouped = {}
    seen = set()
    for row in rows:
        instance = row[instance_column].strip()
        method = row["method"].strip()
        if not instance or method not in LABELS:
            raise ValueError(f"{path}: missing instance ID or unknown method {method!r}.")
        if row["test_budget"].strip() != budget:
            raise ValueError(f"{path}: a row has the wrong test budget.")
        # Baselines have no training budget: NEW exports use 'none', older
        # exports leave it blank. Check training budgets only for learned methods.
        if method in LEARNED and row["training_budget"].strip() != budget:
            raise ValueError(f"{path}: this plot requires matched training/test budgets, not budget-transfer results.")
        key = (instance, method)
        if key in seen:
            raise ValueError(f"{path}: duplicate result for instance {instance}, method {method}.")
        seen.add(key)
        mean = float(row[REGRET_COLUMN])
        if not math.isfinite(mean) or mean < 0:
            raise ValueError(f"{path}: invalid regret for instance {instance}, method {method}.")
        grouped.setdefault(instance, []).append({"method": method, "mean": mean})

    for instance, results in grouped.items():
        missing = set(LABELS) - {row["method"] for row in results}
        if missing:
            raise ValueError(f"{path}: instance {instance} is missing methods {sorted(missing)}.")
        # Sorting happens before values are rounded for the figure labels.
        results.sort(key=lambda row: (row["mean"], row["method"]))
    return {instance: grouped[instance] for instance in sorted(grouped)}


def format_regret(value):
    # Avoid displaying a tiny positive regret as an exact zero.
    return f"{value:.2g}" if 0 < value < 0.0001 else f"{value:.4f}"


def plot_budget(budget, results, x_limit, output_dir):
    columns = min(2, len(results))
    rows = math.ceil(len(results) / columns)
    fig, axes = plt.subplots(
        rows, columns,
        figsize=(12 if columns == 2 else 6.5, 2.9 * rows + 0.8),
        sharex=True, squeeze=False, layout="constrained",
    )

    for index, (instance, scores) in enumerate(results.items()):
        ax = axes.flat[index]
        positions = list(range(len(scores)))
        values = [row["mean"] for row in scores]
        colors = [LEARNED_COLOR if row["method"] in LEARNED else BASELINE_COLOR for row in scores]
        bars = ax.barh(positions, values, color=colors, height=0.68)
        ax.set_yticks(positions, [LABELS[row["method"]] for row in scores], fontsize=10)
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[format_regret(value) for value in values], padding=3, fontsize=8)
        ax.set(xlim=(0, x_limit), title=f"Instance {instance}")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DDDFE1", linewidth=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9)
        # With an odd instance count, show x ticks on the lowest real panel
        # in each column; the unused final panel is hidden below.
        if index + columns >= len(results):
            ax.tick_params(axis="x", labelbottom=True)

    for index in range(len(results), axes.size):
        axes.flat[index].set_visible(False)
    fig.suptitle(f"Matched {budget} budget", fontsize=16, fontweight="bold")
    fig.supxlabel("Mean final simple regret (lower is better)", fontsize=12)
    for extension in ("pdf", "png"):
        path = output_dir / f"regret_by_instance_{budget}.{extension}"
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Matched-budget folder containing the instance_specific subfolder.")
    args = parser.parse_args()
    # Read all inputs before writing figures so incomplete exports fail early.
    results = {budget: read_results(args.results_dir, budget) for budget in BUDGETS}
    instances = set(results["low"])
    if any(set(by_instance) != instances for by_instance in results.values()):
        raise ValueError("The instance IDs differ between the low, medium, and high CSVs.")
    largest = max(row["mean"] for by_instance in results.values()
                  for scores in by_instance.values() for row in scores)
    # Leave room for numeric labels and retain a valid axis if every mean is zero.
    x_limit = 1.25 * largest if largest > 0 else 0.01
    output_dir = args.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for budget in BUDGETS:
        plot_budget(budget, results[budget], x_limit, output_dir)
        print(f"Saved {budget} figure: {len(instances)} instances, PDF and PNG in {output_dir}")


if __name__ == "__main__":
    main()