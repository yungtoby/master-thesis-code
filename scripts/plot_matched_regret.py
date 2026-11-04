"""Rank the seven methods separately for each held-out instance and test budget.

Run from the repository root: python scripts/plot_matched_regret.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATCHED = Path("results/final_testing/matched_budgets")
BUDGETS = ("low", "medium", "high")
REGRET = "mean_final_simple_regret"
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


def read_means(budget):
    path = MATCHED / "instance_specific" / f"{budget}.csv"
    rows = pd.read_csv(path, dtype={"instance": str, "instance_id": str})
    if "instance_id" not in rows and "instance" in rows:
        rows = rows.rename(columns={"instance": "instance_id"})
    if "instance_id" not in rows:
        raise ValueError(f"{path} has no instance ID. Use the instance_specific CSV.")
    if not rows["test_budget"].eq(budget).all():
        raise ValueError(f"{path} contains rows for a different test budget.")
    trained = rows["training_budget"].dropna()
    if not trained.eq(budget).all():
        raise ValueError(f"{path} contains a training budget different from the test budget.")
    if rows[["method", "instance_id"]].isna().any().any():
        raise ValueError(f"{path} contains a missing method or instance ID.")

    # One stored mean per instance and method. pivot also catches duplicate rows.
    rows[REGRET] = pd.to_numeric(rows[REGRET])
    means = rows.pivot(index="instance_id", columns="method", values=REGRET)
    if len(means) != 6 or set(means.columns) != set(LABELS):
        raise ValueError(f"{path} should contain six instances and all seven matched methods.")
    if not np.isfinite(means.to_numpy()).all() or (means < 0).any().any():
        raise ValueError(f"{path} contains a missing, non-finite, or negative regret.")
    return means


def instance_table(budget, instance, ordered):
    rows = [f"{LABELS[m]} & {value:.4f} " + r"\\" for m, value in ordered.items()]
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Mean final simple regret on instance {instance} at the {budget} budget.}}",
        rf"\label{{tab:matched-{budget}-{instance}}}",
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"Method & Mean final simple regret \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def plot_budget(budget, means, x_limit, output_dir):
    fig, axes = plt.subplots(3, 2, figsize=(10.6, 8.6), sharex=True, layout="constrained")
    tables = []
    for ax, (instance, scores) in zip(axes.flat, means.iterrows()):
        # Lower regret is better. Sort before rounding and put the first bar at the top.
        ordered = scores.sort_values(kind="stable")
        tables.append(instance_table(budget, instance, ordered))
        colors = ["#32698A" if method in LEARNED else "#686C71" for method in ordered.index]
        bars = ax.barh([LABELS[m] for m in ordered.index], ordered.values, color=colors, height=0.68)
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set(xlim=(0, x_limit), title=f"Instance {instance}")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DDDFE1", linewidth=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(f"Matched {budget} budget", fontsize=15, fontweight="bold")
    fig.supxlabel("Mean final simple regret (lower is better)")
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"regret_by_instance_{budget}.{extension}", dpi=200)
    plt.close(fig)
    (output_dir / f"regret_by_instance_{budget}.tex").write_text("\n\n".join(tables) + "\n", encoding="utf-8")


def main():
    means = {budget: read_means(budget) for budget in BUDGETS}
    if any(not table.index.equals(means["low"].index) for table in means.values()):
        raise ValueError("The instance IDs differ between the three budget files.")
    # A common horizontal scale makes bar lengths comparable across all three figures.
    largest = max(table.to_numpy().max() for table in means.values())
    x_limit = 1.25 * largest if largest > 0 else 0.01
    output_dir = MATCHED / "figures"
    output_dir.mkdir(exist_ok=True)
    for budget, table in means.items():
        plot_budget(budget, table, x_limit, output_dir)
    print(f"Saved the ranked figures (PDF/PNG) and per-instance LaTeX tables to {output_dir}")


if __name__ == "__main__":
    main()
