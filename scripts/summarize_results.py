from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Evaluation CSV files.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Method names corresponding to input files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/summary.csv",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1e-3,
    )
    return parser.parse_args()


def summarize_one(path: str, name: str, success_threshold: float) -> dict:
    df = pd.read_csv(path)

    if "regret" not in df.columns:
        raise ValueError(f"{path} does not contain a regret column.")

    regret = df["regret"].dropna()

    row = {
        "method": name,
        "file": path,
        "n_episodes": len(df),
        "mean_regret": regret.mean(),
        "median_regret": regret.median(),
        "std_regret": regret.std(),
        "min_regret": regret.min(),
        "max_regret": regret.max(),
        "success_rate": (regret <= success_threshold).mean(),
    }

    optional_means = [
        "budget_used",
        "budget_overshoot",
        "remaining_budget",
        "episode_length",
        "best_value",
        "best_oracle_value",
        "ground_truth",
    ]

    for col in optional_means:
        if col in df.columns:
            row[f"mean_{col}"] = df[col].mean()
            row[f"median_{col}"] = df[col].median()

    return row


def main():
    args = parse_args()

    if len(args.inputs) != len(args.names):
        raise ValueError("--inputs and --names must have the same length.")

    rows = [
        summarize_one(path, name, args.success_threshold)
        for path, name in zip(args.inputs, args.names)
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()