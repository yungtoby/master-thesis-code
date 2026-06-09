from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from yahpo_gym import BenchmarkSet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", required=True)
    parser.add_argument("--samples-per-instance", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=51)
    parser.add_argument("--output", type=str, default="results/yahpo_cost_samples.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []

    for instance in args.instances:
        print(f"Sampling instance {instance}")

        b = BenchmarkSet("lcbench", instance=str(instance))
        cs = b.get_opt_space(drop_fidelity_params=True)

        configs = cs.sample_configuration(args.samples_per_instance)
        if args.samples_per_instance == 1:
            configs = [configs]

        dicts = []
        for cfg in configs:
            d = cfg.get_dictionary()
            d["epoch"] = args.epoch
            d["OpenML_task_id"] = str(instance)
            dicts.append(d)

        outs = b.objective_function(dicts)
        if isinstance(outs, dict):
            outs = [outs]

        for cfg, out in zip(dicts, outs):
            row = {
                "instance": str(instance),
                "time": float(out["time"]),
                "val_accuracy": float(out["val_accuracy"]),
                "val_cross_entropy": float(out["val_cross_entropy"]),
                "batch_size": cfg["batch_size"],
                "learning_rate": cfg["learning_rate"],
                "max_dropout": cfg["max_dropout"],
                "max_units": cfg["max_units"],
                "momentum": cfg["momentum"],
                "num_layers": cfg["num_layers"],
                "weight_decay": cfg["weight_decay"],
                "epoch": cfg["epoch"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} rows to {output_path}")
    print("\nOverall cost summary:")
    print(df["time"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))

    print("\nPer-instance cost summary:")
    summary = df.groupby("instance")["time"].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    )
    print(summary)


if __name__ == "__main__":
    main()