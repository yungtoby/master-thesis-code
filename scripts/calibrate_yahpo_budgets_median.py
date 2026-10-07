from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CALIBRATION_RULE = "pooled_median_cumulative_cost_at_target"
DEFAULT_CAPS = [30, 40, 50, 60, 64]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--target-total-evals", type=int, nargs="+",
                      help="Reference evaluation counts including warm start, e.g. 10 20 30; fit the median cumulative acquisition cost at each count.")
    mode.add_argument("--budget-file", type=Path,
                      help="Check previously fitted budgets without recalibrating them.")
    parser.add_argument("--caps", type=int, nargs="+", default=None,
                        help="Acquisition caps to compare; defaults to 30 40 50 60 64. Each must be <= 64 and <= env.n_candidates.")
    parser.add_argument("--episodes-per-instance", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.seed < 0 or args.episodes_per_instance < 1:
        parser.error("Use a non-negative seed and a positive episode count.")
    if args.caps is not None and any(c < 1 for c in args.caps):
        parser.error("Acquisition caps must be positive.")
    if args.caps is not None:
        args.caps = sorted(set(args.caps))
    return args


def draw_random_actions(rng, n_episodes, n_candidates, path_length):
    """Draw by step so extending the cap preserves every episode's prefix."""
    return rng.integers(0, n_candidates, size=(path_length, n_episodes)).T


def fit_budgets(scenarios, targets, n_init):
    """Choose the pooled median cumulative cost after exactly k acquisitions.

    k = target total evaluations minus n_init. Each trajectory contributes one
    cumulative-cost total at k; pooling is across trajectories, not across
    per-instance medians. The stopping rule can make the realized median episode
    length differ from k, so report that length separately rather than adjusting
    the requested budget definition.
    """
    paths = []
    for _, costs, actions, _ in scenarios:
        charged = np.take_along_axis(costs, actions, axis=1)
        paths.append(np.cumsum(charged, axis=1, dtype=np.float64))
    cumulative = np.concatenate(paths, axis=0)
    labels = ["low", "medium", "high"] if len(targets) == 3 else [f"total_{t}" for t in targets]
    fitted = []
    for label, total in zip(labels, targets):
        k = total - n_init
        at_target = float(np.median(cumulative[:, k - 1]))
        if not np.isfinite(at_target) or at_target <= 0:
            raise ValueError(f"Median cumulative cost must be finite and positive for target {total}.")
        fitted.append({
            "level": label, "target_total_evaluations": total,
            "target_acquisitions": k, "budget": at_target,
            "median_cost_at_target": at_target,
        })
    return fitted


def simulate_random(costs, actions, initial_indices, budget, cap):
    """Mirror BO_env's float32 subtraction, crossing evaluation, and length cap."""
    n_episodes = len(costs)
    rows = np.arange(n_episodes)
    charged = costs[rows[:, None], actions[:, :cap]]
    remaining = np.full(n_episodes, budget, dtype=np.float32)
    lengths = np.zeros(n_episodes, dtype=np.int64)

    for step in range(cap):
        active = remaining > 0
        if not active.any():
            break
        remaining[active] -= charged[active, step]
        lengths[active] += 1

    unique_points = np.array([
        np.unique(np.concatenate((initial_indices[e], actions[e, :lengths[e]]))).size
        for e in range(n_episodes)
    ])
    overshoot = np.maximum(-remaining, 0.0)
    return {
        "episode_length": lengths,
        "total_evaluations": lengths + initial_indices.shape[1],
        "budget_used": np.float32(budget) - remaining,
        "remaining_budget": remaining,
        "budget_overshoot": overshoot,
        "overshoot_fraction": overshoot / budget,
        "budget_hit": remaining <= 0,
        # Simultaneous budget/cap termination is a budget hit, not cap-only.
        "cap_only": (lengths == cap) & (remaining > 0),
        "at_cap": lengths == cap,
        "one_acquisition": lengths == 1,
        "unique_points": unique_points,
        # Initial observations are counted here, but never subtracted above.
        "initialization_cost": costs[rows[:, None], initial_indices].sum(axis=1, dtype=np.float64),
    }


def summarize(data, costs, instance, target, cap, seed, config_path):
    row = {
        "config": str(config_path), "seed": seed, "instance": instance,
        "level": target["level"], "budget": target["budget"], "max_acquisitions": cap,
        "target_total_evaluations": target["target_total_evaluations"],
        "target_acquisitions": target["target_acquisitions"],
        "n_episodes": len(data["episode_length"]),
        "n_cost_samples": costs.size,
        "cost_median": float(np.median(costs)),
        "cost_p90": float(np.quantile(costs, 0.9)),
    }
    for key in ["episode_length", "total_evaluations", "unique_points",
                "budget_used", "remaining_budget", "budget_overshoot"]:
        row[f"mean_{key}"] = float(np.mean(data[key]))
        row[f"median_{key}"] = float(np.median(data[key]))
    row["median_target_error"] = row["median_total_evaluations"] - target["target_total_evaluations"]
    row["max_total_evaluations"] = int(np.max(data["total_evaluations"]))
    row["max_unique_points"] = int(np.max(data["unique_points"]))
    row["p90_budget_overshoot"] = float(np.quantile(data["budget_overshoot"], 0.9))
    row["mean_overshoot_fraction"] = float(np.mean(data["overshoot_fraction"]))
    row["median_overshoot_fraction"] = float(np.median(data["overshoot_fraction"]))
    for key in ["budget_hit", "cap_only", "at_cap", "one_acquisition"]:
        row[f"{key}_rate"] = float(np.mean(data[key]))
    row["mean_initialization_cost"] = float(np.mean(data["initialization_cost"]))
    return row


def main():
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as f:
        cfg = json.load(f)
    problem = cfg["problem_family"]
    env = cfg["env"]
    if problem["type"] != "yahpo_lcbench":
        raise ValueError("This calibration requires a YAHPO LCBench config.")
    if env["mask_visited_actions"]:
        raise ValueError("This calibration matches mask_visited_actions=false only.")
    instances = [str(x) for x in problem["instances"]]
    if not instances or len(set(instances)) != len(instances):
        raise ValueError("The config must contain distinct, nonempty instance IDs.")
    n_candidates, n_init = env["n_candidates"], env["n_init"]
    if not 1 <= n_init <= n_candidates:
        raise ValueError("Require 1 <= n_init <= n_candidates.")
    half_limit = n_candidates // 2
    if half_limit <= n_init:
        raise ValueError("Warm start leaves no acquisition below the median target limit.")

    settings = {
        "n_candidates": n_candidates, "n_init": n_init,
        "epoch": int(problem.get("epoch", 51)),
        "cost_key": problem.get("cost_key", "time"),
        "mask_visited_actions": False,
    }
    if args.budget_file is None:
        targets = sorted(set(args.target_total_evals))
        if any(not n_init < t <= half_limit for t in targets):
            raise ValueError(f"Total-evaluation targets must be in [{n_init + 1}, {half_limit}].")
        fitted = None
    else:
        with args.budget_file.open(encoding="utf-8") as f:
            spec = json.load(f)
        if spec.get("calibration_rule") != CALIBRATION_RULE:
            raise ValueError(
                "This budget file uses a previous calibration rule. Refit with "
                "--target-total-evals 10 20 30 and save a new budget file."
            )
        if spec["settings"] != settings:
            raise ValueError("Candidate count, warm start, fidelity, cost key, or masking differs from the budget file.")
        fitted = spec["budgets"]
        targets = [int(t["target_total_evaluations"]) for t in fitted]
        if not targets or any(not n_init < t <= half_limit for t in targets):
            raise ValueError("Budget-file targets violate the half-candidate median target limit.")
        if any(not np.isfinite(t["budget"]) or t["budget"] <= 0 for t in fitted):
            raise ValueError("Budget-file values must be finite and positive.")
    if args.caps is None:
        args.caps = DEFAULT_CAPS.copy()
    max_allowed_cap = min(64, n_candidates)
    if any(cap > max_allowed_cap for cap in args.caps):
        raise ValueError(
            f"Acquisition caps must not exceed {max_allowed_cap} "
            "(the smaller of 64 and n_candidates)."
        )
    path_length = max(max(args.caps), max(targets) - n_init)

    # Keep imports here so --help and the pure NumPy stopping calculation do
    # not require a local PyTorch/YAHPO installation.
    import torch
    from bo.problems.yahpo_lcbench import YAHPOLCBenchProblemFamily

    n_episodes = args.episodes_per_instance
    scenarios = []
    print("calibrate_yahpo_budgets_new.py: direct pooled median cumulative-cost rule.")
    print(f"Sampling {n_episodes} grids of {n_candidates} candidates per instance.")
    print("Costs exclude initialization; final crossing is included; actions may repeat.")
    print(f"Budget reference totals: {targets}; acquisition caps: {args.caps}.")
    print("Budgets use the pooled median cumulative cost at each reference acquisition count.")
    print(f"Reference totals are at most half the candidate count ({half_limit}).")
    print(f"Individual episodes may reach acquisition cap + {n_init} warm-start evaluations.")

    for instance in instances:
        print(f"  Instance {instance}", flush=True)
        # Stable per-instance streams; unrelated instances and budget loops
        # cannot change these candidates, actions, or initial designs.
        streams = np.random.SeedSequence([args.seed, int(instance)]).spawn(3)
        cache_seed = int(streams[0].generate_state(1)[0])
        action_rng = np.random.default_rng(streams[1])
        init_rng = np.random.default_rng(streams[2])
        family_kwargs = {k: v for k, v in problem.items() if k not in {"type", "instances"}}
        family = YAHPOLCBenchProblemFamily(
            device=torch.device("cpu"), dtype=torch.float32,
            instances=[instance], **family_kwargs,
        )
        _, _, cost_tensor, _ = family.build_candidate_cache(
            B=n_episodes, n_candidates=n_candidates, seed=cache_seed,
        )
        costs = cost_tensor.cpu().numpy()
        if costs.shape != (n_episodes, n_candidates):
            raise RuntimeError(f"Unexpected cost shape: {costs.shape}")
        if not np.isfinite(costs).all() or (costs < 0).any():
            raise RuntimeError("Costs must be finite and non-negative.")
        actions = draw_random_actions(action_rng, n_episodes, n_candidates, path_length)
        # Uniform distinct initialization per episode. BO_env shares an index
        # draw across lanes; the per-episode marginal distribution is the same.
        initial_indices = np.argsort(init_rng.random((n_episodes, n_candidates)), axis=1)[:, :n_init]
        scenarios.append((instance, costs, actions, initial_indices))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fitted is None:
        fitted = fit_budgets(scenarios, targets, n_init)
        spec = {
            "calibration_config": str(args.config), "calibration_seed": args.seed,
            "episodes_per_instance": n_episodes, "settings": settings,
            "calibration_rule": CALIBRATION_RULE,
            "random_action_layout": "step_major_v1",
            "caps": args.caps, "budgets": fitted,
        }
        budget_path = output.with_suffix(".budgets.json")
        budget_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved fitted budgets to {budget_path}")
    else:
        print(f"\nChecking fixed budgets from {args.budget_file}; no refitting.")
    print(pd.DataFrame(fitted)[[
        "level", "target_total_evaluations", "target_acquisitions", "budget",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    per_instance = []
    pooled = {(target["level"], c): [] for target in fitted for c in args.caps}
    for instance, costs, actions, initial_indices in scenarios:
        for target in fitted:
            for cap in args.caps:
                data = simulate_random(costs, actions, initial_indices, target["budget"], cap)
                pooled[target["level"], cap].append(data)
                per_instance.append(summarize(
                    data, costs, instance, target, cap, args.seed, args.config,
                ))

    all_costs = np.concatenate([scenario[1] for scenario in scenarios], axis=0)
    overall = []
    for target in fitted:
        for cap in args.caps:
            parts = pooled[target["level"], cap]
            data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
            overall.append(summarize(data, all_costs, "ALL", target, cap, args.seed, args.config))
    # ALL rows use balanced episodes, hence equal instance weight for means.
    # Medians and p90s are computed from pooled episodes, not averaged quantiles.
    pd.DataFrame(overall + per_instance).to_csv(output, index=False)
    print("\nFresh candidate-cost quantiles:")
    print(pd.Series(all_costs.ravel()).quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    print("\nBalanced Random calibration (rates are fractions):")
    columns = [
        "level", "budget", "max_acquisitions", "n_episodes", "target_total_evaluations",
        "median_episode_length", "median_total_evaluations", "median_unique_points",
        "cap_only_rate", "one_acquisition_rate", "median_budget_overshoot",
        "p90_budget_overshoot",
    ]
    print(pd.DataFrame(overall)[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved overall and per-instance rows to {output}")


if __name__ == "__main__":
    main()