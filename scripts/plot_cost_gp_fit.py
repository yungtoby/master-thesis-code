"""Inspect a cost GP before and after a low-budget validation episode.

Save this file in scripts/ and run from the repository root:
    py scripts/plot_cost_gp_fit.py
    py scripts/plot_cost_gp_fit.py rbf_matern52_mul__linear__lr0.01
    py scripts/plot_cost_gp_fit.py rbf__constant__lr0.1 --episode 7
    py scripts/plot_cost_gp_fit.py all

The default is the selected cost GP and episode 0 on every validation instance.
'all' runs every setting in the saved selection grid. Each episode is replayed
in its original full batch, using EI per unit cost, the saved objective GP,
cached warm starts/noise, and the frozen low budget. The final snapshot includes
the last acquisition, even when that acquisition crosses the budget.

Each candidate is a seven-dimensional configuration. The figure compares its
true and predicted log1p(cost), with a diagonal for perfect prediction and
vertical predictive +/-1 SD bars. Snapshot RMSE/NLPD cover all candidates after
fitting, including observed candidates. They differ from the query-by-query
predictive metrics in the selection summaries.

PDF/PNG figures go under results/gp_selection/diagnostics/cost_fit/.
Dependencies are the same as select_gp.py, plus Matplotlib.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make the repository modules available when called as 'py scripts/<file>.py'.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_saved_selection(results_dir):
    selected = json.loads((results_dir / "cost" / "selected.json").read_text())
    if selected["stage"] != "cost" or selected["cost_model"]["target_transform"] != "log1p":
        raise ValueError("Expected the saved log1p-cost GP selection.")
    bank_path = Path(selected["phase_config"]["episode_bank"])
    if not bank_path.is_file():
        raise FileNotFoundError(f"The original validation episode bank is required: {bank_path}")
    if hashlib.sha256(bank_path.read_bytes()).hexdigest() != selected["episode_bank_sha256"]:
        raise ValueError("The validation bank differs from the bank used for GP selection.")
    with np.load(bank_path, allow_pickle=False) as stored:
        bank = {key: stored[key] for key in stored.files if key != "signature"}
    return selected, bank


def snapshot(env, lane, observed_indices):
    """Copy the current predictions for one episode; this function never refits."""
    pred = env.cost_gp.last_prediction
    mean = pred.mean[lane].detach().cpu().numpy().copy()
    variance = pred.variance[lane].detach().cpu().numpy().copy()
    costs = env.costs[lane].detach().cpu().numpy().copy()
    target = np.log1p(costs)
    if not np.isfinite([mean, variance, target]).all() or (variance < 0).any():
        raise RuntimeError("Non-finite cost predictions or invalid predictive variance.")
    if int(env.cost_gp.t_idx[lane].item()) != len(observed_indices):
        raise RuntimeError("The observation history and cost-GP training data disagree.")
    # Use the predictive variance returned by the repository GP, including its
    # likelihood noise. Match the selection code's small numerical floor.
    variance = np.maximum(variance.astype(np.float64), 1e-12)
    squared_error = (target.astype(np.float64) - mean.astype(np.float64)) ** 2
    return {
        "mean": mean,
        "std": np.sqrt(variance),
        "target": target,
        "counts": np.bincount(observed_indices, minlength=len(costs)),
        "n_observations": len(observed_indices),
        "rmse": float(np.sqrt(squared_error.mean())),
        "nlpd": float(0.5 * np.mean(np.log(2 * np.pi * variance) + squared_error / variance)),
    }


def replay_episode(env, cache, init, noise, seed, lane):
    """Replay the full original batch, retaining snapshots for the requested row."""
    import torch
    from scripts.evaluate_baselines import select_baseline_action

    obs = env.reset(seed=seed, deterministic=True, candidate_cache=cache,
                    init_indices=init, observation_noise=noise, auto_reset=False)
    observed = init[lane].detach().cpu().tolist()
    initial = snapshot(env, lane, observed)
    final = None
    while not bool(env.done.all()):
        with torch.no_grad():
            action = select_baseline_action(env, obs, env.get_action_mask(), "eipu")
        if not bool(env.done[lane]):
            observed.append(int(action[lane].item()))
        obs, _, terminal, _ = env.step(action)
        if bool(terminal[lane]):
            if bool(env.done.all()):
                # The environment skips prediction when the whole batch ends.
                # Refit once so this snapshot includes the last observation.
                # Keep gradient tracking enabled: predict() fits the GP internally.
                env.cost_gp.predict(env.X)
            # When other rows remain active, step() has already refitted both GPs
            # on the updated observations. Capture this row at its own stopping time.
            final = snapshot(env, lane, observed)
    if final is None:
        raise RuntimeError("The requested episode did not produce a final snapshot.")
    return initial, final


def plot_fit(instance, episode, setting_name, budget, initial, final, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), sharex=True, sharey=True)
    bounds = np.concatenate([
        initial["target"], initial["mean"] - initial["std"], initial["mean"] + initial["std"],
        final["target"], final["mean"] - final["std"], final["mean"] + final["std"],
    ])
    padding = max(float(np.ptp(bounds)), 1.0) * 0.06
    limits = (float(bounds.min()) - padding, float(bounds.max()) + padding)
    for ax, data, label in zip(axes, (initial, final), ("After initialization", "After final acquisition")):
        ax.plot(limits, limits, "--", color="#656A70", linewidth=1, label="Perfect prediction")
        seen = data["counts"] > 0
        for mask, color, name in ((~seen, "#32698A", "Unobserved candidates"),
                                  (seen, "#C65D13", "Observed candidates")):
            ax.errorbar(data["target"][mask], data["mean"][mask], yerr=data["std"][mask],
                        fmt="o", markersize=4, color=color, ecolor=color, alpha=0.70,
                        elinewidth=0.7, capsize=1.5, label=name)
        ax.set(xlim=limits, ylim=limits, xlabel="True log1p(cost)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{label}\n{data['n_observations']} observations, {int(seen.sum())} unique candidates\n"
            f"All-candidate RMSE {data['rmse']:.3g} | NLPD {data['nlpd']:.3g}", fontsize=10,
        )
        ax.grid(color="#E2E4E7", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Predicted log1p(cost)")
    fig.suptitle(
        f"Cost GP fit | validation instance {instance} | episode {episode}\n"
        f"{setting_name} | low budget {budget:.2f} s | EI per unit cost", fontsize=12,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3, fontsize=9)
    fig.text(
        0.5, 0.025,
        "Vertical bars show predictive +/-1 SD in log1p-cost space. Cost is measured in benchmark seconds.\n"
        "Each point is one candidate configuration. Observation counts include warm starts and repeated queries.",
        ha="center", fontsize=8,
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.74, bottom=0.21, wspace=0.15)
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"instance_{instance}.{extension}", dpi=200,
                    bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gp", nargs="?", help="Setting ID, or 'all'. Default: the selected cost GP.")
    parser.add_argument("--episode", type=int, default=0, help="Zero-based episode index, used on every instance.")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="Default: the original selection device.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/gp_selection"))
    args = parser.parse_args()
    selected, bank = load_saved_selection(args.results_dir)
    phase, cfg = selected["phase_config"], selected["base_config"]
    if not 0 <= args.episode < phase["episodes_per_instance"]:
        parser.error(f"--episode must be between 0 and {phase['episodes_per_instance'] - 1}.")

    # Delay the repository imports so --help also works without the GP dependencies.
    import torch
    from scripts.select_gp import batch_data, candidate_settings, configured_env, setting_id
    from utils.seeding import seed_everything

    settings = {setting_id(setting): setting for setting in candidate_settings(phase)}
    requested = args.gp or selected["setting"]
    if requested == "all":
        chosen = settings
    elif requested in settings:
        chosen = {requested: settings[requested]}
    else:
        parser.error("Unknown GP setting. Available IDs:\n" + "\n".join(settings))
    device = torch.device(args.device or phase["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Pass --device cpu to run CPU diagnostics.")
    if str(device) != phase["device"]:
        print(f"Using {device}; the original selection used {phase['device']}. Numerical results may differ.")
    seed_everything(phase["seed"], deterministic=True)
    budget = next(item for item in selected["budgets"] if item["level"] == "low")["budget"]
    dtype = getattr(torch, cfg["env"]["dtype"])
    start = (args.episode // phase["batch_size"]) * phase["batch_size"]
    end = min(start + phase["batch_size"], phase["episodes_per_instance"])
    lane = args.episode - start

    for name, setting in chosen.items():
        output_dir = args.results_dir / "diagnostics" / "cost_fit" / name / f"episode_{args.episode:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for instance in map(str, cfg["problem_family"]["instances"]):
            print(f"Replaying {name} / instance {instance} / episodes {start}-{end - 1}", flush=True)
            cache, init, noise = batch_data(bank, instance, start, end, device, dtype)
            env = configured_env(cfg, "cost", setting, selected["gp"], end - start,
                                 budget, phase["max_acquisitions"], device)
            initial, final = replay_episode(env, cache, init, noise, phase["seed"], lane)
            plot_fit(instance, args.episode, name, budget, initial, final, output_dir)
            print(f"  Final observations {final['n_observations']}; "
                  f"RMSE {initial['rmse']:.4g} -> {final['rmse']:.4g}; "
                  f"NLPD {initial['nlpd']:.4g} -> {final['nlpd']:.4g}", flush=True)
            del env
        print(f"Saved PDF/PNG fit figures to {output_dir}", flush=True)


if __name__ == "__main__":
    main()