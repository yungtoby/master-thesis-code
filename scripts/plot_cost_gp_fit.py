import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_archive(path, expected_hash):
    """Load an existing archive; diagnostics must never generate a new bank."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"The saved GP-selection archive is required: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Archive differs from the one used for selection: {path}")
    with np.load(path, allow_pickle=False) as stored:
        signature = json.loads(str(stored["signature"].item()))
        arrays = {key: stored[key] for key in stored.files if key != "signature"}
    return arrays, signature


def load_saved_selection(results_dir):
    from bo import gaussian_process_batch_NEW as gp_module
    from scripts.select_gp_NEW import bank_signature

    path = results_dir / "cost" / "selected.json"
    selected = json.loads(path.read_text(encoding="utf-8"))
    if selected["stage"] != "cost" or selected["method"] != "random_without_replacement":
        raise ValueError("Expected the NEW random-acquisition cost-GP selection.")
    if selected["fitting_recipe"] != gp_module.FITTING_RECIPE:
        raise ValueError("The GP fitting recipe differs from the saved selection.")
    expected = selected["source_hashes"]["bo/gaussian_process_batch_NEW.py"]
    if hashlib.sha256(Path(gp_module.__file__).read_bytes()).hexdigest() != expected:
        raise ValueError("The cost-GP implementation changed after selection.")

    phase, cfg = selected["phase_config"], selected["base_config"]
    if cfg.get("cost_model", {}).get("target_transform", "log1p") != "log1p":
        raise ValueError("This diagnostic expects log1p(cost) targets.")
    bank, signature = read_archive(phase["episode_bank"], selected["episode_bank_sha256"])
    if signature != bank_signature(phase, cfg):
        raise ValueError("The validation-bank signature differs from the selection.")
    orders, signature = read_archive(phase["random_orders"], selected["random_orders_sha256"])
    expected = dict(format="random_without_replacement_v1", seed=phase["seed"],
                    episode_bank_sha256=selected["episode_bank_sha256"], cap=phase["max_acquisitions"])
    if signature != expected:
        raise ValueError("The random-order signature differs from the selection.")
    return selected, bank, orders


def snapshot(gp, X, costs, lane, observed_indices):
    """Fit the full batch once and copy one lane's observation predictions."""
    # predict() performs ten fitting steps internally, so do not use no_grad here.
    prediction = gp.predict(X)
    mean = prediction.mean[lane].detach().cpu().numpy().astype(np.float64)
    variance = prediction.variance[lane].detach().cpu().numpy().astype(np.float64)
    # Match selection's target transformation in the original tensor dtype.
    target = costs[lane].log1p().detach().cpu().numpy().astype(np.float64)
    if not np.isfinite([mean, variance, target]).all() or (variance <= 0).any():
        raise RuntimeError("Non-finite cost prediction or invalid likelihood variance.")
    if int(gp.t_idx[lane].item()) != len(observed_indices):
        raise RuntimeError("The observation history and GP buffers disagree.")
    variance = np.maximum(variance, 1e-12)
    squared_error = (target - mean) ** 2
    point_nlpd = 0.5 * (np.log(2 * np.pi * variance) + squared_error / variance)
    counts = np.bincount(observed_indices, minlength=len(target))
    unseen = counts == 0
    return dict(
        mean=mean, std=np.sqrt(variance), target=target, counts=counts,
        n_observations=len(observed_indices),
        rmse=float(np.sqrt(squared_error.mean())), nlpd=float(point_nlpd.mean()),
        unseen_rmse=float(np.sqrt(squared_error[unseen].mean())) if unseen.any() else None,
        unseen_nlpd=float(point_nlpd[unseen].mean()) if unseen.any() else None,
    )


def replay_episode(cache, init, order, setting, budget, lane):
    """Reconstruct batch histories up to the requested episode's own endpoint."""
    import torch
    from bo.gaussian_process_batch_NEW import RepeatedPadGPWrapper

    X, _, costs, _ = cache
    B, _, d = X.shape
    n_init, cap = init.shape[1], order.shape[1]
    lanes = torch.arange(B, device=X.device)
    gp = RepeatedPadGPWrapper(X.device, X.dtype, B, n_init + cap, d, **setting)
    gp.set_lane_data(
        torch.ones(B, device=X.device, dtype=torch.bool),
        X[lanes[:, None], init], costs[lanes[:, None], init].log1p(),
    )
    observed = init[lane].detach().cpu().tolist()
    initial = snapshot(gp, X, costs, lane, observed)
    remaining = torch.full((B,), budget, device=X.device, dtype=X.dtype)
    done = torch.zeros(B, device=X.device, dtype=torch.bool)

    # Orders and stopping depend only on saved data and costs. Every prediction
    # fits a fresh GP, so intermediate predictions would not change either
    # endpoint. We need only the initial and final fits for these figures.
    for step in range(cap):
        active = ~done
        action = order[:, step]
        queried_cost = costs[lanes, action]
        gp.add_data(X[lanes, action][:, None, :], queried_cost.log1p(), active)
        remaining[active] -= queried_cost[active]
        observed.append(int(action[lane].item()))
        done |= (remaining <= 0) | (step + 1 >= cap)
        if bool(done[lane]):
            # Keep the other lanes at this same time, including already-done
            # lanes. This preserves the original padding and joint gradient clip.
            final = snapshot(gp, X, costs, lane, observed)
            return initial, final
    raise RuntimeError("The requested episode did not reach its endpoint.")


def plot_fit(instance, episode, setting_name, budget, initial, final, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharex=True, sharey=True)
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
        if data["unseen_rmse"] is None:
            unseen_label = "Unobserved: none remaining"
        else:
            unseen_label = f"Unobserved: RMSE {data['unseen_rmse']:.3g} | NLPD {data['unseen_nlpd']:.3g}"
        ax.set(xlim=limits, ylim=limits, xlabel="True log1p(cost)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{label}\n{data['n_observations']} observed candidates\n"
            f"All candidates: RMSE {data['rmse']:.3g} | NLPD {data['nlpd']:.3g}\n"
            f"{unseen_label}", fontsize=10,
        )
        ax.grid(color="#E2E4E7", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Predicted log1p(cost)")
    fig.suptitle(
        f"Cost GP fit | validation instance {instance} | episode {episode}\n"
        f"{setting_name} | low budget {budget:.2f} s\n"
        "Saved random acquisitions without replacement", fontsize=12,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3, fontsize=9)
    fig.text(
        0.5, 0.025,
        "Bars: likelihood predictive +/-1 SD in log1p-cost space. Counts include the initial observations.\n"
        "All-candidate scores include observed points; these snapshot scores are not the selection scores.",
        ha="center", fontsize=8,
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.69, bottom=0.21, wspace=0.15)
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"instance_{instance}.{extension}", dpi=200,
                    bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gp", nargs="?", help="Setting ID, or 'all'. Default: the selected cost GP.")
    parser.add_argument("--episode", type=int, default=0, help="Zero-based episode index on every instance.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu", help="Default: cpu, for use during GPU training.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/gp_selection_NEW"))
    args = parser.parse_args()

    # --help exits before importing PyTorch or requiring any selection files.
    import torch
    from scripts.select_gp_NEW import batch_data, candidate_settings, setting_id
    from utils.seeding import seed_everything

    selected, bank, orders = load_saved_selection(args.results_dir)
    phase, cfg = selected["phase_config"], selected["base_config"]
    if not 0 <= args.episode < phase["episodes_per_instance"]:
        parser.error(f"--episode must be between 0 and {phase['episodes_per_instance'] - 1}.")
    settings = {setting_id(setting): setting for setting in candidate_settings(phase)}
    if settings.get(selected["setting"]) != selected["gp"]:
        raise ValueError("The saved cost GP disagrees with its setting ID or candidate grid.")
    requested = args.gp or selected["setting"]
    if requested == "all":
        chosen = settings
    elif requested in settings:
        chosen = {requested: settings[requested]}
    else:
        parser.error("Unknown GP setting. Available IDs:\n" + "\n".join(settings))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the default --device cpu.")
    if str(device) != phase["device"]:
        print(f"Using {device}; selection used {phase['device']}. Numerical results may differ.")
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
            print(f"Fitting {name} / instance {instance} / original batch {start}-{end - 1}", flush=True)
            cache, init, _ = batch_data(bank, instance, start, end, device, dtype)
            order = torch.as_tensor(orders[instance][start:end], device=device, dtype=torch.long)
            initial, final = replay_episode(cache, init, order, setting, budget, lane)
            plot_fit(instance, args.episode, name, budget, initial, final, output_dir)
            print(f"  Final observations {final['n_observations']}; "
                  f"RMSE {initial['rmse']:.4g} -> {final['rmse']:.4g}; "
                  f"NLPD {initial['nlpd']:.4g} -> {final['nlpd']:.4g}", flush=True)
        print(f"Saved PDF/PNG fit figures to {output_dir}", flush=True)


if __name__ == "__main__":
    main()