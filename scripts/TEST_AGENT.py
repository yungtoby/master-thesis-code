"""Test the validation-selected agents; do not train or select any new models.

    py scripts/TEST_AGENT.py bank
    py scripts/TEST_AGENT.py matched
    py scripts/TEST_AGENT.py transfer

To run only one TEST budget, add its name: e.g. "matched low".
"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
from statistics import fmean, stdev
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import TRAIN_AGENTS as training

RESULTS = ROOT / "results" / "final_testing"
INSTANCES = training.TEST
BUDGETS = ("low", "medium", "high")
COST_AGENTS = ("cost_mean", "cost_uncertainty")
BASELINES = ("random", "ei", "eipu", "ei_cool")
EPISODES = 128                       # Per instance, for every checkpoint or baseline.
BATCH = 32                           # Four batches, each containing only one instance.
BANK_SEED = 20261103
SUCCESS_THRESHOLD = 0.01


# 1. Read the completed training and validation decisions.

def load_training():
    """Return the config and selected checkpoint for each training run."""
    protocol = training.read_json(training.DEFAULT_PROTOCOL)
    selected = training.read_json(Path(training.RESULTS_ROOT) / "selected_checkpoints.json")
    protocol_id = protocol.pop("protocol_sha256")
    if training.object_digest(protocol) != protocol_id or selected["protocol_sha256"] != protocol_id:
        raise ValueError("The checkpoint selection does not match the frozen training protocol.")

    # Phase 4 hashed text files on Windows. An unchanged Linux checkout can use
    # LF line endings instead of CRLF. Check both representations of the same text.
    files = dict(protocol["source_hashes"])
    configs = {}
    for run in protocol["runs"]:
        files[run["config"]] = run["config_sha256"]
        configs[run["run_id"]] = training.read_json(run["config"])
    for path, expected in files.items():
        lf = Path(path).read_bytes().replace(b"\r\n", b"\n")
        if not any(hashlib.sha256(data).hexdigest() == expected
                   for data in (lf, lf.replace(b"\n", b"\r\n"))):
            raise ValueError(f"File changed after training: {path}")

    winners = {}
    for row in selected["selected_checkpoints"]:
        checkpoint = row["checkpoint"]
        if training.digest(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError(f"Selected weights have changed: {checkpoint['path']}")
        winners[row["run_id"]] = checkpoint
    expected_runs = {f"{m}__{b}__seed{s}" for m in training.VARIANTS
                     for b in BUDGETS for s in training.SEEDS}
    if set(configs) != expected_runs or set(winners) != expected_runs:
        raise ValueError("Expected the 27 runs from the completed validation.")
    return configs, winners


# 2. Prepare one paired bank for both experiments.

def bank_settings(config):
    """Keep the same bank format as the previous test script, so it can be reused."""
    env = config["env"]
    return {
        "split": "test", "seed": BANK_SEED, "episodes_per_instance": EPISODES,
        "problem_family": {**config["problem_family"], "instances": list(INSTANCES)},
        "n_candidates": env["n_candidates"], "n_init": env["n_init"], "features": 7,
        "max_acquisitions": env["max_acquisitions"], "dtype": env["dtype"],
        "objective_noise_std": env["objective_noise_std"],
        "objective_noise_clip": env["objective_noise_clip"],
    }


def episode_draws(instance, settings):
    """Give candidates, warm starts and noise independent streams for this instance."""
    streams = np.random.SeedSequence([settings["seed"], int(instance)]).spawn(3)
    candidate_seed = int(streams[0].generate_state(1)[0])
    warm_rng = np.random.default_rng(streams[1])
    noise_rng = np.random.default_rng(streams[2])
    count, n, k = settings["episodes_per_instance"], settings["n_candidates"], settings["n_init"]
    starts = [warm_rng.choice(n, k, replace=False) for _ in range(count)]
    # The first k draws are for warm starts. The remaining draws are indexed by
    # query number: different agents get the same noise at their j-th query.
    noise = noise_rng.standard_normal((count, k + settings["max_acquisitions"]))
    return candidate_seed, np.asarray(starts, dtype=np.int64), noise.astype(np.float32)


def read_bank(directory, settings):
    manifest = training.read_json(directory / "manifest.json")
    path = directory / "episodes.npz"
    if manifest["spec"] != settings or training.digest(path) != manifest["sha256"]:
        raise ValueError(f"Saved bank differs from the planned test bank: {directory}")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    return arrays, manifest["sha256"]


def prepare_bank(directory, settings):
    if directory.exists():
        read_bank(directory, settings)
        print(f"Reusing test bank: {directory}")
        return

    import torch
    from bo.problems.registry import build_problem_family

    arrays = {}
    for instance in settings["problem_family"]["instances"]:
        problem = dict(settings["problem_family"], instances=[instance])
        family = build_problem_family(problem, device="cpu", dtype=torch.float32)
        seed, starts, noise = episode_draws(instance, settings)
        x, y, cost, _ = family.build_candidate_cache(
            settings["episodes_per_instance"], settings["n_candidates"], seed=seed)
        for name, values in (("X", x.numpy()), ("y", y.numpy()), ("cost", cost.numpy()),
                             ("init", starts), ("noise", noise)):
            if not np.isfinite(values).all():
                raise ValueError(f"Invalid {name} values for instance {instance}.")
            arrays[f"{instance}_{name}"] = values
        print(f"Prepared {len(starts)} episodes for {instance}.", flush=True)

    # Publish the data and manifest together. An interrupted bank build leaves
    # no half-written bank that a later experiment might try to reuse.
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as temp:
        ready = Path(temp) / "bank"
        ready.mkdir()
        path = ready / "episodes.npz"
        np.savez_compressed(path, **arrays)
        training.write_json(ready / "manifest.json", {"spec": settings, "sha256": training.digest(path)})
        ready.rename(directory)
    print(f"Saved test bank: {directory}")


# 3. Specify who is tested and at which budget.

def experiment_groups(experiment, target):
    """Each pair is (method, training budget). Baselines have no training budget."""
    if experiment == "matched":
        return ([(method, target) for method in training.VARIANTS]
                + [(method, None) for method in BASELINES])
    return [(method, source) for method in COST_AGENTS for source in BUDGETS if source != target]


def setup_case(method, source, target, seed, configs, winners, device):
    if source is None:
        # Borrow environment settings only. EIpu and EI-cool need the cost GP;
        # Random and EI do not. No baseline loads a trained policy.
        template = "cost_mean" if method in ("eipu", "ei_cool") else "no_cost"
        run_id = f"{template}__{target}__seed1"
        checkpoint = None
    else:
        run_id = f"{method}__{source}__seed{seed}"
        checkpoint = winners[run_id]

    # Source determines the weights and feature definitions. Target determines
    # both when BO stops and how the environment normalizes cost and progress.
    cfg = copy.deepcopy(configs[run_id])
    cfg["env"]["budget"] = configs[f"no_cost__{target}__seed1"]["env"]["budget"]
    cfg["env"]["device"] = str(device)
    cfg["ppo"]["num_envs"] = BATCH
    cfg["problem_family"]["instances"] = list(INSTANCES)
    return cfg, checkpoint


def load_agent(config, checkpoint, device):
    import torch
    from reinforcement_learning.agents.neural_AF import Agent

    a = config["agent"]
    actor = Agent(7, 7, 1, 1, a["num_layers"], a["layer_size"], a["critic_feature_mode"])
    weights = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    actor.load_state_dict(weights)
    return actor.to(device).eval()


# 4. Replay fixed episodes and save each completed case.

def play_batch(env, actor, method, cache, starts, noise, seed):
    if actor is not None:
        # Use exactly the learned-policy argmax evaluation used during validation.
        return training.run_policy_batch(env, actor, cache, starts, noise, seed)

    import torch
    from scripts.evaluate_baselines import select_baseline_action

    obs = env.reset(seed, True, candidate_cache=cache, init_indices=starts,
                    observation_noise=noise, auto_reset=False)
    rng = torch.Generator(device=env.device).manual_seed(seed)
    rows = [None] * env.num_batches
    while not bool(env.done.all()):
        mask = env.get_action_mask()
        with torch.no_grad():
            if not torch.isfinite(obs[~env.done]).all():
                raise ValueError("Non-finite GP observation in a running episode.")
            if method == "random":
                # A private stream keeps Random reproducible even if GP code draws random numbers.
                actions = torch.multinomial(mask.float(), 1, generator=rng).squeeze(1)
            else:
                actions = select_baseline_action(env, obs, mask, method)
        # GPs are fitted inside step(), so this line needs gradients.
        obs, _, terminal, info = env.step(actions)
        for lane in torch.where(terminal)[0].tolist():
            end = info["final_info"][lane]
            rows[lane] = {"regret": end["regret"], "episode_length": end["episode"]["l"],
                          "budget_overshoot": end["budget_overshoot"]}
    return rows


def evaluate(method, cfg, checkpoint, bank, device):
    import torch
    from reinforcement_learning.ppo_no_gym import make_env
    from scripts.select_gp import batch_data
    from utils.seeding import seed_everything

    seed_everything(BANK_SEED, deterministic=True)
    actor = load_agent(cfg, checkpoint, device) if checkpoint is not None else None
    env = make_env(cfg, device)
    rows = []
    for instance in INSTANCES:
        for start in range(0, EPISODES, BATCH):
            # The same instance and block use the same seed at every test budget.
            seed = int(np.random.SeedSequence([BANK_SEED, int(instance), start // BATCH, 4])
                       .generate_state(1)[0])
            cache, starts, noise = batch_data(bank, instance, start, start + BATCH, device, torch.float32)
            outcomes = play_batch(env, actor, method, cache, starts, noise, seed)
            for lane, end in enumerate(outcomes):
                rows.append({"instance": instance, "episode_id": start + lane,
                             "regret": end["regret"], "episode_length": end["episode_length"],
                             "budget_overshoot": end["budget_overshoot"]})
        print(f"  {instance}: {EPISODES} episodes", flush=True)
    return rows


def case_results(path, inputs, method, cfg, checkpoint, bank, device):
    """Reuse a complete case, or replay it once. File checks stay out of the BO loop."""
    if path.exists():
        saved = training.read_json(path)
        rows = saved["episodes"]
        if saved["inputs"] != inputs or saved["checksum"] != training.object_digest(rows):
            raise ValueError(f"Saved case has different inputs or damaged results: {path}")
        print(f"  Reusing {path.stem}")
        return rows

    rows = evaluate(method, cfg, checkpoint, bank, device)
    expected = {(i, j) for i in INSTANCES for j in range(EPISODES)}
    actual = {(r["instance"], r["episode_id"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"Missing or duplicated episodes: {path.stem}")
    training.write_json(path, {"inputs": inputs, "checksum": training.object_digest(rows),
                               "episodes": rows})
    return rows


# 5. Report the same statistics for individual instances and the complete bank.

def episode_statistics(rows):
    """Summarize individual episode outcomes. stdev uses the sample denominator n - 1."""
    regrets = [r["regret"] for r in rows]
    lengths = [r["episode_length"] for r in rows]
    overshoots = [r["budget_overshoot"] for r in rows]
    return {
        "mean_final_simple_regret": fmean(regrets),
        "sd_final_simple_regret": stdev(regrets),
        "mean_episode_length": fmean(lengths),
        "sd_episode_length": stdev(lengths),
        "mean_budget_overshoot": fmean(overshoots),
        "sd_budget_overshoot": stdev(overshoots),
        "success_rate": fmean(r <= SUCCESS_THRESHOLD for r in regrets),
    }


def summarize_group(method, source, target, rows):
    labels = {"method": method, "training_budget": source, "test_budget": target}
    by_instance = []
    for instance in INSTANCES:
        episodes = [r for r in rows if r["instance"] == instance]
        by_instance.append({**labels, "instance": instance, **episode_statistics(episodes)})
    # Each seed completed the same 128 episodes on every instance. Pooling all
    # rows therefore gives the six instances equal weight in the aggregate.
    # Calculate the aggregate SD from those same individual rows; averaging
    # the instance SDs would discard differences between instances.
    return {**labels, **episode_statistics(rows)}, by_instance


def run_budget(experiment, target, configs, winners, bank, common_inputs, device):
    report_dir = RESULTS / (experiment + "_budgets")
    records = RESULTS / "episode_records" / experiment / target
    aggregated, instance_specific = [], []
    for method, source in experiment_groups(experiment, target):
        pooled = []
        seeds = training.SEEDS if source is not None else [None]
        for seed in seeds:
            cfg, checkpoint = setup_case(method, source, target, seed, configs, winners, device)
            name = method if seed is None else f"{method}__{source}__seed{seed}"
            print(f"{experiment}: {name} at {target} test budget", flush=True)
            inputs = {**common_inputs, "method": method, "training_budget": source,
                      "training_seed": seed, "test_budget": target, "checkpoint": checkpoint,
                      "config_sha256": training.object_digest(cfg)}
            pooled.extend(case_results(records / (name + ".json"), inputs, method,
                                       cfg, checkpoint, bank, device))
        total, instances = summarize_group(method, source, target, pooled)
        aggregated.append(total)
        instance_specific.extend(instances)

    training.write_csv(report_dir / "aggregated" / (target + ".csv"), aggregated)
    training.write_csv(report_dir / "instance_specific" / (target + ".csv"), instance_specific)
    print(f"Saved aggregate and per-instance results for {target} in {report_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run the paired, held-out BO experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bank", help="Create or reuse the test episode bank.")
    for experiment in ("matched", "transfer"):
        cmd = commands.add_parser(experiment, help=f"Run the {experiment} budget experiment.")
        cmd.add_argument("budget", nargs="?", choices=BUDGETS, help="Optional TEST budget.")
    args = parser.parse_args()
    os.chdir(ROOT)
    configs, winners = load_training()
    settings = bank_settings(configs["cost_mean__low__seed1"])
    if args.command == "bank":
        prepare_bank(RESULTS / "bank", settings)
        return

    import torch
    import gpytorch
    bank, bank_hash = read_bank(RESULTS / "bank", settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # A paused experiment may resume later. Record what produced the episode
    # outcomes so that results from different banks, evaluators or GPUs cannot mix.
    common_inputs = {"bank_sha256": bank_hash, "device": str(device), "torch": str(torch.__version__),
                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     "gpytorch": gpytorch.__version__,
                     "code": {p.name: training.object_digest(p.read_text(encoding="utf-8"))
                              for p in (Path(__file__), ROOT / "scripts/select_gp.py",
                                        ROOT / "scripts/evaluate_baselines.py")}}
    targets = [args.budget] if args.budget is not None else BUDGETS
    for target in targets:
        run_budget(args.command, target, configs, winners, bank, common_inputs, device)


if __name__ == "__main__":
    main()
