"""Prepare, time, train, and validate the final thesis experiment.

Commands, run from the repository root:

    py scripts/TRAIN_AGENTS.py prepare
    py scripts/TRAIN_AGENTS.py check
    py scripts/TRAIN_AGENTS.py timing
    py scripts/TRAIN_AGENTS.py train
    py scripts/TRAIN_AGENTS.py validate

To run one case:

    py scripts/TRAIN_AGENTS.py train --run-id cost_mean__low__seed1

The experiment consists of:
    3 policy variants x 3 budgets x 3 training seeds = 27 training runs.

This file controls the experiment. PPO training itself remains in
reinforcement_learning/ppo_no_gym.py.

The protocol is an automatically generated record of configurations and file
fingerprints. It is not a separate place to manually change experiment settings.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# 1. Experiment settings
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG_DIR = "configs/final_experiments_training"
DEFAULT_PROTOCOL = f"{DEFAULT_CONFIG_DIR}/protocol.json"
RESULTS_ROOT = "results/final_training"

# Use the audited PPO setup and the completed budget/GP selection.
TEMPLATE = (
    "configs/pilots_and_audits/debug/"
    "debug_cost_mean_longrun_1m_new_critic.json"
)
PHASE2 = "configs/gp_hyperparameter_selection/gp_configs.json"
OBJECTIVE = "results/gp_selection/objective/selected.json"
COST = "results/gp_selection/cost/selected.json"

TRAIN = [
    "3945", "7593", "34539", "126025", "126026", "126029", "146212",
    "167104", "167149", "167152", "167161", "167168", "167181", "167184",
    "167185", "167190", "167200", "167201", "168329", "168330",
]
VALIDATION = [
    "168331", "168335", "168868", "168908",
    "168910", "189354", "189862", "189865",
]
# Record the reserved IDs to check split separation. Do not load test data.
TEST = ["189866", "189873", "189905", "189906", "189908", "189909"]

VARIANTS = ["no_cost", "cost_mean", "cost_uncertainty"]
SEEDS = [1, 2, 3]

CHECKPOINT_STEPS = [262144, 524288, 786432, 1048576]
TOTAL_STEPS = CHECKPOINT_STEPS[-1]

# One PPO rollout contains 256 environments x 32 transitions.
ROLLOUT_SIZE = 256 * 32

# Short timing runs use six PPO updates. Exclude the first two when estimating
# the sustained update rate.
WARMUP_UPDATES = 2
MEASURED_UPDATES = 4
TIMING_STEPS = (WARMUP_UPDATES + MEASURED_UPDATES) * ROLLOUT_SIZE

SELECTION_RULE = (
    "For each variant, budget, and training seed, select the checkpoint "
    "with the lowest unrounded mean of validation-instance means. "
    "An exact tie selects the earlier checkpoint. Retain every seed."
)


# ---------------------------------------------------------------------------
# 2. Small file helpers
# ---------------------------------------------------------------------------

def require(condition, message):
    """Reject invalid experiment inputs or incomplete results."""
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Write through a temporary file to avoid leaving half-written reports."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def digest(path):
    """Fingerprint file contents, including saved checkpoints."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_digest(value):
    """Fingerprint a JSON-compatible object independently of key order."""
    contents = json.dumps(value, sort_keys=True, allow_nan=False)
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 3. Read the previous phase decisions and prepare the experiment
# ---------------------------------------------------------------------------

def load_experiment_settings():
    """Read the existing decisions instead of restating the chosen GP settings."""
    template = read_json(TEMPLATE)
    phase = read_json(PHASE2)
    objective = read_json(OBJECTIVE)
    cost = read_json(COST)
    calibration = read_json(phase["budgets_file"])
    validation_config = read_json(phase["base_config"])

    # These checks concern separately produced input files. They prevent, for
    # example, combining a cost selection with a different objective selection.
    require(
        objective["stage"] == "objective" and cost["stage"] == "cost",
        "The objective/cost selection files contain the wrong stages.",
    )
    require(
        not objective["failed_settings"] and not cost["failed_settings"],
        "A GP selection contains failed settings.",
    )
    require(
        objective["gp"] == cost["gp"],
        "Cost selection used a different objective GP.",
    )
    require(
        objective["phase_config"] == cost["phase_config"] == phase,
        "The GP selection settings do not match.",
    )
    require(
        objective["budgets"] == cost["budgets"] == calibration["budgets"],
        "The selections and budget calibration use different budgets.",
    )
    require(
        objective["episode_bank_sha256"] == cost["episode_bank_sha256"],
        "The GP selections used different validation episode banks.",
    )

    require(
        template["problem_family"]["instances"] == TRAIN,
        "The training template does not use the agreed 20 instances.",
    )
    require(
        validation_config["problem_family"]["instances"] == VALIDATION,
        "The validation configuration does not use the agreed eight instances.",
    )
    require(
        len(set(TRAIN + VALIDATION + TEST)) == 34,
        "The training, validation, and test splits overlap.",
    )

    # Preserve the candidate and cost-accounting setup used in earlier phases.
    for key in ("n_candidates", "n_init", "mask_visited_actions"):
        require(
            template["env"][key]
            == validation_config["env"][key]
            == calibration["settings"][key],
            f"Training, validation, and calibration disagree about {key}.",
        )

    training_problem = {
        key: value
        for key, value in template["problem_family"].items()
        if key != "instances"
    }
    validation_problem = {
        key: value
        for key, value in validation_config["problem_family"].items()
        if key != "instances"
    }
    require(
        training_problem == validation_problem,
        "Training and validation use different benchmark definitions.",
    )

    for key in (
        "dtype", "objective_noise_std", "objective_noise_clip",
        "observation_format", "reward_type", "reward_scale",
    ):
        require(
            template["env"][key] == validation_config["env"][key],
            f"Training and validation disagree about {key}.",
        )

    require(
        (template["ppo"]["num_envs"], template["ppo"]["num_steps"]) == (256, 32),
        "The agreed PPO rollout uses 256 environments and 32 steps.",
    )
    require(
        template["ppo"]["anneal_lr"] is False,
        "The timing estimate assumes the existing constant PPO learning rate.",
    )
    require(
        (
            phase["episodes_per_instance"],
            phase["batch_size"],
            phase["max_acquisitions"],
        ) == (32, 32, 61),
        "The agreed validation setup uses 32 episodes, batch size 32, and cap 61.",
    )
    require(
        [budget["level"] for budget in calibration["budgets"]]
        == ["low", "medium", "high"],
        "Expected low, medium, and high calibrated budgets.",
    )
    require(
        all(
            math.isfinite(budget["budget"]) and budget["budget"] > 0
            for budget in calibration["budgets"]
        ),
        "A calibrated budget is invalid.",
    )

    return {
        "template": template,
        "phase": phase,
        "validation_config": validation_config,
        "objective_gp": objective["gp"],
        "cost_model": cost["cost_model"],
        "budgets": calibration["budgets"],
        "bank_hash": objective["episode_bank_sha256"],
        "input_paths": [
            TEMPLATE, PHASE2, OBJECTIVE, COST,
            phase["budgets_file"], phase["base_config"],
        ],
    }


def prepare_run(settings, variant, budget, seed, steps, group, run_id):
    """Write one complete configuration and return its experiment record.

    group determines output locations: either "train" or "timing".
    steps explicitly determines the length; there is no separate timing flag.
    """
    cfg = copy.deepcopy(settings["template"])
    case = f"{variant}__{budget['level']}"

    cfg["experiment"].update(
        name=f"phase4_{group}__{case}",
        seed=seed,
        save_every=CHECKPOINT_STEPS[0],
        cuda=True,
        checkpoint_dir=f"checkpoints/final_training/{group}",
        run_dir=f"runs/final_training/{group}",
    )
    cfg["ppo"]["total_timesteps"] = steps

    # The trainer derives these values from the rollout dimensions.
    for key in ("batch_size", "minibatch_size", "num_iterations"):
        cfg["ppo"].pop(key, None)

    cfg["problem_family"]["instances"] = list(TRAIN)
    cfg["gp"] = copy.deepcopy(settings["objective_gp"])
    cfg["env"].update(
        budget=budget["budget"],
        max_acquisitions=settings["phase"]["max_acquisitions"],
        device="cuda",
        cost_feature_mode="none" if variant == "no_cost" else "predicted",
        use_cost_uncertainty_feature=variant == "cost_uncertainty",
    )

    # The no-cost policy receives zero cost features and does not fit a cost GP.
    # Actual evaluation costs still consume its budget.
    if variant == "no_cost":
        cfg["cost_model"] = {"type": "known", "target_transform": "log1p"}
    else:
        cfg["cost_model"] = copy.deepcopy(settings["cost_model"])
    cfg["cost_model"]["include_uncertainty"] = variant == "cost_uncertainty"

    path = f"{DEFAULT_CONFIG_DIR}/{group}/{run_id}.json"
    write_json(path, cfg)

    return {
        "run_id": run_id,
        "variant": variant,
        "budget_level": budget["level"],
        "training_seed": seed,
        "config": path,
        "config_sha256": digest(path),
    }


def prepare():
    """Generate the 27 full configurations, nine timing cases, and their record."""
    for folder in (DEFAULT_CONFIG_DIR, RESULTS_ROOT):
        path = Path(folder)
        require(
            not path.exists() or not any(path.iterdir()),
            f"{folder} is not empty. Archive it before preparing a new experiment.",
        )

    settings = load_experiment_settings()
    phase = settings["phase"]
    validation_config = settings["validation_config"]

    # This signature describes the existing paired bank; it does not generate
    # new validation episodes.
    signature = {
        "format": "phase2_step_noise_v1",
        "seed": phase["seed"],
        "episodes_per_instance": phase["episodes_per_instance"],
        "max_acquisitions": phase["max_acquisitions"],
        "problem_family": validation_config["problem_family"],
        "n_candidates": validation_config["env"]["n_candidates"],
        "n_init": validation_config["env"]["n_init"],
        "dtype": validation_config["env"]["dtype"],
    }

    # Store only information needed to run or identify this experiment.
    # No manually maintained version number or reviewed-commit label is needed.
    protocol = {
        "results_root": RESULTS_ROOT,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "selection_rule": SELECTION_RULE,
        "validation": {
            "instances": list(VALIDATION),
            "episodes_per_instance": phase["episodes_per_instance"],
            "batch_size": phase["batch_size"],
            "seed": phase["seed"],
            "action_mode": "argmax",
            "episode_bank": phase["episode_bank"],
            "episode_bank_sha256": settings["bank_hash"],
            "bank_signature": signature,
        },
        "runs": [],
        "timing_runs": [],
        "inputs": {
            path: digest(path) for path in settings["input_paths"]
        },
        "source_hashes": {},
    }

    read_bank(protocol)

    source_files = [SCRIPT_PATH.relative_to(ROOT)]
    for directory in ("bo", "envs", "reinforcement_learning", "utils"):
        source_files.extend(
            path
            for path in Path(directory).rglob("*.py")
            if not path.stem.endswith("_new")
        )
    protocol["source_hashes"] = {
        path.as_posix(): digest(path) for path in sorted(source_files)
    }

    for variant in VARIANTS:
        for budget in settings["budgets"]:
            case = f"{variant}__{budget['level']}"

            for seed in SEEDS:
                run = prepare_run(
                    settings, variant, budget, seed,
                    steps=TOTAL_STEPS,
                    group="train",
                    run_id=f"{case}__seed{seed}",
                )
                protocol["runs"].append(run)

            timing_run = prepare_run(
                settings, variant, budget, seed=1,
                steps=TIMING_STEPS,
                group="timing",
                run_id=case,
            )
            protocol["timing_runs"].append(timing_run)

    protocol["protocol_sha256"] = object_digest(protocol)
    write_json(DEFAULT_PROTOCOL, protocol)
    print(f"Prepared 27 training configs and nine timing configs: {DEFAULT_PROTOCOL}")
    print("No training has started.")


# ---------------------------------------------------------------------------
# 4. Check prepared inputs and the paired validation bank
# ---------------------------------------------------------------------------

def read_bank(protocol):
    """Read and check the fixed candidate sets, warm starts, and noise."""
    import numpy as np

    validation = protocol["validation"]
    signature = validation["bank_signature"]
    path = validation["episode_bank"]

    require(
        digest(path) == validation["episode_bank_sha256"],
        "The validation episode bank has changed.",
    )
    with np.load(path, allow_pickle=False) as stored:
        require(
            json.loads(str(stored["signature"].item())) == signature,
            "The validation-bank signature does not match.",
        )
        bank = {
            key: stored[key]
            for key in stored.files
            if key != "signature"
        }

    episodes = validation["episodes_per_instance"]
    candidates = signature["n_candidates"]
    initial = signature["n_init"]
    noise_length = initial + signature["max_acquisitions"]

    shapes = {
        "X": (episodes, candidates, 7),
        "y": (episodes, candidates),
        "cost": (episodes, candidates),
        "init": (episodes, initial),
        "noise": (episodes, noise_length),
    }
    expected_keys = {
        f"{instance}_{suffix}"
        for instance in validation["instances"]
        for suffix in shapes
    }
    require(set(bank) == expected_keys, "Validation-bank arrays are missing or unexpected.")

    for instance in validation["instances"]:
        for suffix, shape in shapes.items():
            array = bank[f"{instance}_{suffix}"]
            expected_dtype = np.dtype(
                "int64" if suffix == "init" else signature["dtype"]
            )
            require(
                array.shape == shape
                and array.dtype == expected_dtype
                and np.isfinite(array).all(),
                f"Invalid bank array: {instance}_{suffix}",
            )

        indices = bank[f"{instance}_init"]
        require(
            ((indices >= 0) & (indices < candidates)).all(),
            f"An initial candidate index is out of range for {instance}.",
        )
        require(
            all(len(set(row)) == initial for row in indices.tolist()),
            f"Initial candidates are duplicated for {instance}.",
        )
        require(
            (bank[f"{instance}_cost"] >= 0).all(),
            f"The bank contains negative costs for {instance}.",
        )

    return bank


def load_protocol(path=DEFAULT_PROTOCOL):
    """Reject changed files before starting or reusing an experiment."""
    protocol = read_json(path)
    contents = {
        key: value
        for key, value in protocol.items()
        if key != "protocol_sha256"
    }
    require(
        object_digest(contents) == protocol["protocol_sha256"],
        "The prepared protocol was edited.",
    )

    for section in ("inputs", "source_hashes"):
        for file, expected in protocol[section].items():
            require(digest(file) == expected, f"File changed after preparation: {file}")

    for run in protocol["runs"] + protocol["timing_runs"]:
        require(
            digest(run["config"]) == run["config_sha256"],
            f"Configuration changed: {run['config']}",
        )

    read_bank(protocol)
    return protocol


def check():
    protocol = load_protocol()
    print(
        f"PASS: {len(protocol['runs'])} training configs; "
        f"{len(protocol['timing_runs'])} timing configs; "
        "prepared inputs, code, and validation bank match."
    )


def select_runs(runs, run_id):
    """An omitted run ID means all runs in the supplied group."""
    if run_id is None:
        return runs
    selected = [run for run in runs if run["run_id"] == run_id]
    require(bool(selected), f"Unknown run ID: {run_id}")
    return selected


# ---------------------------------------------------------------------------
# 5. Train one independent seed
# ---------------------------------------------------------------------------

def checked_training_report(path, run):
    """A run is complete only when its report and saved files agree."""
    report = read_json(path)
    cfg = read_json(run["config"])
    rollout_size = cfg["ppo"]["num_envs"] * cfg["ppo"]["num_steps"]

    require(
        report["status"] == "complete"
        and report["config_sha256"] == run["config_sha256"],
        f"Invalid or incomplete training report: {path}",
    )
    require(
        report["global_step"] == cfg["ppo"]["total_timesteps"],
        f"Training did not finish the full schedule: {run['run_id']}",
    )
    require(
        len(report["iteration_seconds"])
        == cfg["ppo"]["total_timesteps"] // rollout_size,
        f"Incomplete update timings: {run['run_id']}",
    )
    for checkpoint in report["checkpoints"]:
        require(
            digest(checkpoint["path"]) == checkpoint["sha256"],
            f"Saved checkpoint changed: {checkpoint['path']}",
        )
    require(
        any(
            checkpoint["kind"] == "final"
            and checkpoint["step"] == report["global_step"]
            for checkpoint in report["checkpoints"]
        ),
        f"Final checkpoint is missing: {run['run_id']}",
    )
    return report


def train_run(protocol, run, output):
    """Train one run, or reuse its verified completed report.

    An interrupted run restarts from its original seed. The trainer saves
    weights, not the complete state needed for exact training continuation.
    """
    output = Path(output)
    job_path = output / "job.json"
    report_path = output / "training.json"

    if job_path.exists():
        previous_job = read_json(job_path)
        require(
            previous_job["protocol_sha256"] == protocol["protocol_sha256"],
            f"{output} belongs to a different prepared experiment.",
        )

    if report_path.exists():
        job = read_json(job_path)
        report = checked_training_report(report_path, run)
        print(f"Reusing completed run: {run['run_id']}", flush=True)
        return report, job.get("process_seconds", report["total_seconds"])

    if job_path.exists():
        print(
            f"Restarting incomplete run from its seed: {run['run_id']}",
            flush=True,
        )

    job = {
        "protocol_sha256": protocol["protocol_sha256"],
        "run_id": run["run_id"],
        "status": "started",
    }
    write_json(job_path, job)

    print(f"Starting training: {run['run_id']}", flush=True)
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable, "-u",
            "reinforcement_learning/ppo_no_gym.py",
            "--config", run["config"],
            "--report", report_path.as_posix(),
            "--require-cuda",
        ],
        cwd=ROOT,
        check=True,
    )
    elapsed = time.perf_counter() - start

    report = checked_training_report(report_path, run)
    job.update(status="complete", process_seconds=elapsed)
    write_json(job_path, job)
    return report, elapsed


def train(run_id=None):
    """Run the requested training seeds sequentially."""
    protocol = load_protocol()
    root = Path(protocol["results_root"]) / "train"
    for run in select_runs(protocol["runs"], run_id):
        train_run(protocol, run, root / run["run_id"])


# ---------------------------------------------------------------------------
# 6. Evaluate a saved policy on the paired validation episodes
# ---------------------------------------------------------------------------

def balanced_score(rows, instances, episodes):
    """Average episodes within each instance, then weight instances equally."""
    values = {}
    for row in rows:
        key = (str(row["instance"]), int(row["episode_id"]))
        require(key not in values, f"Duplicate validation episode: {key}")
        regret = float(row["regret"])
        require(
            math.isfinite(regret) and regret >= 0,
            f"Invalid regret for validation episode: {key}",
        )
        values[key] = regret

    expected = {
        (instance, episode)
        for instance in instances
        for episode in range(episodes)
    }
    require(
        set(values) == expected,
        "Validation episodes are missing or unexpected.",
    )
    means = {
        instance: math.fsum(
            values[(instance, episode)] for episode in range(episodes)
        ) / episodes
        for instance in instances
    }
    return math.fsum(means.values()) / len(instances), means


def choose_checkpoint(scores, expected_steps):
    """Compare unrounded scores; an exact tie chooses the earlier checkpoint."""
    require(
        len(scores) == len(expected_steps)
        and {score["step"] for score in scores} == set(expected_steps),
        "Every scheduled checkpoint must be evaluated before selection.",
    )
    require(
        all(
            math.isfinite(score["mean_regret"]) and score["mean_regret"] >= 0
            for score in scores
        ),
        "A checkpoint score is invalid.",
    )
    return min(scores, key=lambda score: (score["mean_regret"], score["step"]))


def run_policy_batch(env, agent, cache, init, noise, seed):
    """Run fixed episodes with argmax actions and no replacement of finished lanes."""
    import torch as t

    obs = env.reset(
        seed, True,
        candidate_cache=cache,
        init_indices=init,
        observation_noise=noise,
        auto_reset=False,
    )
    rows = [None] * env.num_batches

    for _ in range(env.max_acquisitions):
        if bool(env.done.all()):
            break

        active = ~env.done
        require(bool(t.isfinite(obs[active]).all()), "Non-finite policy observation.")
        with t.no_grad():
            logits = agent.get_logits(obs, action_mask=env.get_action_mask())
            require(bool(t.isfinite(logits[active]).all()), "Non-finite action scores.")
            actions = logits.argmax(dim=1)

        # GP fitting inside env.step requires gradients. Only policy inference
        # belongs inside the no_grad block above.
        obs, _, terminal, infos = env.step(actions)

        for lane in t.where(terminal)[0].tolist():
            require(rows[lane] is None, "A finished validation lane was reused.")
            info = infos["final_info"][lane]
            row = {
                key: info[key]
                for key in (
                    "regret", "best_value", "best_oracle_value", "ground_truth",
                    "budget_used", "remaining_budget", "budget_overshoot",
                )
            }
            row["episode_length"] = info["episode"]["l"]
            row["return"] = info["episode"]["r"]
            require(
                all(math.isfinite(float(value)) for value in row.values()),
                "A validation episode produced a non-finite result.",
            )
            rows[lane] = row

    require(all(row is not None for row in rows), "Some validation episodes did not finish.")
    return rows


def evaluate_checkpoint(protocol, run, checkpoint, output):
    """Worker function: load one checkpoint and evaluate all validation instances.

    This function runs in its own process so its GPU memory is released before
    the controller starts another training or evaluation job.
    """
    import numpy as np
    import torch as t

    from reinforcement_learning.agents.neural_AF import Agent
    from reinforcement_learning.ppo_no_gym import make_env
    from utils.seeding import seed_everything

    require(t.cuda.is_available(), "Checkpoint evaluation requires CUDA.")
    start = time.perf_counter()
    validation = protocol["validation"]
    seed_everything(validation["seed"], deterministic=True)

    bank = read_bank(protocol)
    cfg = read_json(run["config"])
    cfg["problem_family"]["instances"] = list(validation["instances"])
    cfg["ppo"]["num_envs"] = validation["batch_size"]

    device = t.device("cuda")
    architecture = cfg["agent"]
    agent = Agent(
        7, 7, 1, 1,
        architecture["num_layers"],
        architecture["layer_size"],
        architecture["critic_feature_mode"],
    ).to(device)

    weights = t.load(checkpoint["path"], map_location=device, weights_only=True)
    require(
        all(bool(t.isfinite(value).all()) for value in weights.values()),
        "The checkpoint contains non-finite weights.",
    )
    agent.load_state_dict(weights)
    agent.eval()

    rows = []
    episodes = validation["episodes_per_instance"]

    # Keep one instance per batch, as in GP selection. This also preserves the
    # grouping used by the batched GP fitting procedure.
    for instance in validation["instances"]:
        def tensor(suffix, dtype=t.float32):
            return t.as_tensor(
                bank[f"{instance}_{suffix}"],
                device=device,
                dtype=dtype,
            )

        cache = (
            tensor("X"), tensor("y"), tensor("cost"),
            {"instances": [instance] * episodes},
        )
        env = make_env(cfg, device)
        instance_rows = run_policy_batch(
            env, agent, cache,
            tensor("init", t.long),
            tensor("noise"),
            validation["seed"],
        )
        for episode_id, row in enumerate(instance_rows):
            row.update(
                run_id=run["run_id"],
                variant=run["variant"],
                budget_level=run["budget_level"],
                training_seed=run["training_seed"],
                checkpoint_step=checkpoint["step"],
                instance=instance,
                episode_id=episode_id,
                budget=cfg["env"]["budget"],
                max_acquisitions=cfg["env"]["max_acquisitions"],
            )
        rows.extend(instance_rows)
        print(f"  {instance}: {episodes} paired episodes", flush=True)

    score, instance_means = balanced_score(
        rows, validation["instances"], episodes
    )
    t.cuda.synchronize()

    output = Path(output)
    episode_path = output / "episodes.csv"
    write_csv(episode_path, rows)
    write_json(
        output / "summary.json",
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "config_sha256": run["config_sha256"],
            "checkpoint": checkpoint,
            "episode_bank_sha256": validation["episode_bank_sha256"],
            "episodes_csv": episode_path.as_posix(),
            "episodes_sha256": digest(episode_path),
            "mean_regret": score,
            "instance_means": instance_means,
            "n_episodes": len(rows),
            "worker_seconds": time.perf_counter() - start,
            "device_name": t.cuda.get_device_name(device),
            "torch_version": str(t.__version__),
            "numpy_version": str(np.__version__),
        },
    )


def checked_evaluation(protocol, run, checkpoint, output):
    """Verify a saved evaluation before using its score or timing."""
    result = read_json(Path(output) / "summary.json")
    validation = protocol["validation"]

    require(
        result["protocol_sha256"] == protocol["protocol_sha256"]
        and result["config_sha256"] == run["config_sha256"]
        and result["checkpoint"] == checkpoint,
        "The saved evaluation belongs to a different experiment or checkpoint.",
    )
    require(
        result["episode_bank_sha256"] == validation["episode_bank_sha256"],
        "The evaluation used a different validation bank.",
    )
    require(
        digest(result["episodes_csv"]) == result["episodes_sha256"],
        "The saved episode results have changed.",
    )

    with Path(result["episodes_csv"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    score, means = balanced_score(
        rows,
        validation["instances"],
        validation["episodes_per_instance"],
    )
    require(
        score == result["mean_regret"] and means == result["instance_means"],
        "The saved validation summary disagrees with its episode results.",
    )
    return result


def run_evaluation(protocol, run, checkpoint, output):
    """Start an isolated evaluation process, or reuse its completed result."""
    output = Path(output)
    summary_path = output / "summary.json"

    if not summary_path.exists():
        start = time.perf_counter()
        worker = mp.get_context("spawn").Process(
            target=evaluate_checkpoint,
            args=(protocol, run, checkpoint, output.as_posix()),
        )
        worker.start()
        try:
            worker.join()
        except KeyboardInterrupt:
            worker.terminate()
            worker.join()
            worker.close()
            raise

        exit_code = worker.exitcode
        worker.close()
        if exit_code != 0:
            raise RuntimeError(
                f"Checkpoint evaluation failed for {run['run_id']} "
                f"at step {checkpoint['step']}."
            )

        result = checked_evaluation(protocol, run, checkpoint, output)
        result["process_seconds"] = time.perf_counter() - start
        write_json(summary_path, result)

    return checked_evaluation(protocol, run, checkpoint, output)


# ---------------------------------------------------------------------------
# 7. Validate the scheduled checkpoints and retain every seed
# ---------------------------------------------------------------------------

def validate(run_id=None):
    protocol = load_protocol()
    runs = select_runs(protocol["runs"], run_id)
    root = Path(protocol["results_root"])

    # Check that the requested training runs finished before starting GPU work.
    reports = {
        run["run_id"]: checked_training_report(
            root / "train" / run["run_id"] / "training.json", run
        )
        for run in runs
    }

    for run in runs:
        checkpoints = [
            checkpoint
            for checkpoint in reports[run["run_id"]]["checkpoints"]
            if checkpoint["kind"] == "periodic"
        ]
        require(
            sorted(checkpoint["step"] for checkpoint in checkpoints)
            == protocol["checkpoint_steps"],
            f"Scheduled checkpoints are missing for {run['run_id']}.",
        )

        # model_final duplicates the last periodic checkpoint. Evaluate that
        # training position once, using the periodic file.
        scores = []
        for checkpoint in sorted(checkpoints, key=lambda item: item["step"]):
            print(
                f"Validating {run['run_id']}: step {checkpoint['step']}",
                flush=True,
            )
            output = (
                root / "validation" / run["run_id"]
                / f"step_{checkpoint['step']}"
            )
            result = run_evaluation(protocol, run, checkpoint, output)
            scores.append({
                "step": checkpoint["step"],
                "mean_regret": result["mean_regret"],
                "instance_means": result["instance_means"],
                "checkpoint": checkpoint,
                "episodes_csv": result["episodes_csv"],
            })

        winner = choose_checkpoint(scores, protocol["checkpoint_steps"])
        write_json(
            root / "validation" / run["run_id"] / "selected.json",
            {
                "protocol_sha256": protocol["protocol_sha256"],
                **run,
                "scores": scores,
                "selected": winner,
            },
        )
        print(
            f"Selected {run['run_id']}: step {winner['step']}; "
            f"validation regret {winner['mean_regret']:.8f}"
        )

    combine_selections(protocol)


def combine_selections(protocol):
    """Write the final manifest once all 27 runs have a selected checkpoint."""
    root = Path(protocol["results_root"])
    winners = []
    score_rows = []

    for run in protocol["runs"]:
        path = root / "validation" / run["run_id"] / "selected.json"
        if not path.exists():
            print("Some runs still need validation; the combined selection is not ready.")
            return

        result = read_json(path)
        require(
            result["protocol_sha256"] == protocol["protocol_sha256"],
            "Cannot combine selections from different experiments.",
        )
        winner = choose_checkpoint(result["scores"], protocol["checkpoint_steps"])
        require(
            winner == result["selected"],
            f"The stored selection is inconsistent for {run['run_id']}.",
        )

        winners.append({
            "run_id": run["run_id"],
            "variant": run["variant"],
            "budget_level": run["budget_level"],
            "training_seed": run["training_seed"],
            **winner,
        })
        for score in result["scores"]:
            score_rows.append({
                "run_id": run["run_id"],
                "variant": run["variant"],
                "budget_level": run["budget_level"],
                "training_seed": run["training_seed"],
                "checkpoint_step": score["step"],
                "mean_regret": score["mean_regret"],
                "selected": int(score["step"] == winner["step"]),
            })

    write_csv(root / "checkpoint_scores.csv", score_rows)
    write_json(
        root / "selected_checkpoints.json",
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "selection_rule": protocol["selection_rule"],
            "selected_checkpoints": winners,
        },
    )
    print(f"Selected one checkpoint for every run: {root / 'selected_checkpoints.json'}")


# ---------------------------------------------------------------------------
# 8. Measure short runs and estimate the full experiment's duration
# ---------------------------------------------------------------------------

def timing(run_id=None):
    protocol = load_protocol()
    root = Path(protocol["results_root"]) / "timing"

    for run in select_runs(protocol["timing_runs"], run_id):
        output = root / run["run_id"]
        report, training_seconds = train_run(protocol, run, output)
        checkpoint = next(
            item for item in report["checkpoints"] if item["kind"] == "final"
        )
        result = run_evaluation(
            protocol, run, checkpoint,
            output / f"step_{checkpoint['step']}",
        )
        validation_seconds = result.get("process_seconds", result["worker_seconds"])
        print(
            f"Timing complete: {run['run_id']}; "
            f"training {training_seconds / 60:.1f} min; "
            f"validation {validation_seconds / 60:.1f} min."
        )

    write_timing_estimate(protocol)


def write_timing_estimate(protocol):
    """Extrapolate measured update times, keeping startup and validation separate."""
    root = Path(protocol["results_root"])
    rows = []

    for run in protocol["timing_runs"]:
        output = root / "timing" / run["run_id"]
        report_path = output / "training.json"
        evaluation_path = output / f"step_{TIMING_STEPS}"

        if (
            not report_path.exists()
            or not (evaluation_path / "summary.json").exists()
        ):
            print("Timing is incomplete. Finish all nine cases for the combined estimate.")
            return

        report = checked_training_report(report_path, run)
        job = read_json(output / "job.json")
        checkpoint = next(
            item for item in report["checkpoints"] if item["kind"] == "final"
        )
        result = checked_evaluation(
            protocol, run, checkpoint, evaluation_path
        )

        measured = report["iteration_seconds"][WARMUP_UPDATES:]
        require(
            len(measured) == MEASURED_UPDATES
            and all(math.isfinite(value) and value > 0 for value in measured),
            f"Invalid timing measurements for {run['run_id']}.",
        )
        mean_update = math.fsum(measured) / len(measured)
        remaining_updates = TOTAL_STEPS // ROLLOUT_SIZE - WARMUP_UPDATES

        # Import/startup time lies outside the trainer's internal timer.
        # Keep the measured warmup updates, then extrapolate the remaining ones.
        startup = max(
            0.0,
            job.get("process_seconds", report["total_seconds"])
            - report["total_seconds"],
        )
        fixed_training_time = (
            startup
            + report["setup_seconds"]
            + math.fsum(report["iteration_seconds"][:WARMUP_UPDATES])
            + report["finalize_seconds"]
            + len(protocol["checkpoint_steps"]) * report["final_save_seconds"]
        )

        validation_seconds = result.get("process_seconds", result["worker_seconds"])
        validation_per_seed = len(protocol["checkpoint_steps"]) * validation_seconds

        def estimate(update_seconds):
            training_seconds = (
                fixed_training_time + remaining_updates * update_seconds
            )
            return len(SEEDS) * (training_seconds + validation_per_seed)

        rows.append({
            "variant": run["variant"],
            "budget_level": run["budget_level"],
            "mean_update_seconds": mean_update,
            "min_update_seconds": min(measured),
            "max_update_seconds": max(measured),
            "estimated_training_seconds_per_seed": (
                fixed_training_time + remaining_updates * mean_update
            ),
            "validation_seconds_per_checkpoint": validation_seconds,
            "estimated_seconds_three_seeds": estimate(mean_update),
            "faster_observed_rate_seconds": estimate(min(measured)),
            "slower_observed_rate_seconds": estimate(max(measured)),
            "device_name": report["device_name"],
        })

    require(
        len({row["device_name"] for row in rows}) == 1,
        "Timing cases used different GPU models.",
    )

    estimate = {
        "protocol_sha256": protocol["protocol_sha256"],
        "estimated_total_hours": (
            math.fsum(row["estimated_seconds_three_seeds"] for row in rows) / 3600
        ),
        "faster_observed_rate_hours": (
            math.fsum(row["faster_observed_rate_seconds"] for row in rows) / 3600
        ),
        "slower_observed_rate_hours": (
            math.fsum(row["slower_observed_rate_seconds"] for row in rows) / 3600
        ),
        "scope": (
            "27 training runs and 108 paired checkpoint evaluations, "
            "executed sequentially on one GPU."
        ),
        "limitation": (
            "A planning estimate, not a confidence interval. "
            "Later policies can change episode lengths and runtime. "
            "Timing runs do not count as final training seeds."
        ),
        "cases": rows,
    }
    write_csv(root / "timing_estimate.csv", rows)
    write_json(root / "timing_estimate.json", estimate)

    print(
        "Estimated full training + checkpoint validation: "
        f"{estimate['estimated_total_hours']:.2f} hours."
    )
    print(
        "Planning range from observed update speeds: "
        f"{estimate['faster_observed_rate_hours']:.2f} to "
        f"{estimate['slower_observed_rate_hours']:.2f} hours "
        "(not a confidence interval)."
    )


# ---------------------------------------------------------------------------
# 9. Command-line entry point
# ---------------------------------------------------------------------------

def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)

    # Each command receives only arguments it actually uses.
    commands.add_parser("prepare", help="Generate configs and the protocol.").set_defaults(
        handler=prepare
    )
    commands.add_parser("check", help="Check the prepared experiment.").set_defaults(
        handler=check
    )

    for name, handler in (
        ("timing", timing),
        ("train", train),
        ("validate", validate),
    ):
        command = commands.add_parser(name)
        command.add_argument(
            "--run-id",
            help="Run only this case; omit to process the full group.",
        )
        command.set_defaults(handler=handler)

    arguments = vars(parser.parse_args())
    handler = arguments.pop("handler")
    handler(**arguments)


if __name__ == "__main__":
    main()