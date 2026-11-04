"""Check the two test experiments using synthetic data only.

    py scripts/smoke_TEST_AGENT.py

Run in the same Python environment as validation. CUDA is used if available.
No test bank, held-out YAHPO instance, or experiment output is needed or written.
"""
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import TEST_AGENT as testing


def check_design(configs, winners):
    """All three learned seeds use their own selected weights at the target budget."""
    counts = {"matched": 0, "transfer": 0}
    for experiment in counts:
        for target in testing.BUDGETS:
            for method, source in testing.experiment_groups(experiment, target):
                if experiment == "transfer":
                    assert method in testing.COST_AGENTS and source != target
                seeds = testing.training.SEEDS if source is not None else [None]
                for seed in seeds:
                    cfg, checkpoint = testing.setup_case(method, source, target, seed, configs, winners, "cpu")
                    assert cfg["env"]["budget"] == configs[f"no_cost__{target}__seed1"]["env"]["budget"]
                    assert cfg["ppo"]["num_envs"] == 32
                    assert cfg["problem_family"]["instances"] == testing.INSTANCES
                    if seed is not None:
                        assert checkpoint == winners[f"{method}__{source}__seed{seed}"]
                    else:
                        assert checkpoint is None
                    if method in ("eipu", "ei_cool"):
                        assert cfg["cost_model"]["type"] == "gp" and cfg["env"]["cost_feature_mode"] == "predicted"
                    counts[experiment] += 1
    assert counts == {"matched": 39, "transfer": 36}

    spec = testing.bank_settings(configs["cost_mean__low__seed1"])
    seed, starts, noise = testing.episode_draws(testing.INSTANCES[0], spec)
    repeat = testing.episode_draws(testing.INSTANCES[0], spec)
    assert seed == repeat[0] and np.array_equal(starts, repeat[1]) and np.array_equal(noise, repeat[2])
    assert starts.shape == (128, 3) and noise.shape == (128, 64)
    assert np.all(np.diff(np.sort(starts, axis=1), axis=1) > 0)
    assert not np.array_equal(noise, testing.episode_draws(testing.INSTANCES[1], spec)[2])
    print("PASS: selected checkpoints, matched/transfer budgets and reproducible bank draws.")


def check_reports():
    """Make one instance easier than the others, so an accidental global mean is visible."""
    pooled = []
    for seed in (1, 2, 3):
        for i, instance in enumerate(testing.INSTANCES):
            regret = (0.01 if i == 0 else 0.04) + (seed - 1) * 0.01
            for episode in range(128):
                pooled.append({"instance": instance, "episode_id": episode, "regret": regret,
                               "episode_length": i + 1, "budget_overshoot": 0 if i == 0 else 2})
    total, by_instance = testing.summarize_group("cost_mean", "low", "low", pooled)
    assert len(pooled) == 2304 and len(by_instance) == 6
    assert math.isclose(total["mean_final_simple_regret"], 0.045)
    assert math.isclose(by_instance[0]["mean_final_simple_regret"], 0.02)
    assert math.isclose(by_instance[1]["mean_final_simple_regret"], 0.05)
    # Exactly 0.01 counts as success: one seed on one of the six instances.
    assert math.isclose(total["success_rate"], 1 / 18)
    assert total["mean_episode_length"] == 3.5
    assert math.isclose(total["mean_budget_overshoot"], 10 / 6)
    first_instance = [r for r in pooled if r["instance"] == testing.INSTANCES[0]]
    for field, sd in (("regret", "sd_final_simple_regret"), ("episode_length", "sd_episode_length"),
                      ("budget_overshoot", "sd_budget_overshoot")):
        assert math.isclose(total[sd], np.std([r[field] for r in pooled], ddof=1))
        assert math.isclose(by_instance[0][sd], np.std([r[field] for r in first_instance], ddof=1))
    assert set(testing.episode_statistics(pooled)) == {
        "mean_final_simple_regret", "sd_final_simple_regret", "mean_episode_length",
        "sd_episode_length", "mean_budget_overshoot", "sd_budget_overshoot", "success_rate"}
    print("PASS: pooled means and sample SDs, individual-instance results and inclusive success.")


def check_rollouts(configs, winners):
    """Two toy episodes cost 1 and 2 per query; their lengths and overshoots are known."""
    import torch
    from reinforcement_learning.ppo_no_gym import make_env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)
    x = torch.rand(2, 8, 7, device=device)
    y = 0.2 + 0.6 * x[:, :, 0]
    cost = torch.ones_like(y)
    cost[1] = 2
    cache = (x, y, cost, {"instances": ["toy_A", "toy_B"]})
    starts = torch.tensor([[0, 2, 4], [1, 3, 5]], device=device)
    noise = torch.randn(2, 6, device=device)

    cases = testing.experiment_groups("matched", "low") + [("cost_uncertainty", "high")]
    for method, source in cases:
        seed = None if source is None else 1
        cfg, cp = testing.setup_case(method, source, "low", seed, configs, winners, device)
        cfg["env"].update(n_candidates=8, max_acquisitions=3, budget=2.5)
        cfg["ppo"]["num_envs"] = 2
        env = make_env(cfg, device)
        actor = testing.load_agent(cfg, cp, device) if cp is not None else None
        rows = testing.play_batch(env, actor, method, cache, starts, noise, 711)
        assert [r["episode_length"] for r in rows] == [3, 2]
        assert [r["budget_overshoot"] for r in rows] == [0.5, 1.5]
        assert all(math.isfinite(r["regret"]) and r["regret"] >= 0 for r in rows)
        assert rows == testing.play_batch(env, actor, method, cache, starts, noise, 711)
    print(f"PASS: real weights and GP fits, all actors/baselines, transfer and fixed episodes ({device}).")


def main():
    os.chdir(ROOT)
    configs, winners = testing.load_training()
    check_design(configs, winners)
    check_reports()
    check_rollouts(configs, winners)
    print("No held-out experiments were run.")


if __name__ == "__main__":
    main()