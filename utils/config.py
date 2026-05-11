from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def load_config(path):
    '''Load config through path'''
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    validate_config(cfg)
    add_derived_ppo_values(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]):
    '''validate config to ensure all params are present'''
    required_sections = [
        "experiment",
        "ppo",
        "agent",
        "env",
        "candidate_set",
        "problem_family",
        "gp",
    ]

    for section in required_sections:
        if section not in cfg:
            raise KeyError(f"Missing config section: {section}")

    ppo = cfg["ppo"]
    if ppo["num_envs"] <= 0:
        raise ValueError("ppo.num_envs must be positive")
    if ppo["num_steps"] <= 0:
        raise ValueError("ppo.num_steps must be positive")
    if ppo["num_minibatches"] <= 0:
        raise ValueError("ppo.num_minibatches must be positive")


def add_derived_ppo_values(cfg: dict[str, Any]) -> dict[str, Any]:
    ppo = cfg["ppo"]

    batch_size = int(ppo["num_envs"] * ppo["num_steps"])
    minibatch_size = int(batch_size // ppo["num_minibatches"])
    num_iterations = int(ppo["total_timesteps"] // batch_size)

    ppo["batch_size"] = batch_size
    ppo["minibatch_size"] = minibatch_size
    ppo["num_iterations"] = num_iterations

    return cfg


def save_config_copy(config_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.json")