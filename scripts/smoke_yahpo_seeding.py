"""Small CPU check of initial-cache and lane-reset sampling; no PPO or GP fit."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from bo.problems.yahpo_lcbench import YAHPOLCBenchProblemFamily


def draw_sequence(seed):
    family = YAHPOLCBenchProblemFamily(
        device=torch.device("cpu"), dtype=torch.float32,
        instances=["3945", "7593"],
    )
    # seed=None is the call used by BO_env._reset_lanes.
    return [
        family.build_candidate_cache(B=2, n_candidates=8, seed=seed),
        family.build_candidate_cache(B=2, n_candidates=8, seed=None),
    ]


def main():
    first, repeat, different = draw_sequence(1), draw_sequence(1), draw_sequence(2)
    for original, replay in zip(first, repeat):
        assert original[0].shape == (2, 8, 7)
        assert original[1].shape == original[2].shape == (2, 8)
        for a, b in zip(original[:3], replay[:3]):
            assert torch.isfinite(a).all()
            assert torch.equal(a, b), "Same-seed cache replay differs"
        assert original[3] == replay[3], "Instances or raw configs differ"
    assert not torch.equal(first[0][0], first[1][0]), "Lane reset reused the initial cache"
    assert not torch.equal(first[0][0], different[0][0]), "Different seed reused the cache"
    print("PASS: same-seed initial/reset caches match; reset advances; different seed differs.")
    print("Shapes: X=(2, 8, 7), y=(2, 8), cost=(2, 8).")


if __name__ == "__main__":
    main()