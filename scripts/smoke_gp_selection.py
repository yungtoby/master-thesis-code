"""Small synthetic smoke test; requires installed dependencies, not YAHPO data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch as t

from scripts.select_gp import KERNELS, configured_env, load_protocol, run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    if args.device == 'cuda' and not t.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    device = t.device(args.device)
    _, cfg, _, _, settings = load_protocol('configs/gp_hyperparameter_selection/gp_configs.json')
    assert len(settings) == 30
    cfg['env'].update(n_candidates=8, n_init=3, mask_visited_actions=False)
    t.manual_seed(42)
    X = t.rand((2, 8, 7), device=device)
    y = 0.2 + 0.6 * X[..., 0]
    costs = t.ones((2, 8), device=device)
    costs[1] *= 2
    cache = (X, y, costs, {'instances': ['synthetic_a', 'synthetic_b']})
    init = t.tensor([[0, 1, 2], [1, 3, 5]], device=device)
    noise = t.randn((2, 6), device=device)
    expected_warm = (y[t.arange(2, device=device)[:, None], init] + noise[:, :3] * 0.01).clamp(0, 1)
    cfg['env']['objective_noise_std'] = 0.01
    cfg['env']['objective_noise_clip'] = True
    for kernel in KERNELS:
        for mean in ('constant', 'linear'):
            objective_gp = dict(kernel=kernel, mean=mean, lr=0.01, training_iter=10)
            cost_gp = {**objective_gp, 'lr': 0.1}
            env = configured_env(cfg, 'cost', cost_gp, objective_gp, 2, 2.5, 3, device)
            obs = env.reset(1, True, candidate_cache=cache, init_indices=init,
                            observation_noise=noise, auto_reset=False)
            assert obs.shape == (2, 8, 7) and t.isfinite(obs).all()
            assert env.obj_gp.lr == 0.01 and env.cost_gp.lr == 0.1
            assert t.equal(env.obj_gp.train_y[:, :3], expected_warm)
            assert env.obj_gp.last_prediction.mean.shape == (2, 8)
            assert env.cost_gp.last_prediction.mean.shape == (2, 8)
    first = run_batch(env, cache, init, noise, 1, 'eipu')
    t.randn(100, device=device)  # Unrelated RNG draws must not alter the episodes.
    second = run_batch(env, cache, init, noise, 1, 'eipu')
    assert first == second
    assert [r['episode_length'] for r in first] == [3, 2]
    assert t.equal(env.X, X) and t.equal(env.y_grid, y)
    ei_env = configured_env(cfg, 'objective', objective_gp, None, 2, 2.5, 3, device)
    ei_rows = run_batch(ei_env, cache, init, noise, 1, 'ei')
    assert [r['episode_length'] for r in ei_rows] == [3, 2]
    print('PASS: 10 kernel/mean combinations; finite [2, 8] GP predictions; separate GP learning rates.')
    print('PASS: identical paired warm starts/noise; repeated EIpu run matches; EI/EIpu lengths [3, 2].')
    print(f'Device: {device}. GP fitting iterations: 10. No YAHPO data loaded.')


if __name__ == '__main__':
    main()