"""Check Phase 4 selection logic and the actual policy/GP path on synthetic data.

--protocol-only needs NumPy, but neither PyTorch nor benchmark data.
The full smoke test also runs two tiny PPO updates; it never loads YAHPO data.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.TRAIN_AGENTS import (CHECKPOINT_STEPS, DEFAULT_PROTOCOL, SEEDS, TRAIN, VALIDATION,
                           VARIANTS, balanced_score, choose_checkpoint, digest, load_protocol,
                           read_json, run_policy_batch, write_json)


def must_reject(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError('Invalid input was accepted.')


def check_protocol(protocol):
    combinations = {(r['variant'], r['budget_level'], r['training_seed']) for r in protocol['runs']}
    assert combinations == {(v, b, s) for v in VARIANTS for b in ('low', 'medium', 'high') for s in SEEDS}
    for run in protocol['runs']:
        cfg = read_json(run['config'])
        assert cfg['problem_family']['instances'] == TRAIN
        assert cfg['env']['max_acquisitions'] == 61 and cfg['env']['n_init'] == 3
        assert cfg['ppo']['total_timesteps'] == CHECKPOINT_STEPS[-1]
        assert cfg['gp']['lr'] == 0.1 and cfg['gp']['mean'] == 'constant'
        if run['variant'] == 'no_cost':
            assert cfg['env']['cost_feature_mode'] == 'none' and cfg['cost_model']['type'] == 'known'
        else:
            assert cfg['cost_model']['gp']['lr'] == 0.01 and cfg['cost_model']['gp']['mean'] == 'linear'
        assert cfg['env']['use_cost_uncertainty_feature'] == (run['variant'] == 'cost_uncertainty')
    rows = [dict(instance=i, episode_id=e, regret=r) for i, r in [('a', 0.1), ('b', 0.5)] for e in range(2)]
    score, means = balanced_score(list(reversed(rows)), ['a', 'b'], 2)
    assert math.isclose(score, 0.3) and means == {'a': 0.1, 'b': 0.5}
    must_reject(balanced_score, rows[:-1], ['a', 'b'], 2)
    must_reject(balanced_score, rows + rows[:1], ['a', 'b'], 2)
    for bad in (float('nan'), float('inf'), -0.01):
        corrupt = copy.deepcopy(rows)
        corrupt[0]['regret'] = bad
        must_reject(balanced_score, corrupt, ['a', 'b'], 2)
    scores = [dict(step=s, mean_regret=v) for s, v in zip(CHECKPOINT_STEPS, (0.2, 0.100000004, 0.100000003, 0.3))]
    assert choose_checkpoint(scores, CHECKPOINT_STEPS)['step'] == 786432  # Do not round first.
    scores[1]['mean_regret'] = scores[2]['mean_regret']
    assert choose_checkpoint(list(reversed(scores)), CHECKPOINT_STEPS)['step'] == 524288
    must_reject(choose_checkpoint, scores[:-1], CHECKPOINT_STEPS)
    print('PASS: all 27 configs; correct GP/feature settings; balanced scores; unrounded selection; earlier exact ties; invalid/incomplete results rejected.')


def check_gpu_path(protocol, device_name):
    import torch as t
    from reinforcement_learning.agents.neural_AF import Agent
    from reinforcement_learning.ppo_no_gym import make_env

    if device_name == 'cuda' and not t.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    device = t.device(device_name)
    t.manual_seed(42)
    X = t.rand((2, 8, 7), device=device)
    y = 0.2 + 0.6 * X[..., 0]
    costs = t.ones((2, 8), device=device)
    costs[1] *= 2
    cache = X, y, costs, {'instances': ['synthetic_a', 'synthetic_b']}
    init = t.tensor([[0, 1, 2], [1, 3, 5]], device=device)
    noise = t.randn((2, 6), device=device)
    expected = (y[t.arange(2, device=device)[:, None], init] + 0.01 * noise[:, :3]).clamp(0, 1)
    for variant in VARIANTS:
        run = next(r for r in protocol['runs'] if r['variant'] == variant)
        cfg = read_json(run['config'])
        cfg['ppo']['num_envs'] = 2
        cfg['env'].update(n_candidates=8, budget=2.5, max_acquisitions=3)
        env = make_env(cfg, device)
        obs = env.reset(1, True, candidate_cache=cache, init_indices=init, observation_noise=noise, auto_reset=False)
        assert obs.shape == (2, 8, 7) and t.isfinite(obs).all()
        assert t.equal(env.obj_gp.train_y[:, :3], expected)
        assert env.obj_gp.lr == 0.1
        if variant == 'no_cost':
            assert not env.use_cost_gp and t.count_nonzero(obs[..., [2, 3, 6]]) == 0
        else:
            assert env.cost_gp.lr == 0.01 and env.cost_gp.mean == 'linear'
            assert (obs[..., 2] >= 0).all() and t.count_nonzero(obs[..., 2]) > 0
        if variant == 'cost_uncertainty':
            assert (obs[..., 3] > 0).all()
        else:
            assert t.count_nonzero(obs[..., 3]) == 0
        a = cfg['agent']
        agent = Agent(7, 7, 1, 1, a['num_layers'], a['layer_size'], a['critic_feature_mode']).to(device).eval()
        first = run_policy_batch(env, agent, cache, init, noise, 1)
        t.randn(100, device=device)
        second = run_policy_batch(env, agent, cache, init, noise, 1)
        assert first == second
        assert [r['episode_length'] for r in first] == [3, 2]
        assert [r['budget_used'] for r in first] == [3.0, 4.0]
        assert t.equal(env.X, X) and t.equal(env.y_grid, y)
    print('PASS: all three real policy/GP paths; shared starts/noise; repeatable argmax; no lane replacement; actual cost charged in every variant.')

    # Exercise the modified trainer and its report/checkpoint serialization too.
    cfg = read_json(next(r['config'] for r in protocol['runs'] if r['variant'] == 'cost_uncertainty'))
    with tempfile.TemporaryDirectory(prefix='phase4_smoke_', dir=ROOT) as temporary:
        directory = Path(temporary)
        cfg['experiment'].update(cuda=device_name == 'cuda', save_every=16,
                                 run_dir=str(directory / 'runs'), checkpoint_dir=str(directory / 'checkpoints'))
        cfg['ppo'].update(total_timesteps=32, num_envs=4, num_steps=4, num_minibatches=2, update_epochs=1)
        cfg['env'].update(n_candidates=8, max_acquisitions=3, budget=2.5, device=device_name)
        cfg['problem_family'] = dict(type='toy_rbf', lb=[0.0], ub=[1.0], M=2, min_cost=1.0)
        cfg['candidate_set'] = dict(type='jittered_grid', res=8, D=1, minimum=0.0, maximum=1.0)
        path, report_path = directory / 'config.json', directory / 'training.json'
        write_json(path, cfg)
        command = [sys.executable, '-u', 'reinforcement_learning/ppo_no_gym.py', '--config', str(path), '--report', str(report_path)]
        if device_name == 'cuda':
            command.append('--require-cuda')
        subprocess.run(command, cwd=ROOT, check=True)
        report = read_json(report_path)
        assert report['global_step'] == 32 and len(report['iteration_seconds']) == 2
        assert [c['step'] for c in report['checkpoints'] if c['kind'] == 'periodic'] == [16, 32]
        assert report['config_sha256'] == digest(path)
        for checkpoint in report['checkpoints']:
            assert digest(checkpoint['path']) == checkpoint['sha256']
            weights = t.load(checkpoint['path'], map_location='cpu', weights_only=True)
            assert all(t.isfinite(v).all() for v in weights.values())
        assert all(x > 0 for x in report['iteration_seconds'])
    print('PASS: two tiny PPO updates; finite saved weights; checkpoint hashes; completed-run timing report. No YAHPO data loaded.')


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default=DEFAULT_PROTOCOL)
    parser.add_argument('--protocol-only', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    check_protocol(protocol)
    if not args.protocol_only:
        check_gpu_path(protocol, args.device)


if __name__ == '__main__':
    main()