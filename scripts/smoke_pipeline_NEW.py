"""Check the NEW GP / selection / training / testing pipeline on synthetic data.

No YAHPO installation or benchmark download is needed. Temporary experiments
are removed automatically; production selections and checkpoints are untouched.
"""
from __future__ import annotations
import argparse
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch as t
from bo.gaussian_process_batch_NEW import RepeatedPadGPWrapper, FITTING_RECIPE
from reinforcement_learning.ppo_no_gym_NEW import make_env
from reinforcement_learning.agents.neural_AF import Agent
from scripts import select_gp_NEW as selection, TRAIN_AGENTS_NEW as training, TEST_AGENT_NEW as testing


def must_reject(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError('Invalid input was accepted.')


def gp_checks(device):
    t.manual_seed(8)
    X = t.rand((2, 64, 7), device=device)
    y = 0.7 + 0.09 * X[..., 0]
    for kernel in ('rbf', 'matern52', 'rbf_matern52_add', 'rbf_matern52_mul'):
        for mean in ('constant', 'linear'):
            predictions = []
            for shift in (0., 5.):
                gp = RepeatedPadGPWrapper(device, t.float32, 2, 64, 7, lr=.01,
                                         kernel=kernel, mean=mean, record_diagnostics=True)
                gp.train_x[:] = X
                gp.train_y[:] = y + shift
                gp.t_idx[:] = t.tensor([3, 5], device=device)
                raw = gp.train_y.clone()
                offset, variance, noise = gp._target_statistics()
                assert t.allclose(offset, t.stack([raw[0, :3].mean(), raw[1, :5].mean()]))
                assert t.allclose(variance, t.stack([raw[0, :3].var(), raw[1, :5].var()]), atol=1e-7)
                assert t.allclose(noise, (.02 * variance).clamp_min(1.01e-4))
                _, padded = gp._repeat_pad_prefixes()
                assert t.equal(padded[0], raw[0, t.tensor([0, 1, 2, 0, 1], device=device)])
                pred = gp.predict(X[:, 8:12])
                assert t.equal(raw, gp.train_y)
                assert bool((pred.variance >= gp.last_fit_parameters['likelihood.noise'] - 1e-6).all())
                repeated = gp.predict(X[:, 8:12])
                assert t.allclose(repeated.mean, pred.mean) and t.allclose(repeated.variance, pred.variance)
                predictions.append((pred.mean - shift, pred.variance))
            assert t.allclose(predictions[0][0], predictions[1][0], atol=2e-5)
            assert t.allclose(predictions[0][1], predictions[1][1], atol=2e-5)
            # Unequal long histories exercise the production padding path.
            for target in (y, 5. + 2. * X[..., 0]):
                gp = RepeatedPadGPWrapper(device, t.float32, 2, 64, 7, lr=.1, kernel=kernel, mean=mean)
                gp.train_x[:] = X
                gp.train_y[:] = target
                gp.t_idx[:] = t.tensor([3, 63], device=device)
                pred = gp.predict(X[:, -1:])
                assert bool(t.isfinite(pred.mean).all() & t.isfinite(pred.variance).all())
    print('PASS: all eight kernel/mean structures; centering, noise, padding, fresh fits, likelihood variance, 3-vs-63 histories.')


def synthetic_batch(device):
    t.manual_seed(42)
    X = t.rand((2, 8, 7), device=device)
    y = .2 + .6 * X[..., 0]
    costs = t.ones((2, 8), device=device)
    costs[1] *= 2
    init = t.tensor([[0, 1, 2], [1, 3, 5]], device=device)
    noise = t.randn((2, 6), device=device)
    order = t.tensor([[3, 4, 5], [0, 2, 4]], device=device)
    return (X, y, costs, {'instances': ['a', 'b']}), init, noise, order


def selection_checks(device):
    cache, init, noise, order = synthetic_batch(device)
    env_cfg = dict(objective_noise_std=.01, objective_noise_clip=True)
    settings = [dict(kernel='rbf', mean='constant', lr=.01, training_iter=10),
                dict(kernel='matern52', mean='linear', lr=.1, training_iter=10)]
    for stage in ('objective', 'cost'):
        results = [selection.run_batch(cache, init, noise, order, s, stage, 2.5, env_cfg) for s in settings]
        for rows in results:
            assert [r['episode_length'] for r in rows] == [3, 2]
            assert [r['budget_used'] for r in rows] == [3., 4.]
            for lane, row in enumerate(rows):
                actions = json.loads(row['actions'])
                assert len(set(actions)) == len(actions) and not set(actions) & set(init[lane].tolist())
        assert [r['actions'] for r in results[0]] == [r['actions'] for r in results[1]]
        X, y, costs, _ = cache
        lane = t.arange(2, device=device)
        initial = (y[lane[:, None], init] + .01 * noise[:, :3]).clamp(0, 1)
        target = (y[lane, order[:, 0]] + .01 * noise[:, 3]).clamp(0, 1)
        if stage == 'cost':
            initial, target = t.log1p(costs[lane[:, None], init]), t.log1p(costs[lane, order[:, 0]])
        gp = RepeatedPadGPWrapper(device, t.float32, 2, 6, 7, **settings[0])
        gp.set_lane_data(t.ones(2, device=device, dtype=t.bool), X[lane[:, None], init], initial)
        pred = gp.predict(X[lane, order[:, 0]][:, None])
        nlpd = selection.gaussian_nlpd(target.double(), pred.mean[:, 0].double(), pred.variance[:, 0].double())
        assert np.allclose([r['initial_nlpd'] for r in results[0]], nlpd.cpu().numpy())
    rows = []
    for setting, nlpd, rmse in zip(settings, [.1, .100000001], [2., .01]):
        for instance in ('a', 'b'):
            rows.append(dict(setting=selection.setting_id(setting), instance=instance, episode_id=0,
                budget_level='low', nlpd=nlpd, rmse=rmse, initial_nlpd=nlpd, initial_abs_error=rmse,
                episode_length=1, budget_overshoot=0, cap_only=0))
    budgets = [dict(level='low', budget=1)]
    _, winner = selection.summarize(rows, settings, set(), budgets, ['a', 'b'], {'episodes_per_instance': 1})
    assert winner == selection.setting_id(settings[0])  # Lower RMSE must not override NLPD.
    must_reject(selection.summarize, rows[:-1], settings, set(), budgets, ['a', 'b'], {'episodes_per_instance': 1})
    print('PASS: identical random trajectories; no replacement; warm-only first prediction; NLPD selection; missing episodes rejected.')


def policy_checks(device):
    cache, init, noise, _ = synthetic_batch(device)
    cfg = training.read_json(training.TEMPLATE)
    cfg['ppo']['num_envs'] = 2
    cfg['env'].update(n_candidates=8, budget=2.5, max_acquisitions=3)
    cfg['gp'] = dict(kernel='rbf', mean='constant', lr=.01, training_iter=10)
    cfg['cost_model'] = dict(type='gp', target_transform='log1p', gp=cfg['gp'])
    architecture = cfg['agent']
    agent = Agent(7, 7, 1, 1, architecture['num_layers'], architecture['layer_size'],
                  architecture['critic_feature_mode']).to(device).eval()
    for variant in training.VARIANTS:
        cfg['env']['cost_feature_mode'] = 'none' if variant == 'no_cost' else 'predicted'
        cfg['env']['use_cost_uncertainty_feature'] = variant == 'cost_uncertainty'
        cfg['cost_model']['type'] = 'known' if variant == 'no_cost' else 'gp'
        env = make_env(cfg, device)
        first = training.run_policy_batch(env, agent, cache, init, noise, 1)
        second = testing.run_batch(make_env(cfg, device), cache, init, noise, 1, variant, agent)
        assert first == second and [r['episode_length'] for r in first] == [3, 2]
    for method in testing.BASELINES:
        uniforms = t.as_tensor(testing.random_uniforms(1, '42', 0, 2, 3), device=device)
        first = testing.run_batch(make_env(cfg, device), cache, init, noise, 1, method, uniforms=uniforms)
        assert [r['episode_length'] for r in first] == [3, 2]
    values = [dict(regret=0., episode_length=1, budget_overshoot=0.),
              dict(regret=.2, episode_length=3, budget_overshoot=2.)]
    summary = testing.aggregate(values, 1e-3)
    assert math.isclose(summary['sd_final_simple_regret'], math.sqrt(.02))
    assert summary['success_rate'] == .5
    print('PASS: three policy variants; identical training/test evaluation; four baselines; mean/sample SD.')
    return cfg


def small_training(cfg, directory, device):
    cfg = copy.deepcopy(cfg)
    cfg['experiment'].update(cuda=device.type == 'cuda', save_every=16,
                             run_dir=str(directory / 'runs'), checkpoint_dir=str(directory / 'checkpoints'))
    cfg['ppo'].update(total_timesteps=32, num_envs=4, num_steps=4, num_minibatches=2, update_epochs=1)
    cfg['env'].update(n_candidates=8, max_acquisitions=3, budget=2.5, device=str(device))
    cfg['problem_family'] = dict(type='toy_rbf', lb=[0.], ub=[1.], M=2, min_cost=1.)
    cfg['candidate_set'] = dict(type='jittered_grid', res=8, D=1, minimum=0., maximum=1.)
    config_path, report_path = directory / 'tiny_config.json', directory / 'tiny_training.json'
    training.write_json(config_path, cfg)
    command = [sys.executable, '-u', 'reinforcement_learning/ppo_no_gym_NEW.py', '--config', str(config_path),
               '--report', str(report_path)]
    if device.type == 'cuda':
        command.append('--require-cuda')
    with (directory / 'training.log').open('w') as stream:
        subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    report = training.read_json(report_path)
    assert report['global_step'] == 32 and len(report['iteration_seconds']) == 2
    assert [c['step'] for c in report['checkpoints'] if c['kind'] == 'periodic'] == [16, 32]
    checkpoint = report['checkpoints'][-1]
    weights = t.load(checkpoint['path'], weights_only=True, map_location='cpu')
    assert all(bool(t.isfinite(value).all()) for value in weights.values())
    assert training.digest(checkpoint['path']) == checkpoint['sha256']
    print('PASS: two actual PPO updates; finite saved weights; checkpoint/report hashes.')
    return checkpoint


def write_synthetic_bank(phase, cfg, instances, cost_scale):
    rng = np.random.default_rng(83)
    bank = {}
    n = phase['episodes_per_instance']
    for instance in instances:
        X = rng.random((n, 64, 7), dtype=np.float32)
        bank[f'{instance}_X'] = X
        bank[f'{instance}_y'] = .2 + .6 * X[..., 0]
        bank[f'{instance}_cost'] = np.full((n, 64), cost_scale, dtype=np.float32)
        bank[f'{instance}_init'] = np.tile(np.arange(3, dtype=np.int64), (n, 1))
        bank[f'{instance}_noise'] = rng.standard_normal((n, 64)).astype(np.float32)
    path = Path(phase['episode_bank'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        np.savez_compressed(stream, signature=json.dumps(selection.bank_signature(phase, cfg), sort_keys=True), **bank)
    return bank, selection.digest(path)


def handoff_checks(directory, checkpoint):
    phase = training.read_json(selection.PHASE_CONFIG)
    cfg = training.read_json(phase['base_config'])
    budgets = training.read_json(phase['budgets_file'])['budgets']
    phase.update(device='cpu', kernels=['rbf', 'matern52'], means=['constant'], learning_rates=[.01],
                 episode_bank=str(directory / 'validation_bank.npz'), random_orders=str(directory / 'orders.npz'))
    phase_path = directory / 'phase.json'
    training.write_json(phase_path, phase)
    bank, bank_hash = write_synthetic_bank(phase, cfg, training.VALIDATION, 2 * max(b['budget'] for b in budgets))
    orders, orders_hash = selection.load_or_make_orders(phase, cfg, training.VALIDATION, bank, bank_hash)
    selection_root = str(directory / 'selection')
    with patch.object(selection, 'RESULTS_ROOT', selection_root), redirect_stdout(io.StringIO()):
        for stage in ('objective', 'cost'):
            selection.run_stage(stage, phase, cfg, budgets, training.VALIDATION,
                                selection.candidate_settings(phase), bank, bank_hash, orders, orders_hash)
    patches = dict(PHASE2=str(phase_path), OBJECTIVE=selection_root + '/objective/selected.json',
                   COST=selection_root + '/cost/selected.json', DEFAULT_CONFIG_DIR=str(directory / 'configs'),
                   DEFAULT_PROTOCOL=str(directory / 'configs/protocol.json'), RESULTS_ROOT=str(directory / 'train'))
    with patch.multiple(training, **patches), redirect_stdout(io.StringIO()):
        training.prepare()
        protocol = training.load_protocol()
        assert len(protocol['runs']) == 27 and len(protocol['timing_runs']) == 9
        settings = training.load_experiment_settings()
        selected_entries = []
        for run in protocol['runs']:
            run_cfg = training.read_json(run['config'])
            assert run_cfg['gp'] == settings['objective_gp']
            assert run_cfg['gp_fitting_recipe'] == FITTING_RECIPE
            if run['variant'] != 'no_cost':
                assert run_cfg['cost_model']['gp'] == settings['cost_model']['gp']
            # Isolated fixture only: reuse tiny weights to check manifest wiring.
            scores = [dict(step=step, mean_regret=.1 + index, instance_means={},
                           checkpoint={**checkpoint, 'step': step}, episodes_csv='fixture')
                      for index, step in enumerate(protocol['checkpoint_steps'])]
            winner = training.choose_checkpoint(scores, protocol['checkpoint_steps'])
            training.write_json(Path(protocol['results_root']) / 'validation' / run['run_id'] / 'selected.json',
                                dict(protocol_sha256=protocol['protocol_sha256'], scores=scores, selected=winner))
            selected_entries.append(dict(run_id=run['run_id'], variant=run['variant'], budget_level=run['budget_level'],
                                         training_seed=run['training_seed'], **winner))
        training.write_json(Path(protocol['results_root']) / 'selected_checkpoints.json',
                            dict(protocol_sha256=protocol['protocol_sha256'], selected_checkpoints=selected_entries))
        test_root = directory / 'testing'
        # Prepare creates the bank. Supply synthetic arrays through its existing maker.
        def make_test_bank(test_phase, test_cfg, instances):
            return write_synthetic_bank(test_phase, test_cfg, instances, 2 * max(b['budget'] for b in budgets))
        with patch.object(testing, 'RESULTS_ROOT', str(test_root)), patch.object(testing, 'EPISODES', 4), \
                patch.object(testing, 'BATCH_SIZE', 2), patch.object(selection, 'load_or_make_bank', make_test_bank):
            testing.prepare()
            test_protocol = testing.load_protocol()
            assert len(test_protocol['jobs']) == 93
            assert len(testing.jobs_for(test_protocol, 'matched_budgets')) == 39
            assert len(testing.jobs_for(test_protocol, 'budget_transfer')) == 66
            for method in ('no_cost', 'eipu'):
                job = next(j for j in test_protocol['jobs'] if j['method'] == method)
                testing.evaluate_job(test_protocol, job, device_name='cpu')
                assert len(testing.checked_job(test_protocol, job)) == 24
    print('PASS: actual selection artifacts; all 27 configs; test manifest; learned-policy/baseline test workers.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    os.chdir(ROOT)
    device = t.device(args.device)
    if device.type == 'cuda' and not t.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    t.set_num_threads(1)
    gp_checks(device)
    selection_checks(device)
    cfg = policy_checks(device)
    with tempfile.TemporaryDirectory(prefix='gp_NEW_smoke_', dir=ROOT) as temporary:
        directory = Path(temporary)
        checkpoint = small_training(cfg, directory, device)
        handoff_checks(directory, checkpoint)
    print('ALL NEW PIPELINE CHECKS PASSED.')


if __name__ == '__main__':
    main()