"""Phase 2: paired EI objective-GP selection, then paired EIpu cost-GP selection.

Run from the repository root. Each candidate has the same fitting budget (10
Adam steps per fit). The frozen budgets come directly from Phase 1's JSON.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch as t

from bo.problems.registry import build_problem_family
from reinforcement_learning.ppo_no_gym import make_env
from scripts.evaluate_baselines import select_baseline_action
from utils.config import load_config
from utils.seeding import seed_everything


# Kernels to try and episode fields
KERNELS = ('rbf', 'matern32', 'matern52', 'rbf_matern52_add', 'rbf_matern52_mul')
EPISODE_FIELDS = [
    'setting', 'stage', 'method', 'kernel', 'mean', 'lr', 'instance',
    'episode_id', 'budget_level', 'budget', 'max_acquisitions', 'regret',
    'best_value', 'best_oracle_value', 'ground_truth', 'return',
    'episode_length', 'budget_used', 'remaining_budget', 'budget_overshoot',
    'cap_only', 'objective_nlpd', 'objective_rmse', 'cost_nlpd_log1p',
    'cost_rmse_log1p', 'objective_parameters', 'cost_parameters',
]


# Read json helper func
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# Get setting id helper func
def setting_id(setting):
    return f"{setting['kernel']}__{setting['mean']}__lr{setting['lr']:g}"


# Return a dict of candidate settings to choose utilizing itertools.prod
def candidate_settings(phase):
    return [
        dict(kernel=k, mean=m, lr=float(lr), training_iter=10)
        for k, m, lr in itertools.product(
            phase['kernels'], phase['means'], phase['learning_rates'])
    ]


# Load the protocol
def load_protocol(path):
    # Retrieve phase, config and pre calibrated budgets
    phase = read_json(path)
    cfg = load_config(phase['base_config'])
    calibrated = read_json(phase['budgets_file'])
    budgets = calibrated['budgets']

    if [b['level'] for b in budgets] != ['low', 'medium', 'high']:
        raise ValueError('Expected the frozen low, medium, high budgets, in that order.')
    if any(not math.isfinite(b['budget']) or b['budget'] <= 0 for b in budgets):
        raise ValueError('Budgets must be finite and positive.')
    if phase['training_iter'] != 10:
        raise ValueError('Phase 2 fixes GP fitting at 10 iterations.')
    if cfg['problem_family']['type'] != 'yahpo_lcbench':
        raise ValueError('This selection protocol expects YAHPO LCBench.')
    for key in ('n_candidates', 'n_init', 'mask_visited_actions'):
        if cfg['env'][key] != calibrated['settings'][key]:
            raise ValueError(f'Phase 1 / Phase 2 mismatch: {key}')
    for key in ('epoch', 'cost_key'):
        if cfg['problem_family'][key] != calibrated['settings'][key]:
            raise ValueError(f'Phase 1 / Phase 2 mismatch: {key}')
    if cfg.get('cost_model', {}).get('target_transform', 'log1p') != 'log1p':
        raise ValueError('The approved cost-GP comparison uses log1p(cost).')
    if not 1 <= phase['max_acquisitions'] <= cfg['env']['n_candidates']:
        raise ValueError('The acquisition cap must be between 1 and n_candidates.')
    for key in ('episodes_per_instance', 'batch_size', 'bootstrap_replicates'):
        if not isinstance(phase[key], int) or phase[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if not phase['kernels'] or set(phase['kernels']) - set(KERNELS):
        raise ValueError('Unknown or empty kernel list.')
    if not phase['means'] or set(phase['means']) - {'constant', 'linear'}:
        raise ValueError('Unknown or empty mean list.')
    if not phase['learning_rates'] or any(not math.isfinite(x) or x <= 0 for x in phase['learning_rates']):
        raise ValueError('Learning rates must be finite and positive.')
    instances = [str(i) for i in cfg['problem_family']['instances']]
    if not instances or len(instances) != len(set(instances)):
        raise ValueError('Validation instances must be nonempty and distinct.')
    settings = candidate_settings(phase)
    if len({setting_id(s) for s in settings}) != len(settings):
        raise ValueError('Duplicate GP settings.')
    return phase, cfg, budgets, instances, settings


# Make signature for episode bank
def bank_signature(phase, cfg):
    # Batch size and GP settings do not determine the episode data.
    return dict(
        format='phase2_step_noise_v1', seed=phase['seed'],
        episodes_per_instance=phase['episodes_per_instance'],
        max_acquisitions=phase['max_acquisitions'],
        problem_family=cfg['problem_family'], n_candidates=cfg['env']['n_candidates'],
        n_init=cfg['env']['n_init'], dtype=cfg['env']['dtype'],
    )


# Create or load episode bank
def load_or_make_bank(phase, cfg, instances):
    """One immutable cache and independent noise schedule per episode.

    Common random numbers: noise is indexed by episode and acquisition step.
    Different methods can evaluate different points at a given step; their
    independent Gaussian innovations at that step are shared across methods.
    """
    path = Path(phase['episode_bank'])
    signature = bank_signature(phase, cfg)
    if path.exists():
        with np.load(path) as stored:
            if json.loads(str(stored['signature'].item())) != signature:
                raise ValueError(f'Episode-bank settings differ: {path}. Use a new bank path.')
            bank = {key: stored[key] for key in stored.files if key != 'signature'}
        print(f'Reusing paired episode bank: {path}', flush=True)
    else:
        bank = {}
        n = phase['episodes_per_instance']
        n_init = cfg['env']['n_init']
        dtype = getattr(t, cfg['env']['dtype'])
        for instance in instances:
            problem = copy.deepcopy(cfg['problem_family'])
            problem['instances'] = [instance]
            family = build_problem_family(problem, device='cpu', dtype=dtype)
            streams = np.random.SeedSequence([phase['seed'], int(instance)]).spawn(3)
            cache_seed = int(streams[0].generate_state(1)[0])
            X, y, cost, _ = family.build_candidate_cache(n, cfg['env']['n_candidates'], seed=cache_seed)
            rng_init, rng_noise = (np.random.default_rng(s) for s in streams[1:])
            bank[f'{instance}_X'] = X.numpy()
            bank[f'{instance}_y'] = y.numpy()
            bank[f'{instance}_cost'] = cost.numpy()
            bank[f'{instance}_init'] = np.stack([
                rng_init.choice(cfg['env']['n_candidates'], size=n_init, replace=False)
                for _ in range(n)
            ])
            bank[f'{instance}_noise'] = rng_noise.standard_normal(
                (n, n_init + phase['max_acquisitions'])
            ).astype(X.numpy().dtype)
            print(f'Cached {n} validation episodes for instance {instance}', flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            np.savez_compressed(f, signature=json.dumps(signature, sort_keys=True), **bank)
        print(f'Saved paired episode bank: {path}', flush=True)

    for value in bank.values():
        if not np.isfinite(value).all():
            raise ValueError('Episode bank contains non-finite values.')
    return bank, hashlib.sha256(path.read_bytes()).hexdigest()


# Batch the episode bank data
def batch_data(bank, instance, start, end, device, dtype):
    def tensor(suffix, tensor_dtype=dtype):
        return t.as_tensor(bank[f'{instance}_{suffix}'][start:end], dtype=tensor_dtype, device=device)
    cache = (tensor('X'), tensor('y'), tensor('cost'), {'instances': [instance] * (end - start)})
    return cache, tensor('init', t.long), tensor('noise')


def configured_env(cfg, stage, setting, objective_gp, batch_size, budget, cap, device):
    local = copy.deepcopy(cfg)
    local['ppo']['num_envs'] = batch_size
    local['env']['budget'] = budget
    local['env']['max_acquisitions'] = cap
    local['gp'] = {**(setting if stage == 'objective' else objective_gp), 'record_diagnostics': True}
    if stage == 'objective':
        # EI never uses a cost prediction; actual costs still end episodes.
        local['cost_model'] = {'type': 'known', 'target_transform': 'log1p'}
        local['env']['cost_feature_mode'] = 'oracle'
    else:
        local['cost_model'] = {
            'type': 'gp', 'target_transform': 'log1p',
            'gp': {**setting, 'record_diagnostics': True},
        }
        local['env']['cost_feature_mode'] = 'predicted'
    return make_env(local, device)


def gaussian_nlpd(target, mean, variance):
    variance = variance.clamp_min(1e-12)
    return 0.5 * (math.log(2.0 * math.pi) + variance.log() + (target - mean).square() / variance)


def parameter_json(parameters, lane):
    return json.dumps({name: value[lane].cpu().tolist() for name, value in parameters.items()}, sort_keys=True)


def run_batch(env, cache, init, noise, seed, method):
    """Complete one fixed batch; never replace a finished lane with a new episode.

    Diagnostics are prequential: the prediction before each acquisition is
    scored against its subsequently observed outcome. They include repeated
    queries and diagnose the visited trajectory; they are not a CV ranking.
    """
    obs = env.reset(seed=seed, deterministic=True, candidate_cache=cache,
                    init_indices=init, observation_noise=noise, auto_reset=False)
    B = env.num_batches
    lanes = t.arange(B, device=env.device)
    sums = t.zeros((B, 4), device=env.device, dtype=env.dtype)
    results = [None] * B
    while not bool(env.done.all()):
        active = ~env.done
        if not t.isfinite(obs[active]).all():
            raise RuntimeError('Non-finite GP prediction / observation.')
        with t.no_grad():
            action = select_baseline_action(env, obs, env.get_action_mask(), method)
            obj_pred = env.obj_gp.last_prediction
            obj_mu = obj_pred.mean[lanes, action]
            obj_var = obj_pred.variance[lanes, action]
            obj_parameters = env.obj_gp.last_fit_parameters
            cost_parameters = {}
            if env.use_cost_gp:
                cost_pred = env.cost_gp.last_prediction
                cost_mu = cost_pred.mean[lanes, action]
                cost_var = cost_pred.variance[lanes, action]
                cost_target = env._cost_to_gp_target(env.costs[lanes, action])
                cost_parameters = env.cost_gp.last_fit_parameters
            latent_y = env.y_grid[lanes, action]
        obs, _, terminal, infos = env.step(action)
        with t.no_grad():
            observed_y = env.obj_gp.train_y[lanes, env.obj_gp.t_idx - 1]
            sums[active, 0] += gaussian_nlpd(observed_y, obj_mu, obj_var)[active]
            sums[active, 1] += (latent_y - obj_mu).square()[active]
            if env.use_cost_gp:
                sums[active, 2] += gaussian_nlpd(cost_target, cost_mu, cost_var)[active]
                sums[active, 3] += (cost_target - cost_mu).square()[active]
        for lane in t.where(terminal)[0].tolist():
            info = infos['final_info'][lane]
            length = info['episode']['l']
            diagnostics = (sums[lane] / length).cpu().tolist()
            if not all(math.isfinite(v) for v in diagnostics):
                raise RuntimeError('Non-finite predictive diagnostic.')
            results[lane] = {
                **{key: info[key] for key in (
                    'regret', 'best_value', 'best_oracle_value', 'ground_truth',
                    'budget_used', 'remaining_budget', 'budget_overshoot')},
                'return': info['episode']['r'], 'episode_length': length,
                'cap_only': int(length >= env.max_acquisitions and info['remaining_budget'] > 0),
                'objective_nlpd': diagnostics[0], 'objective_rmse': math.sqrt(diagnostics[1]),
                'cost_nlpd_log1p': diagnostics[2] if env.use_cost_gp else '',
                'cost_rmse_log1p': math.sqrt(diagnostics[3]) if env.use_cost_gp else '',
                'objective_parameters': parameter_json(obj_parameters, lane),
                'cost_parameters': parameter_json(cost_parameters, lane),
            }
    return results


def paired_interval(values, indices):
    """Paired instance-cluster bootstrap, retaining complete episodes/budgets.

    GP fitting clips gradients across a batch, so episodes in one batch are
    not strictly independent. Keep each instance's entire evaluation intact.
    """
    by_instance = values.mean(axis=(0, 2))
    return np.quantile(by_instance[indices].mean(axis=1), [0.025, 0.975]).tolist()


def regret_cube(rows, budgets, instances, episodes):
    lookup = {(r['budget_level'], str(r['instance']), r['episode_id']): r['regret'] for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate paired episode keys.')
    return np.array([
        [[lookup[(b['level'], instance, episode)] for episode in range(episodes)] for instance in instances]
        for b in budgets
    ], dtype=float)


def summarize(all_rows, settings, failed, budgets, instances, phase):
    grouped = {setting_id(s): [] for s in settings}
    for row in all_rows:
        grouped[row['setting']].append(row)
    expected = len(budgets) * len(instances) * phase['episodes_per_instance']
    cubes = {
        key: regret_cube(rows, budgets, instances, phase['episodes_per_instance'])
        for key, rows in grouped.items() if key not in failed and len(rows) == expected
    }
    winner = min(cubes, key=lambda key: (float(cubes[key].mean()), key)) if cubes else None
    rng = np.random.default_rng(phase['seed'])
    indices = rng.integers(0, len(instances), size=(phase['bootstrap_replicates'], len(instances)))
    output = []
    for setting in settings:
        key = setting_id(setting)
        for level in ['ALL'] + [b['level'] for b in budgets]:
            rows = [r for r in grouped[key] if level == 'ALL' or r['budget_level'] == level]
            summary = {
                'setting': key, **setting, 'budget_level': level,
                'status': 'complete' if key in cubes else 'failed', 'n_episodes': len(rows),
                'mean_regret': '', 'median_regret': '', 'regret_ci_low': '', 'regret_ci_high': '',
                'difference_vs_selected': '', 'difference_ci_low': '', 'difference_ci_high': '',
                'success_rate': '', 'mean_episode_length': '', 'mean_budget_overshoot': '',
                'cap_only_rate': '', 'mean_objective_nlpd': '', 'mean_objective_rmse': '',
                'mean_cost_nlpd_log1p': '', 'mean_cost_rmse_log1p': '',
            }
            if key in cubes:
                cube, selected_cube = cubes[key], cubes[winner]
                if level != 'ALL':
                    index = next(i for i, b in enumerate(budgets) if b['level'] == level)
                    cube, selected_cube = cube[index:index + 1], selected_cube[index:index + 1]
                summary['mean_regret'] = float(cube.mean())
                summary['median_regret'] = float(np.median(cube))
                summary['regret_ci_low'], summary['regret_ci_high'] = paired_interval(cube, indices)
                difference = cube - selected_cube
                summary['difference_vs_selected'] = float(difference.mean())
                summary['difference_ci_low'], summary['difference_ci_high'] = paired_interval(difference, indices)
                # Same default threshold as scripts/summarize_results.py.
                summary['success_rate'] = float(np.mean(cube <= 1e-3))
                for source, target in (
                    ('episode_length', 'mean_episode_length'), ('budget_overshoot', 'mean_budget_overshoot'),
                    ('cap_only', 'cap_only_rate'), ('objective_nlpd', 'mean_objective_nlpd'),
                    ('objective_rmse', 'mean_objective_rmse'), ('cost_nlpd_log1p', 'mean_cost_nlpd_log1p'),
                    ('cost_rmse_log1p', 'mean_cost_rmse_log1p'),
                ):
                    values = [r[source] for r in rows if r[source] != '']
                    summary[target] = float(np.mean(values)) if values else ''
            output.append(summary)
    return output, winner


def write_csv(path, rows, fieldnames=None):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-config', default='configs/gp_selection/phase2.json')
    parser.add_argument('--stage', choices=('objective', 'cost'), required=True)
    parser.add_argument('--objective-selection', help='Objective-stage selected.json; required for the cost stage.')
    parser.add_argument('--output-dir', help='Default: results/gp_selection/<stage>')
    args = parser.parse_args()

    phase, cfg, budgets, instances, settings = load_protocol(args.phase_config)
    
    if args.stage == 'cost' and not args.objective_selection:
        parser.error('--stage cost requires --objective-selection')
    
    device = t.device(phase['device'])
    if device.type == 'cuda' and not t.cuda.is_available():
        raise RuntimeError('Phase 2 requests CUDA, but CUDA is unavailable; no automatic CPU fallback.')
    
    seed_everything(phase['seed'], deterministic=True)
    
    output_dir = Path(args.output_dir or f'results/gp_selection/{args.stage}')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}. Choose a new --output-dir.')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bank, bank_hash = load_or_make_bank(phase, cfg, instances)
    objective_gp = None
    if args.stage == 'cost':
        selected = read_json(args.objective_selection)
        if selected['stage'] != 'objective' or selected['episode_bank_sha256'] != bank_hash:
            raise ValueError('Cost stage requires an objective selection using this same episode bank.')
        if selected['phase_config'] != phase or selected['base_config'] != cfg or selected['budgets'] != budgets:
            raise ValueError('Objective selection used a different protocol/config/budget file.')
        objective_gp = selected['gp']
    
    print(f'Device: {device}; {len(settings)} settings; cap: {phase["max_acquisitions"]}', flush=True)
    print(f'Validation instances: {", ".join(instances)}', flush=True)
    print('Budgets: ' + ', '.join(f'{b["level"]}={b["budget"]:.12g}' for b in budgets), flush=True)
    
    method = 'ei' if args.stage == 'objective' else 'eipu'
    all_rows, failed, errors = [], set(), []
    begin = time.perf_counter()
    
    with (output_dir / 'episodes.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        for index, setting in enumerate(settings, 1):
            key = setting_id(setting)
            print(f'[{index}/{len(settings)}] {method}: {key}', flush=True)
            try:
                for budget in budgets:
                    for instance in instances:
                        for start in range(0, phase['episodes_per_instance'], phase['batch_size']):
                            end = min(start + phase['batch_size'], phase['episodes_per_instance'])
                            dtype = getattr(t, cfg['env']['dtype'])
                            cache, init, noise = batch_data(bank, instance, start, end, device, dtype)
                            env = configured_env(cfg, args.stage, setting, objective_gp, end - start,
                                                 budget['budget'], phase['max_acquisitions'], device)
                            rows = run_batch(env, cache, init, noise, phase['seed'], method)
                            for episode, row in enumerate(rows, start):
                                row.update(setting=key, stage=args.stage, method=method,
                                           kernel=setting['kernel'], mean=setting['mean'], lr=setting['lr'],
                                           instance=instance, episode_id=episode, budget_level=budget['level'],
                                           budget=budget['budget'], max_acquisitions=phase['max_acquisitions'])
                            writer.writerows(rows)
                            f.flush()
                            all_rows.extend(rows)
                        print(f'  {budget["level"]} / {instance}: {phase["episodes_per_instance"]} episodes', flush=True)
            except t.cuda.OutOfMemoryError:
                raise RuntimeError('CUDA out of memory. Reduce phase2.json batch_size and rerun BOTH stages in new output directories.') from None
            except RuntimeError as error:
                # A failed setting cannot win by dropping its difficult episodes.
                failed.add(key)
                errors.append(dict(setting=key, budget_level=budget['level'], instance=instance,
                                   first_episode=start, error=str(error)))
                write_csv(output_dir / 'failures.csv', errors)
                print(f'  FAILED; excluded from selection: {error}', flush=True)
    
    summary, winner = summarize(all_rows, settings, failed, budgets, instances, phase)
    write_csv(output_dir / 'summary.csv', summary)
    
    if winner is None:
        raise RuntimeError('No setting completed every paired episode; inspect failures.csv.')
    
    chosen = next(s for s in settings if setting_id(s) == winner)
    payload = dict(
        stage=args.stage, method=method, setting=winner,
        gp=chosen if args.stage == 'objective' else objective_gp,
        phase_config=phase, base_config=cfg, budgets=budgets,
        episode_bank_sha256=bank_hash,
        selection_rule='Minimum equally weighted validation mean final regret over instances and budgets; exact ties by setting ID.',
        ci_scope='95% paired instance-cluster percentile bootstrap; all complete episodes and budgets stay together within an instance. Only eight validation instances; descriptive after selection.',
        elapsed_seconds=time.perf_counter() - begin, failed_settings=sorted(failed),
    
    )
    if args.stage == 'cost':
        payload['cost_model'] = {'type': 'gp', 'target_transform': 'log1p', 'gp': chosen}
    
    (output_dir / 'selected.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    
    leaders = sorted((r for r in summary if r['budget_level'] == 'ALL' and r['status'] == 'complete'),
                     key=lambda r: (r['mean_regret'], r['setting']))
    
    print('\nTop settings by validation mean regret:')
    
    for row in leaders[:5]:
        print(f"  {row['setting']}: {row['mean_regret']:.6f} "
              f"[95% CI {row['regret_ci_low']:.6f}, {row['regret_ci_high']:.6f}]")
    
    print(f'Selected: {winner}\nResults: {output_dir}\nElapsed: {(time.perf_counter() - begin) / 60:.1f} minutes')
    
    if failed:
        print(f'{len(failed)} setting(s) failed; inspect failures.csv before accepting the selection.')


if __name__ == '__main__':
    main()