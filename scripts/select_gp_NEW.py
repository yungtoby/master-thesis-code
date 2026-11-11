"""Select objective and cost GPs independently using common random episodes.

Predict the next randomly chosen observation before adding it to the GP.
Choose by mean per-episode NLPD, equally weighted across instances/budgets.
RMSE is a diagnostic; neither EI/EIpu nor regret determines the GP winner.
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
from bo.gaussian_process_batch_NEW import RepeatedPadGPWrapper, FITTING_RECIPE
from bo.problems.registry_NEW import build_problem_family
from utils.config import load_config
from utils.seeding import seed_everything

PHASE_CONFIG = 'configs/gp_hyperparameter_selection/gp_configs_NEW.json'
RESULTS_ROOT = 'results/gp_selection_NEW'
KERNELS = ('rbf', 'matern32', 'matern52', 'rbf_matern52_add', 'rbf_matern52_mul')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))



def setting_id(setting):
    return f"{setting['kernel']}__{setting['mean']}__lr{setting['lr']:g}"



def candidate_settings(phase):
    return [
        dict(kernel=k, mean=m, lr=float(lr), training_iter=10)
        for k, m, lr in itertools.product(
            phase['kernels'], phase['means'], phase['learning_rates'])
    ]



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
    if not 1 <= phase['max_acquisitions'] <= cfg['env']['n_candidates'] - cfg['env']['n_init']:
        raise ValueError('The acquisition cap must be between 1 and the number of non-initial candidates.')
    for key in ('episodes_per_instance', 'batch_size'):
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



def bank_signature(phase, cfg):
    # Batch size and GP settings do not determine the episode data.
    return dict(
        format='phase2_step_noise_v1', seed=phase['seed'],
        episodes_per_instance=phase['episodes_per_instance'],
        max_acquisitions=phase['max_acquisitions'],
        problem_family=cfg['problem_family'], n_candidates=cfg['env']['n_candidates'],
        n_init=cfg['env']['n_init'], dtype=cfg['env']['dtype'],
    )



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



def batch_data(bank, instance, start, end, device, dtype):
    def tensor(suffix, tensor_dtype=dtype):
        return t.as_tensor(bank[f'{instance}_{suffix}'][start:end], dtype=tensor_dtype, device=device)
    cache = (tensor('X'), tensor('y'), tensor('cost'), {'instances': [instance] * (end - start)})
    return cache, tensor('init', t.long), tensor('noise')



def gaussian_nlpd(target, mean, variance):
    variance = variance.clamp_min(1e-12)
    return 0.5 * (math.log(2.0 * math.pi) + variance.log() + (target - mean).square() / variance)



def parameter_json(parameters, lane):
    return json.dumps({name: value[lane].cpu().tolist() for name, value in parameters.items()}, sort_keys=True)



def write_csv(path, rows, fieldnames=None):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {path: digest(path) for path in (
        'scripts/select_gp_NEW.py', 'bo/gaussian_process_batch_NEW.py',
        'bo/problems/registry_NEW.py', 'utils/seeding.py',
        'bo/problems/yahpo_lcbench.py')}


def load_or_make_orders(phase, cfg, instances, bank, bank_hash):
    """Save random acquisition orders independently of all GP random state."""
    signature = dict(format='random_without_replacement_v1', seed=phase['seed'],
                     episode_bank_sha256=bank_hash, cap=phase['max_acquisitions'])
    orders = {}
    candidates = np.arange(cfg['env']['n_candidates'])
    for instance in instances:
        rows = []
        for episode, initial in enumerate(bank[f'{instance}_init']):
            rng = np.random.default_rng(np.random.SeedSequence(
                [phase['seed'], int(instance), episode, 2026]))
            remaining = candidates[~np.isin(candidates, initial)]
            rows.append(rng.permutation(remaining)[:phase['max_acquisitions']])
        orders[instance] = np.stack(rows).astype(np.int64)
    path = Path(phase['random_orders'])
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if json.loads(str(stored['signature'].item())) != signature:
                raise ValueError('Random-order archive belongs to a different protocol.')
            if set(stored.files) != {'signature', *instances}:
                raise ValueError('Random-order archive has unexpected arrays.')
            for instance in instances:
                if not np.array_equal(stored[instance], orders[instance]):
                    raise ValueError('Saved random orders differ from the seeded schedule.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            np.savez_compressed(stream, signature=json.dumps(signature, sort_keys=True), **orders)
    return orders, digest(path)


def run_batch(cache, init, noise, order, setting, stage, budget, env_cfg):
    """Score next observations before adding them, on the same random trajectory.

    Finished lanes remain in the batch, preserving repeat padding and the
    original joint gradient clipping. Warm-start costs remain free.
    """
    X, y, cost, _ = cache
    B, _, d = X.shape
    n_init, cap = init.shape[1], order.shape[1]
    lanes = t.arange(B, device=X.device)
    gp = RepeatedPadGPWrapper(X.device, X.dtype, B, n_init + cap, d,
                             **setting, record_diagnostics=True)
    observed = y[lanes[:, None], init] + env_cfg.get('objective_noise_std', 0.) * noise[:, :n_init]
    if env_cfg.get('objective_noise_clip', True):
        observed = observed.clamp(0, 1)
    initial_y = observed if stage == 'objective' else t.log1p(cost[lanes[:, None], init])
    gp.set_lane_data(t.ones(B, device=X.device, dtype=t.bool), X[lanes[:, None], init], initial_y)
    sums = t.zeros((B, 2), device=X.device, dtype=t.float64)
    remaining = t.full((B,), budget, device=X.device, dtype=X.dtype)
    lengths = t.zeros(B, device=X.device, dtype=t.long)
    done = t.zeros(B, device=X.device, dtype=t.bool)
    results = [None] * B
    first_nlpd = first_error = None
    for step in range(cap):
        active = ~done
        if not bool(active.any()):
            break
        action = order[:, step]
        query = X[lanes, action][:, None, :]
        prediction = gp.predict(query)
        mean, variance = prediction.mean[:, 0].double(), prediction.variance[:, 0].double()
        if not bool((t.isfinite(mean[active]) & t.isfinite(variance[active]) & (variance[active] > 0)).all()):
            raise RuntimeError('Invalid observation predictive distribution.')
        target = y[lanes, action] + env_cfg.get('objective_noise_std', 0.) * noise[:, n_init + step]
        if env_cfg.get('objective_noise_clip', True):
            target = target.clamp(0, 1)
        if stage == 'cost':
            target = t.log1p(cost[lanes, action])
        error = target.double() - mean
        nlpd = gaussian_nlpd(target.double(), mean, variance)
        if step == 0:
            first_nlpd, first_error = nlpd.clone(), error.abs().clone()
        sums[active, 0] += nlpd[active]
        sums[active, 1] += error[active].square()
        # Outcome is added only after its predictive score has been recorded.
        gp.add_data(query, target, active)
        remaining[active] -= cost[lanes, action][active]
        lengths[active] += 1
        terminal = active & ((remaining <= 0) | (lengths >= cap))
        for lane in t.where(terminal)[0].tolist():
            n = int(lengths[lane])
            results[lane] = dict(
                nlpd=float(sums[lane, 0] / n), rmse=math.sqrt(float(sums[lane, 1] / n)),
                initial_nlpd=float(first_nlpd[lane]), initial_abs_error=float(first_error[lane]),
                episode_length=n, budget_used=budget - float(remaining[lane]),
                remaining_budget=float(remaining[lane]),
                budget_overshoot=max(0., -float(remaining[lane])),
                cap_only=int(n == cap and float(remaining[lane]) > 0),
                actions=json.dumps(order[lane, :n].cpu().tolist()),
                parameters=parameter_json(gp.last_fit_parameters, lane))
        done |= terminal
    if any(row is None for row in results):
        raise RuntimeError('Some random episodes did not finish.')
    return results


def balanced_metrics(rows, budgets, instances, episodes):
    expected = {(b['level'], str(i), e) for b in budgets for i in instances for e in range(episodes)}
    keys = [(r['budget_level'], str(r['instance']), int(r['episode_id'])) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Missing, duplicated, or unexpected paired episodes.')
    fields = ('nlpd', 'rmse', 'initial_nlpd', 'initial_abs_error',
              'episode_length', 'budget_overshoot', 'cap_only')
    output = {}
    for field in fields:
        means = [np.mean([float(r[field]) for r in rows
                          if r['budget_level'] == b['level'] and str(r['instance']) == str(i)])
                 for b in budgets for i in instances]
        value = float(np.mean(means))
        if not math.isfinite(value):
            raise ValueError(f'Non-finite mean {field}.')
        output['mean_' + field] = value
    return output


def summarize(all_rows, settings, failed, budgets, instances, phase):
    output, scores = [], {}
    for setting in settings:
        key = setting_id(setting)
        rows = [row for row in all_rows if row['setting'] == key]
        if key not in failed:
            scores[key] = balanced_metrics(rows, budgets, instances, phase['episodes_per_instance'])['mean_nlpd']
        for level in ['ALL'] + [b['level'] for b in budgets]:
            selected_budgets = budgets if level == 'ALL' else [b for b in budgets if b['level'] == level]
            subset = rows if level == 'ALL' else [r for r in rows if r['budget_level'] == level]
            metrics = {f'mean_{f}': '' for f in ('nlpd', 'rmse', 'initial_nlpd', 'initial_abs_error',
                                              'episode_length', 'budget_overshoot', 'cap_only')}
            if key not in failed:
                metrics = balanced_metrics(subset, selected_budgets, instances, phase['episodes_per_instance'])
            output.append(dict(setting=key, **setting, budget_level=level,
                               status='failed' if key in failed else 'complete',
                               n_episodes=len(subset), **metrics))
    winner = min(scores, key=lambda key: (scores[key], key)) if scores else None
    return output, winner


def run_stage(stage, phase, cfg, budgets, instances, settings, bank, bank_hash, orders, orders_hash):
    output = Path(RESULTS_ROOT) / stage
    identity = dict(stage=stage, method='random_without_replacement',
                    phase_config=phase, base_config=cfg, budgets=budgets,
                    episode_bank_sha256=bank_hash, random_orders_sha256=orders_hash,
                    fitting_recipe=FITTING_RECIPE, source_hashes=source_hashes())
    if (output / 'selected.json').exists():
        saved = read_json(output / 'selected.json')
        if any(saved.get(key) != value for key, value in identity.items()) or saved['failed_settings']:
            raise ValueError(f'Existing selection is different or failed: {output}. Archive it first.')
        for path, expected in saved['result_hashes'].items():
            if digest(path) != expected:
                raise ValueError(f'Selection result changed: {path}')
        print(f'Reusing verified {stage} selection: {saved["setting"]}')
        return saved
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Incomplete selection: {output}. Archive this stage folder and rerun.')
    output.mkdir(parents=True, exist_ok=True)
    all_rows, failed, errors = [], set(), []
    begin = time.perf_counter()
    device, dtype = t.device(phase['device']), getattr(t, cfg['env']['dtype'])
    with (output / 'episodes.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = None
        for index, setting in enumerate(settings, 1):
            key = setting_id(setting)
            print(f'[{stage} {index}/{len(settings)}] {key}', flush=True)
            try:
                for budget in budgets:
                    for instance in instances:
                        for start in range(0, phase['episodes_per_instance'], phase['batch_size']):
                            end = min(start + phase['batch_size'], phase['episodes_per_instance'])
                            cache, init, noise = batch_data(bank, instance, start, end, device, dtype)
                            order = t.as_tensor(orders[instance][start:end], device=device, dtype=t.long)
                            rows = run_batch(cache, init, noise, order, setting, stage, budget['budget'], cfg['env'])
                            for episode_id, row in enumerate(rows, start):
                                row.update(setting=key, stage=stage, method='random_without_replacement',
                                           kernel=setting['kernel'], mean=setting['mean'], lr=setting['lr'],
                                           instance=instance, episode_id=episode_id, budget_level=budget['level'],
                                           budget=budget['budget'], max_acquisitions=phase['max_acquisitions'])
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                                writer.writeheader()
                            writer.writerows(rows)
                            stream.flush()
                            all_rows.extend(rows)
            except t.cuda.OutOfMemoryError:
                raise RuntimeError('CUDA out of memory; review batch size and rerun both selections.') from None
            except RuntimeError as error:
                failed.add(key)
                errors.append(dict(setting=key, error=str(error)))
                write_csv(output / 'failures.csv', errors)
                print(f'FAILED: {key}: {error}', flush=True)
    summary, winner = summarize(all_rows, settings, failed, budgets, instances, phase)
    write_csv(output / 'summary.csv', summary)
    if winner is None:
        raise RuntimeError('No setting completed every episode; inspect failures.csv.')
    chosen = next(s for s in settings if setting_id(s) == winner)
    saved = dict(**identity, setting=winner, gp=chosen,
                 selection_rule='Minimum unrounded mean per-episode NLPD, equal weight per instance and budget; exact ties by setting ID.',
                 elapsed_seconds=time.perf_counter() - begin, failed_settings=sorted(failed),
                 result_hashes={str(output / name): digest(output / name) for name in ('episodes.csv', 'summary.csv')})
    (output / 'selected.json').write_text(json.dumps(saved, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Selected {stage}: {winner}; results: {output}', flush=True)
    if failed:
        raise RuntimeError('Some GP candidates failed. Resolve failures before phase 4.')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-config', default=PHASE_CONFIG)
    parser.add_argument('--stage', choices=('both', 'objective', 'cost'), default='both')
    args = parser.parse_args()
    phase, cfg, budgets, instances, settings = load_protocol(args.phase_config)
    if t.device(phase['device']).type == 'cuda' and not t.cuda.is_available():
        raise RuntimeError('Phase 2 requests CUDA; no automatic CPU fallback.')
    seed_everything(phase['seed'], deterministic=True)
    bank, bank_hash = load_or_make_bank(phase, cfg, instances)
    orders, orders_hash = load_or_make_orders(phase, cfg, instances, bank, bank_hash)
    for stage in ('objective', 'cost') if args.stage == 'both' else (args.stage,):
        run_stage(stage, phase, cfg, budgets, instances, settings, bank, bank_hash, orders, orders_hash)


if __name__ == '__main__':
    main()