"""Evaluate the newly trained policies and baselines on reserved test instances.

Run prepare, check, then run --experiment matched_budgets / budget_transfer.
Completed baseline jobs are shared between those two reports.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import TRAIN_AGENTS_NEW as training
from scripts.TRAIN_AGENTS_NEW import require, read_json, write_json, write_csv, digest, object_digest

RESULTS_ROOT = 'results/final_testing_NEW'
EPISODES = 128
BATCH_SIZE = 32
TEST_SEED = 1
SUCCESS_THRESHOLD = 1e-2
BASELINES = ('random', 'ei', 'eipu', 'ei_cool')


def checked_selections(protocol):
    path = Path(protocol['results_root']) / 'selected_checkpoints.json'
    manifest = read_json(path)
    require(manifest['protocol_sha256'] == protocol['protocol_sha256'],
            'The selected checkpoints belong to a different training protocol.')
    entries = manifest['selected_checkpoints']
    by_id = {entry['run_id']: entry for entry in entries}
    require(len(by_id) == len(entries) == len(protocol['runs'])
            and set(by_id) == {run['run_id'] for run in protocol['runs']},
            'Every training seed needs its own validated checkpoint.')
    for run in protocol['runs']:
        entry = by_id[run['run_id']]
        for field in ('variant', 'budget_level', 'training_seed'):
            require(entry[field] == run[field], f'Checkpoint metadata mismatch: {field}')
        selected_path = Path(protocol['results_root']) / 'validation' / run['run_id'] / 'selected.json'
        selection = read_json(selected_path)
        require(selection['protocol_sha256'] == protocol['protocol_sha256'],
                'Per-run checkpoint selection uses different code/configuration.')
        winner = training.choose_checkpoint(selection['scores'], protocol['checkpoint_steps'])
        require(winner == selection['selected'], 'Stored checkpoint choice is inconsistent.')
        require(all(entry[key] == value for key, value in winner.items()),
                'Combined checkpoint manifest differs from per-run selection.')
        checkpoint = entry['checkpoint']
        require(digest(checkpoint['path']) == checkpoint['sha256'], 'Checkpoint bytes changed.')
    return entries, path


def prepare():
    """Freeze testing after all training/validation choices have been made."""
    from scripts.select_gp_NEW import load_or_make_bank, bank_signature
    root = Path(RESULTS_ROOT)
    require(not root.exists() or not any(root.iterdir()),
            f'{root} is not empty; use check/run to continue, or archive it first.')
    trained = training.load_protocol()
    selections, manifest_path = checked_selections(trained)
    settings = training.load_experiment_settings()
    cfg = read_json(trained['runs'][0]['config'])
    cfg['problem_family']['instances'] = list(training.TEST)
    phase = copy.deepcopy(settings['phase'])
    phase.update(seed=TEST_SEED, episodes_per_instance=EPISODES, batch_size=BATCH_SIZE,
                 episode_bank=str(root / 'paired_test_episodes.npz'))
    _, bank_hash = load_or_make_bank(phase, cfg, training.TEST)
    test = dict(instances=list(training.TEST), episodes_per_instance=EPISODES,
                batch_size=BATCH_SIZE, seed=TEST_SEED, action_mode='argmax',
                episode_bank=phase['episode_bank'], episode_bank_sha256=bank_hash,
                bank_signature=bank_signature(phase, cfg))
    jobs = []
    run_by_id = {run['run_id']: run for run in trained['runs']}
    for selected in selections:
        run = run_by_id[selected['run_id']]
        for budget in settings['budgets']:
            jobs.append(dict(job_id=f"{run['run_id']}__test_{budget['level']}",
                             method=run['variant'], training_budget=run['budget_level'],
                             test_budget=budget['level'], budget=budget['budget'],
                             training_seed=run['training_seed'], config=run['config'],
                             checkpoint=selected['checkpoint']))
    # The same frozen GP choices apply to every method. EI/Random need no cost GP.
    base_cfg = copy.deepcopy(settings['template'])
    base_cfg['gp'] = copy.deepcopy(settings['objective_gp'])
    base_cfg['cost_model'] = copy.deepcopy(settings['cost_model'])
    base_cfg['gp_fitting_recipe'] = settings['fitting_recipe']
    base_cfg['env']['max_acquisitions'] = phase['max_acquisitions']
    base_path = root / 'baseline_config.json'
    write_json(base_path, base_cfg)
    for method in BASELINES:
        for budget in settings['budgets']:
            jobs.append(dict(job_id=f"{method}__test_{budget['level']}", method=method,
                             training_budget='none', test_budget=budget['level'],
                             budget=budget['budget'], training_seed='', config=str(base_path),
                             checkpoint=None))
    inputs = {str(manifest_path): digest(manifest_path), str(base_path): digest(base_path)}
    for job in jobs:
        inputs[job['config']] = digest(job['config'])
        if job['checkpoint']:
            inputs[job['checkpoint']['path']] = job['checkpoint']['sha256']
    protocol = dict(results_root=RESULTS_ROOT, test=test, jobs=jobs, inputs=inputs,
                    training_protocol=training.DEFAULT_PROTOCOL,
                    training_protocol_sha256=trained['protocol_sha256'],
                    source_hashes={'scripts/TEST_AGENT_NEW.py': digest('scripts/TEST_AGENT_NEW.py')},
                    success_threshold=SUCCESS_THRESHOLD)
    protocol['protocol_sha256'] = object_digest(protocol)
    write_json(root / 'protocol.json', protocol)
    print(f'Prepared {len(jobs)} testing jobs: {root / "protocol.json"}')


def read_bank(protocol):
    return training.read_bank({'validation': protocol['test']})


def load_protocol():
    protocol = read_json(Path(RESULTS_ROOT) / 'protocol.json')
    contents = {k: v for k, v in protocol.items() if k != 'protocol_sha256'}
    require(object_digest(contents) == protocol['protocol_sha256'], 'Test protocol was edited.')
    trained = training.load_protocol(protocol['training_protocol'])
    require(trained['protocol_sha256'] == protocol['training_protocol_sha256'],
            'Training protocol changed after test preparation.')
    for section in ('inputs', 'source_hashes'):
        for path, expected in protocol[section].items():
            require(digest(path) == expected, f'Frozen test input changed: {path}')
    read_bank(protocol)
    return protocol


def jobs_for(protocol, experiment):
    if experiment == 'all':
        return protocol['jobs']
    matched = experiment == 'matched_budgets'
    return [job for job in protocol['jobs'] if job['method'] in BASELINES
            or (job['training_budget'] == job['test_budget']) == matched]


def expected_improvement(mu, sigma, best):
    import torch as t
    sigma = sigma.clamp_min(1e-9)
    difference = mu - best[:, None]
    z = difference / sigma
    normal = t.distributions.Normal(t.zeros((), device=mu.device, dtype=mu.dtype),
                                    t.ones((), device=mu.device, dtype=mu.dtype))
    return (difference * normal.cdf(z) + sigma * normal.log_prob(z).exp()).clamp_min(0.)


def baseline_action(env, obs, method, uniform=None):
    """Keep the existing EI/EIpu/EI-cool definitions and action masking."""
    import torch as t
    mask = env.get_action_mask()
    if method == 'random':
        # External uniforms keep Random independent of GP fitting and batching.
        rank = (uniform * mask.sum(dim=1)).long()
        return (mask & (mask.long().cumsum(dim=1) == rank[:, None] + 1)).long().argmax(dim=1)
    ei = expected_improvement(obs[..., 0], obs[..., 1], env.best_current_value)
    cost = (obs[..., 2] * env.budget).clamp_min(1e-8)
    if method == 'ei':
        scores = ei
    elif method == 'eipu':
        scores = ei / cost
    elif method == 'ei_cool':
        alpha = (env.remaining_budget / env.budget).clamp(0, 1)[:, None]
        scores = ei / cost.pow(alpha)
    else:
        raise ValueError(f'Unknown baseline: {method}')
    return scores.masked_fill(~mask, -t.inf).argmax(dim=1)


def random_uniforms(seed, instance, start, end, cap):
    import numpy as np
    return np.stack([np.random.default_rng(np.random.SeedSequence(
        [seed, int(instance), episode, 2027])).random(cap) for episode in range(start, end)])


def run_batch(env, cache, init, noise, seed, method, agent=None, uniforms=None):
    import torch as t
    obs = env.reset(seed, True, candidate_cache=cache, init_indices=init,
                    observation_noise=noise, auto_reset=False)
    rows = [None] * env.num_batches
    for step in range(env.max_acquisitions):
        if bool(env.done.all()):
            break
        active = ~env.done
        require(bool(t.isfinite(obs[active]).all()), 'Non-finite test observation.')
        with t.no_grad():
            if agent is not None:
                logits = agent.get_logits(obs, action_mask=env.get_action_mask())
                require(bool(t.isfinite(logits[active]).all()), 'Non-finite policy logits.')
                actions = logits.argmax(dim=1)
            else:
                actions = baseline_action(env, obs, method,
                                          uniforms[:, step] if uniforms is not None else None)
        obs, _, terminal, info = env.step(actions)
        for lane in t.where(terminal)[0].tolist():
            require(rows[lane] is None, 'Finished test lane was reused.')
            result = info['final_info'][lane]
            row = {key: result[key] for key in ('regret', 'best_value', 'best_oracle_value',
                   'ground_truth', 'budget_used', 'remaining_budget', 'budget_overshoot')}
            row.update(episode_length=result['episode']['l'], **{'return': result['episode']['r']})
            require(all(math.isfinite(float(v)) for v in row.values()), 'Non-finite test result.')
            rows[lane] = row
    require(all(row is not None for row in rows), 'Some test episodes did not finish.')
    return rows


def evaluate_job(protocol, job, device_name='cuda'):
    """A worker handles one policy/budget pair; CPU is available for smoke checks."""
    import torch as t
    from reinforcement_learning.ppo_no_gym_NEW import make_env
    from reinforcement_learning.agents.neural_AF import Agent
    from scripts.select_gp_NEW import batch_data
    from utils.seeding import seed_everything
    require(device_name != 'cuda' or t.cuda.is_available(), 'Final testing requires CUDA.')
    device = t.device(device_name)
    test = protocol['test']
    seed_everything(test['seed'], deterministic=True)
    cfg = read_json(job['config'])
    cfg['problem_family']['instances'] = test['instances']
    cfg['env'].update(budget=job['budget'], max_acquisitions=test['bank_signature']['max_acquisitions'],
                      device=device_name)
    agent = None
    if job['checkpoint']:
        architecture = cfg['agent']
        agent = Agent(7, 7, 1, 1, architecture['num_layers'], architecture['layer_size'],
                      architecture['critic_feature_mode']).to(device)
        weights = t.load(job['checkpoint']['path'], map_location=device, weights_only=True)
        require(all(bool(t.isfinite(v).all()) for v in weights.values()), 'Non-finite checkpoint weights.')
        agent.load_state_dict(weights)
        agent.eval()
    else:
        uses_cost = job['method'] in ('eipu', 'ei_cool')
        cfg['env'].update(cost_feature_mode='predicted' if uses_cost else 'none',
                          use_cost_uncertainty_feature=False)
        if not uses_cost:
            cfg['cost_model'] = {'type': 'known', 'target_transform': 'log1p'}
    bank = read_bank(protocol)
    rows = []
    begin = time.perf_counter()
    dtype = getattr(t, cfg['env']['dtype'])
    for instance in test['instances']:
        for start in range(0, test['episodes_per_instance'], test['batch_size']):
            end = min(start + test['batch_size'], test['episodes_per_instance'])
            cfg['ppo']['num_envs'] = end - start
            cache, init, noise = batch_data(bank, instance, start, end, device, dtype)
            uniforms = t.as_tensor(random_uniforms(test['seed'], instance, start, end,
                                    cfg['env']['max_acquisitions']), device=device, dtype=t.float64)
            batch = run_batch(make_env(cfg, device), cache, init, noise, test['seed'],
                              job['method'], agent, uniforms)
            for episode, row in enumerate(batch, start):
                row.update(job_id=job['job_id'], method=job['method'], training_budget=job['training_budget'],
                           test_budget=job['test_budget'], training_seed=job['training_seed'],
                           instance=instance, episode_id=episode)
            rows.extend(batch)
        print(f"{job['job_id']} / {instance}: complete", flush=True)
    output = Path(protocol['results_root']) / 'jobs' / job['job_id']
    path = output / 'episodes.csv'
    write_csv(path, rows)
    write_json(output / 'summary.json', dict(protocol_sha256=protocol['protocol_sha256'], job=job,
               episode_bank_sha256=test['episode_bank_sha256'], episodes_csv=str(path),
               episodes_sha256=digest(path), n_episodes=len(rows), worker_seconds=time.perf_counter() - begin))


def checked_job(protocol, job):
    result = read_json(Path(protocol['results_root']) / 'jobs' / job['job_id'] / 'summary.json')
    require(result['protocol_sha256'] == protocol['protocol_sha256'] and result['job'] == job,
            'Saved test job belongs to a different protocol.')
    require(result['episode_bank_sha256'] == protocol['test']['episode_bank_sha256'], 'Test bank changed.')
    require(digest(result['episodes_csv']) == result['episodes_sha256'], 'Saved test episodes changed.')
    with Path(result['episodes_csv']).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    test = protocol['test']
    training.balanced_score(rows, test['instances'], test['episodes_per_instance'])
    require(len(rows) == result['n_episodes'], 'Saved episode count disagrees with summary.')
    for row in rows:
        for field in ('job_id', 'method', 'training_budget', 'test_budget', 'training_seed'):
            require(row[field] == str(job[field]), f'Test row has mismatched {field}.')
        require(all(math.isfinite(float(row[field])) for field in ('regret', 'episode_length', 'budget_overshoot')),
                'Non-finite test metrics.')
    return rows


def aggregate(rows, threshold):
    import numpy as np
    result = {}
    for source, name in (('regret', 'final_simple_regret'), ('episode_length', 'episode_length'),
                         ('budget_overshoot', 'budget_overshoot')):
        values = np.array([float(row[source]) for row in rows])
        result['mean_' + name] = float(values.mean())
        result['sd_' + name] = float(values.std(ddof=1)) if len(values) > 1 else 0.
    result['success_rate'] = float(np.mean([float(row['regret']) <= threshold for row in rows]))
    return result


def summarize(protocol, experiment):
    rows = [row for job in jobs_for(protocol, experiment) for row in checked_job(protocol, job)]
    folder = Path(protocol['results_root']) / experiment
    for level in ('low', 'medium', 'high'):
        subset = [r for r in rows if r['test_budget'] == level]
        groups = sorted({(r['method'], r['training_budget']) for r in subset})
        overall, individual = [], []
        for method, trained_budget in groups:
            selected = [r for r in subset if (r['method'], r['training_budget']) == (method, trained_budget)]
            labels = dict(method=method, training_budget=trained_budget, test_budget=level)
            overall.append(dict(**labels, **aggregate(selected, protocol['success_threshold'])))
            for instance in protocol['test']['instances']:
                individual.append(dict(**labels, instance=instance, **aggregate(
                    [r for r in selected if r['instance'] == instance], protocol['success_threshold'])))
        write_csv(folder / 'aggregated' / f'{level}.csv', overall)
        write_csv(folder / 'instance_specific' / f'{level}.csv', individual)
    print(f'Wrote mean/sample-SD tables: {folder}')


def run(experiment):
    protocol = load_protocol()
    for job in jobs_for(protocol, experiment):
        path = Path(protocol['results_root']) / 'jobs' / job['job_id'] / 'summary.json'
        if path.exists():
            checked_job(protocol, job)
            print(f"Reusing completed test job: {job['job_id']}", flush=True)
            continue
        worker = mp.get_context('spawn').Process(target=evaluate_job, args=(protocol, job))
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
        require(exit_code == 0, f"Test worker failed: {job['job_id']}")
        checked_job(protocol, job)
    for name in ('matched_budgets', 'budget_transfer') if experiment == 'all' else (experiment,):
        summarize(protocol, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'run', 'summarize'))
    parser.add_argument('--experiment', choices=('matched_budgets', 'budget_transfer', 'all'), default='matched_budgets')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'check':
        protocol = load_protocol()
        print(f"PASS: {len(protocol['jobs'])} frozen testing jobs and test bank.")
    elif args.command == 'run':
        run(args.experiment)
    else:
        protocol = load_protocol()
        for name in ('matched_budgets', 'budget_transfer') if args.experiment == 'all' else (args.experiment,):
            summarize(protocol, name)


if __name__ == '__main__':
    main()