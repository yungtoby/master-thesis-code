"""Prepare, time, train and validate the frozen 27-run Phase 4 experiment.

Run from the repository root; see docs/phase4_training_new.md.
The prepare/check commands do not import PyTorch or load YAHPO.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN = ['3945', '7593', '34539', '126025', '126026', '126029', '146212',
         '167104', '167149', '167152', '167161', '167168', '167181', '167184',
         '167185', '167190', '167200', '167201', '168329', '168330']
VALIDATION = ['168331', '168335', '168868', '168908', '168910', '189354', '189862', '189865']
# IDs only: no test data are loaded anywhere in this script.
TEST = ['189866', '189873', '189905', '189906', '189908', '189909']
VARIANTS = ['no_cost', 'cost_mean', 'cost_uncertainty']
SEEDS = [1, 2, 3]
CHECKPOINT_STEPS = [262144, 524288, 786432, 1048576]
WARMUP_UPDATES, MEASURED_UPDATES = 2, 4
TEMPLATE = 'configs/pilots_and_audits/debug/debug_cost_mean_longrun_1m_new_critic.json'
PHASE2 = 'configs/gp_hyperparameter_selection/gp_configs.json'
OBJECTIVE = 'results/gp_selection/objective/selected.json'
COST = 'results/gp_selection/cost/selected.json'
DEFAULT_PROTOCOL = 'configs/final_training/protocol.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_path(path):
    # Keep manifests portable between computers/checkouts of this repository.
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_bank(protocol):
    """Check all fixed episode data without constructing a benchmark backend."""
    import numpy as np

    validation = protocol['validation']
    path = validation['episode_bank']
    require(digest(path) == validation['episode_bank_sha256'], 'Validation bank hash changed.')
    with np.load(path, allow_pickle=False) as stored:
        require(json.loads(str(stored['signature'].item())) == validation['bank_signature'],
                'Validation bank signature differs from the frozen protocol.')
        bank = {k: stored[k] for k in stored.files if k != 'signature'}
    expected = {f'{i}_{suffix}' for i in VALIDATION for suffix in ('X', 'y', 'cost', 'init', 'noise')}
    require(set(bank) == expected, 'Unexpected/missing validation instances or bank arrays.')
    for instance in VALIDATION:
        shapes = {'X': (32, 64, 7), 'y': (32, 64), 'cost': (32, 64),
                  'init': (32, 3), 'noise': (32, 64)}
        for suffix, shape in shapes.items():
            array = bank[f'{instance}_{suffix}']
            require(array.shape == shape and np.isfinite(array).all(), f'Invalid bank array: {instance}_{suffix}')
            expected_dtype = np.dtype('int64' if suffix == 'init' else 'float32')
            require(array.dtype == expected_dtype, f'Unexpected bank dtype: {instance}_{suffix}')
        init = bank[f'{instance}_init']
        require(((0 <= init) & (init < 64)).all(), 'Initial index outside the candidate set.')
        require(all(len(set(row)) == 3 for row in init.tolist()), 'Duplicate initial points.')
        require((bank[f'{instance}_cost'] >= 0).all(), 'Negative benchmark costs.')
    return bank


def training_config(template, objective_gp, cost_model, variant, budget, seed, timing=False):
    cfg = copy.deepcopy(template)
    case = f'{variant}__{budget["level"]}'
    profile = 'timing' if timing else 'train'
    cfg['experiment'].update(name=f'phase4_{profile}__{case}', seed=seed,
                             save_every=CHECKPOINT_STEPS[0], cuda=True,
                             checkpoint_dir=f'checkpoints/final_training/{profile}',
                             run_dir=f'runs/final_training/{profile}')
    cfg['ppo']['total_timesteps'] = ((WARMUP_UPDATES + MEASURED_UPDATES) * 8192
                                    if timing else CHECKPOINT_STEPS[-1])
    for key in ('batch_size', 'minibatch_size', 'num_iterations'):
        cfg['ppo'].pop(key, None)
    cfg['problem_family']['instances'] = list(TRAIN)
    cfg['gp'] = copy.deepcopy(objective_gp)
    cfg['env'].update(budget=budget['budget'], max_acquisitions=61, device='cuda',
                      cost_feature_mode='none' if variant == 'no_cost' else 'predicted',
                      use_cost_uncertainty_feature=variant == 'cost_uncertainty')
    cfg['cost_model'] = ({'type': 'known', 'target_transform': 'log1p'} if variant == 'no_cost'
                         else copy.deepcopy(cost_model))
    cfg['cost_model']['include_uncertainty'] = variant == 'cost_uncertainty'
    return cfg


def prepare(output, results_root):
    output, results_root = relative_path(output), relative_path(results_root)
    require(not Path(output).exists() or not any(Path(output).iterdir()),
            f'{output} is not empty. Use check; do not overwrite a prepared protocol.')
    require(not Path(results_root).exists() or not any(Path(results_root).iterdir()),
            f'{results_root} already contains results. Choose a new results root.')
    template, phase, objective, cost = map(read_json, (TEMPLATE, PHASE2, OBJECTIVE, COST))
    calibrated = read_json(phase['budgets_file'])
    validation_cfg = read_json(phase['base_config'])
    require(template['problem_family']['instances'] == TRAIN, 'Training split changed.')
    require(validation_cfg['problem_family']['instances'] == VALIDATION, 'Validation split changed.')
    require(len(set(TRAIN + VALIDATION + TEST)) == 34, 'The 20/8/6 splits overlap.')
    require(objective['stage'] == 'objective' and cost['stage'] == 'cost', 'Wrong selection files.')
    require(not objective['failed_settings'] and not cost['failed_settings'], 'A GP selection has failures.')
    require(objective['gp'] == cost['gp'], 'Cost selection used a different objective GP.')
    require(objective['phase_config'] == cost['phase_config'] == phase, 'GP selection protocols differ.')
    require(objective['budgets'] == cost['budgets'] == calibrated['budgets'], 'Frozen budgets differ.')
    require(objective['episode_bank_sha256'] == cost['episode_bank_sha256'], 'Selection banks differ.')
    require(objective['gp'] == dict(kernel='rbf_matern52_mul', mean='constant', lr=0.1, training_iter=10),
            'Unexpected objective GP: review the protocol before changing it.')
    require(cost['cost_model'] == dict(type='gp', target_transform='log1p',
            gp=dict(kernel='rbf_matern52_mul', mean='linear', lr=0.01, training_iter=10)),
            'Unexpected cost GP: review the protocol before changing it.')
    require([b['level'] for b in calibrated['budgets']] == ['low', 'medium', 'high'], 'Budget order changed.')
    require(all(math.isfinite(b['budget']) and b['budget'] > 0 for b in calibrated['budgets']), 'Invalid budgets.')
    require((phase['episodes_per_instance'], phase['batch_size'], phase['max_acquisitions'], phase['seed'])
            == (32, 32, 61, 1), 'This protocol preserves the 32-episode Phase 2 bank and batch size.')
    for key in ('n_candidates', 'n_init', 'mask_visited_actions'):
        require(template['env'][key] == validation_cfg['env'][key] == calibrated['settings'][key],
                f'Training/calibration/validation mismatch: {key}')
    require({k: v for k, v in template['problem_family'].items() if k != 'instances'} ==
            {k: v for k, v in validation_cfg['problem_family'].items() if k != 'instances'},
            'Benchmark objective, scale, epoch or cost definition differs.')
    for key in ('dtype', 'objective_noise_std', 'objective_noise_clip', 'observation_format',
                'reward_type', 'reward_scale'):
        require(template['env'][key] == validation_cfg['env'][key], f'Validation mismatch: {key}')
    require((template['ppo']['num_envs'], template['ppo']['num_steps'], template['ppo']['num_minibatches'])
            == (256, 32, 4), 'The audited PPO rollout dimensions changed.')
    require(template['ppo']['anneal_lr'] is False, 'Timing assumes the same constant PPO learning rate.')
    signature = dict(format='phase2_step_noise_v1', seed=1, episodes_per_instance=32,
                     max_acquisitions=61, problem_family=validation_cfg['problem_family'],
                     n_candidates=64, n_init=3, dtype='float32')
    protocol = dict(
        version=1, prepared_utc=datetime.now(timezone.utc).isoformat(),
        reviewed_commit='37ea8b80ed05d93105603abe18e1d7dedd7e0af4',
        results_root=results_root, splits=dict(train=TRAIN, validation=VALIDATION, test=TEST),
        variants=VARIANTS, training_seeds=SEEDS, budgets=calibrated['budgets'],
        checkpoint_steps=CHECKPOINT_STEPS,
        selection_rule='For each variant/budget/training seed, minimum unrounded mean of instance means; exact ties choose earlier step. Keep every seed.',
        validation=dict(instances=VALIDATION, episodes_per_instance=32, batch_size=32,
                        seed=1, action_mode='argmax', episode_bank=phase['episode_bank'],
                        episode_bank_sha256=objective['episode_bank_sha256'], bank_signature=signature),
        timing=dict(warmup_updates=WARMUP_UPDATES, measured_updates=MEASURED_UPDATES),
        runs=[], timing_runs=[], inputs={}, source_hashes={},
    )
    read_bank(protocol)
    for path in (TEMPLATE, PHASE2, OBJECTIVE, COST, phase['budgets_file'], phase['base_config']):
        protocol['inputs'][path] = digest(path)
    source_files = [Path('scripts/phase4.py')]
    for directory in ('bo', 'envs', 'reinforcement_learning', 'utils'):
        source_files += [p for p in Path(directory).rglob('*.py') if not p.stem.endswith('_new')]
    protocol['source_hashes'] = {p.as_posix(): digest(p) for p in sorted(source_files)}
    for variant in VARIANTS:
        for budget in calibrated['budgets']:
            for seed in SEEDS:
                run_id = f'{variant}__{budget["level"]}__seed{seed}'
                path = f'{output}/train/{run_id}.json'
                write_json(path, training_config(template, objective['gp'], cost['cost_model'], variant, budget, seed))
                protocol['runs'].append(dict(run_id=run_id, variant=variant, budget_level=budget['level'],
                                             training_seed=seed, config=path, config_sha256=digest(path)))
            run_id = f'{variant}__{budget["level"]}'
            path = f'{output}/timing/{run_id}.json'
            write_json(path, training_config(template, objective['gp'], cost['cost_model'], variant, budget, 1, timing=True))
            protocol['timing_runs'].append(dict(run_id=run_id, variant=variant, budget_level=budget['level'],
                                                training_seed=1, config=path, config_sha256=digest(path)))
    protocol['protocol_sha256'] = object_digest(protocol)
    path = f'{output}/protocol.json'
    write_json(path, protocol)
    print(f'Prepared 27 training configs and 9 timing configs. Protocol: {path}')
    print('No training or benchmark evaluation has been started.')


def load_protocol(path):
    protocol = read_json(path)
    payload = {k: v for k, v in protocol.items() if k != 'protocol_sha256'}
    require(object_digest(payload) == protocol['protocol_sha256'], 'Protocol was edited after preparation.')
    require(protocol['checkpoint_steps'] == CHECKPOINT_STEPS and protocol['training_seeds'] == SEEDS,
            'Unexpected training schedule.')
    require(protocol['splits'] == dict(train=TRAIN, validation=VALIDATION, test=TEST), 'Split changed.')
    for section in ('inputs', 'source_hashes'):
        for file, expected in protocol[section].items():
            require(digest(file) == expected, f'{file} changed after preparation; review before starting a new protocol.')
    require(len(protocol['runs']) == 27 and len(protocol['timing_runs']) == 9, 'Incomplete run matrix.')
    for run in protocol['runs'] + protocol['timing_runs']:
        require(digest(run['config']) == run['config_sha256'], f'Config changed: {run["config"]}')
    read_bank(protocol)
    return protocol


def balanced_score(rows, instances, episodes):
    """Require one finite result for each fixed episode, then weight instances equally."""
    lookup = {}
    for row in rows:
        key = (str(row['instance']), int(row['episode_id']))
        require(key not in lookup, f'Duplicate validation episode: {key}')
        value = float(row['regret'])
        require(math.isfinite(value) and value >= 0, f'Invalid regret for {key}')
        lookup[key] = value
    expected = {(i, e) for i in instances for e in range(episodes)}
    require(set(lookup) == expected, 'Missing or unexpected validation episodes; selection stopped.')
    means = {i: math.fsum(lookup[(i, e)] for e in range(episodes)) / episodes for i in instances}
    return math.fsum(means.values()) / len(instances), means


def choose_checkpoint(scores, expected_steps):
    require(len(scores) == len(expected_steps) and {s['step'] for s in scores} == set(expected_steps),
            'All scheduled checkpoints must be evaluated before selecting a winner.')
    require(all(math.isfinite(s['mean_regret']) and s['mean_regret'] >= 0 for s in scores), 'Invalid checkpoint score.')
    return min(scores, key=lambda s: (s['mean_regret'], s['step']))


def run_policy_batch(env, agent, cache, init, noise, seed):
    """Fixed lanes, shared step-indexed noise, and deterministic policy maximization."""
    import torch as t

    obs = env.reset(seed, True, candidate_cache=cache, init_indices=init,
                    observation_noise=noise, auto_reset=False)
    rows = [None] * env.num_batches
    for _ in range(env.max_acquisitions):
        if bool(env.done.all()):
            break
        active = ~env.done
        require(bool(t.isfinite(obs[active]).all()), 'Non-finite policy observation.')
        with t.no_grad():
            logits = agent.get_logits(obs, action_mask=env.get_action_mask())
            require(bool(t.isfinite(logits[active]).all()), 'Non-finite action scores.')
            actions = logits.argmax(dim=1)
        # GP fitting inside step needs gradients; do not wrap it in no_grad.
        obs, _, terminal, infos = env.step(actions)
        for lane in t.where(terminal)[0].tolist():
            require(rows[lane] is None, 'A completed validation lane was reused.')
            info = infos['final_info'][lane]
            row = {key: info[key] for key in ('regret', 'best_value', 'best_oracle_value',
                    'ground_truth', 'budget_used', 'remaining_budget', 'budget_overshoot')}
            row.update(episode_length=info['episode']['l'], **{'return': info['episode']['r']})
            require(all(math.isfinite(float(v)) for v in row.values()), 'Non-finite episode result.')
            rows[lane] = row
    require(all(row is not None for row in rows), 'Validation did not finish every episode.')
    return rows


def evaluate_worker(protocol, run, checkpoint, output):
    """One isolated CUDA process: frees GPU memory before the next training job."""
    import numpy as np
    import torch as t
    from reinforcement_learning.agents.neural_AF import Agent
    from reinforcement_learning.ppo_no_gym import make_env
    from utils.seeding import seed_everything

    require(t.cuda.is_available(), 'CUDA is required; no automatic CPU fallback.')
    start_time = time.perf_counter()
    seed_everything(protocol['validation']['seed'], deterministic=True)
    bank = read_bank(protocol)
    cfg = read_json(run['config'])
    cfg['problem_family']['instances'] = list(VALIDATION)
    cfg['ppo']['num_envs'] = 32
    device = t.device('cuda')
    a = cfg['agent']
    agent = Agent(7, 7, 1, 1, a['num_layers'], a['layer_size'], a['critic_feature_mode']).to(device)
    weights = t.load(checkpoint['path'], map_location=device, weights_only=True)
    require(all(bool(t.isfinite(v).all()) for v in weights.values()), 'Checkpoint contains non-finite weights.')
    agent.load_state_dict(weights)
    agent.eval()
    all_rows = []
    for instance in VALIDATION:
        def tensor(suffix, dtype=t.float32):
            return t.as_tensor(bank[f'{instance}_{suffix}'], device=device, dtype=dtype)
        cache = (tensor('X'), tensor('y'), tensor('cost'), {'instances': [instance] * 32})
        env = make_env(cfg, device)
        rows = run_policy_batch(env, agent, cache, tensor('init', t.long), tensor('noise'), protocol['validation']['seed'])
        for episode_id, row in enumerate(rows):
            row.update(run_id=run['run_id'], variant=run['variant'], budget_level=run['budget_level'],
                       training_seed=run['training_seed'], checkpoint_step=checkpoint['step'],
                       instance=instance, episode_id=episode_id, budget=cfg['env']['budget'],
                       max_acquisitions=cfg['env']['max_acquisitions'])
        all_rows.extend(rows)
        print(f'  {instance}: 32 paired episodes', flush=True)
    score, by_instance = balanced_score(all_rows, VALIDATION, 32)
    t.cuda.synchronize()
    episode_path = Path(output) / 'episodes.csv'
    write_csv(episode_path, all_rows)
    result = dict(protocol_sha256=protocol['protocol_sha256'], config_sha256=run['config_sha256'],
                  checkpoint=checkpoint, episode_bank_sha256=protocol['validation']['episode_bank_sha256'],
                  episodes_csv=episode_path.as_posix(), episodes_sha256=digest(episode_path),
                  mean_regret=score, instance_means=by_instance, n_episodes=len(all_rows),
                  worker_seconds=time.perf_counter() - start_time, device_name=t.cuda.get_device_name(device),
                  torch_version=str(t.__version__), numpy_version=str(np.__version__))
    write_json(Path(output) / 'summary.json', result)


def checked_training_report(path, run):
    report = read_json(path)
    require(report['status'] == 'complete' and report['config_sha256'] == run['config_sha256'],
            'Training report belongs to another config or is incomplete.')
    cfg = read_json(run['config'])
    require(report['global_step'] == cfg['ppo']['total_timesteps'], 'Training ended before the full schedule.')
    require(len(report['iteration_seconds']) == cfg['ppo']['total_timesteps'] // 8192, 'Incomplete timing report.')
    for checkpoint in report['checkpoints']:
        require(digest(checkpoint['path']) == checkpoint['sha256'], f'Checkpoint changed: {checkpoint["path"]}')
    require(any(c['kind'] == 'final' and c['step'] == report['global_step'] for c in report['checkpoints']),
            'Missing final checkpoint.')
    return report


def launch_training(protocol, run, profile):
    output = Path(protocol['results_root']) / profile / run['run_id']
    report_path, job_path = output / 'training.json', output / 'job.json'
    if job_path.exists():
        job = read_json(job_path)
        require(job['protocol_sha256'] == protocol['protocol_sha256'], 'This result directory belongs to another protocol.')
    if report_path.exists():
        report = checked_training_report(report_path, run)
        print(f'Reusing completed {profile} run: {run["run_id"]}', flush=True)
        # A process can finish writing its report immediately before interruption.
        wall = read_json(job_path).get('process_seconds', report['total_seconds'])
        return report, wall
    if job_path.exists():
        print(f'Restarting incomplete run {run["run_id"]} from its seed; previous checkpoints are retained.', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    job = dict(protocol_sha256=protocol['protocol_sha256'], run_id=run['run_id'],
               started_utc=datetime.now(timezone.utc).isoformat())
    write_json(job_path, job)
    print(f'Starting {profile}: {run["run_id"]}', flush=True)
    start = time.perf_counter()
    subprocess.run([sys.executable, '-u', 'reinforcement_learning/ppo_no_gym.py', '--config', run['config'],
                    '--report', report_path.as_posix(), '--require-cuda'], cwd=ROOT, check=True)
    wall = time.perf_counter() - start
    report = checked_training_report(report_path, run)
    job.update(process_seconds=wall, status='complete')
    write_json(job_path, job)
    return report, wall


def launch_evaluation(protocol_path, protocol, run, checkpoint, profile):
    output = Path(protocol['results_root']) / profile / run['run_id'] / f'step_{checkpoint["step"]}'
    summary_path = output / 'summary.json'
    if not summary_path.exists():
        start = time.perf_counter()
        subprocess.run([sys.executable, '-u', 'scripts/phase4.py', '_evaluate', '--protocol', protocol_path,
                        '--run-id', run['run_id'], '--checkpoint', checkpoint['path'],
                        '--step', str(checkpoint['step']), '--profile', profile], cwd=ROOT, check=True)
        summary = read_json(summary_path)
        summary['process_seconds'] = time.perf_counter() - start
        write_json(summary_path, summary)
    summary = read_json(summary_path)
    require(summary['protocol_sha256'] == protocol['protocol_sha256'] and
            summary['config_sha256'] == run['config_sha256'] and summary['checkpoint'] == checkpoint,
            'Cached validation result does not match this checkpoint/protocol.')
    require(summary['episode_bank_sha256'] == protocol['validation']['episode_bank_sha256'], 'Validation bank differs.')
    require(digest(summary['episodes_csv']) == summary['episodes_sha256'], 'Validation CSV changed.')
    with Path(summary['episodes_csv']).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    score, by_instance = balanced_score(rows, VALIDATION, 32)
    require(score == summary['mean_regret'] and by_instance == summary['instance_means'], 'Validation summary mismatch.')
    return summary


def select_runs(runs, run_id):
    selected = [r for r in runs if run_id is None or r['run_id'] == run_id]
    require(bool(selected), f'Unknown run ID: {run_id}')
    return selected


def timing(protocol_path, protocol, run_id):
    for run in select_runs(protocol['timing_runs'], run_id):
        report, wall = launch_training(protocol, run, 'timing')
        checkpoint = next(c for c in report['checkpoints'] if c['kind'] == 'final')
        result = launch_evaluation(protocol_path, protocol, run, checkpoint, 'timing')
        print(f'Timing complete: {run["run_id"]}; training {wall / 60:.1f} min; validation '
              f'{result.get("process_seconds", result["worker_seconds"]) / 60:.1f} min.', flush=True)
    rows = []
    for run in protocol['timing_runs']:
        output = Path(protocol['results_root']) / 'timing' / run['run_id']
        report_path, summary_path = output / 'training.json', output / 'step_49152' / 'summary.json'
        if not report_path.exists() or not summary_path.exists():
            print('Timing is incomplete. Rerun timing without --run-id to finish all nine cases.')
            return
        report = checked_training_report(report_path, run)
        job, summary = read_json(output / 'job.json'), read_json(summary_path)
        measured = report['iteration_seconds'][WARMUP_UPDATES:]
        require(len(measured) == MEASURED_UPDATES and all(math.isfinite(x) and x > 0 for x in measured),
                'Invalid measured update times.')
        remaining_updates = 128 - WARMUP_UPDATES
        fixed = (max(0, job.get('process_seconds', report['total_seconds']) - report['total_seconds'])
                 + report['setup_seconds'] + sum(report['iteration_seconds'][:WARMUP_UPDATES])
                 + report['finalize_seconds'] + len(CHECKPOINT_STEPS) * report['final_save_seconds'])
        validation = summary.get('process_seconds', summary['worker_seconds'])
        training = fixed + remaining_updates * (math.fsum(measured) / len(measured))
        rows.append(dict(variant=run['variant'], budget_level=run['budget_level'],
                         mean_update_seconds=math.fsum(measured) / len(measured),
                         min_update_seconds=min(measured), max_update_seconds=max(measured),
                         estimated_training_seconds_per_seed=training,
                         validation_seconds_per_checkpoint=validation,
                         estimated_seconds_three_seeds=3 * (training + 4 * validation),
                         faster_observed_rate_seconds=3 * (fixed + remaining_updates * min(measured) + 4 * validation),
                         slower_observed_rate_seconds=3 * (fixed + remaining_updates * max(measured) + 4 * validation),
                         device_name=report['device_name']))
    require(len({r['device_name'] for r in rows}) == 1, 'Timing cases used different GPU models.')
    root = Path(protocol['results_root'])
    write_csv(root / 'timing_estimate.csv', rows)
    hours = sum(r['estimated_seconds_three_seeds'] for r in rows) / 3600
    estimate = dict(protocol_sha256=protocol['protocol_sha256'], estimated_total_hours=hours,
                    faster_observed_rate_hours=sum(r['faster_observed_rate_seconds'] for r in rows) / 3600,
                    slower_observed_rate_hours=sum(r['slower_observed_rate_seconds'] for r in rows) / 3600,
                    scope='27 full training runs plus 108 paired checkpoint evaluations; sequential on one GPU.',
                    limitation='Planning estimate, not a confidence interval. Later policies can change episode lengths and runtime. Timing runs are separate and are not final seeds.',
                    cases=rows)
    write_json(root / 'timing_estimate.json', estimate)
    print(f'Estimated full training + checkpoint validation: {hours:.2f} hours.')
    print(f'Planning range from observed update speeds: {estimate["faster_observed_rate_hours"]:.2f} to '
          f'{estimate["slower_observed_rate_hours"]:.2f} hours (not a confidence interval).')
    print(f'Report: {root / "timing_estimate.json"}')


def validate(protocol_path, protocol, run_id):
    selected_runs = select_runs(protocol['runs'], run_id)
    reports = {}
    for run in selected_runs:
        path = Path(protocol['results_root']) / 'train' / run['run_id'] / 'training.json'
        reports[run['run_id']] = checked_training_report(path, run)
    for run in selected_runs:
        report = reports[run['run_id']]
        # The periodic final-step file and model_final file contain the same weights.
        # Evaluate each scheduled step once, using its periodic file.
        checkpoints = [c for c in report['checkpoints'] if c['kind'] == 'periodic']
        require(sorted(c['step'] for c in checkpoints) == CHECKPOINT_STEPS, 'A scheduled checkpoint is missing.')
        scores = []
        for checkpoint in checkpoints:
            print(f'Validating {run["run_id"]}: step {checkpoint["step"]}', flush=True)
            result = launch_evaluation(protocol_path, protocol, run, checkpoint, 'validation')
            scores.append(dict(step=checkpoint['step'], mean_regret=result['mean_regret'],
                               instance_means=result['instance_means'], checkpoint=checkpoint,
                               episodes_csv=result['episodes_csv']))
        winner = choose_checkpoint(scores, CHECKPOINT_STEPS)
        payload = dict(protocol_sha256=protocol['protocol_sha256'], **run, scores=scores, selected=winner)
        write_json(Path(protocol['results_root']) / 'validation' / run['run_id'] / 'selected.json', payload)
        print(f'Selected {run["run_id"]}: step {winner["step"]}; validation mean regret {winner["mean_regret"]:.8f}')
    winners, score_rows = [], []
    for run in protocol['runs']:
        path = Path(protocol['results_root']) / 'validation' / run['run_id'] / 'selected.json'
        if not path.exists():
            print('Some runs still need validation; the combined 27-checkpoint manifest is not ready.')
            return
        result = read_json(path)
        require(result['protocol_sha256'] == protocol['protocol_sha256'], 'Mixed selection protocols.')
        winner = choose_checkpoint(result['scores'], CHECKPOINT_STEPS)
        require(winner == result['selected'], 'Stored selection is inconsistent.')
        winners.append(dict(run_id=run['run_id'], variant=run['variant'], budget_level=run['budget_level'],
                            training_seed=run['training_seed'], **winner))
        for score in result['scores']:
            score_rows.append(dict(run_id=run['run_id'], variant=run['variant'], budget_level=run['budget_level'],
                                   training_seed=run['training_seed'], checkpoint_step=score['step'],
                                   mean_regret=score['mean_regret'], selected=int(score['step'] == winner['step'])))
    root = Path(protocol['results_root'])
    write_csv(root / 'checkpoint_scores.csv', score_rows)
    write_json(root / 'selected_checkpoints.json', dict(protocol_sha256=protocol['protocol_sha256'],
               selection_rule=protocol['selection_rule'], selected_checkpoints=winners))
    print(f'Selected one checkpoint for each of all 27 runs: {root / "selected_checkpoints.json"}')


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Write the protocol and configs; no training.')
    prep.add_argument('--output', default='configs/final_training')
    prep.add_argument('--results-root', default='results/final_training')
    for command in ('check', 'timing', 'train', 'validate', '_evaluate'):
        sub = commands.add_parser(command)
        sub.add_argument('--protocol', default=DEFAULT_PROTOCOL)
        if command != 'check':
            sub.add_argument('--run-id')
        if command == '_evaluate':
            sub.add_argument('--checkpoint', required=True)
            sub.add_argument('--step', type=int, required=True)
            sub.add_argument('--profile', choices=('timing', 'validation'), required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output, args.results_root)
        return
    protocol = load_protocol(args.protocol)
    if args.command == 'check':
        print('PASS: 27 configs; selected GPs; 20/8/6 split; fixed budgets; paired bank; unchanged source/config hashes.')
    elif args.command == 'timing':
        timing(args.protocol, protocol, args.run_id)
    elif args.command == 'train':
        for run in select_runs(protocol['runs'], args.run_id):
            launch_training(protocol, run, 'train')
    elif args.command == 'validate':
        validate(args.protocol, protocol, args.run_id)
    else:
        require(args.run_id is not None, 'A worker needs a run ID.')
        runs = protocol['timing_runs'] if args.profile == 'timing' else protocol['runs']
        run = select_runs(runs, args.run_id)[0]
        report_path = Path(protocol['results_root']) / ('timing' if args.profile == 'timing' else 'train') / run['run_id'] / 'training.json'
        report = checked_training_report(report_path, run)
        matches = [c for c in report['checkpoints'] if c['path'] == args.checkpoint and c['step'] == args.step]
        require(len(matches) == 1, 'Checkpoint was not produced by this run.')
        output = Path(protocol['results_root']) / args.profile / run['run_id'] / f'step_{args.step}'
        evaluate_worker(protocol, run, matches[0], output)


if __name__ == '__main__':
    main()