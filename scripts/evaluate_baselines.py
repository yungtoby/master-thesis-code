from __future__ import annotations

################### QUICK FIX FOR IMPORTS: ############################################ 
import os, sys                                                                        #
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          #
#######################################################################################

import argparse
import csv
from pathlib import Path
import torch as t

from reinforcement_learning.ppo_no_gym import make_env
from utils.config import load_config
from utils.seeding import seed_everything



def expected_improvement(mu, sigma, best, eps=1e-9):
    '''
    Maximization EI.

    mu:    [B, N]
    sigma: [B, N]
    best:  [B]
    '''
    sigma = sigma.clamp_min(eps)
    improvement = mu - best.unsqueeze(1)

    z = improvement / sigma
    normal = t.distributions.Normal(
        t.tensor(0.0, device=mu.device, dtype=mu.dtype),
        t.tensor(1.0, device=mu.device, dtype=mu.dtype),
    )

    ei = improvement * normal.cdf(z) + sigma * t.exp(normal.log_prob(z))
    return ei.clamp_min(0.0)



def masked_argmax(scores, mask):
    scores = scores.masked_fill(~mask, -t.inf)
    return t.argmax(scores, dim=1)



def masked_random(mask):
    probs = mask.float()
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1.0)
    return t.multinomial(probs, num_samples=1).squeeze(1)



def select_baseline_action(env, obs, mask, method: str):
    mu = obs[..., 0]    # [..., 0] same as [:, :, ..., :, 0]
    sigma = obs[..., 1]
    cost = (obs[..., 2] * env.budget).clamp_min(1e-8)
    best = env.best_current_value

    if method == 'random':
        return masked_random(mask)

    ei = expected_improvement(mu, sigma, best)

    if method == 'ei':
        scores = ei

    elif method == 'eipu':
        scores = ei / cost

    elif method == 'ei_cool':
        # alpha = remaining budget fraction.
        # Early: alpha approx 1 -> EI / cost.
        # Late:  alpha approx 0 -> EI.
        alpha = (env.remaining_budget / env.budget).clamp(0.0, 1.0).unsqueeze(1)
        scores = ei / cost.pow(alpha)

    else:
        raise ValueError(f'Unknown baseline method: {method}')

    return masked_argmax(scores, mask)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--method', type=str, choices=['random', 'ei', 'eipu', 'ei_cool'], required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def main():
    # Get arguments and load config
    args = parse_args()
    cfg = load_config(args.config)

    # Extract part of config needed and seed everything
    exp_cfg = cfg['experiment']
    seed = args.seed if args.seed is not None else exp_cfg['seed']
    seed_everything(seed=seed, deterministic=exp_cfg['torch_deterministic'])
    device = t.device('cuda' if t.cuda.is_available() and exp_cfg['cuda'] else 'cpu')

    # Create environment and get initial observation
    env = make_env(cfg, device)
    obs = env.reset(seed=seed, deterministic=exp_cfg['torch_deterministic'])

    # Init counters
    rows = []
    completed = 0
    step_count = 0

    # Start evaluation loop
    while completed < args.episodes:

        # Get action and do one step in environment
        mask = env.get_action_mask()
        with t.no_grad():
            action = select_baseline_action(env, obs, mask, args.method)
        obs, reward, terminals, infos = env.step(action)
        step_count += 1

        # If an episode is not done, continue
        if 'final_info' not in infos:
            continue

        # If an episode is done, append results to row list
        for lane, info in enumerate(infos['final_info']):
            if info is None:
                continue

            row = {
                'instance': info.get('instance'),
                'episode_index': completed,
                'lane': lane,
                'step_count': step_count,
                'method': args.method,
                'return': info['episode']['r'],
                'episode_length': info['episode']['l'],
                'regret': info.get('regret'),
                'best_value': info.get('best_value'),
                'best_oracle_value': info.get('best_oracle_value'),
                'ground_truth': info.get('ground_truth'),
                'budget_used': info.get('budget_used'),
                'remaining_budget': info.get('remaining_budget'),
                'budget_overshoot': max(0.0, -float(info.get('remaining_budget', 0.0))),
            }

            rows.append(row)
            completed += 1

            if completed >= args.episodes:
                break

    # Make directory to save CSV File
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'instance',
        'episode_index',
        'lane',
        'step_count',
        'method',
        'return',
        'episode_length',
        'regret',
        'best_value',
        'best_oracle_value',
        'ground_truth',
        'budget_used',
        'remaining_budget',
        'budget_overshoot',
    ]

    # Write to csv file and save
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    regrets = t.tensor([r['regret'] for r in rows if r['regret'] is not None])
    lengths = t.tensor([r['episode_length'] for r in rows])
    overshoot = t.tensor([r['budget_overshoot'] for r in rows])

    print(f'Saved {len(rows)} episodes to {output_path}')
    if len(regrets) > 0:
        print(f'Mean regret: {regrets.mean().item():.6f}')
        print(f'Median regret: {regrets.median().item():.6f}')
    print(f'Mean episode length: {lengths.float().mean().item():.2f}')
    print(f'Mean budget overshoot: {overshoot.float().mean().item():.2f}')


if __name__ == '__main__':
    main()