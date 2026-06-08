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
from reinforcement_learning.agents.neural_AF import Agent
from utils.config import load_config
from utils.seeding import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--action-mode', type=str, choices=['sample', 'argmax'], default='argmax')
    parser.add_argument("--episodes-per-instance", type=int, default=None, help="If set, collect this many completed episodes per instance.")
    parser.add_argument('--output', type=str, default='eval_results.csv')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def select_action(agent, obs, mask, action_mode: str):
    with t.no_grad():
        if action_mode == 'sample':
            action, _, _, _ = agent.get_action_and_value(obs, action_mask=mask)
            return action

        if action_mode == 'argmax':
            logits = agent.get_logits(obs, action_mask=mask)
            return t.argmax(logits, dim=1)

    raise ValueError(f'Unknown action mode: {action_mode}')


def main():
    # Get arguments and load config
    args = parse_args()
    cfg = load_config(args.config)

    # Extract main parts of config
    exp_cfg = cfg['experiment']
    ppo_cfg = cfg['ppo']
    agent_cfg = cfg['agent']

    # Set seed and device
    seed = args.seed if args.seed is not None else exp_cfg['seed']
    seed_everything(seed=seed, deterministic=exp_cfg['torch_deterministic'])
    device = t.device('cuda' if t.cuda.is_available() and exp_cfg['cuda'] else 'cpu')

    # Create environment and get initial observation
    env = make_env(cfg, device)
    obs = env.reset(seed=seed, deterministic=exp_cfg['torch_deterministic'])

    # Retrieve environment observation dimension shape for further computation
    _, _, d = obs.shape

    # Initialize agent
    agent = Agent(d, int(d / 2), 1, 1, agent_cfg['num_layers'], agent_cfg['layer_size']).to(device)
    
    # Load checkpoint if not none
    if args.checkpoint is not None:
        state_dict = t.load(args.checkpoint, map_location=device)
        agent.load_state_dict(state_dict)

    # Set agent to eval mode 
    agent.eval()

    # Init counters
    rows = []
    completed = 0
    step_count = 0

    # Initialize counter dict for completed instances
    if args.episodes_per_instance is not None:
        problem_cfg = cfg["problem_family"]
        if "instances" not in problem_cfg:
            raise ValueError(
                "--episodes-per-instance requires problem_family.instances in the config."
            )

        target_instances = [str(x) for x in problem_cfg["instances"]]
        completed_by_instance = {inst: 0 for inst in target_instances}

    done_collecting = False

    # Start evaluation loop
    while not done_collecting:

        # Get action and do one step in environment
        mask = env.get_action_mask()
        action = select_action(agent, obs, mask, args.action_mode)
        obs, reward, terminals, infos = env.step(action)
        step_count += 1

        # If an episode is not done, continue
        if 'final_info' not in infos:
            continue
        
        # If an episode is done, append results to row list
        for lane, info in enumerate(infos['final_info']):
            if info is None:
                continue
            
            # If episodes per instance and specific instance is completed, continue.
            instance = str(info.get("instance"))
            if args.episodes_per_instance is not None:
                if instance not in completed_by_instance:
                    continue

                if completed_by_instance[instance] >= args.episodes_per_instance:
                    continue
            
            # If not, increment count
            completed_by_instance[instance] += 1


            row = {
                'instance': info.get('instance'),
                'episode_index': completed,
                'lane': lane,
                'step_count': step_count,
                'action_mode': args.action_mode,
                'return': info['episode']['r'],
                'episode_length': info['episode']['l'],
                'regret': info.get('regret'),
                'best_value': info.get('best_value'),
                'best_oracle_value': info.get('best_oracle_value'),
                'ground_truth': info.get('ground_truth'),
                'budget_used': info.get('budget_used'),
                'remaining_budget': info.get('remaining_budget'),
                'budget_overshoot': max(0.0, -float(info.get('remaining_budget', 0.0)))
            }

            rows.append(row)
            completed += 1

            if args.episodes_per_instance is None and completed >= args.episodes:
                break

            if args.episodes_per_instance is not None:
                done_collecting = all(count >= args.episodes_per_instance for count in completed_by_instance.values())
            else:
                done_collecting = completed >= args.episodes
    
    # Make directory to save CSV File
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'instance',
        'episode_index',
        'lane',
        'step_count',
        'action_mode',
        'return',
        'episode_length',
        'regret',
        'best_value',
        'best_oracle_value',
        'ground_truth',
        'budget_used',
        'remaining_budget',
        'budget_overshoot'
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