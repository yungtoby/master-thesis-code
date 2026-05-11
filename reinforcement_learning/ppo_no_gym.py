#########################################################################################################################
# THE FOLLOWING CODE (LINE 5 - 316) IS FROM CLEANRL. URL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py #
# DISCLAIMER: MAJOR CHANGES HAS BEEN MADE TO FIT THESIS PROJECT.                                                        #
#########################################################################################################################
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy



################### QUICK FIX FOR IMPORTS: ############################################ 
import os, sys                                                                        #
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          #
#######################################################################################



# IMPORTS
import os
import random
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from envs.BO_env import BatchedBOEnv
from reinforcement_learning.agents.neural_AF import Agent
from utils.config import load_config, add_derived_ppo_values, save_config_copy
from utils.seeding import seed_everything



# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy_rbf_debug.json",
        help="Path to experiment config JSON.",
    )
    return parser.parse_args()



def make_env(cfg, device):
    env_cfg = cfg['env']
    ppo_cfg = cfg['ppo']

    dtype_map = {
        "float32" : torch.float32,
        "float64" : torch.float64
    }

    return BatchedBOEnv(
        device=str(device),
        dtype=dtype_map[env_cfg["dtype"]],
        num_batches=ppo_cfg["num_envs"],
        n_candidates=env_cfg["n_candidates"],
        n_init=env_cfg["n_init"],
        budget=env_cfg["budget"],
        max_acquisitions=env_cfg["max_acquisitions"],
        reward_type=env_cfg["reward_type"],
        candidate_set_cfg=cfg["candidate_set"],
        problem_family_cfg=cfg["problem_family"],
        gp_cfg=cfg["gp"],
    )




if __name__ == "__main__":
    # Parse arguments given to CLI and load corresponding config
    cli_args = parse_args()
    cfg = load_config(cli_args.config)

    # Divide config into different parts PPO, Experiment, etc etc
    exp_cfg = cfg["experiment"]
    ppo_cfg = cfg["ppo"]
    agent_cfg = cfg["agent"]
    
    # Seed everything
    seed_everything(
        seed=exp_cfg["seed"],
        deterministic=exp_cfg["torch_deterministic"],
    )

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and exp_cfg["cuda"] else "cpu"
    )
    print(f"Using device {device}")

    # Create experiment name
    run_name = (
        f"{exp_cfg['env_id']}__"
        f"{exp_cfg['name']}__"
        f"{exp_cfg['seed']}__"
        f"{int(time.time())}"
    )

    # Save run to its directory and save config for run
    run_dir = Path(exp_cfg.get("run_dir", "runs")) / run_name
    writer = SummaryWriter(str(run_dir))
    save_config_copy(cli_args.config, run_dir)


    # Environement setup 
    env = make_env(cfg, device)

    
    
    # START PPO:
    next_obs = env.reset(seed=exp_cfg['seed'], deterministic=exp_cfg['torch_deterministic'])
    B, N, d = next_obs.shape

    # Agent setup
    agent = Agent(d, int(d/2), 1, 1, agent_cfg['num_layers'], agent_cfg['layer_size']).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=ppo_cfg['learning_rate'], eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((ppo_cfg['num_steps'], B, N, d), dtype=next_obs.dtype, device=device)
    actions = torch.zeros((ppo_cfg['num_steps'], B), dtype=torch.long, device=device)
    logprobs = torch.zeros((ppo_cfg['num_steps'], B), dtype=next_obs.dtype, device=device)
    rewards = torch.zeros((ppo_cfg['num_steps'], B), dtype=next_obs.dtype, device=device)
    dones = torch.zeros((ppo_cfg['num_steps'], B), dtype=next_obs.dtype, device=device)
    values = torch.zeros((ppo_cfg['num_steps'], B), dtype=next_obs.dtype, device=device)
    next_done = torch.zeros(B, dtype=next_obs.dtype, device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()


    # CHECKPOINT SETUP
    save_every = exp_cfg['save_every']
    next_save_step = save_every
    os.makedirs("checkpoints", exist_ok=True)

    for iteration in range(1, ppo_cfg['num_iterations'] + 1):
        # Annealing the rate if instructed to do so.
        if ppo_cfg['anneal_lr']:
            frac = 1.0 - (iteration - 1.0) / ppo_cfg['num_iterations']
            lrnow = frac * ppo_cfg['learning_rate']
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, ppo_cfg['num_steps']):
            global_step += ppo_cfg['num_envs']
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminals, infos = env.step(action)
            next_done = terminals
            rewards[step] = reward.view(-1)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        if "episode" in info:
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                        if "regret" in info:
                            writer.add_scalar("eval/final_regret", info["regret"], global_step)

                        if "best_value" in info:
                            writer.add_scalar("eval/best_value", info["best_value"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros(B, dtype=next_obs.dtype, device=device)

            for t in reversed(range(ppo_cfg['num_steps'])):
                if t == ppo_cfg['num_steps'] - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + ppo_cfg['gamma'] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + ppo_cfg['gamma'] * ppo_cfg['gae_lambda'] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, N, d))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(ppo_cfg['batch_size'])
        clipfracs = []
        for epoch in range(ppo_cfg['update_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, ppo_cfg['batch_size'], ppo_cfg['minibatch_size']):
                end = start + ppo_cfg['minibatch_size']
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > ppo_cfg['clip_coef']).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if ppo_cfg['norm_adv']:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - ppo_cfg['clip_coef'], 1 + ppo_cfg['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if ppo_cfg['clip_vloss']:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -ppo_cfg['clip_coef'],
                        ppo_cfg['clip_coef'],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ppo_cfg['ent_coef'] * entropy_loss + v_loss * ppo_cfg['vf_coef']

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo_cfg['max_grad_norm'])
                optimizer.step()

            if ppo_cfg['target_kl'] is not None and approx_kl > ppo_cfg['target_kl']:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(f"Total time used: {(time.time() - start_time)/60:.2f} min")
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if global_step >= next_save_step:
            torch.save(agent.state_dict(), f"checkpoints/model_step_{global_step}.pt")
            next_save_step += save_every
        

    torch.save(agent.state_dict(), f"checkpoints/model_step_{global_step}.pt")
    writer.close()