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
from utils.config import load_config, save_config_copy
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

# Function to format seconds to hours, min and sec
def format_duration(seconds):
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def grad_l2_norm(parameters):
    """Return the combined L2 norm of all currently populated gradients."""
    grad_norms = [
        p.grad.detach().norm(2)
        for p in parameters
        if p.grad is not None
    ]
    if not grad_norms:
        return 0.0
    return torch.stack(grad_norms).norm(2).item()


def snapshot_parameters(module):
    """Clone a module's parameters for measuring one PPO update."""
    return [p.detach().clone() for p in module.parameters()]


def relative_parameter_update(module, before):
    """Return ||theta_after - theta_before||_2 / ||theta_before||_2."""
    delta_norms = []
    before_norms = []
    for current, previous in zip(module.parameters(), before):
        delta_norms.append((current.detach() - previous).norm(2))
        before_norms.append(previous.norm(2))

    delta_norm = torch.stack(delta_norms).norm(2)
    before_norm = torch.stack(before_norms).norm(2)
    return (delta_norm / before_norm.clamp_min(1e-12)).item()


def categorical_diagnostics(logits, action_mask):
    """Summarize policy concentration while excluding invalid actions."""
    valid_count = action_mask.sum(dim=1).clamp_min(1)
    valid_logits = logits.masked_fill(~action_mask, 0.0)
    mean_logits = valid_logits.sum(dim=1) / valid_count
    centered = (logits - mean_logits.unsqueeze(1)).masked_fill(~action_mask, 0.0)
    logit_std = torch.sqrt(
        centered.square().sum(dim=1) / valid_count
    )

    min_logits = logits.masked_fill(~action_mask, float("inf")).min(dim=1).values
    max_logits = logits.masked_fill(~action_mask, float("-inf")).max(dim=1).values

    log_probs = torch.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=1)
    max_entropy = valid_count.to(logits.dtype).log()
    normalized_entropy = torch.where(
        valid_count > 1,
        entropy / max_entropy.clamp_min(1e-12),
        torch.ones_like(entropy),
    )

    return {
        "mean_max_probability": probs.max(dim=1).values.mean().item(),
        "mean_logit_std": logit_std.mean().item(),
        "mean_logit_range": (max_logits - min_logits).mean().item(),
        "mean_entropy": entropy.mean().item(),
        "mean_normalized_entropy": normalized_entropy.mean().item(),
    }


def categorical_kl(old_logits, new_logits):
    """Mean exact KL(old policy || new policy) over a batch of states."""
    old_log_probs = torch.log_softmax(old_logits, dim=1)
    new_log_probs = torch.log_softmax(new_logits, dim=1)
    old_probs = old_log_probs.exp()
    return (old_probs * (old_log_probs - new_log_probs)).sum(dim=1).mean().item()


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
        cost_model_cfg = cfg.get("cost_model", {"type" : "known"}),
        mask_visited_actions=env_cfg.get("mask_visited_actions", False),
        objective_noise_std=env_cfg.get("objective_noise_std", 0.0),
        objective_noise_clip=env_cfg.get("objective_noise_clip", True),
        cost_feature_mode=env_cfg.get("cost_feature_mode", "predicted"),
        use_cost_uncertainty_feature=env_cfg.get("use_cost_uncertainty_feature", False),
        observation_format=env_cfg.get("observation_format", "cost6")
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
    action_masks = torch.ones((ppo_cfg["num_steps"], B, N), dtype=torch.bool, device=device)
    next_done = torch.zeros(B, dtype=next_obs.dtype, device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()


    # CHECKPOINT SETUP
    save_every = exp_cfg['save_every']
    next_save_step = save_every

    checkpoint_root = Path(exp_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir = checkpoint_root / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    last_saved_step = -1

    for iteration in range(1, ppo_cfg['num_iterations'] + 1):
        rollout_episode_returns = []
        rollout_episode_lengths = []
        rollout_regrets = []

        # Annealing the rate if instructed to do so.
        if ppo_cfg['anneal_lr']:
            frac = 1.0 - (iteration - 1.0) / ppo_cfg['num_iterations']
            lrnow = frac * ppo_cfg['learning_rate']
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, ppo_cfg['num_steps']):
            global_step += ppo_cfg['num_envs']
            obs[step] = next_obs
            dones[step] = next_done

            # Get mask for available actions
            mask = env.get_action_mask()
            action_masks[step] = mask

            # ALGO LOGIC: action logic
            with torch.no_grad():
                #action, logprob, _, value = agent.get_action_and_value(next_obs) (old without mask)
                action, logprob, _, value = agent.get_action_and_value(next_obs, action_mask=mask)
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
                            rollout_episode_returns.append(info["episode"]["r"])
                            rollout_episode_lengths.append(info["episode"]["l"])

                        if "regret" in info:
                            writer.add_scalar("eval/final_regret", info["regret"], global_step)
                            rollout_regrets.append(info["regret"])

                        if "best_value" in info:
                            writer.add_scalar("eval/best_value", info["best_value"], global_step)

                        if "best_oracle_value" in info:
                            writer.add_scalar("eval/best_oracle_value", info["best_oracle_value"], global_step)

                        if "ground_truth" in info:
                            writer.add_scalar("eval/ground_truth", info["ground_truth"], global_step)
                            
                        if "budget_used" in info:
                            writer.add_scalar("eval/budget_used", info["budget_used"], global_step)

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
        b_action_masks = action_masks.reshape((-1, N))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        with torch.no_grad():
            pre_update_logits = agent.get_logits(
                b_obs,
                action_mask=b_action_masks,
            ).detach()
            pre_policy_stats = categorical_diagnostics(
                pre_update_logits,
                b_action_masks,
            )

        actor_before = snapshot_parameters(agent.actor)
        critic_before = snapshot_parameters(agent.critic)

        # Optimizing the policy and value network
        b_inds = np.arange(ppo_cfg['batch_size'])
        clipfracs = []
        value_losses = []
        policy_losses = []
        entropy_losses = []
        old_approx_kls = []
        approx_kls = []
        actor_grad_norms_pre_clip = []
        critic_grad_norms_pre_clip = []
        total_grad_norms_pre_clip = []
        actor_grad_norms_post_clip = []
        critic_grad_norms_post_clip = []
        grad_was_clipped = []
        for epoch in range(ppo_cfg['update_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, ppo_cfg['batch_size'], ppo_cfg['minibatch_size']):
                end = start + ppo_cfg['minibatch_size']
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], action_mask=b_action_masks[mb_inds])
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

                actor_grad_pre = grad_l2_norm(agent.actor.parameters())
                critic_grad_pre = grad_l2_norm(agent.critic.parameters())
                total_grad_pre = (actor_grad_pre ** 2 + critic_grad_pre ** 2) ** 0.5

                nn.utils.clip_grad_norm_(agent.parameters(), ppo_cfg['max_grad_norm'])

                actor_grad_post = grad_l2_norm(agent.actor.parameters())
                critic_grad_post = grad_l2_norm(agent.critic.parameters())
                optimizer.step()

                value_losses.append(v_loss.item())
                policy_losses.append(pg_loss.item())
                entropy_losses.append(entropy_loss.item())
                old_approx_kls.append(old_approx_kl.item())
                approx_kls.append(approx_kl.item())
                actor_grad_norms_pre_clip.append(actor_grad_pre)
                critic_grad_norms_pre_clip.append(critic_grad_pre)
                total_grad_norms_pre_clip.append(total_grad_pre)
                actor_grad_norms_post_clip.append(actor_grad_post)
                critic_grad_norms_post_clip.append(critic_grad_post)
                grad_was_clipped.append(
                    float(total_grad_pre > ppo_cfg['max_grad_norm'])
                )

            if ppo_cfg['target_kl'] is not None and approx_kl > ppo_cfg['target_kl']:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        with torch.no_grad():
            post_update_logits = agent.get_logits(
                b_obs,
                action_mask=b_action_masks,
            ).detach()
            post_policy_stats = categorical_diagnostics(
                post_update_logits,
                b_action_masks,
            )
            full_policy_kl = categorical_kl(
                pre_update_logits,
                post_update_logits,
            )

        mean_actor_grad_pre = float(np.mean(actor_grad_norms_pre_clip))
        mean_critic_grad_pre = float(np.mean(critic_grad_norms_pre_clip))

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", np.mean(value_losses), global_step)
        writer.add_scalar("losses/policy_loss", np.mean(policy_losses), global_step)
        writer.add_scalar("losses/entropy", np.mean(entropy_losses), global_step)
        writer.add_scalar("losses/old_approx_kl", np.mean(old_approx_kls), global_step)
        writer.add_scalar("losses/approx_kl", np.mean(approx_kls), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        for name, value in pre_policy_stats.items():
            writer.add_scalar(f"policy/pre_update_{name}", value, global_step)
        for name, value in post_policy_stats.items():
            writer.add_scalar(f"policy/post_update_{name}", value, global_step)
        writer.add_scalar("policy/full_batch_kl", full_policy_kl, global_step)

        writer.add_scalar("rollout/advantage_mean", b_advantages.mean().item(), global_step)
        writer.add_scalar("rollout/advantage_std", b_advantages.std(unbiased=False).item(), global_step)
        writer.add_scalar("rollout/advantage_min", b_advantages.min().item(), global_step)
        writer.add_scalar("rollout/advantage_max", b_advantages.max().item(), global_step)
        writer.add_scalar("rollout/return_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("rollout/return_std", b_returns.std(unbiased=False).item(), global_step)
        writer.add_scalar("rollout/value_mean", b_values.mean().item(), global_step)
        writer.add_scalar("rollout/value_std", b_values.std(unbiased=False).item(), global_step)
        writer.add_scalar("rollout/episode_count", len(rollout_regrets), global_step)
        writer.add_scalar(
            "rollout/terminal_transition_fraction",
            len(rollout_regrets) / ppo_cfg['batch_size'],
            global_step,
        )

        if rollout_regrets:
            writer.add_scalar("rollout/mean_episode_return", np.mean(rollout_episode_returns), global_step)
            writer.add_scalar("rollout/mean_episode_length", np.mean(rollout_episode_lengths), global_step)
            writer.add_scalar("rollout/mean_regret", np.mean(rollout_regrets), global_step)
            writer.add_scalar("rollout/median_regret", np.median(rollout_regrets), global_step)
            writer.add_scalar(
                "rollout/reward_saturation_fraction",
                np.mean(np.asarray(rollout_regrets) <= 1e-12),
                global_step,
            )

        writer.add_scalar("diagnostics/actor_grad_norm_pre_clip", mean_actor_grad_pre, global_step)
        writer.add_scalar("diagnostics/critic_grad_norm_pre_clip", mean_critic_grad_pre, global_step)
        writer.add_scalar(
            "diagnostics/critic_actor_grad_ratio",
            mean_critic_grad_pre / max(mean_actor_grad_pre, 1e-12),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/total_grad_norm_pre_clip",
            np.mean(total_grad_norms_pre_clip),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/max_total_grad_norm_pre_clip",
            np.max(total_grad_norms_pre_clip),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/actor_grad_norm_post_clip",
            np.mean(actor_grad_norms_post_clip),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/critic_grad_norm_post_clip",
            np.mean(critic_grad_norms_post_clip),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/grad_clip_fraction",
            np.mean(grad_was_clipped),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/actor_relative_parameter_update",
            relative_parameter_update(agent.actor, actor_before),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/critic_relative_parameter_update",
            relative_parameter_update(agent.critic, critic_before),
            global_step,
        )
        elapsed = time.time() - start_time
        sps = global_step / max(elapsed, 1e-8)
        writer.add_scalar("charts/SPS", sps, global_step)

        total_train_steps = (
            ppo_cfg["num_iterations"]
            * ppo_cfg["num_envs"]
            * ppo_cfg["num_steps"]
        )

        remaining_steps = max(total_train_steps - global_step, 0)
        eta_seconds = remaining_steps / max(sps, 1e-8)

        print(
            f"Step {global_step}/{total_train_steps} "
            f"({100.0 * global_step / total_train_steps:.1f}%) | "
            f"Elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(eta_seconds)} | "
            f"SPS {int(sps)}"
        )

        if global_step >= next_save_step:
            ckpt_path = checkpoint_dir / f"model_step_{global_step}.pt"
            torch.save(agent.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            last_saved_step = global_step
            next_save_step += save_every
        

    final_ckpt_path = checkpoint_dir / f"model_final_step_{global_step}.pt"
    torch.save(agent.state_dict(), final_ckpt_path)
    print(f"Saved final checkpoint: {final_ckpt_path}")

    writer.close()
