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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from envs.BO_no_gym_env_FINAL import BatchedBOEnv
from reinforcement_learning.agents.neural_AF import Agent


# CURRENT PLACEHOLDER FOR ARGS COMMAND USED BY CLEANRL
args = {
    'exp_name' : os.path.basename(__file__)[: -len(".py")],
    'seed' : 1,
    'torch_deterministic' : True,
    'cuda' : True,
    'track' : False,
    'capture_video' : False,
    'env_id' : "COST-AWARE-BO-PROTOTYPE",
    'total_timesteps' : 5000000,
    'learning_rate' : 1e-4,
    'num_envs' : 256,
    'num_steps' : 32,
    'anneal_lr' : True,
    'gamma' : 1,
    'gae_lambda' : 0.98,
    'num_minibatches' : 4,
    'update_epochs' : 4,
    'norm_adv' : True,
    'clip_coef' : 0.2,
    'clip_vloss' : True,
    'ent_coef' : 0.01,
    'vf_coef' : 1,
    'max_grad_norm' : 0.5,
    'target_kl' : 0.3,
    'batch_size': 0,
    'minibatch_size' : 0,
    'num_iterations' : 0
}



def make_env():
    env = BatchedBOEnv(
        device='cuda',
        dtype=torch.float32,
        num_batches=args['num_envs'],
        n_candidates=900,
        n_init=3,
        budget=30,
        max_acquistions=200,
        reward_type='final_neglog_regret'
    )
    return env




if __name__ == "__main__":
    # Calculating batch size, minibatch size and number of iterations
    args['batch_size'] = int(args['num_envs'] * args['num_steps'])
    args['minibatch_size'] = int(args['batch_size'] // args['num_minibatches'])
    args['num_iterations'] = args['total_timesteps'] // args['batch_size']


    # Name of the current run
    run_name = f"{args['env_id']}__{args['exp_name']}__{args['seed']}__{int(time.time())}"


    # Initializing summarywriter (tensorboard logic)
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in args.items()])),
    )


    # TRY NOT TO MODIFY: seeding
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = args['torch_deterministic']


    # Selecting device
    device = torch.device('cuda' if torch.cuda.is_available() and args['cuda'] else 'cpu')
    print(f"Using device {device}")

    # Environement setup 
    env = make_env()
    
    # START PPO:
    next_obs = env.reset(seed=args['seed'], deterministic=args['torch_deterministic'])
    B, N, d = next_obs.shape

    # Agent setup
    agent = Agent(d, int(d/2), 1, 1, 3, 64).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args['learning_rate'], eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args['num_steps'], B, N, d), dtype=next_obs.dtype, device=device)
    actions = torch.zeros((args['num_steps'], B), dtype=torch.long, device=device)
    logprobs = torch.zeros((args['num_steps'], B), dtype=next_obs.dtype, device=device)
    rewards = torch.zeros((args['num_steps'], B), dtype=next_obs.dtype, device=device)
    dones = torch.zeros((args['num_steps'], B), dtype=next_obs.dtype, device=device)
    values = torch.zeros((args['num_steps'], B), dtype=next_obs.dtype, device=device)
    next_done = torch.zeros(B, dtype=next_obs.dtype, device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()


    # CHECKPOINT SETUP
    save_every = 250000
    next_save_step = save_every
    os.makedirs("checkpoints", exist_ok=True)

    for iteration in range(1, args['num_iterations'] + 1):
        # Annealing the rate if instructed to do so.
        if args['anneal_lr']:
            frac = 1.0 - (iteration - 1.0) / args['num_iterations']
            lrnow = frac * args['learning_rate']
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args['num_steps']):
            global_step += args['num_envs']
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

            for t in reversed(range(args['num_steps'])):
                if t == args['num_steps'] - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args['gamma'] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args['gamma'] * args['gae_lambda'] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, N, d))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args['batch_size'])
        clipfracs = []
        for epoch in range(args['update_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, args['batch_size'], args['minibatch_size']):
                end = start + args['minibatch_size']
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args['clip_coef']).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args['norm_adv']:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args['clip_coef'], 1 + args['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args['clip_vloss']:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args['clip_coef'],
                        args['clip_coef'],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args['ent_coef'] * entropy_loss + v_loss * args['vf_coef']

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args['max_grad_norm'])
                optimizer.step()

            if args['target_kl'] is not None and approx_kl > args['target_kl']:
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