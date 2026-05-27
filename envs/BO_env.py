################### QUICK FIX FOR IMPORTS: ############################################ 
import os, sys                                                                        #
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          #
#######################################################################################

import torch as t
from bo.gaussian_process_batch import RepeatedPadGPWrapper
#from bo.candidate_set_batch import BatchedCandidateSet
#from bo.problems.toy_rbf import ToyRBFProblemFamily
from bo.candidate_sets import build_candidate_set
from bo.problems.registry import build_problem_family



class BatchedBOEnv():
    '''Batched environment for one BO episode.'''
    def __init__(self, device, dtype, num_batches, n_candidates, n_init, budget, max_acquisitions, reward_type,
                 candidate_set_cfg, problem_family_cfg, gp_cfg, cost_model_cfg=None, mask_visited_actions=False,
                 objective_noise_std=0.0, objective_noise_clip=True):
        # Device and dtype
        self.device = t.device(device)
        self.dtype = dtype


        # Environment 
        self.num_batches = num_batches
        self.n_candidates = n_candidates
        self.n_init = n_init
        self.budget = budget
        self.remaining_budget = None
        self.reward_type = reward_type 
        self.max_acquisitions = max_acquisitions
        self.T_max = self.n_init + max_acquisitions
        self.last_obs = None
        self.problem_family = None
        self.params = None


        # Configs
        self.candidate_set_cfg = candidate_set_cfg
        self.problem_family_cfg = problem_family_cfg
        self.gp_cfg = gp_cfg
        self.cost_model_cfg = cost_model_cfg or {'type':'known'}
        self.use_cost_gp = self.cost_model_cfg.get('type', 'known') == 'gp'
        self.mask_visited_actions = mask_visited_actions
        self.objective_noise_std = float(objective_noise_std)
        self.objective_noise_clip = bool(objective_noise_clip)
        

        # Internal state
        self.X = None
        self.y_grid = None
        self.costs = None
        self.obj_gp = None
        self.cost_gp = None
        self.ep_return = None
        self.ep_len = None
        self.best_current_value = None
        self.best_oracle_value = None
        self.done = None
        self.visited = None



    def reset(self, seed, deterministic):
        '''Reset the environment'''
        # Set seed
        t.manual_seed(seed)
        t.backends.cudnn.deterministic = deterministic

        # Initialize problem family
        self.problem_family = build_problem_family(
            self.problem_family_cfg,
            device=self.device,
            dtype=self.dtype
        )

        if getattr(self.problem_family, 'provides_candidate_cache', False):
            self.candidate_set = None
            self.X, self.y_grid, self.costs, self.params = (
                self.problem_family.build_candidate_cache(
                    B=self.num_batches,
                    n_candidates=self.n_candidates,
                    seed=seed,
                )
            )

        else:
            # Initialize Candidate set
            self.candidate_set = build_candidate_set(
                self.candidate_set_cfg,
                device=self.device,
                dtype=self.dtype,
                B=self.num_batches,
            )

            self.X = self.candidate_set.get_grid()
            assert self.X.shape[1] == self.n_candidates


            # Sample parameters
            self.params = self.problem_family.sample_params(B=self.num_batches, seed=seed) 

            # Update y_grid with parameters
            self.y_grid = self.problem_family.evaluate(self.X, self.params)
            if self.y_grid.dim() == 3 and self.y_grid.size(-1) == 1:
                self.y_grid = self.y_grid.squeeze(-1)

            # Update costs with parameters
            self.costs = self.problem_family.costs(self.X, self.params)
            if self.costs.dim() == 3 and self.costs.size(-1) == 1:
                self.costs = self.costs.squeeze(-1)

        # Check dimensions
        assert self.X.shape[0] == self.num_batches
        assert self.X.shape[1] == self.n_candidates
        assert self.y_grid.shape == (self.num_batches, self.n_candidates)
        assert self.costs.shape == (self.num_batches, self.n_candidates)


        # Initialize gaussian process   
        d = self.X.shape[-1]   
        self.obj_gp = RepeatedPadGPWrapper(
            device=self.device,
            dtype=self.dtype,
            B=self.num_batches,
            T_max=self.T_max,
            d=d,
            lr=self.gp_cfg.get('lr', 1e-3),
            training_iter=self.gp_cfg.get('training_iter', 10),
            kernel=self.gp_cfg.get('kernel', 'rbf'),
        )

        # If cost gp, then initialize cost gp
        if self.use_cost_gp:
            self.cost_gp = RepeatedPadGPWrapper(
                device=self.device,
                dtype=self.dtype,
                B=self.num_batches,
                T_max=self.T_max,
                d=d,
                lr=self.gp_cfg.get('lr', 1e-3),
                training_iter=self.gp_cfg.get('training_iter', 10),
                kernel=self.gp_cfg.get('kernel', 'rbf'),
            )

        # Initialize initial points for each lane in the batch
        x_init, y_init, c_init, init_idx = self._sample_init_design()
        
        batch_idx = t.arange(self.num_batches, device=self.device).unsqueeze(1)
        y_init_oracle = self.y_grid[batch_idx, init_idx]

        self.visited = t.zeros((self.num_batches, self.n_candidates), device=self.device, dtype=t.bool)

        if self.mask_visited_actions:
            batch_idx = t.arange(self.num_batches, device=self.device).unsqueeze(1)
            self.visited[batch_idx, init_idx] = True

        self.obj_gp.set_lane_data(t.ones((self.num_batches,), device=self.device, dtype=t.bool), x_init, y_init)

        if self.use_cost_gp:
            self.cost_gp.set_lane_data(t.ones((self.num_batches,), device=self.device, dtype=t.bool), x_init, self._cost_to_gp_target(c_init))
                
        # Reset environment scalars and find best current values among init
        self.remaining_budget = t.full((self.num_batches,), self.budget, device=self.device, dtype=self.dtype)
        self.done = t.full((self.num_batches,), False, device=self.device, dtype=t.bool)
        self.best_current_value = y_init.max(dim=1).values
        self.best_oracle_value = y_init_oracle.max(dim=1).values

        # Compute initial mu, sigma and build observation
        obs = self._build_obs(*self._gp_predict_on_candidates())
        self.last_obs = obs

        self.ep_return = t.zeros((self.num_batches,), device=self.device, dtype=self.dtype) 
        self.ep_len = t.zeros((self.num_batches,), device=self.device, dtype=t.long) 

        return obs
    


    def _sample_init_design(self, lane_mask=None):
        '''
        Sample the n_init number of initial training points
        for lanes chosen by lane_mask (Default: all lanes)
        '''
        # Choose n_init random indices from length N
        B, N, d = self.X.shape

        if lane_mask is None:
            lane_mask = t.ones((B,), device=self.device, dtype=t.bool)

        lanes = t.where(lane_mask)[0]
        
        # Allocate full-shaped outputs
        train_x = t.zeros((B, self.n_init, d), device=self.device, dtype=self.dtype)
        train_y = t.zeros((B, self.n_init), device=self.device, dtype=self.dtype)
        train_cost = t.zeros((B, self.n_init), device=self.device, dtype=self.dtype)
        train_idx = t.zeros((B, self.n_init), device=self.device, dtype=t.long)

        # Saftery in case it is called with 0 mask
        if lanes.numel() == 0:
            return train_x, train_y, train_cost, train_idx

        # Choose init indices (shared across lanes for simplicity)
        idx = t.randperm(N, device=self.device)[: self.n_init]  # [n_init]

        train_x[lanes] = self.X[lanes][:, idx, :]

        y_mean = self.y_grid[lanes][:, idx]
        train_y[lanes] = self._observe_y(y_mean)

        train_cost[lanes] = self.costs[lanes][:, idx]

        train_idx[lanes] = idx.unsqueeze(0).expand(lanes.numel(), -1)

        return train_x, train_y, train_cost, train_idx

    

    def _reset_lanes(self, lane_mask):
        # Find which lanes to reset, if none, return
        lanes = t.where(lane_mask)[0]
        if lanes.numel() == 0:
            return
        
        if getattr(self.problem_family, "provides_candidate_cache", False):
            X_new, y_new, c_new, params_new = self.problem_family.build_candidate_cache(
                B=lanes.numel(),
                n_candidates=self.n_candidates,
                seed=None,
            )

            self.X[lanes] = X_new
            self.y_grid[lanes] = y_new
            self.costs[lanes] = c_new

            if hasattr(self.problem_family, "update_lane_params"):
                self.params = self.problem_family.update_lane_params(
                    self.params,
                    lanes,
                    params_new,
                )

        else:
            # Regenerate candidates for these lanes
            self.candidate_set.reset_lanes(lane_mask)
            self.X = self.candidate_set.get_grid()

            # Regenerate params for these lanes
            new_params = self.problem_family.sample_params(B=lanes.numel())
            for k in self.params:
                self.params[k][lanes] = new_params[k]

            # Recompute cost and y_grid for these lanes
            params_lanes = {k: v[lanes] for k,v in self.params.items()}

            y_lanes = self.problem_family.evaluate(self.X[lanes], params_lanes)
            if y_lanes.dim() == 3 and y_lanes.size(-1) == 1:
                y_lanes = y_lanes.squeeze(-1)

            c_lanes = self.problem_family.costs(self.X[lanes], params_lanes)
            if c_lanes.dim() == 3 and c_lanes.size(-1) == 1:
                c_lanes = c_lanes.squeeze(-1)

            self.y_grid[lanes] = y_lanes
            self.costs[lanes] = c_lanes


        # Reset budget / done
        self.remaining_budget[lanes] = self.budget
        self.done[lanes] = False

        # Reset info
        self.ep_return[lanes] = 0.0
        self.ep_len[lanes] = 0

        # New init design and load into GP buffers
        x_init, y_init, c_init, init_idx = self._sample_init_design(lane_mask) 
        y_init_oracle = self.y_grid[lanes.unsqueeze(1), init_idx[lanes]]

        if self.mask_visited_actions:
            self.visited[lanes] = False
            self.visited[lanes.unsqueeze(1), init_idx[lanes]] = True

        self.obj_gp.set_lane_data(lane_mask, x_init, y_init)

        if self.use_cost_gp:
            self.cost_gp.set_lane_data(lane_mask, x_init, self._cost_to_gp_target(c_init),)

        # Update best current values after new init
        self.best_current_value[lanes] = y_init[lanes].max(dim=1).values
        self.best_oracle_value[lanes] = y_init_oracle.max(dim=1).values
    


    def step(self, actions):
        B, _, d = self.X.shape
        active = ~self.done
        actions = actions.to(dtype=t.long)

        # Default outputs 
        reward = t.zeros((B,),  device=self.device, dtype=self.dtype)
        info = {}

        if active.any():
            # Retrieve points chosen in each batch by acquisition function, and cost
                # Reshape such that we can use torch.gather()
            x_idx = actions.view(B, 1, 1).expand(B, 1, d)
            x = t.gather(self.X, 1, x_idx)

                # Reshape such that we can use torch.gather()
            c_idx = actions.view(B, 1)
            cost = t.gather(self.costs, 1, c_idx).squeeze(1)

            y1_oracle = t.gather(self.y_grid, 1, c_idx).squeeze(1)
            y1 = self._observe_y(y1_oracle)
            y = y1.unsqueeze(1)

            if self.mask_visited_actions:
                active_lanes = t.where(active)[0]
                self.visited[active_lanes, actions[active_lanes]] = True

            # Add data for the active lanes
            self.obj_gp.add_data(x, y, active_mask=active)

            if self.use_cost_gp:
                self.cost_gp.add_data(x, self._cost_to_gp_target(cost), active_mask=active)

            # Update state for active lanes in the batch
            self.best_current_value[active] = t.maximum(self.best_current_value[active], y1[active])
            self.best_oracle_value[active] = t.maximum(self.best_oracle_value[active], y1_oracle[active])

            self.remaining_budget[active] -= cost[active]
            self.ep_len[active] += 1

            budget_done = self.remaining_budget <= 0
            length_done = self.ep_len >= self.max_acquisitions
            self.done = budget_done | length_done

        # Terminal mask for PPO, which lanes ended THIS step
        #terminal = self.done.clone()
        terminal = active & self.done

        if (self.reward_type == 'final_neglog_regret') and terminal.any():
            ground_truth = self.y_grid.max(dim=1).values
            regret = ground_truth - self.best_oracle_value
            reward[terminal] = -t.log(t.clamp(regret[terminal], min=1e-12))

        self.ep_return[active] += reward[active]

        info = {}

        if terminal.any():
            final_info = [None] * B
            terminal_lanes = t.where(terminal)[0]

            ground_truth = self.y_grid.max(dim=1).values
            regret = ground_truth - self.best_oracle_value

            for i in terminal_lanes.tolist():
                final_info[i] = {
                    'episode': {
                        'r': self.ep_return[i].item(),
                        'l': int(self.ep_len[i].item()),
                    },
                    'regret': regret[i].item(),
                    'best_value': self.best_current_value[i].item(),
                }

            info['final_info'] = final_info
            self._reset_lanes(terminal)

        obs = self._build_obs(*self._gp_predict_on_candidates())
        self.last_obs = obs

        return obs, reward, terminal, info
    


    def get_action_mask(self):
        """
        Returns:
            mask: [B, N] bool tensor.
                True means action is available.
                False means action is unavailable.
        """
        if not self.mask_visited_actions:
            return t.ones((self.num_batches, self.n_candidates), device=self.device, dtype=t.bool)

        mask = ~self.visited

        # Safety fallback: avoid all-invalid categorical distributions.
        no_valid = mask.sum(dim=1) == 0
        if no_valid.any():
            mask[no_valid] = True

        return mask
    

    def _observe_y(self, y_mean):
        """
        Convert latent objective values into observed objective values.

        y_mean is the deterministic benchmark mean, usually from self.y_grid.
        """
        if self.objective_noise_std <= 0.0:
            return y_mean

        noise = t.randn_like(y_mean) * self.objective_noise_std
        y_obs = y_mean + noise

        if self.objective_noise_clip:
            y_obs = y_obs.clamp(0.0, 1.0)

        return y_obs



    def _gp_predict_on_candidates(self):
        pred = self.obj_gp.predict(self.X)  
        mu = pred.mean      
        sigma = pred.stddev

        if self.use_cost_gp:
            cost_pred = self.cost_gp.predict(self.X)
            cost = self._gp_target_to_cost_mean(cost_pred)
        
        else:
            cost = self.costs

        max_cost = cost.max(dim=1, keepdim=True).values.expand_as(cost)

        return mu, sigma, cost, max_cost


    def _cost_to_gp_target(self, cost):
        '''Transform cost to log cost if specified'''
        transform = self.cost_model_cfg.get('target_transform', 'log1p')

        if transform == 'log1p':
            return t.log1p(cost.clamp_min(0.0))

        if transform == 'none':
            return cost

        raise ValueError(f'Unknown cost target transform: {transform}')
    

    def _gp_target_to_cost_mean(self, pred):
        '''...'''
        transform = self.cost_model_cfg.get('target_transform', 'log1p')

        if transform == 'log1p':
            return t.expm1(pred.mean).clamp_min(0.0)

        if transform == 'none':
            return pred.mean.clamp_min(0.0)

        raise ValueError(f'Unknown cost target transform: {transform}')
    

    def _build_obs(self, mu, sigma, cost, max_cost):
        B, N = mu.shape

        # global progress (time spent fraction) or remaining fraction
        progress = (1.0 - self.remaining_budget / self.budget).unsqueeze(1).expand(B, N)

        # best so far
        best = self.best_current_value.unsqueeze(1).expand(B, N)


        obs = t.stack([mu, sigma, cost / self.budget, progress, best, max_cost / self.budget], dim=-1)  # [B,N,6]
        return obs
    


























    ############
    # OLD CODE #
    ############
    def _sample_init_design_ARCHIVED(self, lane_mask=None):
        '''
        Sample the n_init number of initial training points
        for lanes chosen by lane_mask (Default: all lanes)
        '''
        # Choose n_init random indices from length N
        B, N, d = self.X.shape

        if lane_mask is None:
            lane_mask = t.ones((B,), device=self.device, dtype=t.bool)

        lanes = t.where(lane_mask)[0]
        
        # Allocate full-shaped outputs
        train_x = t.zeros((B, self.n_init, d), device=self.device, dtype=self.dtype)
        train_y = t.zeros((B, self.n_init), device=self.device, dtype=self.dtype)
        train_cost = t.zeros((B, self.n_init), device=self.device, dtype=self.dtype)

        # Saftery in case it is called with 0 mask
        if lanes.numel() == 0:
            return train_x, train_y, train_cost

        # Choose init indices (shared across lanes for simplicity)
        idx = t.randperm(N, device=self.device)[: self.n_init]  # [n_init]

        # Fill only selected lanes
        train_x[lanes] = self.X[lanes][:, idx, :]  # [n_lanes, n_init, d]

        # Evaluate only selected lanes using matching params slice
        params_lanes = {k: v[lanes] for k, v in self.params.items()}
        y = self.problem_family.evaluate(train_x[lanes], params_lanes)  # [n_lanes, n_init]
        c = self.problem_family.costs(train_x[lanes], params_lanes)

        if y.dim() == 3 and y.size(-1) == 1:
            y = y.squeeze(-1)

        if c.dim() == 3 and c.size(-1) == 1:
            c = c.squeeze(-1)


        train_y[lanes] = y
        train_cost[lanes] = c

        return train_x, train_y, train_cost