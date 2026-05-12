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
                 candidate_set_cfg, problem_family_cfg, gp_cfg, cost_model_cfg=None):
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
        self.T_max = self.n_init + max_acquisitions
        self.last_obs = None
        self.problem_family = None
        self.params = None

        # Configs
        self.candidate_set_cfg = candidate_set_cfg
        self.problem_family_cfg = problem_family_cfg
        self.gp_cfg = gp_cfg
        self.cost_model_cfg = cost_model_cfg or {"type":"known"}
        self.use_cost_gp = self.cost_model_cfg.get("type", "known") == "gp"

        # Internal state
        self.X = None
        self.costs = None
        self.obj_gp = None
        self.cost_gp = None
        self.ep_return = None
        self.ep_len = None
        self.best_current_value = None
        self.done = None



    def reset(self, seed, deterministic):
        '''Reset the environment'''
        # Set seed
        t.manual_seed(seed)
        t.backends.cudnn.deterministic = deterministic


        # Initialize Candidate set
        self.candidate_set = build_candidate_set(
            self.candidate_set_cfg,
            device=self.device,
            dtype=self.dtype,
            B=self.num_batches,
        )

        self.X = self.candidate_set.get_grid()
        
        assert self.X.shape[1] == self.n_candidates

        # Initialize cost and objective func params
        self.problem_family = build_problem_family(
            self.problem_family_cfg,
            device=self.device,
            dtype=self.dtype
        )

        self.params = self.problem_family.sample_params(B=self.num_batches, seed=seed) 
        self.costs = self.problem_family.costs(self.X, self.params)

        # And gaussian process   
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
        x_init, y_init, c_init = self._sample_init_design()
        self.obj_gp.set_lane_data(t.ones((self.num_batches,), device=self.device, dtype=t.bool), x_init, y_init)

        if self.use_cost_gp:
            self.cost_gp.set_lane_data(t.ones((self.num_batches,), device=self.device, dtype=t.bool), x_init, self._cost_to_gp_target(c_init))
                
        # Reset environment scalars and find best current values among init
        self.remaining_budget = t.full((self.num_batches,), self.budget, device=self.device, dtype=self.dtype)
        self.done = t.full((self.num_batches,), False, device=self.device, dtype=t.bool)
        self.best_current_value = y_init.max(dim=1).values

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
    


    def _reset_lanes(self, lane_mask):
        # Find which lanes to reset, if none, return
        lanes = t.where(lane_mask)[0]
        if lanes.numel() == 0:
            return

        # Regenerate candidates for these lanes
        self.candidate_set.reset_lanes(lane_mask)
        self.X = self.candidate_set.get_grid()

        # Regenerate params for these lanes
        new_params = self.problem_family.sample_params(B=lanes.numel())
        for k in self.params:
            self.params[k][lanes] = new_params[k]

        # Recompute cost for these lanes
        params_lanes = {k: v[lanes] for k,v in self.params.items()}
        self.costs[lanes] = self.problem_family.costs(self.X[lanes], params_lanes)

        # Reset budget / done
        self.remaining_budget[lanes] = self.budget
        self.done[lanes] = False

        # Reset info
        self.ep_return[lanes] = 0.0
        self.ep_len[lanes] = 0

        # New init design and load into GP buffers
        x_init, y_init, c_init = self._sample_init_design(lane_mask) 
        self.obj_gp.set_lane_data(lane_mask, x_init, y_init)

        if self.use_cost_gp:
            self.cost_gp.set_lane_data(lane_mask, x_init, self._cost_to_gp_target(c_init),)

        # Update best current values after new init
        self.best_current_value[lanes] = y_init[lanes].max(dim=1).values
    


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

            # Evaluate for active lanes 
            y = t.zeros((B,1), device=self.device, dtype=self.dtype)
            params_active = {k: v[active] for k, v in self.params.items()}
            y[active] = self.problem_family.evaluate(x[active], params_active).squeeze(-1)
            y1 = y.squeeze(1)

            # Add data for the active lanes
            self.obj_gp.add_data(x, y, active_mask=active)
            if self.use_cost_gp:
                self.cost_gp.add_data(
                    x,
                    self._cost_to_gp_target(cost),
                    active_mask=active,
                )

            # Update state for active lanes in the batch
            self.best_current_value[active] = t.maximum(self.best_current_value[active], y1[active])
            self.remaining_budget[active] -= cost[active]
            self.done[active] = self.remaining_budget[active] <= 0

        # Terminal mask for PPO, which lanes ended THIS step
        terminal = self.done.clone()

        if (self.reward_type == "final_neglog_regret") and terminal.any():
            ground_truth = self.problem_family.optimal_value_on_grid(self.X, self.params) 
            regret = ground_truth - self.best_current_value
            reward[terminal] = -t.log(t.clamp(regret[terminal], min=1e-12))

        self.ep_return[active] += reward[active]
        self.ep_len[active] += 1

        info = {}

        if terminal.any():
            final_info = [None] * B
            terminal_lanes = t.where(terminal)[0]

            ground_truth = self.problem_family.optimal_value_on_grid(self.X, self.params)
            regret = ground_truth - self.best_current_value

            for i in terminal_lanes.tolist():
                final_info[i] = {
                    "episode": {
                        "r": self.ep_return[i].item(),
                        "l": int(self.ep_len[i].item()),
                    },
                    "regret": regret[i].item(),
                    "best_value": self.best_current_value[i].item(),
                }

            info["final_info"] = final_info
            self._reset_lanes(terminal)

        obs = self._build_obs(*self._gp_predict_on_candidates())
        self.last_obs = obs

        return obs, reward, terminal, info



    def _gp_predict_on_candidates(self):
        pred = self.obj_gp.predict(self.X)  
        mu = pred.mean      
        sigma = pred.stddev

        if self.use_cost_gp:
            cost_pred = self.cost_gp.predict(self.X)
            cost = self._gp_target_to_cost_mean(cost_pred)
        
        else:
            cost = self.problem_family.costs(self.X, self.params)
            
            if cost.dim() == 3 and cost.size(-1) == 1:
                cost = cost.squeeze(-1)

        max_cost = cost.max(dim=1, keepdim=True).values.expand_as(cost)

        return mu, sigma, cost, max_cost


    def _cost_to_gp_target(self, cost):
        '''Transform cost to log cost if specified'''
        transform = self.cost_model_cfg.get("target_transform", "log1p")

        if transform == "log1p":
            return t.log1p(cost.clamp_min(0.0))

        if transform == "none":
            return cost

        raise ValueError(f"Unknown cost target transform: {transform}")
    

    def _gp_target_to_cost_mean(self, pred):
        '''...'''
        transform = self.cost_model_cfg.get("target_transform", "log1p")

        if transform == "log1p":
            return t.expm1(pred.mean).clamp_min(0.0)

        if transform == "none":
            return pred.mean.clamp_min(0.0)

        raise ValueError(f"Unknown cost target transform: {transform}")
    

    def _build_obs(self, mu, sigma, cost, max_cost):
        B, N = mu.shape

        # global progress (time spent fraction) or remaining fraction
        progress = (1.0 - self.remaining_budget / self.budget).unsqueeze(1).expand(B, N)

        # best so far
        best = self.best_current_value.unsqueeze(1).expand(B, N)


        obs = t.stack([mu, sigma, cost / self.budget, progress, best, max_cost / self.budget], dim=-1)  # [B,N,6]
        return obs