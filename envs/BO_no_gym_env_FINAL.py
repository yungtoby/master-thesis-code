################### QUICK FIX FOR IMPORTS: ############################################ 
import os, sys                                                                        #
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          #
#######################################################################################

import torch as t
from bo.gaussian_process_batch_FINAL import RepeatedPadGPWrapper # MaskedGPWrapper
from bo.candidate_set_batch import BatchedCandidateSet
from bo.problems.toy_rbf import ToyRBFProblemFamily



class BatchedBOEnv():
    '''Gymansium environment for one BO episode.'''
    def __init__(self, device, dtype, num_batches, n_candidates, n_init, budget, max_acquistions, reward_type):
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
        self.T_max = self.n_init + max_acquistions
        self.last_obs = None
        self.problem_family = None
        self.params = None

        # RL specifics (observation and action space)
        self.observation_dim = (num_batches, n_candidates, 6)

        # Internal state
        self.X = None
        self.costs = None
        self.gp = None
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
        self.candidate_set = BatchedCandidateSet(
            device=self.device,
            dtype=self.dtype,
            B=self.num_batches,
            res=30,
            D=2,
            minimum=0,
            maximum=10
        )

        self.X = self.candidate_set.get_grid()
        
        assert self.X.shape[1] == self.n_candidates

        # Initialize cost and objective func params
        self.problem_family = ToyRBFProblemFamily(
            device=self.device,
            dtype=self.dtype,
            lb = [0, 0],
            ub = [10, 10]
        )

        self.params = self.problem_family.sample_params(B=self.num_batches, seed=seed) 
        self.costs = self.problem_family.costs(self.X, self.params)

        # And gaussian process   
        d = self.X.shape[-1]
        self.gp = RepeatedPadGPWrapper(
            device=self.device,
            dtype=self.dtype,
            B=self.num_batches,
            T_max=self.T_max,
            d=d,
            lr=1e-3,
            training_iter=10
        )                               

        # Initialize initial points for each lane in the batch
        x_init, y_init = self._sample_init_design()
        self.gp.set_lane_data(t.ones((self.num_batches,), device=self.device, dtype=t.bool), x_init, y_init)
        #self.gp.train()
                
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

        # Saftery in case it is called with 0 mask
        if lanes.numel() == 0:
            return train_x, train_y

        # Choose init indices (shared across lanes for simplicity)
        idx = t.randperm(N, device=self.device)[: self.n_init]  # [n_init]

        # Fill only selected lanes
        train_x[lanes] = self.X[lanes][:, idx, :]  # [n_lanes, n_init, d]

        # Evaluate only selected lanes using matching params slice
        params_lanes = {k: v[lanes] for k, v in self.params.items()}
        y = self.problem_family.evaluate(train_x[lanes], params_lanes).squeeze(-1)  # [n_lanes, n_init]
        train_y[lanes] = y

        return train_x, train_y
    


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
        x_init, y_init = self._sample_init_design(lane_mask) 
        self.gp.set_lane_data(lane_mask, x_init, y_init)
        #self.gp.train()

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
            self.gp.add_data(x, y, active_mask=active)
            #self.gp.train()

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
        pred = self.gp.predict(self.X)  
        mu = pred.mean      
        sigma = pred.stddev

        return mu, sigma



    #def _build_obs(self, mu, sigma):
    #    B, N = mu.shape

    #    budget = self.remaining_budget.unsqueeze(1).expand(B, N)        # [B] -> [B,N]
    #    best   = self.best_current_value.unsqueeze(1).expand(B, N)      # [B] -> [B,N]

    #    obs = t.stack([mu, sigma, self.costs, budget, best], dim=-1)    # [B, N, 5]

    #    return obs
    

    def _build_obs(self, mu, sigma):
        B, N = mu.shape

        # normalized local cost
        cost = self.costs / self.budget

        # global progress (time spent fraction) or remaining fraction
        progress = (1.0 - self.remaining_budget / self.budget).unsqueeze(1).expand(B, N)

        # best so far
        best = self.best_current_value.unsqueeze(1).expand(B, N)

        # global max cost baseline (normalized)
        max_cost = (self.costs.max(dim=1).values / self.budget).unsqueeze(1).expand(B, N)

        obs = t.stack([mu, sigma, cost, progress, best, max_cost], dim=-1)  # [B,N,6]
        return obs