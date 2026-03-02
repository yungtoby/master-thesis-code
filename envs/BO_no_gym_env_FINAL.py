################### QUICK FIX FOR IMPORTS: ############################################ 
import os, sys                                                                        #
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          #
#######################################################################################

import torch as t
from bo.gaussian_process_batch_FINAL import MaskedGPWrapper
from bo.candidate_set_batch import BatchedCandidateSet



class BatchedBOEnv():
    '''Gymansium environment for one BO episode.'''
    def __init__(self, device, dtype, num_batches, n_candidates, n_init, budget, reward_type, max_acquistions, objective_fn):
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
        self.objective_fn = objective_fn

        # RL specifics (observation and action space)
        self.observation_dim = (num_batches, n_candidates, 5)

        # Internal state
        self.X = None
        self.costs = None
        self.gp = None
        self.ep_return = None
        self.ep_len = None
        self.best_current_value = None
        self.done = False


    def reset(self, seed, deterministic):
        '''Reset the environment'''
        # Set seed
        t.manual_seed(seed)
        t.backends.cudnn.deterministic = deterministic

        # Initialize Candidate set, costs
        self.candidate_set = BatchedCandidateSet(
            device=self.device,
            dtype=self.dtype,
            B=self.B,
            res=100,
            D=2,
            minimum=0,
            maximum=10
        )
        self.X = self.candidate_set.get_grid()

        self.costs, self.lane_params = ...  # TODO: Needs work

        # And gaussian process   
        d = self.X.shape[-1]
        self.gp = MaskedGPWrapper(
            device=self.device,
            dtype=self.dtype,
            B=self.num_batches,
            T_max=self.T_max,
            d=d,
            base_noise=1e-4,
            lr=1e-2,
            training_iter=10
        )                               

        # Initialize initial points for each lane in the batch
        x_init, y_init = self._sample_init_design()
        self.gp.set_lane_data(t.ones((self.num_batches,), device=self.device, dtype=t.bool), x_init, y_init)
        self.gp.train()
                
        # Reset environment scalars and find best current values among init
        self.remaining_budget = t.full((self.num_batches,), self.budget, device=self.device, dtype=self.dtype)
        self.done = t.full((self.num_batches,), False, device=self.device, dtype=t.bool)
        self.best_current_value = y_init.max(dim=1).values

        # Compute initial mu, sigma and build observation
        obs = self._build_obs(*self._gp_predict_on_candidates())
        self.last_obs = obs

        # self.ep_return = t.zeros((self.num_batches,), device=self.device, dtype=self.dtype) # TODO: Needs fixing
        # self.ep_len = t.zeros((self.num_batches,), device=self.device, dtype=self.dtype) 

        return obs, {}
    

    def _sample_init_design(self):
        '''Sample the n_init number of initial training points'''
        # Choose n_init random indices from length N
        _, N, _ = self.X.shape
        idx = t.randperm(N, device=self.device)[: self.n_init]

        # Choose the following from X and evaluate on objective function.
        train_x = self.X[:, idx, :]
        train_y = self.objective_fn.evaluate(train_x).squeeze(-1) # TODO: fiks objective func logic

        return train_x, train_y
    

    def _reset_lanes(self, lane_mask):
        # Find which lanes to reset, if none, return
        lanes = t.where(lane_mask)[0]
        if lanes.numel() == 0:
            return
        
        # Regenerate candidates for these lanes
        X_new, c_new = self._initialize_candidate_factory(lane_mask) # TODO: init candidate factory for mask
        self.X[lanes] = X_new[lanes]
        self.costs[lanes] = c_new[lanes]

        # Reset budget / done
        self.remaining_budget[lanes] = self.budget
        self.done[lanes] = False

        # New init design and load into GP buffers
        x_init, y_init = self._sample_init_design(lane_mask) # TODO: Sample init design needs to accept lane mask
        self.gp.set_lane_data(lane_mask, x_init, y_init)
        self.gp.train()

        # Update best current values after new init
        self.best_current_value[lanes] = y_init[lanes].max(dim=1).values
    


    def step(self, actions):
        B, _, d = self.X.shape
        active = ~self.done

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
            y[active] = self.objective_fn.evaluate(x[active])
            y1 = y.squeeze(1) # Dim: [B]

            # Add data for the active lanes
            self.gp.add_data(x, y, active_mask=active)
            self.gp.train()

            # Update state for active lanes in the batch
            self.best_current_value[active] = t.maximum(self.best_current_value[active], y1[active])
            self.remaining_budget[active] -= cost[active]
            self.done[active] = self.remaining_budget[active] <= 0

        # Terminal mask for PPO, which lanes ended THIS step
        terminal = self.done.clone()

        if (self.reward_type == "final_neglog_regret") and (terminal.any()):
            ground_truth = self._get_true_optimum_value() # TODO: Needs extra fixing!
            regret = ground_truth - self.best_current_value
            reward[terminal] = -t.log(t.clamp(regret[terminal], min=1e-12))

        if terminal.any():
            self._reset_lanes(terminal)

        obs = self._build_obs(*self._gp_predict_on_candidates())
        self.last_obs = obs

        return obs, reward, terminal, info




    def _gp_predict_on_candidates(self):
        with t.no_grad():
            pred = self.gp.predict(self.X)  
            mu = pred.mean      
            sigma = pred.stddev

        return mu, sigma


    def _build_obs(self, mu, sigma):
        B, N = mu.shape

        budget = self.remaining_budget.unsqueeze(1).expand(B, N)        # [B] -> [B,N]
        best   = self.best_current_value.unsqueeze(1).expand(B, N)      # [B] -> [B,N]

        obs = t.stack([mu, sigma, self.costs, budget, best], dim=-1)    # [B, N, 5]

        return obs


    def _get_true_optimum_value(self):
        return self.objective_fn.get_optimal_value()