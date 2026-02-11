import gymnasium as gym
import torch as t
import numpy as np
import gpytorch as gpy



class BatchedBOEnv():
    '''Gymansium environment for one BO episode.'''
    def __init__(self, device, dtype, num_batches, n_candidates, n_init, budget, reward_type, candidate_factory, gp_factory, objective_fn):
        # Device and dtype
        self.device = t.device(device)
        self.dtype = dtype

        # Environment 
        self.num_batches = num_batches
        self.n_candidates = n_candidates
        self.n_init = n_init
        self.budget = budget
        self.remaining_budget = budget
        self.reward_type = reward_type 
        self.candidate_factory = candidate_factory(device, dtype)
        self.gp_factory = gp_factory(device, dtype)
        self.objective_fn = objective_fn

        # RL specifics (observation and action space)
        self.observation_dim = (num_batches, n_candidates, 5)

        # Internal state
        self.X = None
        self.costs = None
        self.gp = None
        self.train_history = []
        self.best_current_value = None
        self.done = False


    def reset(self, seed, deterministic):
        '''Reset the environment'''
        # Set seed
        t.manual_seed(seed)
        t.backends.cudnn.deterministic = deterministic

        # Initialize Candidate set, costs and gaussian process
        self.X, self.costs = self._initialize_candidate_factory()     
        self.gp = self._initialize_gp()        
        self.gp.train()                               

        # Reset train history, best val and remaining budget
        self.train_history = []              
        self.remaining_budget = t.tensor(float(self.budget), device=self.device, dtype=self.dtype)
        self.best_current_value = t.max(self.gp.train_y)
        self.terminated = False
        self.truncated = False

        # Compute initial mu, sigma and build observation
        obs = self._build_obs(*self._gp_predict_on_candidates())

        # Convert to numpy as gym expects numpy
        obs_np = obs.detach().cpu().numpy().astype(np.float32)

        return obs_np, {}
    


    def _initialize_candidate_factory(self):
        # Placeholder, need to implement better solution
        kwargs = {
            'res' : 100,
            'D' : 1,
            'minimum' : 0,
            'maximum' : 10
        }

        x_s_and_costs = [(self.candidate_factory(**kwargs)) for _ in range(self.num_batches)]
        x_s = t.stack([x[0] for x in x_s_and_costs], dim=0)
        costs = t.stack([x[1] for x in x_s_and_costs], dim=0)

        self.X, self.costs = x_s, costs
        return self.X, self.costs     


    # TODO: make batchable GP
    def _initialize_gp(self):
        # init points, need to implement better solution
        K = self.X.shape[0]
        idx = self.np_random.choice(K, size=self.n_init, replace=False)
        train_x = self.X[idx]
        

        # Placeholder, need to implement better solution
        kwargs = {
            'train_x' : train_x,
            'train_y' : self.objective_fn.evaluate(train_x).squeeze(-1),
            'likelihood' : gpy.likelihoods.GaussianLikelihood(batch_shape=t.Size([self.num_batches])),
            'mean_module' : gpy.means.ConstantMean(batch_shape=t.Size([self.num_batches])),
            'covar_module' : gpy.kernels.ScaleKernel(
                gpy.kernels.RBFKernel(batch_shape=t.Size([self.num_batches]), ard_num_dims=train_x.shape[2]),
                batch_shape=t.Size([self.num_batches])
                ),
            'optimizer_and_lr' : (t.optim.Adam, 0.1),
            'training_iter' : 10
        }

        return self.gp_factory(**kwargs)












    def _gp_predict_on_candidates(self):
        with t.no_grad():
            pred = self.gp.predict(self.X)  
            mu = pred.mean.reshape(-1)      
            sigma = pred.stddev.reshape(-1)  

        return mu, sigma


    def _build_obs(self, mu, sigma):
        # Build observation based on paper
        obs = t.concatenate([
            mu,
            sigma,
            self.costs,
            t.as_tensor(self.remaining_budget, device=self.device, dtype=self.dtype).view(1),
            t.as_tensor(self.best_current_value, device=self.device, dtype=self.dtype).view(1)
        ])

        return obs.to(dtype=t.float32)


    def step(self, action):
        # If step after termination or truncation, return zero reward.
        if self.terminated or self.truncated:
            obs = self._build_obs(*self._gp_predict_on_candidates())
            obs_np = obs.detach().cpu().numpy().astype(np.float32)

            return obs_np, 0.0, self.terminated, self.truncated, {}

        # Get chosen candidate point and its cost
        idx = int(action)                                               
        x = self.X[idx:idx+1]
        cost = self.costs[idx]


        # Evaluate objective with candidate point
        y = self.objective_fn.evaluate(x).reshape(1)


        # Append new data to gp, train and update current best if better
        self.gp.add_data(x, y)
        self.gp.train() # Might need to only train every k-steps?


        self.best_current_value = t.maximum(self.best_current_value, y.squeeze())
        self.remaining_budget = self.remaining_budget - cost
        if self.remaining_budget <= 0:
            self.terminated = True


        # Build new observation for Agent
        mu, sigma = self._gp_predict_on_candidates()
        obs = self._build_obs(mu, sigma)
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        

        # If terminated or truncated, calculate (reward type specified if we want to try others)
        reward = 0.0
        info = {}
        if self.terminated or self.truncated and self.reward_type == "final_neglog_regret":
            ground_truth = self._get_true_optimum_value()  # implement or pass in; else approximate by best of full objective
            regret = ground_truth - self.best_current_value

            # In case regret is equal to zero (should not this be min() instead of max())
            sr = t.clamp(regret, min=t.tensor(1e-12, device=self.device, dtype=self.dtype))
            reward = float((-t.log(sr)).detach().cpu())

        return obs_np, reward, self.terminated, self.truncated, info


    def _get_true_optimum_value(self):
        return self.objective_fn.get_optimal_value()
    

    ########################
    # UNUSED FUNCTIONALITY #
    ########################

    def render(self, mode="human"):
        pass

    def close(self):
        pass