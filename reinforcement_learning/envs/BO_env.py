import gymnasium as gym
import torch as t
import numpy as np
import gpytorch as gpy



class BOEnv(gym.Env):
    '''Gymansium environment for one BO episode.'''
    def __init__(self, device, dtype, candidate_factory, gp_factory, objective_fn, n_candidates=100, n_init=3, budget=500, reward_type="final_neglog_regret"):
        super().__init__()
        # Device and dtype
        self.device = t.device(device)
        self.dtype = dtype

        # Environment specifics
        self.n_candidates = n_candidates
        self.n_init = n_init
        self.budget = budget
        self.remaining_budget = budget
        self.reward_type = reward_type 
        self.candidate_factory = candidate_factory(device, dtype)
        self.gp_factory = gp_factory(device, dtype)
        self.objective_fn = objective_fn

        # RL specifics (observation and action space)
        observation_dim = 3 * n_candidates + 2  # (mean, std, and cost for each candidate) + remaining budget and best found solution so far
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n_candidates)

        # Internal state
        self.X = None
        self.costs = None
        self.gp = None
        self.train_history = []
        self.best_value = -np.inf
        self.terminated = False
        self.truncated = False

        self.reset() # Start (or restart) environment


    def reset(self, seed=None, options=None):
        '''Reset the environment'''
        super().reset(seed=seed)
        # Initialize Candidate set, costs and gaussian process
        self.X, self.costs = self._initialize_candidate_factory()     
        self.gp = self._initialize_gp()        
        self.gp.train()                               

        # Reset train history, best val and remaining budget
        self.train_history = []              
        self.best_value = -np.inf             
        self.remaining_budget = self.budget
        self.terminated = False
        self.truncated = False

        # Compute initial mu, sigma and build observation
        mu, sigma = self._gp_predict_on_candidates()
        obs = self._build_obs(mu, sigma)

        return obs, {}
    

    def _initialize_candidate_factory(self):
        # Placeholder, need to implement better solution
        kwargs = {
            'res' : 100,
            'D' : 1,
            'minimum' : 0,
            'maximum' : 10
        }

        self.X, self.costs = self.candidate_factory(**kwargs)
        return self.X, self.costs     


    def _initialize_gp(self):
        # init points, need to implement better solution
        K = self.X.shape[0]
        idx = self.np_random.choice(K, size=self.n_init, replace=False)  # uses Gym's seeded RNG
        train_x = self.X[idx].to(self.device, self.dtype)  
        

        # Placeholder, need to implement better solution
        kwargs = {
            'train_x' : train_x,
            'train_y' : self.objective_fn.evaluate(train_x).squeeze(-1),
            'likelihood' : gpy.likelihoods.GaussianLikelihood(),
            'mean_module' : gpy.means.ConstantMean(),
            'covar_module' : gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel()),
            'optimizer_and_lr' : (t.optim.Adam, 0.1),
            'training_iter' : 10
        }

        return self.gp_factory(**kwargs)


    def _gp_predict_on_candidates(self):
        with t.no_grad():
            pred = self.gp.predict(t.as_tensor(self.X, dtype=t.float32, device=self.device))  
            mu = pred.mean.reshape(-1)      
            sigma = pred.stddev.reshape(-1)  

        return np.array(mu), np.array(sigma)


    def _build_obs(self, mu, sigma):
        # Build observation based on paper
        obs = np.concatenate(
            [mu, sigma, self.costs, np.array([self.remaining_budget]), np.array([self.best_value])]
        )

        return obs.astype(np.float32)


    def step(self, action):
        # If step after termination or truncation, return zero reward.
        if np.logical_or(self.terminated, self.truncated):
            return self._build_obs(*self._gp_predict_on_candidates()), 0, self.terminated, self.truncated, {}

        # Get chosen candidate point and its cost
        idx = int(action)                                               
        x = self.X[idx:idx+1]
        cost = self.costs[idx]

        # Evaluate objective with candidate point
        y = self.objective_fn(x)

        # Append new data to gp, train and update current best if better
        self.gp.add_data(t.as_tensor(x, dtype=t.float32), t.as_tensor([y], dtype=t.float32))
        self.gp.train() # Might need to only train every k-steps?

        if y > self.best_value:
            self.best_value = y

        # VAR HER FØR LEGETIME, LURER PÅ OM DET SKAL GÅ AN Å PLUKKE ET PUNKT SOM HAR KOSTNAD UTENFOR BUDJSETT. SKAL DETTE
        # ISÅFALL VÆRE TRUNCATED ELLER TERMINATED?? (LENER MEST MOT TRUNCATED)

        # Check wether the cost is outside the remaining budget
        self.remaining_budget -= cost
        if self.remaining_budget <= 0:
            #unsure if it should be terminated or truncated
            self.terminated = True

        # Build new observation for Agent
        mu, sigma = self._gp_predict_on_candidates()
        obs = self._build_obs(mu, sigma)

        # If terminated or truncated, calculate (reward type specified if we want to try others)
        reward = 0.0
        info = {}
        if np.logical_or(self.terminated, self.truncated) and self.reward_type == "final_neglog_regret":
            ground_truth = self._get_true_optimum_value()  # implement or pass in; else approximate by best of full objective
            regret = ground_truth - self.best_value

            # In case regret is equal to zero (should not this be min() instead of max())
            sr = max(regret, 1e-12)
            reward = -np.log(sr)

        return obs, reward, self.terminated, self.truncated, info
    
    def _get_true_optimum_value(self):
        # If you can compute ground truth optimum on known objective, return it.
        # fallback: maximum value across full candidate set evaluation of objective_fn
        vals = [float(self.objective_fn(x.reshape(1, -1))) for x in self.X]
        return max(vals)
    

    ########################
    # UNUSED FUNCTIONALITY #
    ########################

    def render(self, mode="human"):
        pass

    def close(self):
        pass