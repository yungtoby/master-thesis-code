import gymnasium as gym
import torch as t



class BOEnv(gym.Env):
    '''Gymansium environment for one BO episode.'''
    def __init__(self, candidate_factory, gp_factory, objective_fn, n_candidates=50, budget=1.0, reward_type="final_neglog_regret"):
        super().__init__()
        # Environment specifics
        self.n_candidates = n_candidates
        self.budget = budget
        self.remaining_budget = budget
        self.reward_type = reward_type                 # "final_neglog_regret" as in paper
        self.candidate_factory = candidate_factory     # candidate_factory: callable -> returns (X: np.array (N,D), costs: np.array (N,))
        self.gp_factory = gp_factory                   # callable to create a GPWrapper instance (one per env)
        self.objective_fn = objective_fn               # callable for evaluating true objective

        # RL specifics
        observation_dim = 3 * n_candidates + 2         # (mean, std, and cost for each candidate) + remaining budget and best found solution so far
        self.observation_space = gym.spaces.Box(low=-t.inf, high=t.inf, shape=(observation_dim,), dtype=t.float32)
        self.action_space = gym.spaces.Discrete(n_candidates)

        self.reset() # Start (or restart) environment





    # SHOULD RETURN TUPLE[Observation(ObsType), info(dict)]
    def reset(self):
        '''Reset the environment'''
        self.X, self.costs = self.candidate_factory(self.n_candidates)     
        self.gp = self.gp_factory()                                       

        self.train_history = []                # list of indices chosen by agent
        self.best_value = -t.inf               # (-infinity if maximizing, else infinity)
        self.remaining_budget = self.budget
        self.terminated = False
        self.truncated = False

        # compute initial mu, sigma
        mu, sigma = self._gp_predict_on_candidates()
        return (self._build_obs(mu, sigma), {})





    def _gp_predict_on_candidates(self):
        with t.no_grad():
            pred = self.gp.predict(self.X)   # design your GP to return object with mean, stddev as torch tensors
            mu = pred.mean.reshape(-1)       # TODO: Do we need reshaping here???
            sigma = pred.stddev.reshape(-1)  # TODO: Do we need reshaping here???

        return mu, sigma


    def _build_obs(self, mu, sigma):
        # Normalize mu, sigma, costs, budget, best. Use utils/normalization functions.
        # Here we just concat raw values; recommended to scale them later.
        obs = t.concatenate([mu, sigma, self.costs, t.Tensor([self.remaining_budget]), t.Tensor([self.best_value])]) # TODO: might change dimensionality
        return obs.astype(t.float32) # TODO: is this needed? Should already be float32 i think


    # SHOULD RETURN TUPLE[Observation(obsType), reward(float), terminated(bool), truncated(bool), info(dict)]
    def step(self, action):
        if self.done:
            return self._build_obs(*self._gp_predict_on_candidates()), 0.0, True, {}

        idx = int(action)                                               #TODO: This might invoke cpu usage because of casting
        x = self.X[idx:idx+1]
        cost = float(self.costs[idx])                                   #TODO: This might invoke cpu usage because of casting

        # evaluate objective (simulate real eval, could be noisy)
        y = self.objective_fn(x)  # returns float or array

        # add new data to gp and update best
        self.gp.add_data(t.as_tensor(x, dtype=t.float32).unsqueeze(0) if x.ndim==1 else t.as_tensor(x, dtype=t.float32),
                         t.as_tensor(y, dtype=t.float32))
        if y > self.best_value:
            self.best_value = float(y)

        self.remaining_budget -= cost
        # termination
        if self.remaining_budget <= 0:
            self.done = True

        mu, sigma = self._gp_predict_on_candidates()
        obs = self._build_obs(mu, sigma)

        reward = 0.0
        info = {}
        if self.done and self.reward_type == "final_neglog":
            true_best = self._get_true_optimum_value()  # implement or pass in; else approximate by best of full objective
            simple_regret = true_best - self.best_value
            # guard small sr
            sr = max(simple_regret, 1e-12)
            reward = -t.log(sr)
        elif self.reward_type == "improvement":
            # immediate reward is improvement over previous best
            # define reward as y - previous_best (could be negative)
            reward = float(y - self.best_value)  # small signal; consider scaling

        return obs, float(reward), bool(self.done), info
    
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