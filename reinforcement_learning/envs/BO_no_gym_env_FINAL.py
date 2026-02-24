import torch as t
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
        self.gp_factory = gp_factory(device, dtype, num_batches)
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

        # Initialize Candidate set, costs and gaussian process
        self.X, self.costs = self._initialize_candidate_factory()     
        self.gp = self._initialize_gp()        
        self.gp.train()                               

        # Reset train history, best val and remaining budget
        self.ep_return = t.zeros((self.num_batches,), device=self.device, dtype=self.dtype)      
        self.ep_len = t.zeros((self.num_batches,), device=self.device, dtype=self.dtype)         
        self.remaining_budget = t.full((self.num_batches,), self.budget, device=self.device, dtype=self.dtype)
        self.best_current_value = self.gp.train_y.max(dim=1).values
        self.done = t.full((self.num_batches,), False, device=self.device, dtype=t.bool)

        # Compute initial mu, sigma and build observation
        obs = self._build_obs(*self._gp_predict_on_candidates())

        return obs, {}
    

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


    def _initialize_gp(self):
        # init points, need to implement better solution
        N = self.X.shape[1]
        idx = t.randperm(N, device=self.device)[: self.n_init]   # [n_init]
        train_x = self.X[:, idx, :]                              # [B, n_init, d]

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
            mu = pred.mean      
            sigma = pred.stddev

        return mu, sigma


    def _build_obs(self, mu, sigma):
        B, N = mu.shape

        budget = self.remaining_budget.unsqueeze(1).expand(B, N)        # [B] -> [B,N]
        best   = self.best_current_value.unsqueeze(1).expand(B, N)      # [B] -> [B,N]

        obs = t.stack([mu, sigma, self.costs, budget, best], dim=-1)    # [B, N, 5]

        return obs


    def step(self, actions):
        # Get chosen candidate points and their respective costs
        B, _, d = self.X.shape

        x_idx = actions.view(B, 1, 1).expand(B, 1, d)                     # [B]->[B,1,1]->[B,1,d]
        x = t.gather(self.X, dim=1, index=x_idx)                          # [B,N,d] gather -> [B,1,d]

        c_idx = actions.view(B, 1)                                        # [B]->[B,1]
        cost = t.gather(self.costs, dim=1, index=c_idx).squeeze(1)        # [B,N] gather -> [B,1]->[B]


        # Evaluate objective with candidate point
        y = self.objective_fn.evaluate(x)

        # Append new data to gp, train and update current best if better
        self.gp.add_data(x, y)
        self.gp.train() 

        # Update state
        self.best_current_value = t.maximum(self.best_current_value, y.squeeze(1))
        self.remaining_budget = self.remaining_budget - cost
        self.done = self.remaining_budget <= 0


        # Build new observation for Agent
        obs = self._build_obs(*self._gp_predict_on_candidates())
        

        # If done, calculate (reward type specified if we want to try others)
        reward = t.zeros((B,), device=self.device, dtype=self.dtype)
        info = {}


        if self.reward_type == "final_neglog_regret":
            if self.done.any():

                ground_truth = self._get_true_optimum_value()  # implement or pass in; else approximate by best of full objective
                regret = ground_truth - self.best_current_value                                # [B]
                safe_regret = t.clamp(regret, min=t.tensor(1e-12, device=self.device, dtype=self.dtype))
                reward[self.done] = -t.log(safe_regret[self.done])

        # TODO: misses full implementation

        return obs, reward, self.done, info


    def _get_true_optimum_value(self):
        return self.objective_fn.get_optimal_value()
    

    ########################
    # UNUSED FUNCTIONALITY #
    ########################

    def render(self, mode="human"):
        pass

    def close(self):
        pass