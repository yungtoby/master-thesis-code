import gpytorch as gpy
import torch as t

# Noise for fixed gaussian noise (for masking)
LARGE_NOISE = 1e7


class GaussianProcess(gpy.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)










class RepeatedPadGPWrapper:
    """
    Keep batched buffers with valid-prefix bookkeeping, but rebuild a fresh
    batched GP from repeated real observations when predictions are needed.
    """
    def __init__(self, device, dtype, B, T_max, d, lr=1e-3, training_iter=10):
        self.device = t.device(device)
        self.dtype = dtype
        self.B = B
        self.T_max = T_max
        self.d = d
        self.lr = lr
        self.training_iter = training_iter

        # storage only
        self.train_x = t.zeros((B, T_max, d), device=self.device, dtype=self.dtype)
        self.train_y = t.zeros((B, T_max), device=self.device, dtype=self.dtype)
        self.t_idx = t.zeros((B,), device=self.device, dtype=t.long)


    def set_lane_data(self, lane_mask, x_init, y_init):
        lanes = t.where(lane_mask)[0]
        if lanes.numel() == 0:
            return

        n_init = x_init.shape[1]

        # Reset buffers
        self.train_x[lanes] = 0
        self.train_y[lanes] = 0

        # Set lane data to specific lanes
        self.train_x[lanes, :n_init] = x_init[lanes]
        self.train_y[lanes, :n_init] = y_init[lanes]
        self.t_idx[lanes] = n_init


    def add_data(self, x_new, y_new, active_mask=None):
        if y_new.dim() == 3 and y_new.size(-1) == 1:
            y_new = y_new.squeeze(-1)
        if y_new.dim() == 2 and y_new.size(1) == 1:
            y_new = y_new.squeeze(1)

        if active_mask is None:
            active_mask = t.ones((self.B,), device=self.device, dtype=t.bool)

        lanes = t.where(active_mask)[0]
        if lanes.numel() == 0:
            return

        idx = self.t_idx[lanes]
        valid = idx < self.T_max
        lanes = lanes[valid]
        idx = idx[valid]

        if lanes.numel() == 0:
            return

        self.train_x[lanes, idx] = x_new[lanes, 0]
        self.train_y[lanes, idx] = y_new[lanes]
        self.t_idx[lanes] += 1


    def _repeat_pad_prefixes(self):
        """
        Build batched tensors [B, expand_size, d] and [B, expand_size] using
        repeated real observations from each lane's valid prefix.
        """
        expand_size = self._get_current_expand_size()
        x_pad = t.empty((self.B, expand_size, self.d), device=self.device, dtype=self.dtype)
        y_pad = t.empty((self.B, expand_size), device=self.device, dtype=self.dtype)

        for b in range(self.B):
            n = int(self.t_idx[b].item())
            if n <= 0:
                raise RuntimeError(f"Lane {b} has no observations; cannot build GP batch.")

            tx = self.train_x[b, :n]   # [n, d]
            ty = self.train_y[b, :n]   # [n]

            num_copies = expand_size // n
            rem = expand_size % n

            x_rep = tx.repeat((num_copies, 1))
            y_rep = ty.repeat(num_copies)

            if rem > 0:
                x_rep = t.cat([x_rep, tx[:rem]], dim=0)
                y_rep = t.cat([y_rep, ty[:rem]], dim=0)

            x_pad[b] = x_rep
            y_pad[b] = y_rep

        return x_pad, y_pad
    

    def _get_current_expand_size(self):
        current_max = int(self.t_idx.max().item())
        if current_max <= 0:
            raise RuntimeError("No observations available to build GP batch.")
        return min(current_max, self.T_max)


    def _build_and_fit_model(self, train_x, train_y):
        batch_shape = t.Size([self.B])

        likelihood = gpy.likelihoods.GaussianLikelihood(
            batch_shape=batch_shape
        ).to(self.device)

        mean_module = gpy.means.ConstantMean(batch_shape=batch_shape)
        covar_module = gpy.kernels.ScaleKernel(
            gpy.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=self.d),
            batch_shape=batch_shape,
        )

        model = GaussianProcess(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
        ).to(self.device)

        likelihood.noise_covar.register_constraint("raw_noise", gpy.constraints.GreaterThan(1e-4)) # FOR NUMERICAL STABILITY.
        optimizer = t.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, model)

        model.train()
        likelihood.train()

        for _ in range(self.training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y).sum()

            if not t.isfinite(loss):
                raise RuntimeError("Non-finite GP loss encountered.")

            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) # FOR NUMERICAL STABILITY.
            optimizer.step()

        return model, likelihood


    def predict(self, x):
        """
        x: [B, N, d]
        Returns:
            mu:    [B, N]
            sigma: [B, N]
        """
        train_x_eq, train_y_eq = self._repeat_pad_prefixes()
        model, likelihood = self._build_and_fit_model(train_x_eq, train_y_eq)

        model.eval()
        likelihood.eval()

        with t.no_grad(), gpy.settings.fast_pred_var():
            posterior = model(x)  # latent posterior
            mu = posterior.mean
            sigma = posterior.variance.clamp_min(1e-12).sqrt()

        return mu, sigma














class MaskedGPWrapper:
    '''Gaussian Process wrapper for utilizing GPYTorch's ExactGP implementation.'''
    def __init__(self, device, dtype, B, T_max, d, base_noise=1e-4, lr=1e-2, training_iter=10):
        # Initializing device, dtype, batch size, episode horizon, dimensionality,
        # base noise and number of training iterations. 
        self.device = t.device(device)
        self.dtype = dtype
        self.B = B
        self.T_max = T_max
        self.d = d
        self.base_noise = base_noise
        self.training_iter = training_iter

        # Buffers for training points s.t. we can enable done batches to not wait
        # for a global reset.
        self.train_x = t.zeros((B, T_max, d), device=self.device, dtype=self.dtype)
        self.train_y = t.zeros((B, T_max), device=self.device, dtype=self.dtype)
        self.noise = t.full((B, T_max), LARGE_NOISE, device=self.device, dtype=self.dtype) # mask

        # Per batch pointer, where to write next, and batch_size
        self.t_idx = t.zeros((B,), device=self.device, dtype=t.long)
        batch_shape = t.Size([B])

        # Choose fixed noise gaussian likelihood so that we can mask with high value noise
        self.likelihood = gpy.likelihoods.FixedNoiseGaussianLikelihood(
             noise=self.noise, batch_shape=batch_shape
        ).to(device = self.device)

        mean_module = gpy.means.ConstantMean(batch_shape=batch_shape)
        covar_module = gpy.kernels.ScaleKernel(
            gpy.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=d),
            batch_shape=batch_shape
        )

        # Initialize the GP
        self.GP = GaussianProcess(
            train_x=self.train_x,
            train_y=self.train_y, 
            likelihood=self.likelihood,
            mean_module=mean_module,
            covar_module=covar_module
            ).to(device = self.device)
        
        # Initialize optimizer and number of training iterations
        self.optimizer = t.optim.Adam(self.GP.parameters(), lr=lr)


    def set_lane_data(self, lane_mask, x_init, y_init):
        '''Reset specific lanes in the batches' GP training buffers to provided init'''
        # Get indices of lanes to reset
        lanes = t.where(lane_mask)[0]
        if lanes.numel() == 0:
            return
        
        # Reset buffers for these indices
        self.train_x[lanes] = 0
        self.train_y[lanes] = 0
        self.noise[lanes] = LARGE_NOISE

        # Write init data to these indices
        n_init = x_init.shape[1]
        self.train_x[lanes, :n_init] = x_init[lanes]
        self.train_y[lanes, :n_init] = y_init[lanes]
        self.noise[lanes, :n_init] = self.base_noise

        # Update lane pointers for these indices
        self.t_idx[lanes] = n_init

        # Push update into GP
        self._refresh_train_data()


    def add_data(self, x_new, y_new, active_mask=None):
        '''Add data (x_new, y_new) to active lanes in the batch'''
        # Unsure, but modifying y_new --> [B]
        if y_new.dim() == 3 and y_new.size(-1) == 1:
            y_new = y_new.squeeze(-1)
        if y_new.dim() == 2 and y_new.size(1) == 1:
            y_new = y_new.squeeze(1)

        # If no active mask given, set all to active
        if active_mask is None:
            active_mask = t.ones((self.B, ), device=self.device, dtype=t.bool)
        lanes = t.where(active_mask)[0]
        if lanes.numel() == 0:
            return
        
        idx = self.t_idx[lanes]
        valid = idx < self.T_max   # Safety
        lanes = lanes[valid]
        idx = idx[valid]
        if lanes.numel() == 0:
            return
        
        # write one point per lane
        self.train_x[lanes, idx] = x_new[lanes, 0]
        self.train_y[lanes, idx] = y_new[lanes]
        self.noise[lanes, idx]   = self.base_noise
        self.t_idx[lanes] += 1

        self._refresh_train_data()

    
    def _refresh_train_data(self):
        '''Update likelihood noise reference and GP train data'''
        self.likelihood.noise = self.noise  # TODO: UNSURE IF THIS WORKS: FixedNoiseGaussianLikelihood uses stored noise; assigning updates it.
        self.GP.set_train_data(self.train_x, self.train_y, strict=False)


    def train(self, verbose=False):
        '''Train the GP on current training data'''
        # Set GP and likelihood to training mode:
        self.GP.train()
        self.likelihood.train()

        # Loss for the GP - the marginal log likelihood
        mll = gpy.mlls.ExactMarginalLogLikelihood(self.likelihood, self.GP)

        for i in range(self.training_iter):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()

            # Output from model
            output = self.GP(self.train_x)

            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y).sum()
            loss.backward()

            self.optimizer.step()

            if verbose:
                print('Iter %d/%d - Loss: %.3f' % (
                    i + 1, self.training_iter, loss.item()
                ))


    def predict(self, x):
        '''Predict on newly seen data'''
        # Get into evaluation (predictive posterior) mode
        self.GP.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with t.no_grad(), gpy.settings.fast_pred_var():
            return self.likelihood(self.GP(x))