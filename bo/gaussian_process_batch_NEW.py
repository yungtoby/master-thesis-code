import gpytorch as gpy
import torch as t

# Shared by selection, training, and testing; record this in experiment outputs.
FITTING_RECIPE = {
    'target_centering': 'observed_mean_before_padding', 
    'noise_init_fraction': 0.02, 
    'noise_init_floor': 0.000101, 
    'noise_lower_bound': 0.0001, 
    'sample_variance_correction': 1, 
    'prediction': 'likelihood', 'padding': 
    'repeat_valid_observations', 'initialization': 
    'fresh_each_prediction', 'training_iter': 10
    }


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
    def __init__(self, device, dtype, B, T_max, d, lr=1e-3, training_iter=10,
                 kernel='rbf', mean='constant', record_diagnostics=False):
        self.device = t.device(device)
        self.dtype = dtype
        self.B = B
        self.T_max = T_max
        self.d = d
        self.lr = lr
        if training_iter != 10:
            raise ValueError("The NEW fitting recipe requires exactly 10 training iterations.")
        self.training_iter = training_iter
        self.kernel = kernel
        self.mean = mean
        self.record_diagnostics = record_diagnostics
        self.last_fit_parameters = {}
        self.last_prediction = None

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

    
    def _make_base_kernel(self, batch_shape, name=None):
        name = self.kernel if name is None else name
        if name == 'rbf':
            return gpy.kernels.RBFKernel(
                batch_shape=batch_shape,
                ard_num_dims=self.d
            )
        if name == 'matern32':
            return gpy.kernels.MaternKernel(
                nu=1.5,
                batch_shape=batch_shape,
                ard_num_dims=self.d
            )
        if name == 'matern52':
            return gpy.kernels.MaternKernel(
                nu=2.5,
                batch_shape=batch_shape,
                ard_num_dims=self.d
            )
        raise ValueError(f"Unknown GP kernel choice: {name}")


    def _make_covar_module(self, batch_shape):
        def scaled(kernel):
            return gpy.kernels.ScaleKernel(kernel, batch_shape=batch_shape)

        if self.kernel in {'rbf_matern52_add', 'rbf_matern52_mul'}:
            rbf = self._make_base_kernel(batch_shape, 'rbf')
            matern = self._make_base_kernel(batch_shape, 'matern52')
            if self.kernel == 'rbf_matern52_add':
                # Each component has its own amplitude and ARD length scales.
                return scaled(rbf) + scaled(matern)
            # One overall amplitude avoids redundant product amplitudes.
            return scaled(rbf * matern)
        return scaled(self._make_base_kernel(batch_shape))

    def _make_mean_module(self, batch_shape):
        if self.mean == 'constant':
            return gpy.means.ConstantMean(batch_shape=batch_shape)
        if self.mean == 'linear':
            # LinearMean normally draws random initial weights. Start both
            # candidate means at zero, without consuming the caller's RNG.
            with t.random.fork_rng(devices=[]):
                mean = gpy.means.LinearMean(self.d, batch_shape=batch_shape, bias=True)
            with t.no_grad():
                mean.weights.zero_()
                mean.bias.zero_()
            return mean
        raise ValueError(f"Unknown GP mean choice: {self.mean}")


    def _record_fit_parameters(self, gp, likelihood):
        parameters = {'likelihood.noise': likelihood.noise}
        for name, module in gp.covar_module.named_modules():
            prefix = f'covar.{name}' if name else 'covar'
            if getattr(module, 'has_lengthscale', False):
                parameters[f'{prefix}.lengthscale'] = module.lengthscale
            if isinstance(module, gpy.kernels.ScaleKernel):
                parameters[f'{prefix}.outputscale'] = module.outputscale
        if self.mean == 'constant':
            parameters['mean.constant'] = gp.mean_module.constant
        else:
            parameters['mean.weights'] = gp.mean_module.weights
            parameters['mean.bias'] = gp.mean_module.bias
        self.last_fit_parameters = {
            key: value.detach().reshape(self.B, -1).clone()
            for key, value in parameters.items()
        }


    def _target_statistics(self):
        """Use each lane's actual observed prefix, before repeat padding."""
        if bool((self.t_idx < 1).any()):
            raise RuntimeError("Every GP lane needs at least one observation.")
        valid = t.arange(self.T_max, device=self.device)[None, :] < self.t_idx[:, None]
        counts = self.t_idx.to(self.dtype)
        offset = self.train_y.masked_fill(~valid, 0).sum(dim=1) / counts
        residual = (self.train_y - offset[:, None]).masked_fill(~valid, 0)
        variance = residual.square().sum(dim=1) / (counts - 1).clamp_min(1)
        initial_noise = (FITTING_RECIPE['noise_init_fraction'] * variance).clamp_min(
            FITTING_RECIPE['noise_init_floor'])
        return offset, variance, initial_noise


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


    def _build_and_fit_gp(self, train_x, train_y, initial_noise):
        batch_shape = t.Size([self.B])

        likelihood = gpy.likelihoods.GaussianLikelihood(
            batch_shape=batch_shape
        ).to(device=self.device, dtype=self.dtype)

        mean_module = self._make_mean_module(batch_shape)
        covar_module = self._make_covar_module(batch_shape)

        gp = GaussianProcess(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
        ).to(device=self.device, dtype=self.dtype)

        likelihood.noise_covar.register_constraint("raw_noise", gpy.constraints.GreaterThan(1e-4)) # FOR NUMERICAL STABILITY.
        # Set constrained noise variance, not its unconstrained raw parameter.
        likelihood.noise = initial_noise[:, None]
        optimizer = t.optim.Adam(gp.parameters(), lr=self.lr)
        mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, gp)

        gp.train()
        likelihood.train()

        for _ in range(self.training_iter):
            optimizer.zero_grad()
            output = gp(train_x)
            loss = -mll(output, train_y).sum()

            if not t.isfinite(loss):
                raise RuntimeError("Non-finite GP loss encountered.")

            loss.backward()
            t.nn.utils.clip_grad_norm_(gp.parameters(), max_norm=10.0) # FOR NUMERICAL STABILITY.
            optimizer.step()

        if self.record_diagnostics:
            self._record_fit_parameters(gp, likelihood)
        return gp, likelihood


    def predict(self, x):
        """Return the observation predictive distribution, in original target units.

        Training buffers remain uncentered. A fresh GP and optimizer are fitted
        on every call. Cost targets are already log1p(cost) when supplied here.
        """
        offset, variance, initial_noise = self._target_statistics()
        train_x, train_y = self._repeat_pad_prefixes()
        gp, likelihood = self._build_and_fit_gp(
            train_x, train_y - offset[:, None], initial_noise)
        gp.eval()
        likelihood.eval()
        with t.no_grad(), gpy.settings.fast_pred_var():
            centered = likelihood(gp(x))
            prediction = gpy.distributions.MultivariateNormal(
                centered.mean + offset[:, None], centered.lazy_covariance_matrix)
        if self.record_diagnostics:
            self.last_fit_parameters.update({
                'target.offset': offset.detach()[:, None].clone(),
                'target.sample_variance': variance.detach()[:, None].clone(),
                'likelihood.initial_noise': initial_noise.detach()[:, None].clone(),
            })
            self.last_prediction = prediction
        return prediction