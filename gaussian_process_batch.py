import gpytorch as gpy
import torch as t



class GaussianProcess(gpy.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)


# TODO: MAKE BATCHABLE
class GPWrapper:
    '''Gaussian Process wrapper for utilizing GPYTorch's ExactGP implementation.'''
    def __init__(self, device, dtype, batch_size, train_x, train_y, likelihood, mean_module, covar_module, optimizer_and_lr, training_iter):
        # Initializing device and dtype
        self.device = t.device(device)
        self.dtype = dtype

        # Initializing training data and moving to device + reshaping y to 1D
        self.train_x = train_x.to(device = self.device, dtype = self.dtype).contiguous()
        self.train_y = train_y.to(device = self.device, dtype = self.dtype).contiguous()

        # Initialize likelihood, mean and covar (dont move mean and covar as the get moved with model)
        self.likelihood = likelihood.to(device = self.device)
        self.mean_module = mean_module
        self.covar_module = covar_module

        # Initialize the GP
        self.GP = GaussianProcess(
            train_x=self.train_x,
            train_y=self.train_y, 
            likelihood=self.likelihood,
            mean_module=self.mean_module,
            covar_module=self.covar_module
            ).to(device = self.device)
        
        # Initialize optimizer and number of training iterations
        optimizer_class, self.lr = optimizer_and_lr
        self.optimizer = optimizer_class(self.GP.parameters(), self.lr)
        self.training_iter = training_iter
        

    # NOTE: Might become a bottleneck, need to rethink..
    def add_data(self, x_new, y_new):
        '''Append new data to the GP'''

        # TODO: Might not need to move, might be superflous
        # Move new points to device
        x_new = x_new.to(self.device, dtype=self.dtype).contiguous()
        y_new = y_new.to(self.device, dtype=self.dtype).contiguous()

        # Safeguard in case of wrong dimensions
        if x_new.dim() == 2:
            x_new = x_new.unsqueeze(1)              # [B,d]   -> [B,1,d]
        if y_new.dim() == 1:
            y_new = y_new.unsqueeze(1)              # [B]     -> [B,1]
        if y_new.dim() == 3 and y_new.size(-1) == 1:
            y_new = y_new.squeeze(-1)               # [B,k,1] -> [B,k]

        # Update stored data
        self.train_x = t.cat([self.train_x, x_new], dim=1)
        self.train_y = t.cat([self.train_y, y_new], dim=1)

        # Update GPs data:
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

            if verbose:
                print('Iter %d/%d - Loss: %.3f' % (
                    i + 1, self.training_iter, loss.item()
                ))

            self.optimizer.step()


    def predict(self, x):
        '''Predict on newly seen data'''
        # Get into evaluation (predictive posterior) mode
        self.GP.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with t.no_grad(), gpy.settings.fast_pred_var():
            return self.likelihood(self.GP(x))




    # NOTE: SHOULD NOT BE NECESSARY AFTER INIT!    
    def to(self, device):
        """Move internal tensors and modules to a new device."""
        self.device = t.device(device)
        self.train_x = self.train_x.to(self.device)
        self.train_y = self.train_y.to(self.device)
        self.GP = self.GP.to(self.device)
        self.likelihood = self.likelihood.to(self.device)
        # Recreate optimizer to point to new parameters (optimizer state isn't transferred cleanly)
        optimizer_class = type(self.optimizer)
        self.optimizer = optimizer_class(self.GP.parameters(), lr=self.lr)