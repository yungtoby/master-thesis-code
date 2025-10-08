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



class GPWrapper:
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module, optimizer_and_lr, training_iter):
        self.train_x = train_x
        self.train_y = train_y

        self.likelihood = likelihood
        self.mean_module = mean_module
        self.covar_module = covar_module

        self.GP = GaussianProcess(
            train_x=self.train_x,
            train_y=self.train_y, 
            likelihood=self.likelihood,
            mean_module=self.mean_module,
            covar_module=self.covar_module
            )
        
        optimizer_class, self.lr = optimizer_and_lr
        self.optimizer = optimizer_class(self.GP.parameters(), self.lr)
        self.training_iter = training_iter
        

    def add_data(self, x_new, y_new):
        # Update stored data
        self.train_x = t.cat([self.train_x, x_new])
        self.train_y = t.cat([self.train_y, y_new])

        # Update GPs data:
        self.GP.set_train_data(self.train_x, self.train_y, strict=False)


    def train(self, verbose=False):
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
            loss = -mll(output, self.train_y)
            loss.backward()

            if verbose:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, self.training_iter, loss.item(),
                    self.GP.covar_module.base_kernel.lengthscale.item(),
                    self.likelihood.noise.item()
                ))

            self.optimizer.step()


    def predict(self, x):
        # Get into evaluation (predictive posterior) mode
        self.GP.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with t.no_grad(), gpy.settings.fast_pred_var():
            return self.likelihood(self.GP(x))