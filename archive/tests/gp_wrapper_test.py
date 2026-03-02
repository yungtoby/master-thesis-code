#####################################
# TEST FOR GAUSSIAN PROCESS WRAPPER #
#####################################
def test_GP():
    from gaussian_process import GPWrapper
    import gpytorch as gpy
    import torch as t
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Initialize likelihood and model
    mean_module = gpy.means.ConstantMean()
    covar_module = gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel())
    likelihood = gpy.likelihoods.GaussianLikelihood()

    # Initizlie training data
    train_x = t.linspace(0, 1, 100)
    train_y = t.sin((2*t.pi)*train_x) + (t.randn(train_x.size()) * t.sqrt(t.tensor(0.04)))

    # Initialize optimizer
    optimizer = (t.optim.Adam, 0.1)
    training_iter = 50

    # Initialize model
    model = GPWrapper(t.device('cpu'), t.float32, train_x, train_y, likelihood, mean_module, covar_module, optimizer, training_iter)

    # Train model
    model.train(verbose=True)

    # Predict with model on test data
    test_x = t.linspace(0, 1, 51)
    observed_pred = model.predict(test_x)

    with t.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()


###########################################
# TEST FOR GAUSSIAN PROCESS WRAPPER REFIT #
###########################################
def test_GP_refit():
    from gaussian_process import GPWrapper
    import gpytorch as gpy
    import torch as t
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Initialize likelihood and model
    mean_module = gpy.means.ConstantMean()
    covar_module = gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel())
    likelihood = gpy.likelihoods.GaussianLikelihood()

    # Initizlie training data
    train_x = t.linspace(0, 1, 10).unsqueeze(-1)
    train_y = (t.sin((2*t.pi)*train_x) * train_x) + (t.randn(train_x.size()) * t.sqrt(t.tensor(0.04)))

    # Initialize optimizer
    optimizer = (t.optim.Adam, 0.1)
    training_iter = 50

    # Initialize model
    model = GPWrapper(t.device('cpu'), t.float32, train_x, train_y, likelihood, mean_module, covar_module, optimizer, training_iter)

    # Train model
    model.train(verbose=True)

    # Predict with model on test data
    test_x = t.linspace(0, 2, 51)
    observed_pred = model.predict(test_x)

    with t.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-5, 5])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()
    

    # Newly aquiried training point
    train_x_new = t.linspace(1, 2, 100).unsqueeze(-1)
    train_y_new = (t.cos((2*t.pi)*train_x_new) * train_x_new) + (t.randn(train_x_new.size()) * t.sqrt(t.tensor(0.04)))

    train_x = t.cat([train_x, train_x_new])
    train_y = t.cat([train_y, train_y_new])

    model.add_data(train_x_new, train_y_new)
    model.train(verbose=True)
    observed_pred = model.predict(test_x)

    with t.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-5, 5])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()



###########################################
# TEST FOR GAUSSIAN PROCESS WRAPPER (GPU) #
###########################################
def test_GP_GPU():
    import torch as t
    import gpytorch as gpy
    from functions.blackbox_function import BlackBoxFunc
    from functions.own_functions import weird_func_2
    from candidate_set import CandidateSet
    from gaussian_process import GPWrapper


    # Initalize device:
    if t.mps.is_available():
        device = t.device('mps')
    elif t.cuda.is_available():
        device = t.device('cuda')
    dtype = t.float32

    # Initialize grid and blackbox func:
    c_set = CandidateSet(device=device, dtype=dtype, res=30, D=1, minimum=0.0, maximum=6.28)
    grid = c_set.get_grid() 
    bb_func = BlackBoxFunc(weird_func_2)

    init_inds = t.randperm(grid.shape[0], device=device)[:6]
    train_x = grid[init_inds].clone()
    train_y = bb_func.evaluate(train_x)

    # Initialize GP components:
    likelihood = gpy.likelihoods.GaussianLikelihood().to(device)
    mean_module = gpy.means.ConstantMean()
    covar_module = gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel())

    # Initialize GP Wrapper:
    gpw = GPWrapper(
        device=device,
        dtype=dtype,
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        mean_module=mean_module,
        covar_module=covar_module,
        optimizer_and_lr=(t.optim.Adam, 0.1),
        training_iter=30
    )

    # Try to train the GP Wrapper
    gpw.train(verbose=False)
    observed_pred = gpw.predict(grid)

    means = observed_pred.mean
    stds = observed_pred.stddev

    # Check for device mismatches:
    print(f'Current means device: {means.device.type} stds device: {stds.device.type}')
    assert means.device.type == device.type and stds.device.type == device.type
    print(f'\nGP Wrapper object created, fit and inferred succesfully!\nRunning on device: {gpw.device}')