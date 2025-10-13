##################################
# TEST FOR BAYESIAN OPTIMIZATION #
##################################
def test_BO():
    import gpytorch as gpy 
    import torch as t
    import time

    from gaussian_process import GPWrapper
    from candidate_set import CandidateSet 
    from functions.blackbox_function import BlackBoxFunc
    from functions.own_functions import sin_func
    from acquistion_function import EI
    from bayesian_optimization import BO_Pipeline

    # Initialize Acquistion function, candidate set and blackbox function 
    acq_func = EI()
    can_set = CandidateSet(100, 1, 0, 10)
    bb_func = BlackBoxFunc(sin_func)
    budget = 5
    num_steps = 45

    # Initialize model params and model 
    mean_module = gpy.means.ConstantMean()
    covar_module = gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel())
    likelihood = gpy.likelihoods.GaussianLikelihood()
    train_x = t.linspace(0, 10, 5)
    train_y = bb_func.evaluate(train_x)
    optimizer = (t.optim.Adam, 0.1)
    training_iter = 50
    surr_model = GPWrapper(train_x, train_y, likelihood, mean_module, covar_module, optimizer, training_iter)
    
    # Initialize pipeline:
    pipeline = BO_Pipeline(budget, surr_model, acq_func, can_set, bb_func)
    start_time = time.time()
    best_x, best_y = pipeline.run_BO(num_steps)
    end_time = time.time()

    print(f'BO COMPLETE!\nTime used: {(end_time-start_time):.2f} seconds')
    print('Best x found: %.3f   Best y found: %.3f  iterations: %d' % (best_x, best_y, num_steps))