##################################
# TEST FOR BAYESIAN OPTIMIZATION #
##################################
def test_BO(num_steps, num_training_iter):
    import gpytorch as gpy 
    import torch as t
    import time

    from gaussian_process import GPWrapper
    from candidate_set import CandidateSet 
    from functions.blackbox_function import BlackBoxFunc
    from functions.own_functions import sin_func
    from acquisition_function import EI
    from bayesian_optimization import BO_Pipeline
    

    print('\n\nINITIALIZING BO PIPELINE ON CPU\n\n--------------------------------')
    
    
    # Device and dtype
    device = t.device('cpu')
    dtype = t.float32


    # Initialize Acquistion function, candidate set and blackbox function 
    acq_func = EI()
    can_set = CandidateSet(device, dtype, 100, 1, 0, 10)
    bb_func = BlackBoxFunc(sin_func)
    budget = 5


    # Initialize model params and model 
    mean_module = gpy.means.ConstantMean()
    covar_module = gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel())
    likelihood = gpy.likelihoods.GaussianLikelihood()
    train_x = t.linspace(0, 10, 5).unsqueeze(-1)
    train_y = bb_func.evaluate(train_x).squeeze(-1)
    optimizer = (t.optim.Adam, 0.1)
    training_iter = num_training_iter
    surr_model = GPWrapper(device, dtype, train_x, train_y, likelihood, mean_module, covar_module, optimizer, training_iter)
    

    # Initialize pipeline:
    pipeline = BO_Pipeline(device, dtype, budget, surr_model, acq_func, can_set, bb_func)
    start_time = time.time()
    best_x, best_y = pipeline.run_BO(num_steps)
    end_time = time.time()

    print(f'BO COMPLETE!\nTime used: {(end_time-start_time):.2f} seconds')
    print('Best x found: %.3f   Best y found: %.3f  iterations: %d' % (best_x, best_y, num_steps))



#########################################
# TEST FOR BAYESIAN OPTIMIZATION ON GPU #
#########################################
def test_BO_GPU(num_steps, num_training_iter):
    import gpytorch as gpy 
    import torch as t
    import time

    from gaussian_process import GPWrapper
    from candidate_set import CandidateSet 
    from functions.blackbox_function import BlackBoxFunc
    from functions.own_functions import sin_func
    from acquisition_function import EI
    from bayesian_optimization import BO_Pipeline


    print('\n\nINITIALIZING BO PIPELINE ON GPU\n\n--------------------------------')


    # Torch params
    if t.mps.is_available():
        device = t.device('mps')
    elif t.cuda.is_available():
        device = t.device('cuda')
    else:
        print("NO GPU AVAILABLE!")
        return
    dtype = t.float32

    print(f"Using device: {device}")

    # Initialize Acquistion function, candidate set and blackbox function (pass device/dtype where supported)
    acq_func = EI()
    can_set = CandidateSet(device, dtype, 100, 1, 0, 10)
    bb_func = BlackBoxFunc(sin_func)
    budget = 5

    # Assert candidate grid on correct device
    grid = can_set.get_grid()
    assert grid.device.type == device.type, f"Grid device {grid.device.type} != {device.type}"
    assert grid.dtype == dtype, f"Grid dtype {grid.dtype} != {dtype}"
    print("CandidateSet grid verified on GPU.")

    # Initialize model params and model (pass device/dtype to GPWrapper)
    mean_module = gpy.means.ConstantMean()
    covar_module = gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel())
    likelihood = gpy.likelihoods.GaussianLikelihood()
    train_x = t.linspace(0, 10, 5, device=device, dtype=dtype).unsqueeze(-1)
    train_y = bb_func.evaluate(train_x).squeeze(-1)
    optimizer = (t.optim.Adam, 0.1)
    training_iter = num_training_iter
    surr_model = GPWrapper(device, dtype, train_x, train_y, likelihood, mean_module, covar_module, optimizer, training_iter)
    
    # Assert surrogate on GPU
    assert next(surr_model.GP.parameters()).device.type == device.type, "GP parameters not on GPU"
    assert surr_model.train_x.device.type == device.type, "Train data not on GPU"
    print("GPWrapper verified on GPU.")

    # Initialize pipeline (pass device/dtype)
    pipeline = BO_Pipeline(device, dtype, budget, surr_model, acq_func, can_set, bb_func)
    
    # Run BO
    start_time = time.time()

    best_x, best_y = pipeline.run_BO(num_steps)

    # NOTE: UNCOMMENT TO GET TRACE OF OPERATIONS!
    #with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU, t.profiler.ProfilerActivity.CUDA]) as prof:
    #    best_x, best_y = pipeline.run_BO(num_steps)
    #prof.export_chrome_trace("cuda_trace.json")
    end_time = time.time()

    # Assert outputs on GPU
    assert best_x.device.type == device.type, "Best x not on GPU"
    assert best_y.device.type == device.type, "Best y not on GPU"

    print(f'BO COMPLETE ON GPU!\nTime used: {(end_time-start_time):.2f} seconds')
    print('Best x found: %.3f   Best y found: %.3f  iterations: %d' % (best_x.item(), best_y.item(), num_steps))
    print("All device assertions passed—no CPU transfers detected in key components.")