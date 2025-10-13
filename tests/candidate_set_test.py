##########################
# TEST FOR CANDIDATE SET #
##########################
def candidate_set_test_GPU():
    import torch as t
    from candidate_set import CandidateSet

    # Initialize candidate set params:
    
    ## Torch params
    if t.mps.is_available():
        device = t.device('mps')
    elif t.cuda.is_available():
        device = t.device('cuda')
    dtype = t.float32

    ## Candidate set params
    res = 10
    D = 1
    minimum = 1
    maximum = 10


    # New candidate set:
    try:
        c_set = CandidateSet(
            device=device,
            dtype=dtype,
            res = res,
            D=D,
            minimum=minimum,
            maximum=maximum
        )
        print(f'Candidate set created succesfully!\nRunning on device: {c_set.device}')
        
    except Exception as e:
        print(f'Candiate set failed!\nError: {e}')