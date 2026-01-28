##########################
# TEST FOR BLACKBOX FUNC #
##########################
def test_blackbox(device, dtype):
    from functions.blackbox_function import BlackBoxFunc
    from candidate_set import CandidateSet
    import torch as t

    bb_func = BlackBoxFunc(t.sin)
    c_set = CandidateSet(device=device, dtype=dtype, res=10, D=1, minimum=1, maximum=10)
    grid = c_set.get_grid()

    eval = bb_func.evaluate(c_set.get_grid())

    print(grid)
    print(eval)