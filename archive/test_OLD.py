from tests.blackbox_test import test_blackbox
from tests.candidate_set_test import test_candidate_set, test_candidate_set_GPU
from tests.gp_wrapper_test import test_GP, test_GP_refit, test_GP_GPU
from tests.bo_pipeline_test import test_BO, test_BO_GPU


if __name__ == '__main__':
    import torch as t

    # BLACKBOX TEST:
    #test_blackbox('cpu', t.float32)
    

    # CANDIDATE SET TEST:
    #test_candidate_set()
    #test_candidate_set_GPU()


    # GAUSSIAN PROCESS TEST:
    #test_GP()
    #test_GP_refit()
    #test_GP_GPU()


    # TEST BO ON CPU TEST:
    test_BO(num_steps=50, num_training_iter=10)

    # TEST BO ON GPU TEST:
    training_iters_gp = 10
    number_of_BO_iterations = range(50, 51, 10) 
    
    print('\n\n----------------------------\nSTARTING TEST OF BO PIPELINE - GPU\n----------------------------')
    print(f'Training iterations per re-optimization step: {training_iters_gp}')
    print(f'Number of BO iterations to test: {number_of_BO_iterations}\n----------------------------')
    print
    
    for i in number_of_BO_iterations:
       test_BO_GPU(i, training_iters_gp)