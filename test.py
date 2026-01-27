from tests.blackbox_test import test_blackbox
from tests.bo_pipeline_test import test_BO, test_BO_GPU
from tests.gp_wrapper_test import test_GP, test_GP_refit, test_GP_GPU
from tests.candidate_set_test import candidate_set_test_GPU

if __name__ == '__main__':
    training_iters_gp = 10
    number_of_BO_iterations = range(50, 51, 10) 

    print('\n\n----------------------------\nSTARTING TEST OF BO PIPELINE - GPU\n----------------------------')
    print(f'Training iterations per re-optimization step: {training_iters_gp}')
    print(f'Number of BO iterations to test: {number_of_BO_iterations}\n----------------------------')
    print

    for i in number_of_BO_iterations:
        test_BO_GPU(i, training_iters_gp)