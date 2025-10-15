from tests.blackbox_test import test_blackbox
from tests.bo_pipeline_test import test_BO, test_BO_GPU
from tests.gp_wrapper_test import test_GP, test_GP_refit, test_GP_GPU
from tests.candidate_set_test import candidate_set_test_GPU

if __name__ == '__main__':
    test_BO()
    test_BO_GPU()