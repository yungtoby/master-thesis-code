from tests.blackbox_test import test_blackbox
from tests.bo_pipeline_test import test_BO
from tests.gp_wrapper_test import test_GP, test_GP_refit
from tests.candidate_set_test import candidate_set_test_GPU

if __name__ == '__main__':
    candidate_set_test_GPU()