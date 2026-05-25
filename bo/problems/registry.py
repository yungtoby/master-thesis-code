from bo.problems.toy_rbf import ToyRBFProblemFamily
from bo.problems.yahpo_lcbench import YAHPOLCBenchProblemFamily

def build_problem_family(cfg: dict, device, dtype):
    '''Build problem family from config'''
    problem_type = cfg['type']

    if problem_type == 'toy_rbf':
        return ToyRBFProblemFamily(
            device=device,
            dtype=dtype,
            lb=cfg['lb'],
            ub=cfg['ub'],
            M=cfg.get('M', 3),
            min_cost=cfg.get('min_cost', 1.0),
        )
    
    if problem_type == 'yahpo_lcbench':
        return YAHPOLCBenchProblemFamily(
            device=device,
            dtype=dtype,
            instances=cfg['instances'],
            objective_key=cfg.get('objective_key', 'val_accuracy'),
            cost_key=cfg.get('cost_key', 'time'),
            epoch=cfg.get('epoch', 51),
            objective_scale=cfg.get('objective_scale', 0.01),
        )

    raise ValueError(f'Unknown problem family type: {problem_type}')