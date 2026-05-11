from bo.problems.toy_rbf import ToyRBFProblemFamily


def build_problem_family(cfg: dict, device, dtype):
    '''Build problem family from config'''
    problem_type = cfg["type"]

    if problem_type == "toy_rbf":
        return ToyRBFProblemFamily(
            device=device,
            dtype=dtype,
            lb=cfg["lb"],
            ub=cfg["ub"],
            M=cfg.get("M", 3),
            min_cost=cfg.get("min_cost", 1.0),
        )

    raise ValueError(f"Unknown problem family type: {problem_type}")