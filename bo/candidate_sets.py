from bo.candidate_set_batch import BatchedCandidateSet


def build_candidate_set(cfg: dict, device, dtype, B):
    '''Build candidate set from config'''
    candidate_type = cfg["type"]

    if candidate_type == "jittered_grid":
        return BatchedCandidateSet(
            device=device,
            dtype=dtype,
            B=B,
            res=cfg["res"],
            D=cfg["D"],
            minimum=cfg["minimum"],
            maximum=cfg["maximum"],
            jitter_frac=cfg.get("jitter_frac", 0.35),
        )

    raise ValueError(f"Unknown candidate set type: {candidate_type}")