import random
import numpy as np
import torch


def seed_everything(seed, deterministic):
    '''Seed global RNGs; separately owned RNGs must also be seeded by their owner.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False