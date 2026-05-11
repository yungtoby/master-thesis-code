import random
import numpy as np
import torch


def seed_everything(seed, deterministic):
    '''Function to seed everything'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic