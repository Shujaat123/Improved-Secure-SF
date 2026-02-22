import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
