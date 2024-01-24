# *_*coding:utf-8 *_*
import numpy as np
import torch
import random

def set_seed(seed = 0):
    # print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
