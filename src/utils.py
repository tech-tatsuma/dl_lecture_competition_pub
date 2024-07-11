import random
import numpy as np
import torch

# def set_seed(seed: int = 0) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

# 再現性を担保するためにシードを固定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False