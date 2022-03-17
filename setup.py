import torch

def nondeterministic():
    torch.manual_seed(11)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)