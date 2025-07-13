import torch

def get_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print("Using TPU device")
        except ImportError:
            device = torch.device("cpu")
            print("Using CPU device (no GPU/TPU available)")
    return device