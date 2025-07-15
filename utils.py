import torch
import tiktoken
from train import _print

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

class GPT2DataLoaderLite:
    
    def __init__(self, filepath, B, T, process_rank, num_processess):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processess
        
        with open(filepath, 'r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))
        _print(f"Loaded {len(self.tokens)} tokens")
        
        # state
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + (B*T)+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += (B * T * self.num_processes)
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = (B * T * self.process_rank * self.num_processes)
        return x, y