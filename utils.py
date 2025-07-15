import torch
import tiktoken
from pathlib import Path

def get_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        try:
            import torch_xla.core.xla_model as xm # type: ignore
            device = xm.xla_device()
        except ImportError:
            device = torch.device("cpu")
    return device

tiny_shakespeare_raw_txt = Path("./dataset/tinyshakespeare/raw.txt")
class TinyShakespeareLoader:
    
    def __init__(self, batch_size, block_size, proc_rank, n_proc):
        self.batch_size = batch_size # batch size
        self.block_size = block_size # context length
        self.proc_rank = proc_rank
        self.n_proc = n_proc
        
        with tiny_shakespeare_raw_txt.open('r') as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))
        if proc_rank == 0:
            print(f"Loaded {len(self.tokens)} tokens")
        
        # state
        self.current_position = self.batch_size * self.block_size * self.proc_rank
    
    def next_batch(self):
        B, T = self.batch_size, self.block_size
        buf = self.tokens[self.current_position : self.current_position + (B*T)+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += (B * T * self.n_proc)
        if self.current_position + (B * T * self.n_proc + 1) > len(self.tokens):
            self.current_position = (B * T * self.proc_rank * self.n_proc)
        return x, y