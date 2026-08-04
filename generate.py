import tiktoken
import torch
import torch.nn.functional as F

from model import GPT
from utils import get_torch_device

device = get_torch_device()
print(f"using {device}")

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)

torch.manual_seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed(42)

# generate! right now x is (B, T) where B = 5, T = 8
while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x)  # (B, T, vocab_size)
        logits = logits[:, -1, :]  # last position logits (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (hf pipeline default)
        # topk_probs here becomes (5, 50) topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # selec token from top-k probs
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the squence
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
