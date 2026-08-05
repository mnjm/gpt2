from argparse import ArgumentParser
from time import perf_counter

import tiktoken
import torch
import torch.nn.functional as F

from model import GPT
from utils import get_torch_device

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2")
parser.add_argument("--prompt", type=str, default="Hello, I'm language model")
parser.add_argument("--use_cache", action="store_true", help="Use KV caching during generation")
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--gen_len", type=int, default=100)
args = parser.parse_args()

device = get_torch_device()
print(f"Using {device}")

model_name = args.model
prompt = args.prompt
use_cache = args.use_cache
num_samples = args.num_samples
gen_len = args.gen_len

print(f"Model: {model_name}, prompt: {prompt!r}, use_cache: {use_cache}, num_samples: {num_samples}, gen_len: {gen_len}")

model = GPT.from_pretrained(model_name)
model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long)  # (T,)
tokens = tokens.unsqueeze(0).repeat(num_samples, 1)  # (B, T)
x = tokens.to(device)

torch.manual_seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed(42)

kv_caches = model.init_kv_caches() if use_cache else None
model_input = x  # prompt prefill on the first cached forward

if device.type == "cuda":
    torch.cuda.synchronize()
elif device.type == "mps":
    torch.mps.synchronize()
start_time = perf_counter()

with torch.no_grad():
    while x.size(1) < gen_len:
        logits, _, kv_caches = model(model_input, kv_caches=kv_caches)
        logits = logits[:, -1, :]  # last position logits (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (hf pipeline default)
        # topk_probs here becomes (5, 50) topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select token from top-k probs
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the squence
        x = torch.cat((x, xcol), dim=1)
        model_input = xcol if use_cache else x

if device.type == "cuda":
    torch.cuda.synchronize()
elif device.type == "mps":
    torch.mps.synchronize()
elapsed_time = perf_counter() - start_time

for i in range(num_samples):
    tokens = x[i, :gen_len].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

print("=" * 80)
print(f"Generation time: {elapsed_time:.2f} seconds")
print("=" * 80)
