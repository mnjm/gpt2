import torch
import math
from time import time
import inspect

from utils import get_torch_device, GPT2DataLoaderLite
from model import GPT, GPTConfig

device = get_torch_device()
print(f"using device: {device}")

# ------------- params -------------
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
batch_size = 4
block_size = 1024 # context length
# ----------------------------------

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# vocab_size changed from 50257 (vocab_size of gpt2) to 50304 as it is close to a power of 2
# which aligns well with memory and GPT tensor core optimizations, making it more efficient for
# neural network computations than arbitrary or odd-sized values.
gpt2_config = GPTConfig(vocab_size=50304)

def get_lr(step):
    # 1) linar warmup for warmup_steps
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    # 2) if step > max_steps, return min learning rate
    elif step > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # starts at 1 and goes to 0 gradually
    return min_lr + coeff * (max_lr - min_lr)

def configure_optimizer(model: torch.nn.Module, weight_decay: float, learning_rate: float, device: torch.device):
    params_dict = { pn: p for pn, p in model.named_parameters() }
    params_dict = { pn:p for pn, p in params_dict.items() if p.requires_grad } # filter params that requires grad
    # create optim groups of any params that is 2D or more. This group will be weight decayed ie weight tensors in Linar and embeddings
    decay_params = [ p for p in params_dict.values() if p.dim() >= 2]
    # create optim groups of any params that is 1D. All biases and layernorm params
    no_decay_params = [ p for p in params_dict.values() if p.dim() < 2]
    optim_groups = [
        { 'params': decay_params, 'weight_decay': weight_decay },
        { 'params': no_decay_params, 'weight_decay': 0.0 },
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_no_decay_params = sum(p.numel() for p in no_decay_params)
    print(f"Num decayed params tensors: {len(decay_params)}, with {num_decay_params:,} params")
    print(f"Num non-decayed params tensors: {len(no_decay_params)}, with {num_no_decay_params:,} params")
    # AdamW
    fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    print(f"useing fused AdamW: {fused_avail}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused_avail)
    return optimizer

if __name__ == "__main__":
    # Uses TF32 if available check: https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision("high")

    model = GPT(gpt2_config)
    model.to(device)
    model = torch.compile(model)

    optimizer = configure_optimizer(model, weight_decay=0.1, learning_rate=6e-4, device=device) # lr is updated in the training loop below

    train_loader = GPT2DataLoaderLite("input.txt", B=batch_size, T=block_size)

    for step in range(max_steps):
        t0 = time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # autocast the forward pass to bfloat16 - i think it only applies to matmul ops check: https://docs.pytorch.org/docs/stable/amp.html#torch.autocast
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        # clip the max global norm to 1.0 - stabalizes trainig during unlucky bad data batch causing shocking the model
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set lr this step
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time() - t0)
        tokens_per_sec = (train_loader.B * train_loader.T) / dt
        print(f"step: {step:5d} | loss: {loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")