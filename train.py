import os
import math
from time import time
import inspect
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from pathlib import Path

from utils import get_torch_device, FineWebEduLoader #, TinyShakespeareLoader
from model import GPT, GPTConfig

# ------------- params -------------
max_lr = 1.2e-3 # 6e-4 * 2 Doubled the learning rate compared to GPT2
min_lr = max_lr * 0.1
warmup_steps = 300 # 715
n_steps = 9537 # 19073
total_batch_size_tok = 524288 # 2**19, ~0.5M in number of tokens | desired batch size | for grad accumulation
micro_batch_size = 16
block_size = 1024 # context length
val_per_step = 250
save_ckpt_every = 1000
log_dir = Path("./logs")
# ----------------------------------

# set up DDP (distributed data parallel)
# torchrun command sets the new env vars RANK, LOCAL_RANK and WORLD_SIZE
# RANK = global id of the current process involved in distributed training accross nodes
# LOCAL_RANK = Local id in a node of the current process involved in distributed training
# WORLD_SIZE = Total number of processes involved in training
# Ex. If 2 nodes each with 4 GPUs then WORLD_SIZE = 8, RANK -> {0..7}, LOCAL_RANK in each node -> {0..4}
# Note: This code only runs on a single NODE
# torchrun --standalone --nproc_per_node={N_GPUS} train.py
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available(), "Only CUDA is supported with in DDP mode"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # non-DDP run
    ddp_rank = ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = get_torch_device()
    print(f"using device: {device}")

def _print(text):
    # just a wrappper that print things only in the master process
    if master_process:
        print(text)

assert save_ckpt_every % val_per_step == 0, "val_per_step should be multiple of save_ckpt_every"
tok_per_microbatch = micro_batch_size * block_size * ddp_world_size
assert total_batch_size_tok % tok_per_microbatch == 0, f"Make sure {total_batch_size_tok=} is divisible by (batch_size * block_size * dpp_world_size)"
grad_accum_steps = total_batch_size_tok // tok_per_microbatch
_print(f"Calculated gradient accumulation steps: {grad_accum_steps}")

log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "logs.txt"
log_file.touch()
rng_seed = 1337

torch.manual_seed(rng_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(rng_seed)

# vocab_size changed from 50257 (vocab_size of gpt2) to 50304 as it is close to a power of 2
# which aligns well with memory and GPT tensor core optimizations, making it more efficient for
# neural network computations than arbitrary or odd-sized values.
gpt2_config = GPTConfig(vocab_size=50304)

def get_lr(step):
    # 1) linar warmup for warmup_steps
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    # 2) if step > max_steps, return min learning rate
    elif step > n_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (n_steps - warmup_steps)
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
    _print(f"Num decayed params tensors: {len(decay_params)}, with {num_decay_params:,} params")
    _print(f"Num non-decayed params tensors: {len(no_decay_params)}, with {num_no_decay_params:,} params")
    # AdamW
    fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    _print(f"useing fused AdamW: {fused_avail}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused_avail)
    return optimizer

def main():
    train_loader = FineWebEduLoader(batch_size=micro_batch_size, block_size=block_size, proc_rank=ddp_local_rank, n_proc=ddp_world_size, split="train")
    val_loader = FineWebEduLoader(batch_size=micro_batch_size, block_size=block_size, proc_rank=ddp_local_rank, n_proc=ddp_world_size, split="val")

    # Uses TF32 if available check: https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision("high")

    model = GPT(gpt2_config)
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # raw unwrapped model

    device_type = "cuda" if ddp else device.type

    optimizer = configure_optimizer(raw_model, weight_decay=0.1, learning_rate=6e-4, device=device) # lr is updated in the training loop below

    for step in range(n_steps):
        last_step = step == (n_steps - 1)
        t0 = time()

         # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with log_file.open("a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % save_ckpt_every == 0 or last_step):
                    checkpoint_path = log_dir / f"gpt2_{step:05d}.pt"
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        # 'optimizer_state_dict': optimizer.state_dict,
                        # 'rng_seed': rng_seed
                    }
                    torch.save(checkpoint, str(checkpoint_path))

        # train
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # autocast the forward pass to bfloat16 - i think it only applies to matmul ops check: https://docs.pytorch.org/docs/stable/amp.html#torch.autocast
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for grad accumulation, because the gradients just add on each
            # successive backward(). SUM of grad corresponds to SUM of objective, but instead we want MEAN, so scale the loss by `1/grad_accum_steps`
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            # when using ddp, loss.backward() call will sync the gredients with all the GPU, here we only want to sync in the last micro_batch.
            # The proper way to do this using no_sync check here: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync
            # this no_sync internally resets require_bakward_grad_sync, lets use that directly
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # only sync on the last micro batch
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
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
        tokens_processed = train_loader.batch_size * train_loader.block_size * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        _print(f"step: {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        if master_process:
            with log_file.open("a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()
