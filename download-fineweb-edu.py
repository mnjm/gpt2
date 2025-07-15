import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

save_dir = Path("./dataset/edu-fineweb10B")
remote_ds_path = "HuggingFaceFW/fineweb-edu"
remote_ds_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

save_dir.mkdir(parents=True, exist_ok=True)

# download the dataset
ds = load_dataset(remote_ds_path, name=remote_ds_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # <|endoftext|> token delimits all documents, Note that it is prefixed with doc tokens.
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_shard(filename, tokens_np):
    np.save(filename, tokens_np)
    
# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    tokens_buffer = np.empty((shard_size,), dtype=np.uint16) # buffer
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, ds, chunksize=16):
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # populate tokens into the current shard
            tokens_buffer[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the shard and create a new one with reminder tokens
            split = "val" if shard_index == 0 else "train"
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            tokens_buffer[token_count:token_count+remainder] = tokens[:remainder]
            filename = save_dir / f"edufineweb_{split}_{shard_index:06d}"
            write_shard(filename, tokens_buffer)
            shard_index += 1
            progress_bar = None
            # populate remaining tokens into buffer
            tokens_buffer[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = save_dir / f"edufineweb_{split}_{shard_index:06d}"
        write_shard(filename, tokens_buffer[:token_count])