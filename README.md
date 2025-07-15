# GPT-2 Pretraining

An attempt at GPT2 Pretraining on FineWeb Edu dataset (10B tokens). Follows Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt).

>[!Important]
>Training was stopped due to wallet converging to 0, not loss.

Ran a training run on 4x A100 GPUs until my wallet cried for mercy, so did "Poor man's early stopping" - when GPU bill > bank balance. | SAD FACE

## Files
- `model.py`: Minimal GPT2 implementation
- `train.py`: Training script with multi-gpu training support
- `utils.py`: Mainly contains data loader.
- `download-fineweb-edu.py`: Downloads the FineWeb Edu (10B) dataset from Hugging Face, tokenizes it, and saves it in shards.

## Usage
1. Download dataset
```
python download-fineweb-edu.py
```

2. Start training: 

**If you're rich with multiple GPUs**
```
torchrun --standalone --nproc_per_node=<No.of.GPUs> train.py
```
or **Single GPU / CPU**
```
python train.py
```