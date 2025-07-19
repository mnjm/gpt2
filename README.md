# GPT-2 Pretraining

An attempt at GPT2 Pretraining on FineWeb Edu dataset (20B tokens). Follows Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt).

## Loss

![Loss](./misc/loss.png)

## Sample Output

```
Input: Hello, I'm a language model,
> Hello, I'm a language model, you're asking me if I want a simple way to explain what a simple language will do but you're wondering
> Hello, I'm a language model, so what I'd like to do is to change the parameters and let me try it for my future job.
> Hello, I'm a language model, so I know that when you don't know where to start, I can think of no one. When I
> Hello, I'm a language model, so if you get lost in a language, try to learn English. I used to say English is a language
> Hello, I'm a language model, so here we go here:
I think that there is a need for a language-centered framework that addresses
```

## Files
- `model.py`: Minimal GPT2 implementation
- `train.py`: Training script with multi-gpu training support
- `utils.py`: Mainly contains data loader.
- `download-fineweb-edu.py`: Downloads the FineWeb Edu (20B) dataset from Hugging Face, tokenizes it, and saves it in shards.

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