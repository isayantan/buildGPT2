# buildGPT2

This repository trains a GPT-2 style language model from scratch in PyTorch, evaluates on HellaSwag, generates qualitative samples, writes structured logs/checkpoints, and includes a plotting notebook for run analysis.

## Contents

- [Highlights](#highlights)
- [What This Repository Does](#what-this-repository-does)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Running Training](#running-training)
- [Logs and Checkpoints](#logs-and-checkpoints)
- [Plotting Results](#plotting-results)
- [Generate From Trained Model](#generate-from-trained-model)
- [Repository Layout](#repository-layout)
- [Standalone Utility Scripts](#standalone-utility-scripts)

## Highlights

### Training Curves

![Training/Validation/HellaSwag plots from `plot_results.ipynb`](assets/plots/plot_results_metrics.png)

### Prompt and Generated Samples

#### Sample 1

**Prompt**

```text
Hello, I'm a language model,
```

**Generated Sample**

```text
Hello, I'm a language model, and I like to use them for learning but still use them as long as my students don't do anything crazy. And as a teacher, I've heard quite a few of the words (and sometimes you're lucky to have the opportunity!), but what I'm really afraid of is
```

#### Sample 2

**Prompt**

```text
The future of AI is
```

**Generated Sample**

```text
The future of AI is in the making. The Internet has many good aspects of its own for humanity that are being neglected and neglected by the governments, corporations, and nonprofits that we humans are working to eliminate. These are a few things AI can do that promise us to never forget.
A few years ago, the
```

#### Sample 3

**Prompt**

```text
In a shocking finding, researchers discovered that
```

**Generated Sample**

```text
In a shocking finding, researchers discovered that the brain was not only able to regulate appetite with a healthy appetite hormone, the hormone leptin. Instead it produced a molecule that was able to kill off brain tumor cells.
"Now we see these cells dying and dying and growing. We think this is a proof of the
```

### How These Samples Were Generated (Brief)

- Checkpoint: `log/gpt2_fineweb_default__l12_h12_e768__tbs524288_mb32_seq1024_steps19073__model_10000.pt`
- Script: `scripts/generate_from_checkpoint.py`
- Sampling config: `max_length=64`, `top_k=50`, `sample_seed=42`, `num_return_sequences=1`
- Prompt-specific runs use the same command with different `--prompt` values

## What This Repository Does

- Implements a GPT-2 architecture from scratch (attention, MLP, residual blocks, weight tying, custom init).
- Trains on tokenized FineWeb-EDU shards with gradient accumulation.
- Supports single-process and multi-GPU Distributed Data Parallel (DDP) runs.
- Evaluates validation loss and HellaSwag accuracy during training.
- Generates sample completions during training.
- Saves logs and model checkpoints with descriptive run names derived from config values.
- Plots training/validation/hellaswag metrics from log files in a notebook.

## Environment Setup

Use Python 3.10+ (3.8+ should also work).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy tiktoken datasets tqdm requests transformers matplotlib jupyter
```

## Data Preparation

### FineWeb-EDU Shards

Generate token shards:

```bash
python fineweb.py
```

Expected training path is set by config at `data.data_root` (default: `edu_fineweb10B`).

Important: `fineweb.py` currently defines `local_dir = "edu_fiineweb10B"` (double `i`), while training defaults to `edu_fineweb10B`. Keep these consistent by either renaming the generated folder or updating `config/gpt2_fineweb_default.py`.

### HellaSwag Cache

HellaSwag files are downloaded on demand by `hellaswag.py`/training evaluation into `hellaswag/`.

## Configuration

The main config is `config/gpt2_fineweb_default.py` and contains:

- `model`: GPT shape (`vocab_size`, `block_size`, `n_layer`, `n_head`, `n_embd`)
- `data`: shard location (`data_root`)
- `training`: `total_batch_size`, `micro_batch_size`, `sequence_length`, `max_steps`
- `optimizer`: weight decay and base LR
- `lr_schedule`: warmup and cosine settings
- `evaluation`: validation and HellaSwag intervals
- `generation`: prompt/sampling settings
- `checkpointing`: checkpoint save interval
- `runtime`: `use_compile`, matmul precision, default single-process CUDA index
- `logging`: log directory

Tokens processed per optimizer step are globally fixed by:

`tokens_per_step = total_batch_size`

because:

`grad_accum_steps = total_batch_size / (micro_batch_size * sequence_length * world_size)`

and:

`tokens_per_step = micro_batch_size * sequence_length * grad_accum_steps * world_size`

## Running Training

### Single GPU (non-DDP)

```bash
python scripts/main.py --config config/gpt2_fineweb_default.py
```

### Multi-GPU DDP

```bash
NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=0,4,5,6 torchrun --standalone --nproc_per_node=4 scripts/main.py --config config/gpt2_fineweb_default.py
```

Notes:

- DDP is auto-enabled when `torchrun` sets `RANK/LOCAL_RANK/WORLD_SIZE`.
- `CUDA_VISIBLE_DEVICES` chooses physical GPUs.
- `nproc_per_node` should match the number of visible GPUs.

## Logs and Checkpoints

Run name is built from config filename + key hyperparameters, for example:

`gpt2_fineweb_default__l12_h12_e768__tbs524288_mb32_seq1024_steps10`

Artifacts:

- Log file: `log/<run_name>.log`
- Checkpoint: `log/<run_name>__model_<step>.pt`

Log lines include:

- Metadata: `run_name ...`, `device ...`, `grad_accum_steps ...`
- Metrics: `<step> train <loss>`, `<step> val <loss>`, `<step> hella <acc>`

## Plotting Results

Run this notebook to generate the plots:

`plot_results.ipynb`

```bash
jupyter notebook plot_results.ipynb
```

Then run all cells (or `jupyter lab plot_results.ipynb` if you prefer JupyterLab).

What this notebook does:

- lists available `log/*.log` files,
- selects one run log (`RUN_LOG_NAME` or latest),
- parses `train`, `val`, `hella` lines,
- renders three horizontal plots:
  - train loss vs step,
  - validation loss vs step,
  - HellaSwag accuracy vs step.

The example plot is shown at the top under **Highlights**.

## Generate From Trained Model

Run this with multiple prompts from a saved checkpoint:

```bash
CKPT=log/gpt2_fineweb_default__l12_h12_e768__tbs524288_mb32_seq1024_steps19073__model_10000.pt

python scripts/generate_from_checkpoint.py --checkpoint "$CKPT" \
  --prompt "Hello, I'm a language model," \
  --max-length 64 \
  --top-k 50 \
  --num-return-sequences 1 \
  --sample-seed 42

python scripts/generate_from_checkpoint.py --checkpoint "$CKPT" \
  --prompt "The future of AI is" \
  --max-length 64 \
  --top-k 50 \
  --num-return-sequences 1 \
  --sample-seed 42

python scripts/generate_from_checkpoint.py --checkpoint "$CKPT" \
  --prompt "In a shocking finding, researchers discovered that" \
  --max-length 64 \
  --top-k 50 \
  --num-return-sequences 1 \
  --sample-seed 42
```

Brief generation details:

- Checkpoint used: `...steps19073__model_10000.pt`
- Prompts: `Hello, I'm a language model,`, `The future of AI is`, `In a shocking finding, researchers discovered that`
- Sampling: `top-k=50`, `sample_seed=42`, one return sequence per prompt
- Max token length per sample: `64`
- Prompt-only and sample-only formatted outputs are highlighted at the top under **Highlights**

## Repository Layout

- `scripts/main.py`: Main entrypoint for modular, config-driven training.
- `scripts/training.py`: Training loop, LR schedule, eval cadence, checkpointing, throughput reporting.
- `scripts/model.py`: GPT model, blocks, optimizer setup, optional HF pretrained weight loading helper.
- `scripts/dataloader.py`: Shard-based train/val dataloader over local `.npy` token shards.
- `scripts/evaluation.py`: Validation loss, HellaSwag scoring, and text generation helpers.
- `scripts/ddp.py`: DDP bootstrap/cleanup and device selection logic.
- `scripts/generate_from_checkpoint.py`: Inference entrypoint to sample text from a saved `.pt` checkpoint.
- `scripts/utils.py`: Config loading, run-name construction, log/checkpoint file helpers.
- `config/gpt2_fineweb_default.py`: Default hyperparameter config used by `scripts/main.py`.
- `train_gpt2.py`: Original monolithic training script retained as a baseline/reference.
- `fineweb.py`: Utility script to download/tokenize FineWeb-EDU and write shard files.
- `hellaswag.py`: Utility script to download/render/evaluate HellaSwag examples.
- `plot_results.ipynb`: Notebook that selects a run log from `log/`, parses `train/val/hella` entries, and renders the 3 result plots.
- `assets/plots/plot_results_metrics.png`: Example output figure generated from `plot_results.ipynb` (used in this README).
- `edu_fineweb10B/`: Local training token shards consumed by the dataloader.
- `hellaswag/`: Local cache for downloaded HellaSwag JSONL files.
- `log/`: Run logs (`*.log`) and checkpoint files (`*.pt`).

## Standalone Utility Scripts

- `hellaswag.py`: evaluate HuggingFace GPT-2 checkpoints directly on HellaSwag.
- `fineweb.py`: build local token shards from FineWeb-EDU.
- `train_gpt2.py`: one-file training implementation containing model, data, DDP setup, eval, generation, and training loop.
- `scripts/generate_from_checkpoint.py`: load a trained checkpoint and sample text from a prompt.
