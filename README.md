# buildGPT2

This repository trains a GPT-2 style language model from scratch in PyTorch, evaluates on HellaSwag, generates qualitative samples, writes structured logs/checkpoints, and includes a plotting notebook for run analysis.

## What This Repository Does

- Implements a GPT-2 architecture from scratch (attention, MLP, residual blocks, weight tying, custom init).
- Trains on tokenized FineWeb-EDU shards with gradient accumulation.
- Supports single-process and multi-GPU Distributed Data Parallel (DDP) runs.
- Evaluates validation loss and HellaSwag accuracy during training.
- Generates sample completions during training.
- Saves logs and model checkpoints with descriptive run names derived from config values.
- Plots training/validation/hellaswag metrics from log files in a notebook.

## Repository Layout

- `scripts/main.py`: Main entrypoint for modular, config-driven training.
- `scripts/training.py`: Training loop, LR schedule, eval cadence, checkpointing, throughput reporting.
- `scripts/model.py`: GPT model, blocks, optimizer setup, optional HF pretrained weight loading helper.
- `scripts/dataloader.py`: Shard-based train/val dataloader over local `.npy` token shards.
- `scripts/evaluation.py`: Validation loss, HellaSwag scoring, and text generation helpers.
- `scripts/ddp.py`: DDP bootstrap/cleanup and device selection logic.
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

### Example Plots

![Training/Validation/HellaSwag plots from `plot_results.ipynb`](assets/plots/plot_results_metrics.png)

## Standalone Utility Scripts

- `hellaswag.py`: evaluate HuggingFace GPT-2 checkpoints directly on HellaSwag.
- `fineweb.py`: build local token shards from FineWeb-EDU.
- `train_gpt2.py`: one-file training implementation containing model, data, DDP setup, eval, generation, and training loop.
