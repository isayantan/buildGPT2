# GPT-2 Training with PyTorch

This repository contains a PyTorch implementation for training a GPT-2-like language model. The training script (`train_gpt2.py`) is designed to be efficient and leverages modern techniques such as mixed precision training, fused optimizers, and learning rate scheduling.

## Features
- **Custom GPT-2 Implementation**: The model is implemented from scratch, including components like self-attention, MLP, and layer normalization.
- **Flash Attention**: Implements efficient attention computation using PyTorch's `scaled_dot_product_attention` function. This method reduces memory usage and speeds up training by avoiding the need to explicitly compute and store large attention matrices. It is particularly useful for long sequences and modern GPUs.
- **Mixed Precision Training**: Utilizes `torch.autocast` with `bfloat16` for faster training on modern GPUs.
- **Fused AdamW Optimizer**: Dynamically enables fused kernels for the AdamW optimizer if supported by the hardware.
- **Learning Rate Scheduler**: Implements a warmup and cosine decay learning rate schedule.
- **Gradient Clipping**: Prevents exploding gradients by clipping the gradient norm.
- **Gradient Accumulation**: Simulates larger effective batch sizes by accumulating gradients over multiple smaller micro-batches before updating weights. This allows training with larger batch sizes on GPUs with limited memory.
- **Data Distributed Parallel (DDP)**: Enables multi-GPU training across a single machine or multiple machines. Automatically synchronizes gradients across GPUs and supports configurable batch splitting, allowing efficient scaling of training with minimal code changes.

## Requirements
- Python 3.8+
- PyTorch 2.0+
- A CUDA-enabled GPU (Ampere or newer recommended for optimal performance)

Install the required dependencies:
```bash
pip install torch torchvision tiktoken
```

## File Structure
- `train_gpt2.py`: Main training script.
- `data/tiny_shakespeare.txt`: Example dataset for training.
- `README.md`: Project documentation.

## Usage
### Training the Model
To train the model on a single GPU, run the following command:
```bash
python train_gpt2.py
```

### Distributed Training (Multi-GPU)
To train using multiple GPUs with DDP, use `torchrun`. For example, to train on 2 specific GPUs (3 and 4):
```bash
CUDA_VISIBLE_DEVICES=3,4 torchrun --standalone --nproc_per_node=2 train_gpt2.py
```

For all available GPUs on a single machine:
```bash
torchrun --standalone --nproc_per_node=<num_gpus> train_gpt2.py
```

### Key Configurations
- **Batch Size**: Set in the `DataLoaderLite` class (`B` parameter).
- **Sequence Length**: Set in the `DataLoaderLite` class (`T` parameter).
- **Learning Rate**: Configured in the `configure_optimizers` method.
- **Model Architecture**: Defined in the `GPTConfig` class.

### Example Output
During training, the script will log the following metrics:
- **Step**: Training step number.
- **Loss**: Cross-entropy loss for the language modeling task.
- **Learning Rate (lr)**: Current learning rate (changes with warmup and cosine decay schedule).
- **Gradient Norm**: The norm of the gradients after clipping.
- **Time**: Time taken for each training step in milliseconds.
- **Tokens per Second (tok/sec)**: Throughput in tokens processed per second.

Example log:
```
step 0 | loss: 10.935456 | lr: 0.000060 | norm: 0.9876 | time: 1917.00ms | tok/sec: 8546.68
step 1 | loss: 9.876543 | lr: 0.000120 | norm: 0.8765 | time: 1800.00ms | tok/sec: 9000.00
```

## Model Details
### GPT Architecture
- **Embedding Size**: 768
- **Number of Layers**: 12
- **Number of Attention Heads**: 12
- **Context Length**: 1024 tokens

### Optimizer
- **AdamW**: Includes weight decay for regularization.
- **Fused Kernels**: Enabled if supported by the hardware for faster training.

### Learning Rate Schedule
- **Warmup Steps**: Gradual increase in learning rate for the first few steps.
- **Cosine Decay**: Smooth decay of learning rate after the warmup phase.

## Dataset
The training script uses the `tiny_shakespeare.txt` dataset as an example. Replace this file with your own dataset for custom training. Ensure the dataset is a plain text file.

## Performance Optimization
- **Mixed Precision**: Reduces memory usage and speeds up training using `bfloat16`.
- **Fused Optimizer**: Reduces kernel launch overhead and improves efficiency.
- **Gradient Clipping**: Stabilizes training by preventing exploding gradients.
- **Flash Attention**: Implements efficient attention computation using PyTorch's `scaled_dot_product_attention` function. This method reduces memory usage and speeds up training by avoiding the need to explicitly compute and store large attention matrices. It is particularly useful for long sequences and modern GPUs.

## Future Work
- Implement evaluation and inference scripts.
- Extend the model to support fine-tuning on downstream tasks.

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)



