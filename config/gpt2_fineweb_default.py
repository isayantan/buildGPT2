CONFIG = {
    "seed": 1337,
    "model": {
        "vocab_size": 50257,
        "block_size": 1024,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
    },
    "data": {
        "data_root": "edu_fineweb10B",
    },
    "training": {
        "total_batch_size": 524288,
        "micro_batch_size": 32,
        "sequence_length": 1024,
        "max_steps": 19073,
    },
    "optimizer": {
        "weight_decay": 0.1,
        "learning_rate": 1e-3,
    },
    "lr_schedule": {
        "max_lr": 1e-3,
        "min_lr_ratio": 0.1,
        "warmup_steps": 715,
    },
    "evaluation": {
        "val_interval": 250,
        "val_loss_steps": 20,
        "hellaswag_interval": 250,
    },
    "generation": {
        "sample_interval": 250,
        "num_return_sequences": 4,
        "max_length": 32,
        "prompt": "Hello, I'm a language model,",
        "top_k": 50,
        "sample_seed": 42,
    },
    "checkpointing": {
        "checkpoint_interval": 5000,
    },
    "runtime": {
        "use_compile": False,
        "matmul_precision": "high",
        "default_cuda_device_index": 4,
    },
    "logging": {
        "log_dir": "log",
    },
}
