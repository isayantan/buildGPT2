import math
import time

import tiktoken
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.dataloader import DataLoaderLite
from scripts.evaluation import evaluate_hellaswag, evaluate_validation, generate_samples
from scripts.model import GPT, GPTConfig
from scripts.utils import append_log, checkpoint_path, get_autocast_device_type


def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def run_training(config, ddp_ctx, run_name, log_file):
    seed = config["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_cfg = config["training"]
    model_cfg = config["model"]
    optim_cfg = config["optimizer"]
    lr_cfg = config["lr_schedule"]
    eval_cfg = config["evaluation"]
    gen_cfg = config["generation"]
    ckpt_cfg = config["checkpointing"]
    runtime_cfg = config["runtime"]
    data_cfg = config["data"]
    log_cfg = config["logging"]

    total_batch_size = train_cfg["total_batch_size"]
    B = train_cfg["micro_batch_size"]
    T = train_cfg["sequence_length"]
    max_steps = train_cfg["max_steps"]

    assert total_batch_size % (B * T * ddp_ctx.ddp_world_size) == 0, (
        "total_batch_size must be divisible by B * T * ddp_world_size"
    )
    grad_accum_steps = total_batch_size // (B * T * ddp_ctx.ddp_world_size)
    if ddp_ctx.master_process:
        print(f"total_batch_size = {total_batch_size}")
        print(f"Using grad_accum_steps = {grad_accum_steps}")

    print(
        "I am GPU with ddp_rank = "
        f"{ddp_ctx.ddp_rank} and ddp_local_rank = {ddp_ctx.ddp_local_rank} "
        f"of {ddp_ctx.ddp_world_size}, master process: {ddp_ctx.master_process}"
    )

    train_loader = DataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_ctx.ddp_rank,
        num_processes=ddp_ctx.ddp_world_size,
        split="train",
        data_root=data_cfg.get("data_root"),
        master_process=ddp_ctx.master_process,
    )
    val_loader = DataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_ctx.ddp_rank,
        num_processes=ddp_ctx.ddp_world_size,
        split="val",
        data_root=data_cfg.get("data_root"),
        master_process=ddp_ctx.master_process,
    )

    torch.set_float32_matmul_precision(runtime_cfg["matmul_precision"])

    model = GPT(GPTConfig(**model_cfg))
    model.to(ddp_ctx.device)
    use_compile = runtime_cfg["use_compile"]
    if use_compile:
        model = torch.compile(model)
    if ddp_ctx.ddp:
        model = DDP(model, device_ids=[ddp_ctx.ddp_local_rank])
    raw_model = model.module if ddp_ctx.ddp else model

    max_lr = lr_cfg["max_lr"]
    min_lr = max_lr * lr_cfg["min_lr_ratio"]
    warmup_steps = lr_cfg["warmup_steps"]

    optimizer = raw_model.configure_optimizers(
        weight_decay=optim_cfg["weight_decay"],
        learning_rate=optim_cfg["learning_rate"],
        device=ddp_ctx.device,
    )
    enc = tiktoken.get_encoding("gpt2")
    autocast_device_type = get_autocast_device_type(ddp_ctx.device)

    append_log(log_file, f"run_name {run_name}")
    append_log(log_file, f"device {ddp_ctx.device}")
    append_log(log_file, f"grad_accum_steps {grad_accum_steps}")

    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        if step % eval_cfg["val_interval"] == 0 or last_step:
            val_loss_accum = evaluate_validation(
                model=model,
                val_loader=val_loader,
                device=ddp_ctx.device,
                autocast_device_type=autocast_device_type,
                val_loss_steps=eval_cfg["val_loss_steps"],
                ddp=ddp_ctx.ddp,
                dist_module=dist,
            )
            if ddp_ctx.master_process:
                print(f"Validation loss: {val_loss_accum.item():.4f}")
                append_log(log_file, f"{step} val {val_loss_accum.item():.4f}")
                if step > 0 and (step % ckpt_cfg["checkpoint_interval"] == 0 or last_step):
                    cpath = checkpoint_path(log_cfg["log_dir"], run_name, step)
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "config": raw_model.config,
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                    }
                    torch.save(checkpoint, cpath)

        if (step % eval_cfg["hellaswag_interval"] == 0 or last_step) and (not use_compile):
            num_correct_norm, num_total, acc_norm = evaluate_hellaswag(
                model=model,
                device=ddp_ctx.device,
                autocast_device_type=autocast_device_type,
                ddp_world_size=ddp_ctx.ddp_world_size,
                ddp_rank=ddp_ctx.ddp_rank,
                ddp=ddp_ctx.ddp,
                dist_module=dist,
            )
            if ddp_ctx.master_process:
                print(f"HellaSwag Accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
                append_log(log_file, f"{step} hella {acc_norm:.4f}")

        if (
            ((step > 0 and step % gen_cfg["sample_interval"] == 0) or last_step)
            and (not use_compile)
        ):
            outputs = generate_samples(
                model=model,
                enc=enc,
                device=ddp_ctx.device,
                ddp_rank=ddp_ctx.ddp_rank,
                num_return_sequences=gen_cfg["num_return_sequences"],
                max_length=gen_cfg["max_length"],
                prompt=gen_cfg["prompt"],
                top_k=gen_cfg["top_k"],
                sample_seed=gen_cfg["sample_seed"],
            )
            for line in outputs:
                print(line)

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(ddp_ctx.device), y.to(ddp_ctx.device)
            with torch.autocast(device_type=autocast_device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp_ctx.ddp:
                model.require_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()
        if ddp_ctx.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if "cuda" in ddp_ctx.device:
            torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_ctx.ddp_world_size
        tokens_per_sec = tokens_processed / (t1 - t0)
        if ddp_ctx.master_process:
            print(
                f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.6f} | "
                f"norm: {norm.item():.4f} | time: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            append_log(log_file, f"{step} train {loss_accum.item():.6f}")
