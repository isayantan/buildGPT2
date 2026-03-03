import torch
from torch.nn import functional as F

from hellaswag import iterate_examples, render_example


def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def evaluate_validation(
    model,
    val_loader,
    device,
    autocast_device_type,
    val_loss_steps,
    ddp=False,
    dist_module=None,
):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=autocast_device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    if ddp:
        dist_module.all_reduce(val_loss_accum, op=dist_module.ReduceOp.AVG)
    return val_loss_accum


def evaluate_hellaswag(
    model,
    device,
    autocast_device_type,
    ddp_world_size,
    ddp_rank,
    ddp=False,
    dist_module=None,
):
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        if i % ddp_world_size != ddp_rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=autocast_device_type, dtype=torch.bfloat16):
                logits, _ = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        # Intentional parity with the original script's reduction sequence.
        dist_module.all_reduce(num_total, op=dist_module.ReduceOp.SUM)
        dist_module.all_reduce(num_total, op=dist_module.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()

    acc_norm = num_correct_norm / num_total
    return num_correct_norm, num_total, acc_norm


def generate_samples(
    model,
    enc,
    device,
    ddp_rank,
    num_return_sequences,
    max_length,
    prompt,
    top_k,
    sample_seed,
):
    model.eval()
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(sample_seed + ddp_rank)
    while xgen.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

    outputs = []
    for i in range(num_return_sequences):
        out_tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(out_tokens)
        outputs.append(f"rank {ddp_rank} sample {i}: {decoded}")
    return outputs
