import argparse
import sys
from pathlib import Path

import tiktoken
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation import generate_samples
from scripts.model import GPT, GPTConfig
from scripts.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint file (for example: log/<run_name>__model_10000.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/gpt2_fineweb_default.py",
        help="Config path used to build the model shape and default generation params.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text. Defaults to config['generation']['prompt'].",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Total token length after generation. Defaults to config value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling size. Defaults to config value.",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="Number of sampled sequences to return.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Sampling seed. Defaults to config value.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (for example: cpu, cuda, cuda:0). Auto-detects if not set.",
    )
    return parser.parse_args()


def pick_device(device_override):
    if device_override:
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    args = parse_args()
    config, _ = load_config(args.config)
    model_cfg = config["model"]
    gen_cfg = config["generation"]

    prompt = args.prompt if args.prompt is not None else gen_cfg["prompt"]
    max_length = args.max_length if args.max_length is not None else gen_cfg["max_length"]
    top_k = args.top_k if args.top_k is not None else gen_cfg["top_k"]
    sample_seed = (
        args.sample_seed if args.sample_seed is not None else gen_cfg["sample_seed"]
    )
    device = pick_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**model_cfg))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    outputs = generate_samples(
        model=model,
        enc=enc,
        device=device,
        ddp_rank=0,
        num_return_sequences=args.num_return_sequences,
        max_length=max_length,
        prompt=prompt,
        top_k=top_k,
        sample_seed=sample_seed,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(f"prompt: {prompt!r}")
    print(f"max_length: {max_length}, top_k: {top_k}, sample_seed: {sample_seed}")
    for line in outputs:
        print(line)


if __name__ == "__main__":
    main()
