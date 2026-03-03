import argparse
import os
import sys
from pathlib import Path

# Single GPU: python scripts/main.py --config config/gpt2_fineweb_default.py
# Multi GPU (DDP): NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=0,4,5,6 torchrun --standalone --nproc_per_node=4 scripts/main.py --config config/gpt2_fineweb_default.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import build_run_name, load_config, prepare_log_file


def parse_args():
    parser = argparse.ArgumentParser(description="Modular GPT-2 training entrypoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/gpt2_fineweb_default.py",
        help="Path to a python config file that defines CONFIG dict.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from scripts.ddp import cleanup_distributed, setup_distributed
    from scripts.training import run_training

    config, config_path = load_config(args.config)
    config_stem = config_path.stem
    run_name = build_run_name(config_stem, config)
    log_dir = config["logging"]["log_dir"]
    log_file = prepare_log_file(log_dir, run_name)

    ddp_ctx = setup_distributed(
        default_cuda_device_index=config["runtime"]["default_cuda_device_index"]
    )
    try:
        if ddp_ctx.master_process:
            print(f"Using config: {config_path}")
            print(f"Writing logs to: {os.path.abspath(log_file)}")
        run_training(config=config, ddp_ctx=ddp_ctx, run_name=run_name, log_file=log_file)
    finally:
        cleanup_distributed(ddp_ctx)


if __name__ == "__main__":
    main()
