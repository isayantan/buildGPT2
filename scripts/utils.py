import importlib.util
import os
from pathlib import Path


def load_config(config_path):
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"Config module {path} must define CONFIG")
    config = module.CONFIG
    if not isinstance(config, dict):
        raise TypeError("CONFIG must be a dict")
    return config, path


def get_autocast_device_type(device):
    return device.split(":")[0]


def build_run_name(config_stem, config):
    model_cfg = config["model"]
    train_cfg = config["training"]
    return (
        f"{config_stem}"
        f"__l{model_cfg['n_layer']}_h{model_cfg['n_head']}_e{model_cfg['n_embd']}"
        f"__tbs{train_cfg['total_batch_size']}_mb{train_cfg['micro_batch_size']}"
        f"_seq{train_cfg['sequence_length']}_steps{train_cfg['max_steps']}"
    )


def prepare_log_file(log_dir, run_name):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{run_name}.log")
    with open(log_file, "w", encoding="utf-8"):
        pass
    return log_file


def append_log(log_file, line):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def checkpoint_path(log_dir, run_name, step):
    return os.path.join(log_dir, f"{run_name}__model_{step:05d}.pt")
