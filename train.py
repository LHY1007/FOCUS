"""Train FOCUS from a YAML configuration, optionally with torchrun.

One worker uses one visible GPU. ``torchrun --nproc_per_node=N train.py``
shards patch indices between ranks and synchronizes trainable gradients.
``batch_size`` is per worker; the effective global batch is N times larger.
"""

import argparse
import os
from pathlib import Path

os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import yaml

from guided_diffusion.script_util import add_dict_to_argparser


def create_argparser():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(Path(__file__).parent / "config/config_train.yaml"))
    known, _ = pre.parse_known_args()
    with open(known.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=known.config, help="YAML training configuration")
    add_dict_to_argparser(parser, config)
    return parser


def main():
    args = create_argparser().parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1 and int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Launch multi-process training with torchrun, not mpirun")
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError("Each torchrun worker needs a distinct visible GPU")
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)
    try:
        # Select the process device before importing the dataset dependencies.
        from train_part import train_fun
        train_fun(args, args.log_dir, args.n, args.pre_model_dir)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
