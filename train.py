"""Train FOCUS.

Reads `config/config_train.yaml` beside this script (or `--config`), builds the data loader, the
conditional U-Net and the diffusion process, then hands control to
`guided_diffusion.train_util.TrainLoop`.

    python train.py
"""

import os, torch.multiprocessing as mp
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import argparse
import yaml
from pathlib import Path
import argparse
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    add_dict_to_argparser
)

import torch
import os
from mpi4py import MPI
comm =MPI.COMM_WORLD
rank = comm.Get_rank()
gpu_ids = [0]
if torch.cuda.is_available():
    torch.cuda.set_device(rank % torch.cuda.device_count())
from train_part import train_fun
def main():
    args = create_argparser().parse_args()
    if comm.size != 1:
        raise RuntimeError("This training entrypoint is single-process; do not launch unsynchronized multi-GPU jobs")
    train_fun(args,args.log_dir,args.n ,args.pre_model_dir)

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config/config_train.yaml"), help="Path to YAML configuration file")
    args, _ = parser.parse_known_args()

    # Load the configuration from the YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Add the configuration values to the argument parser
    add_dict_to_argparser(parser, config)

    return parser


if __name__ == "__main__":

    main()
