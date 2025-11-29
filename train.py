import os, torch.multiprocessing as mp
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import argparse
import yaml
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
torch.cuda.set_device(gpu_ids[rank])
from train_part import train_fun
def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    train_fun(args,args.log_dir,args.n ,args.pre_model_dir)

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--./code/config/config_train.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load the configuration from the YAML file
    with open('./code/config/config_train.yaml', "r") as file:
        config = yaml.safe_load(file)

    # Add the configuration values to the argument parser
    add_dict_to_argparser(parser, config)

    return parser


if __name__ == "__main__":

    main()
