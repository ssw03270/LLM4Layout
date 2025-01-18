import argparse
import random

import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="F:\\Urban\\Preprocessed_Dataset")
    parser.add_argument("--save_dir", type=str, default="F:\\Urban\\Trained_Model")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-11B-Vision")
    parser.add_argument("--seed", type=int, default=327)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)

    args = vars(parser.parse_args())
    print("Args: {}".format(args))

    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Random Seed: ", seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
