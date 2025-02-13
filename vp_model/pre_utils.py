import argparse
import random

import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_dir", type=str, default="C:\\Users\\ttd85\\Documents\\Github\LLM4Layout\\vp_model\\indoor_preprocessing\\outputs\\indoor_layouts")
    # parser.add_argument("--dataset_dir", type=str, default="/local_datasets/llm4layout/outputs/indoor_layouts/")
    parser.add_argument("--dataset_dir", type=str, default="./indoor_preprocessing/outputs/indoor_layouts/")
    parser.add_argument("--save_dir", type=str, default="./vp_model_saves")
    # parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit")
    parser.add_argument("--prompt_path", type=str, default="prompt1.txt")
    parser.add_argument("--seed", type=int, default=327)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--model_path", type=str, default="model_path")

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
