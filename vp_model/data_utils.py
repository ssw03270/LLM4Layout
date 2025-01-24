import os
import glob
import random

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class LayoutDataset(Dataset):
    def __init__(self, split_dataset_paths, dataset_type):
        self.data_paths = split_dataset_paths[dataset_type]

    def __getitem__(self, index):
        real_image_path, target_image_path = self.data_paths[index]

        real_image = Image.open(real_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")
        print(real_image_path, target_image_path)
        return np.array(real_image), np.array(target_image)

    def __len__(self):
        return len(self.data_paths)

def get_dataset_paths(dataset_folder):
    real_image_folder = os.path.join(dataset_folder, "real_images")
    target_image_folder = os.path.join(dataset_folder, "target_images")

    real_image_paths = glob.glob(os.path.join(real_image_folder, "*.png"))
    target_image_paths = glob.glob(os.path.join(target_image_folder, "*.png"))

    print("Get data folders: ", dataset_folder)

    dataset_paths_dict = {
        "real_image_paths": real_image_paths,
        "target_image_paths": target_image_paths,
    }

    return dataset_paths_dict

def split_dataset(dataset_paths_dict, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    real_image_paths = dataset_paths_dict["real_image_paths"]
    target_image_paths = dataset_paths_dict["target_image_paths"]

    total_files = len(real_image_paths)
    indices = list(range(total_files))
    random.shuffle(indices)

    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    split_dataset_paths = {"train": [], "val": [], "test": []}
    for i, file_idx in enumerate(indices):
        real_image_path = real_image_paths[file_idx]
        target_image_path = target_image_paths[file_idx]

        file_path_tuple = (real_image_path, target_image_path)
        if i < train_end:
            split_dataset_paths["train"].append(file_path_tuple)
        elif i < val_end:
            split_dataset_paths["val"].append(file_path_tuple)
        else:
            split_dataset_paths["test"].append(file_path_tuple)

    print("total files: ", total_files, ", train_length: ", len(split_dataset_paths["train"]),
          ", val_length: ", len(split_dataset_paths["val"]), ", test_length: ", len(split_dataset_paths["test"]),
          ", dict_keys: ", split_dataset_paths.keys())

    return split_dataset_paths

def get_dataloader(dataset, args, shuffle=True):
    return DataLoader(dataset, batch_size=args["batch_size"], shuffle=shuffle)