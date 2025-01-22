import os
import glob
import random

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class UrbanDataset(Dataset):
    def __init__(self, split_dataset_paths, dataset_type):
        self.data_paths = split_dataset_paths[dataset_type]
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        satellite_image_file_path, bldg_polygon_image_file_path = self.data_paths[index]

        satellite_image = Image.open(satellite_image_file_path).convert("RGB")
        bldg_polygon_image = Image.open(bldg_polygon_image_file_path).convert("RGB")

        satellite_image_tensor = self.to_tensor(satellite_image)
        bldg_polygon_image_tensor = self.to_tensor(bldg_polygon_image)

        return np.array(satellite_image), np.array(bldg_polygon_image)

    def __len__(self):
        return len(self.data_paths)

def get_dataset_paths(dataset_folder):
    image_folder = os.path.join(dataset_folder, "image_dataset")
    pkl_folder = os.path.join(dataset_folder, "pkl_dataset")

    satellite_image_folder = os.path.join(image_folder, "satellite_image")
    bldg_bbox_image_folder = os.path.join(image_folder, "bldg_bbox_image")
    bldg_polygon_image_folder = os.path.join(image_folder, "bldg_polygon_image")
    blk_image_folder = os.path.join(image_folder, "blk_image")

    satellite_image_paths = glob.glob(os.path.join(satellite_image_folder, "*.png"))
    blk_image_paths = glob.glob(os.path.join(blk_image_folder, "*.png"))

    print("Get data folders: ", dataset_folder)

    dataset_paths_dict = {
        "satellite_image": satellite_image_paths,
        "blk_image": blk_image_paths,
    }

    return dataset_paths_dict

def split_dataset(dataset_paths_dict, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    satellite_image_file_paths = dataset_paths_dict["satellite_image"]
    blk_image_file_paths = dataset_paths_dict["blk_image"]

    total_files = len(satellite_image_file_paths)
    indices = list(range(total_files))
    random.shuffle(indices)

    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    split_dataset_paths = {"train": [], "val": [], "test": []}
    for i, file_idx in enumerate(indices):
        satellite_image_file_path = satellite_image_file_paths[file_idx]
        blk_image_file_path = blk_image_file_paths[file_idx]

        file_path_tuple = (satellite_image_file_path, blk_image_file_path)
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