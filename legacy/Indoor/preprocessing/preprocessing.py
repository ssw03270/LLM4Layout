import os
import numpy as np
import pickle
import json
import random
import csv
from collections import defaultdict
from PIL import Image

from tqdm import tqdm

from utils import *

def process_scene(folder_path, room, split_path):
    """
    Loads split data, statistics, and all relevant scene information for each subfolder.
    Constructs polygons for the room and the objects, then visualizes them.
    """
    valid_split_data = load_room_split_data(split_path)
    stats = load_stats(os.path.join(folder_path, room, "dataset_stats.txt"))

    target_path = os.path.join(folder_path, room)
    subfolders = [
        f for f in os.listdir(target_path)
        if os.path.isdir(os.path.join(target_path, f)) and "test" not in f and "train" not in f
    ]
    train_layout_data = {}
    train_polygon_data = {}
    test_layout_data = {}
    test_polygon_data = {}
    val_layout_data = {}
    val_polygon_data = {}
    for subfolder in tqdm(subfolders):
        file_path = os.path.join(target_path, subfolder)
        file_data_type = parse_data_type(file_path, valid_split_data)

        # Load and parse various data files
        boxes_data = parse_boxes_npz(file_path)
        descriptions_data = parse_descriptions_pkl(file_path)
        models_info_data = parse_models_info_pkl(file_path)
        relations_data = parse_relations_npy(file_path)
        room_mask_data = parse_room_mask_png(file_path)

        # Build room polygon
        room_polygon_raw = construct_room_polygon(
            boxes_data['floor_plan_vertices'],
            boxes_data['floor_plan_faces']
        )
        room_polygon = center_polygon(room_polygon_raw)

        # Build object polygons
        object_polygons, object_infos = build_object_polygons(
            boxes_data['class_labels'],
            boxes_data['sizes'],
            boxes_data['angles'],
            boxes_data['translations'],
            stats
        )

        # Visualization
        # visualize_polygons(room_polygon, object_polygons)
        room_polygon = list(room_polygon.exterior.coords)
        object_polygons = [list(polygon.exterior.coords) for polygon in object_polygons]

        if file_data_type == "train":
            train_layout_data[subfolder] = {
                    "class_labels": boxes_data['class_labels'],
                    "sizes": boxes_data['sizes'],
                    "angles": boxes_data['angles'],
                    "translations": boxes_data['translations'],
                    "room_mask": room_mask_data
                }
            train_polygon_data[subfolder] = {
                    "room_polygon": room_polygon,
                    "object_polygons": object_polygons
                }
        elif file_data_type == "val":
            val_layout_data[subfolder] = {
                    "class_labels": boxes_data['class_labels'],
                    "sizes": boxes_data['sizes'],
                    "angles": boxes_data['angles'],
                    "translations": boxes_data['translations'],
                    "room_mask": room_mask_data
                }
            val_polygon_data[subfolder] = {
                    "room_polygon": room_polygon,
                    "object_polygons": object_polygons
                }

        elif file_data_type == "test":
            test_layout_data[subfolder] = {
                    "class_labels": boxes_data['class_labels'],
                    "sizes": boxes_data['sizes'],
                    "angles": boxes_data['angles'],
                    "translations": boxes_data['translations'],
                    "room_mask": room_mask_data
                }
            test_polygon_data[subfolder] = {
                    "room_polygon": room_polygon,
                    "object_polygons": object_polygons
                }

    dataset = {
        "train": {
            "layout": train_layout_data,
            "polygons": train_polygon_data,
        },
        "val": {
            "layout": val_layout_data,
            "polygons": val_polygon_data,
        },
        "test": {
            "layout": test_layout_data,
            "polygons": test_polygon_data,
        }
    }
    return dataset

def save_dict_list_to_npz(dataset, file_path):
    """
    Saves a list of dictionaries to a .npz file.
    """
    try:
        with open(os.path.join(file_path, "dataset.pkl"), 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Data successfully saved to {file_path}")

    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    """
    Sets up the folder paths, rooms, seeds, and initiates processing for each room type.
    """
    random.seed(42)
    np.random.seed(42)

    folder_path = 'E:/Resources/IndoorSceneSynthesis/InstructScene'
    room_lists = ["threed_front_diningroom", "threed_front_livingroom", "threed_front_bedroom"]
    split_paths = [
        "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes/diningroom_threed_front_splits.csv",
        "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes/livingroom_threed_front_splits.csv",
        "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes/bedroom_threed_front_splits.csv",
    ]

    for room, split_path in zip(room_lists, split_paths):
        dataset = process_scene(folder_path, room, split_path)

        save_path = os.path.join("../datasets", room)
        os.makedirs(save_path, exist_ok=True)
        save_dict_list_to_npz(dataset, save_path)

        train = len(dataset["train"]["layout"])
        val = len(dataset["val"]["layout"])
        test = len(dataset["test"]["layout"])
        print(f"Saved train: {train}, val: {val}, test: {test}")

if __name__ == "__main__":
    main()