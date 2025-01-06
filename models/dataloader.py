import os
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

class Dataset_3Dfront(Dataset):
    def __init__(self, root_dir, data_type, k=5):
        self.root_dir = root_dir
        with open(os.path.join(root_dir, "train.pkl"), 'rb') as f:
            train_dataset = pickle.load(f)
        with open(os.path.join(root_dir, data_type + ".pkl"), 'rb') as f:
            source_dataset = pickle.load(f)
        with open(os.path.join(root_dir, f"retrieval_{data_type}.pkl"), 'rb') as f:
            retrieved_dataset = pickle.load(f)

        data_length = len(source_dataset)
        object_num = 25
        feature_num = 34
        image_size = 64

        self.source_layout_dataset = np.zeros((data_length, object_num, feature_num))
        self.source_mask_dataset = np.zeros((data_length, image_size, image_size, 1))
        self.retrieved_layout_dataset = np.zeros((data_length, k, object_num, feature_num))
        self.retrieved_mask_dataset = np.zeros((data_length, k, image_size, image_size, 1))

        for idx, source_file_path, source_data in enumerate(source_dataset.items()):
            _object_num = len(source_data['translations'])
            for jdx in range(_object_num):
                self.source_layout_dataset[idx, jdx] = np.array(
                    source_data['translations'][jdx] +
                    source_data['sizes'][jdx] +
                    source_data['angles'][jdx] +
                    source_data['class_labels'][jdx] +
                    [0]
                )
            if _object_num < 25:
                self.source_layout_dataset[idx, _object_num:, -1] = 1  # 올바른 범위 설정
            self.source_mask_dataset[idx, :, :] = source_data["room_mask"]

            for jdx, retrieved_file_path in enumerate(retrieved_dataset[source_file_path]):
                _object_num = len(train_dataset[retrieved_file_path]['translations'])
                for kdx in range(_object_num):
                    self.retrieved_layout_dataset[idx, jdx, kdx] = np.array(
                        train_dataset[retrieved_file_path]['translations'][kdx] +
                        train_dataset[retrieved_file_path]['sizes'][kdx] +
                        train_dataset[retrieved_file_path]['angles'][kdx] +
                        train_dataset[retrieved_file_path]['class_labels'][kdx] +
                        [0]
                    )
                if _object_num < 25:
                    self.retrieved_layout_dataset[idx, jdx, _object_num:, -1] = 1  # 올바른 범위 설정
                self.retrieved_mask_dataset[idx, jdx, :, :] = train_dataset[retrieved_file_path]["room_mask"]

        self.data_len = data_length

    def __getitem__(self, index):
        source_layout = self.source_layout_dataset[index]
        source_mask = self.source_mask_dataset[index]
        retrieved_layout = self.retrieved_layout_dataset[index]
        retrieved_mask = self.retrieved_mask_dataset[index]

        return (
            torch.tensor(source_layout, dtype=torch.bfloat16),
            torch.tensor(source_mask, dtype=torch.bfloat16),
            torch.tensor(retrieved_layout, dtype=torch.bfloat16),
            torch.tensor(retrieved_mask, dtype=torch.bfloat16),
        )

    def __len__(self):
        return self.data_len