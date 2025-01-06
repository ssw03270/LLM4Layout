import os
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

class Dataset_3Dfront(Dataset):
    def __init__(self, root_dir, data_type):
        self.root_dir = root_dir
        with open(os.path.join(root_dir, data_type + ".pkl"), 'rb') as f:
            loaded_data = pickle.load(f)

        self.layout_dataset = []
        self.mask_dataset = []
        for file_path, data in loaded_data.items():
            layout_dataset = np.zeros((25, 34))
            for idx in range(len(data['translations'])):
                layout_dataset[idx] = np.array(
                    data['translations'][idx] +
                    data['sizes'][idx] +
                    data['angles'][idx] +
                    data['class_labels'][idx] +
                    [0]
                )

            if len(data['translations']) < 25:
                layout_dataset[len(data['translations']):, -1] = 1  # 올바른 범위 설정

            self.layout_dataset.append(layout_dataset)
            self.mask_dataset.append(np.expand_dims(data['room_mask'][:, :, 0], axis=-1))


        self.data_len = len(self.layout_dataset)

    def __getitem__(self, index):
        layout = self.layout_dataset[index]
        shuffled_layout = layout[np.random.permutation(layout.shape[0])]

        return (
            torch.tensor(shuffled_layout, dtype=torch.bfloat16),
            torch.tensor(self.mask_dataset[index], dtype=torch.bfloat16),
        )

    def __len__(self):
        return self.data_len