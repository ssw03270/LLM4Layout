import os
import numpy as np

import torch
from torch.utils.data import Dataset

class Dataset_3Dfront(Dataset):
    def __init__(self, root_dir, data_type):
        self.root_dir = root_dir
        loaded_data = np.load(os.path.join(root_dir, data_type + ".npy"), allow_pickle=True)

        self.layout_dataset = []
        self.mask_dataset = []
        for data in loaded_data:
            layout_dataset = np.zeros((25, 33))
            for idx in range(len(data['translations'])):
                layout_dataset[idx] = np.array(
                    data['translations'][idx] +
                    data['sizes'][idx] +
                    data['angles'][idx] +
                    data['class_labels'][idx]
                )

            self.layout_dataset.append(layout_dataset)
            self.mask_dataset.append(np.expand_dims(data['room_mask'][:, :, 0], axis=-1))


        self.data_len = len(self.layout_dataset)

    def __getitem__(self, index):
        return (
            torch.tensor(self.layout_dataset[index], dtype=torch.float32),
            torch.tensor(self.mask_dataset[index], dtype=torch.float32),
        )

    def __len__(self):
        return self.data_len