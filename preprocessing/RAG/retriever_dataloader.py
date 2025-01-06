import os
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

from PIL import Image  # PIL 라이브러리 사용
from torchvision import transforms

class RetrieverDataset(Dataset):
    def __init__(self, root_dir, data_type):

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet도 224x224 이미지를 입력으로 받음
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet의 표준 정규화 값
        ])

        self.root_dir = root_dir
        with open(os.path.join(root_dir, data_type + ".pkl"), 'rb') as f:
            loaded_data = pickle.load(f)

        self.layout_dataset = []
        self.mask_dataset = []
        self.path_dataset = []
        for file_path, data in loaded_data.items():
            layout_dataset = np.zeros((25, 33))
            for idx in range(len(data['translations'])):
                layout_dataset[idx] = np.array(
                    data['translations'][idx] +
                    data['sizes'][idx] +
                    data['angles'][idx] +
                    data['class_labels'][idx]
                )

            self.layout_dataset.append(layout_dataset)
            self.mask_dataset.append(data['room_mask'][:, :, 0])
            self.path_dataset.append(file_path)


        self.data_len = len(self.layout_dataset)

    def __getitem__(self, index):
        image_mask = self.mask_dataset[index]
        image_mask = image_mask.astype(np.uint8)
        image_mask = Image.fromarray(image_mask)  # NumPy 배열을 PIL 이미지로 변환
        image_mask = image_mask.convert("RGB")
        image_mask = self.preprocess(image_mask)

        file_path = self.path_dataset[index]
        print(image_mask.shape, file_path)

        return (file_path, image_mask)

    def __len__(self):
        return self.data_len