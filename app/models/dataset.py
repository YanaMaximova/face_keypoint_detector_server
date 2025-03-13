import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

class MyCustomDataset(Dataset):
    def __init__(
        self,
        mode,
        data_dir,
        fraction: float = 0.85,
        transform=None,
        train_gt=None
    ):
        self._items = []
        self._labels = []
        self._transform = transform

        image_files = os.listdir(data_dir)
        split_idx = int(fraction * len(image_files))

        if train_gt is None:
            train_gt = self._read_csv_to_dict('./tests/00_test_img_input/train/gt.csv')

        if mode == "train":
            selected_files = image_files[-split_idx:]
        elif mode == "val":
            selected_files = image_files[:-split_idx]

        for img_file in selected_files:
            img_path = os.path.join(data_dir, img_file)
            points = train_gt[img_file]
            if min(points) >= 0:
                self._items.append(img_path)
                self._labels.append(points)


    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, labels = self._items[index], self._labels[index]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        labels = labels.reshape(-1, 2)

        if self._transform:
            transformed = self._transform(image=image, keypoints=labels)
            if len(transformed['keypoints']) == 14:
                image = transformed['image']
                labels = transformed['keypoints']

        image = np.array(image).astype(np.float32)
        image, labels = self._norm_image(image, labels)

        return image, labels

    @staticmethod
    def _read_csv_to_dict(csv_file):
        df = pd.read_csv(csv_file)
        result = {
            row['filename']: np.array(row[1:].values, dtype=np.float32) for _, row in df.iterrows()
        }
        return result

    @staticmethod
    def _norm_image(image, labels):
        res_image = A.Compose(
                [
                    A.Resize(284, 284),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
              )(image=image)['image']
        labels = torch.tensor(labels, dtype=torch.float32).reshape(-1)
        labels[0::2] *= 284 / image.shape[1]
        labels[1::2] *= 284 / image.shape[0]
        return res_image, labels