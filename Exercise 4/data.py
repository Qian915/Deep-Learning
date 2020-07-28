from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        # information in data.csv
        self.data = data
        # "val" or "train"
        self.mode = mode
        # two transforms for train / val dataset
        if self.mode == "train":
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                     tv.transforms.RandomHorizontalFlip(p=0.5),
                                                     tv.transforms.RandomRotation(degrees=(90, 270)),
                                                     tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(mean=train_mean, std=train_std)])
        if self.mode == "val":
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                     tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(mean=train_mean, std=train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # read image and convert to rgb
        img_path = Path(self.data.iloc[index, 0])
        image = imread(img_path)
        image = gray2rgb(image)
        image = self._transform(image)

        # read class label
        label = self.data.iloc[index, 1:]
        label = np.array([label]).astype('float')
        label = torch.from_numpy(label)

        return image, label
