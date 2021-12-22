from numpy.lib.type_check import _imag_dispatcher
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image as Image

import numpy as np
import torch.nn as nn

class ImageGenreDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.trans = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    def __getitem__(self, index):
        album = self.data.iloc[index]
        img_path = album['img_path']
        img = self.trans(Image.open(img_path).convert("RGB"))
        return img

    def __len__(self):
        return len(self.data)