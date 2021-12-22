import os
import torch
from torch.cuda import random
import torchvision.transforms as transforms
from torchvision.utils import save_image

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random

from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import ImageGenreDataset 
from models import VQ_VAE

class VQ_VAE_Utilizer():
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path).to(self.device)
        self.model.eval()

    def get_latent_z(self, input):
        """
        input: (B, C, W, H) size of img
        """
        input = input.to(self.device)
        encoder_out = self.model._pre_vq_conv(self.model._encoder(input))
        _, quantized_z, _, _ = self.model._vq_vae(encoder_out)
        return encoder_out, quantized_z

    def get_recon_from_z(self, input):
        """
        input: (B, embedding_dim, resized_W, resized_H) size of latent z
        (e.g.) 128x128 img with 16 of embedding_dim -> should have (B, 16, 32, 32) size of latent z
        """
        input = input.to(self.device)
        reconstruction = self.model._decoder(input)
        return reconstruction

    def get_recon_from_img(self, input, img_size=128):
        """
        input: (B, C, W, H) size of img
        """
        input = input.to(self.device)
        _, quantized_z = self.get_latent_z(input)
        reconstructed = self.get_recon_from_z(quantized_z)
        x_concat = torch.cat([input.view(-1, 3, img_size, img_size), reconstructed.view(-1, 3, img_size, img_size)], dim=3)
        save_image(x_concat, './reconst.png')

# Set Model Utilizer
utilizer = VQ_VAE_Utilizer(model_path='128_16_128_128-step8000.pt')

# Load Data
batch_size = 32
data = pd.read_csv('./MuMu_dataset/available_album_ids.csv')
dataset = ImageGenreDataset(data=data)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Use Example
batch = next(iter(dataloader))

## Separated process
_, z = utilizer.get_latent_z(batch)
recon = utilizer.get_recon_from_z(z)
print(z.size(), recon.size())

## Whole process & Save origin/recon images
utilizer.get_recon_from_img(batch)