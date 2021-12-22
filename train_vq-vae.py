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

import logging

# Result Paths
model_name = "vq_vae"
result_path = os.path.join('./results', model_name)
sample_path = os.path.join(result_path, 'recon_images')
model_path = os.path.join(result_path, 'models')
if not os.path.exists(result_path):
    os.makedirs(sample_path)
    os.makedirs(model_path)

# Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(u'%(asctime)s %(message)s')
file_handler = logging.FileHandler(os.path.join(result_path, 'train.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load Data
batch_size = 128
data = pd.read_csv('./MuMu_dataset/available_album_ids.csv')
total_num = len(data)

train_dataset = ImageGenreDataset(data=data[0:int(0.95*total_num)])
test_dataset = ImageGenreDataset(data=data[int(0.95*total_num):])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
logger.info('Total: {} ({}/{})'.format(total_num, len(train_dataset), len(test_dataset)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set params
num_training_updates = 8000 # paper: 250000

num_hiddens = 128 # [128, 256]
num_residual_hiddens = 128 # [128, 256]
num_residual_layers = 2 # [2]

embedding_dim = 16 # [16, 32, 64]
num_embeddings = 128 # [128, 256, 512] paper: 512 for ImageNet data

commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-4 # practical

# Set Model
model = VQ_VAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_res_recon_error = []
train_res_perplexity = []

model.train()
train_res_recon_error = []
train_res_perplexity = []

# Train Model
print("Start Training")
for i in range(num_training_updates):
    data = next(iter(train_loader))
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data)
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()
    
    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    if (i+1) % 1000 == 0:    
        with torch.no_grad():
            for img in test_loader:
                # batch split (to check only 32 samples from the batch)
                sample_img_num = 32
                rand_split = random.randint(0, (batch_size // sample_img_num) - 1)
                split_batch = torch.split(img, sample_img_num)
                img = split_batch[rand_split]

                # Save reconstructed images
                img = img.to(device)
                vq_output_img = model._pre_vq_conv(model._encoder(img))
                _, quantized, _, _ = model._vq_vae(vq_output_img)
                reconstruction = model._decoder(quantized)
                x_concat = torch.cat([img.view(-1, 3, 128, 128), reconstruction.view(-1, 3, 128, 128)], dim=3)
                save_image(x_concat, os.path.join(sample_path, 'reconst-{}.png'.format(i+1)))

                # Logging
                logger.info('[STEP %d] recon_error: %.3f, perplexity: %.3f' % (i+1, np.mean(train_res_recon_error[-100:]), np.mean(train_res_perplexity[-100:])))
                break

        torch.save(model, os.path.join(model_path, 'step-{}.pt'.format(i+1)))