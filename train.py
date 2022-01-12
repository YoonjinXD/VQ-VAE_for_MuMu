import os
import random
import argparse
import pandas as pd
import numpy as np
import logging
import yaml

import torch
from torch.cuda import random
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import ImageGenreDataset 
from models import VQ_VAE

def set_logger(result_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(u'%(asctime)s %(message)s')
    file_handler = logging.FileHandler(os.path.join(result_path, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def load_data(batch_size, logger):
    data = pd.read_csv('./MuMu_dataset/available_album_ids.csv')
    total_num = len(data)

    train_dataset = ImageGenreDataset(data=data[0:int(0.95*total_num)])
    test_dataset = ImageGenreDataset(data=data[int(0.95*total_num):])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    logger.info('Total: {} ({}/{})'.format(total_num, len(train_dataset), len(test_dataset)))
    return train_loader, test_loader

def train(num_steps, batch_size, learning_rate):
    # Set Paths
    result_path = './results'
    sample_path = os.path.join(result_path, 'recon_images')
    model_path = os.path.join(result_path, 'models')
    if not os.path.exists(result_path):
        os.makedirs(sample_path)
        os.makedirs(model_path)
    
    # Set Logger
    logger = set_logger(result_path)

    # Load Data
    train_loader, test_loader = load_data(batch_size, logger)

    # Set Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('./config.yaml') as file:
        config = yaml.safe_load(file)
    model = VQ_VAE(config['num_hiddens'], 
                   config['num_residual_layers'], 
                   config['num_residual_hiddens'],
                   config['num_embeddings'], 
                   config['embedding_dim'], 
                   config['commitment_cost'], 
                   config['decay'])
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_res_recon_error = []
    train_res_perplexity = []

    # Train
    for i in range(num_steps):
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
                    logger.info('[STEP %d] recon_error: %.3f, perplexity: %.3f' % (i+1, np.mean(train_res_recon_error[-1000:]), np.mean(train_res_perplexity[-1000:])))
                    break

            torch.save(model, os.path.join(model_path, 'step-{}.pt'.format(i+1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, help='num of training steps', default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=int, default=1e-4)
    train(**vars(parser.parse_args()))