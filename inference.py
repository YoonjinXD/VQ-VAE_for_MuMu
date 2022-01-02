import os
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import PIL.Image as Image

from models import VQ_VAE

def inference(infer_type, model_dir, img_dir, latent_dir, save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    model = torch.load(model_dir)
    model.to(device)
    
    if infer_type == 'encoding':
        img = trans(Image.open(img_dir).convert("RGB"))
        img = torch.unsqueeze(img, dim=0)
        loss, latent, perplexity = model.get_latent_z(img.to(device))
        torch.save(latent, latent_dir)
        print('VQ Loss: {}, VQ Perplexity: {}'.format(loss, perplexity))

    elif infer_type == 'decoding':
        latent = torch.load(latent_dir)
        recon = model.get_recon_from_z(latent.to(device))
        save_image(recon, save_dir)
    
    else:
        print('Check inference type')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_type', '-t', type=str, help='encoding OR decoding')
    parser.add_argument('--model_dir', '-m', type=str, default='./128_16_128_128-step8000.pt')
    parser.add_argument('--img_dir', '-i', type=str)
    parser.add_argument('--latent_dir', '-l', type=str)
    parser.add_argument('--save_dir', '-s', type=str)

    inference(**vars(parser.parse_args()))