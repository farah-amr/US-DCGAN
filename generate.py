import math
import os
import timeit
import random
import torch
from tqdm import tqdm
import torchvision.utils as vutils
import argparse

from GAN.GANmodels import Generator

def generate_fake(params, G_path, device, size, save_path, batch_size):
    nz, ngf, nc, ngpu = params['nz'], params['ngf'], params['nc'], params['gpu']
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netG.load_state_dict(torch.load(G_path))

    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except OSError as e:
            print(f"Error creating directory: {e}")
            return

    def giveName(iter):  # 7 digit name.
        ans = str(iter)
        return ans.zfill(7)

    def sampleFake(netG, nz, sampleSize, batchSize, saveFolder):
        print('Sampling fake images...')
        noise = torch.FloatTensor(batchSize, nz, 1, 1).to(device)
        iter = 0
        for i in tqdm(range(0, 1 + sampleSize // batchSize)):
            noise.data.normal_(0, 1)
            fake = netG(noise)
            for j in range(len(fake.data)):
                if iter < sampleSize:
                    vutils.save_image(fake.data[j].mul(0.5).add(0.5), os.path.join(saveFolder, giveName(iter) + ".jpg"))
                iter += 1
                if iter >= sampleSize:
                    break

    sampleFake(netG, nz, size, batch_size, save_path)
    print("Finished generating fake data")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path for GAN generator')
    parser.add_argument('--size', type=int, required=True, help='Number of synthetic images to generate')
    parser.add_argument('--save_path', required=True, help='Folder to save generated images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generating images')
    parser.add_argument('--nz', type=int, default=100, help='Dimension of the latent z vector')
    parser.add_argument('--nc', type=int, default=1, help='Number of channels')
    parser.add_argument('--ngf', type=int, default=64, help='Relates to the depth of feature maps carried through the generator')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs to use')

    opt = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = {
        'nz': opt.nz,
        'ngf': opt.ngf,
        'nc': opt.nc,
        'gpu': opt.ngpu
    }
    generate_fake(params, opt.model_path, device, opt.size, opt.save_path, opt.batch_size)
