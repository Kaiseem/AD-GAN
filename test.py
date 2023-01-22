import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from trainer import Trainer
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='adgan')
parser.add_argument('--resume', type=str, default='latest.pth')
parser.add_argument('--seed', type=int, default=10)

#Network
parser.add_argument('--dimensions', type=int, default=2, help='use 2D or 3D data, 2 for 2D, 3 for 3D')
parser.add_argument('--num_c_dim', type=int, default=64, help='the number of dimensions for the channel.')

#Datasets
parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                    help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')

parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--no_synB', action='store_true', help='if specified, do not synthesis datasetsB')
parser.add_argument('--no_inst', action='store_true', help='if specified, do not use instance segmentation')

parser.add_argument('--ellipse_min_radius', type=int, default=20)
parser.add_argument('--ellipse_max_radius', type=int, default=30)
parser.add_argument('--ellipse_min_num', type=int, default=5)
parser.add_argument('--ellipse_max_num', type=int, default=15)

parser.add_argument('--preprocess', type=str, default='crop')
parser.add_argument('--dataroot', default='datasets/YourDATA')

# GAN
parser.add_argument('--lambda_rec', type=float, default=20,help='weight for image-level reconstruction')
parser.add_argument('--lambda_cyc', type=float, default=20,help='weight for cycle consistency loss')
parser.add_argument('--lambda_ctr', type=float, default=1,help='weight for feature-level reconstruction')
parser.add_argument('--no_adt', action='store_true', help='if specified, do not Aligned Disentangling Training')
parser.add_argument('--gan_mode', type=str, default='lsgan',
                    help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of image buffer that stores previously generated images')

# Optimization
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--iter_count', type=int, default=1, help='the starting iteration count')
parser.add_argument('--n_iters', type=int, default=5000, help='number of iterations with the initial learning rate')
parser.add_argument('--n_iters_decay', type=int, default=5000, help='number of iterations to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='linear',
                    help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50,
                    help='multiply by a gamma every lr_decay_iters iterations')


opts = parser.parse_args()

def check_manual_seed(seed):
    """ If manual seed is not specified, choose a
    random one and communicate it to the user.
    Args:
        seed: seed to check
    """
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print("Using manual seed: {seed}".format(seed=seed))
    return

def dice_loss_chill(output, gt):
    num = (output*gt).sum(dim=[2, 3])
    denom = output.sum(dim=[2, 3]) + gt.sum(dim=[2, 3]) + 0.001
    return num, denom

from data.nuclei_dataset import NucleiDataset
if __name__ == '__main__':
    check_manual_seed(opts.seed)
    test_loader = DataLoader(dataset=NucleiDataset(opts,'test'), batch_size=1, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
    ckpt_path=opts.resume
    if not os.path.isfile(ckpt_path):
        print('No such file in '+ckpt_path+', pls check again.')
        exit()
    trainer = Trainer(opts)
    trainer.cuda()
    trainer.load(ckpt_path)
    trainer.evaluate(test_loader)

