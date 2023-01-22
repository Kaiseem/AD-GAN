import os
import torch
import datetime
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.nuclei_dataset import *
from trainer import Trainer
from util import  data_prefetcher,check_manual_seed
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='adgan')
parser.add_argument('--seed', type=int, default=10)

#Network
parser.add_argument('--dimensions', type=int, default=2, help='use 2D or 3D data, 2 for 2D, 3 for 3D')
parser.add_argument('--num_c_dim', type=int, default=64, help='the number of dimensions for the channel.')

#Datasets
parser.add_argument('--crop_size', type=int, default=256, help='crop the images to size')

parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--no_synB', action='store_true', help='if specified, do not synthesis datasetsB')
parser.add_argument('--no_inst', action='store_true', help='if specified, do not use instance segmentation')

parser.add_argument('--ellipse_min_radius', type=int, default=20)
parser.add_argument('--ellipse_max_radius', type=int, default=30)
parser.add_argument('--ellipse_min_num', type=int, default=5)
parser.add_argument('--ellipse_max_num', type=int, default=15)

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

# Misc
parser.add_argument('--save_visual_freq', type=int, default=100)
parser.add_argument('--save_ckpt_freq', type=int, default=2000)
parser.add_argument('--evaluate_freq', type=int, default=1000)

opts = parser.parse_args()
torch.backends.cudnn.benchmark = True




if __name__ == '__main__':
    check_manual_seed(opts.seed)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + '_' + opts.name
    logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    visdir = os.path.join(logdir, "visuals")
    for d in [logdir, ckptdir, visdir]:
        os.makedirs(d, exist_ok=True)
    opts.visdir = visdir
    opts.ckptdir = ckptdir
    if opts.dimensions==2:
        train_loader = DataLoader(dataset=NucleiDataset(opts,'train'), batch_size=8, shuffle=True, drop_last=True, num_workers=4,persistent_workers=True, pin_memory=True)
        test_loader = DataLoader(dataset=NucleiDataset(opts,'test'), batch_size=1, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
    else:
        print('the training for 3D images shall be released soon')
        raise NotImplementedError

    trainer = Trainer(opts)
    trainer.cuda()

    cur_iter=0

    total_iter=opts.n_iters+opts.n_iters_decay

    prefetcher = data_prefetcher(train_loader, 'cuda', prefetch=True)

    loop=tqdm(range(total_iter),desc='Train')
    for _ in loop:
        train_data = prefetcher.next()
        trainer.gan_forward(train_data['A'], train_data['B'])
        trainer.gen_update()
        trainer.dis_update()
        text=trainer.verbose()
        loop.set_description(f'\rIter {cur_iter}/{total_iter}, {text}')

        cur_iter+=1
        trainer.update_learning_rate()

        # visualization
        if (cur_iter+1)%opts.save_visual_freq==0:
            trainer.gan_visual(cur_iter)

        # evaluation
        if (cur_iter+1)%opts.evaluate_freq==0 and cur_iter+1>opts.n_iters:
            trainer.evaluate(test_loader)

        # save checkpoints
        if (cur_iter+1)%opts.save_ckpt_freq==0:
            trainer.save((cur_iter+1))
