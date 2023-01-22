from models import GeneratorDecoder,GeneratorEncoder,DomainDiscriminator
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
import time
import os
import copy
from util import *

from sklearn import metrics
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if not isinstance(predictions,list):
            prediction=predictions
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
        else:
            loss=0
            for prediction in predictions:
                if self.gan_mode in ['lsgan', 'vanilla']:
                    target_tensor = self.get_target_tensor(prediction, target_is_real)
                    loss += self.loss(prediction, target_tensor)
                elif self.gan_mode == 'wgangp':
                    if target_is_real:
                        loss += -prediction.mean()
                    else:
                        loss += prediction.mean()
        return loss

class Trainer(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts=opts
        image_nc=1
        self.device='cuda'

        self.g_enc = GeneratorEncoder(image_nc=image_nc, ngf=opts.num_c_dim, dimensions=opts.dimensions)
        self.g_dec = GeneratorDecoder(image_nc=image_nc, ngf=opts.num_c_dim, dimensions=opts.dimensions)
        self.dis = DomainDiscriminator(image_nc=image_nc, ndf=opts.num_c_dim, dimensions=opts.dimensions)

        init_weights(self.g_enc, 'kaiming', init_gain=0.02)
        init_weights(self.g_dec, 'kaiming', init_gain=0.02)
        init_weights(self.dis, 'normal', init_gain=0.02)

        self.gen_opt=torch.optim.Adam(itertools.chain(self.g_enc.parameters(), self.g_dec.parameters()), lr=opts.lr, betas=(opts.beta1, 0.99),weight_decay=1e-4)
        self.dis_opt=torch.optim.Adam(itertools.chain(self.dis.parameters()), lr=opts.lr, betas=(opts.beta1, 0.99),weight_decay=1e-4)

        self.optimizers=[self.gen_opt,self.dis_opt]
        self.schedulers = [get_scheduler(optimizer, opts) for optimizer in self.optimizers]

        self.lambda_rec=opts.lambda_rec
        self.lambda_cyc=opts.lambda_cyc
        self.lambda_ctr=opts.lambda_ctr
        self.use_adt=not opts.no_adt
        if self.use_adt:
            print('Aligned Disentangling Training activated')

        self.recon_criterion=nn.L1Loss()
        self.criterionGAN = GANLoss('lsgan').to(self.device)

        self.x_a_recon_pool = ImagePool(50)
        self.x_b_recon_pool = ImagePool(50)
        self.x_ab_pool = ImagePool(50)
        self.x_ba_pool = ImagePool(50)

    def forward(self,x):
        self.l_a = torch.zeros(x.size(0)).long().to(self.device)
        self.l_b = torch.ones(x.size(0)).long().to(self.device)
        x = self.g_dec(self.g_enc(x,self.l_a), self.l_b, return_logits_only=True)
        return x

    def gan_forward(self,x_a, x_b):
        self.x_a = x_a
        self.x_b = x_b

        self.l_a = torch.zeros(self.x_a.size(0)).long().to(self.device)
        self.l_b = torch.ones(self.x_b.size(0)).long().to(self.device)

        self.c_a = self.g_enc(self.x_a,self.l_a)
        self.c_b = self.g_enc(self.x_b,self.l_b)

        self.x_a_recon = self.g_dec(self.c_a, self.l_a)
        self.x_b_recon = self.g_dec(self.c_b, self.l_b)

        self.x_ab = self.g_dec(self.c_a, self.l_b)
        self.x_ba = self.g_dec(self.c_b, self.l_a)

        self.c_a_recon = self.g_enc(self.x_ab,self.l_b)
        self.c_b_recon = self.g_enc(self.x_ba,self.l_a)

        self.x_aba = self.g_dec(self.c_a_recon, self.l_a)
        self.x_bab = self.g_dec(self.c_b_recon, self.l_b)

    def gen_update(self):
        self.set_requires_grad([self.dis], False)
        self.gen_opt.zero_grad()

        self.loss_g_rec = (self.recon_criterion(self.x_a, self.x_a_recon) + self.recon_criterion(self.x_b, self.x_b_recon)) * self.lambda_rec
        self.loss_g_cyc = (self.recon_criterion(self.x_a, self.x_aba) + self.recon_criterion(self.x_b, self.x_bab)) * self.lambda_cyc
        self.loss_g_ctr = (self.recon_criterion(self.c_a, self.c_a_recon) + self.recon_criterion(self.c_b, self.c_b_recon)) *self.lambda_ctr

        loss_G_rec = self.criterionGAN(self.dis(torch.cat((self.x_a_recon,self.x_b_recon), 0), torch.cat((self.l_a, self.l_b), 0)), False)
        loss_G_fake = self.criterionGAN(self.dis(torch.cat((self.x_ab, self.x_ba), 0),torch.cat((self.l_b, self.l_a), 0)), True)
        self.loss_g_adv = (loss_G_rec + loss_G_fake)

        if self.use_adt:
            self.loss_g_rec.backward(retain_graph=True)
            self.set_requires_grad([self.g_dec], False)
            self.loss_g_rest = self.loss_g_cyc + self.loss_g_ctr + self.loss_g_adv
            self.loss_g_rest.backward()
            self.set_requires_grad([self.g_dec], True)
        else:
            self.loss_g_total = self.loss_g_rec + self.loss_g_cyc + self.loss_g_ctr + self.loss_g_adv
            self.loss_g_total.backward()
        self.gen_opt.step()

    def dis_update(self):
        self.set_requires_grad([self.dis], True)
        self.dis_opt.zero_grad()

        x_a_recon=self.x_a_recon.detach()
        x_b_recon=self.x_b_recon.detach()
        x_ab=self.x_ab_pool.query(self.x_ab.detach())
        x_ba=self.x_ba_pool.query(self.x_ba.detach())

        loss_D_rec = self.criterionGAN(self.dis(torch.cat((x_a_recon,x_b_recon), 0), torch.cat((self.l_a, self.l_b), 0)), True)
        loss_D_fake = self.criterionGAN(self.dis(torch.cat((x_ab, x_ba), 0),torch.cat((self.l_b, self.l_a), 0)), False)
        self.loss_dis = (loss_D_rec + loss_D_fake)

        self.loss_dis.backward()
        self.dis_opt.step()

    def verbose(self):
        text=''
        lr = self.optimizers[0].param_groups[0]['lr']
        text+='{} {:.6f}  '.format('lr', lr)
        for s in self.__dict__.keys():
            if 'loss_' in s:
                text+='{} {:.3f}  '.format(s.replace('loss_',''),getattr(self,s).item())
        return text

    def gan_visual(self,epoch):
        collections=[]
        for im in [self.x_a, self.x_a_recon, self.x_ab, self.x_aba, self.x_b,self.x_b_recon,self.x_ba, self.x_bab]:
            tim= np.clip(((im[0,0].detach().cpu().numpy())+1)*127.5,0,255).astype(np.uint8)
            collections.append(tim)
        plt.figure(figsize=(12,6))
        titles=['real image','recon image','fake mask','cyclic image','real mask','recon mask','fake image','cyclic mask']
        for i in range(2):
            for j in range(4):
                plt.subplot(2,4,i*4+j+1)
                plt.title(titles[i*4+j])
                plt.imshow(collections[i*4+j],cmap='gray')
                plt.axis('off')
        plt.tight_layout()
        e='%03d'%epoch
        plt.savefig(os.path.join(self.opts.visdir,e+'.png'),dpi=200)
        plt.close()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def moving_average(self, model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opts.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

    def load(self, ckpt_path):
        state_dict=torch.load(ckpt_path)
        self.g_enc.load_state_dict(state_dict['enc'])
        self.g_dec.load_state_dict(state_dict['dec'])

    def save(self,  iteration):
        model_name = os.path.join(self.opts.ckptdir, '%03d.pth' % (iteration))
        state_dict={'enc': self.g_enc.state_dict(), 'dec': self.g_dec.state_dict(),
                    'dis': self.dis.state_dict(), }
        torch.save(state_dict, model_name)
        torch.save(state_dict,  os.path.join(self.opts.ckptdir, 'latest.pth'))

    def evaluate(self,test_dataloader):
        def get_perf(pred, gt):
            def pre_procss(arr):
                return arr.flatten().astype(np.int32)
            pred = pre_procss(pred)
            gt = pre_procss(gt)

            confusion_mat = metrics.confusion_matrix(pred, gt)
            TP = confusion_mat[1, 1]
            FP = confusion_mat[1, 0]
            FN = confusion_mat[0, 1]
            TN = confusion_mat[0, 0]
            total = TP + TN + FP + FN

            iou = TP / (TP + FP + FN)
            acc = (TP + TN) / (total)
            P = TP / (TP + FP)
            R = TP / (TP + FN)
            FS = 2 * P * R / (P + R)
            DICE = 2 * TP / (TP + FP + TP + FN)
            return P, R, FS

        PS,RS,FSS=[],[],[]
        for test_data in test_dataloader:
            img=test_data['A'].cuda()
            gtt= (((test_data['B'].cuda()+1)*127.5)>64).long().cpu().numpy()[0, 0]
            with torch.no_grad():
                predict=self(img)
            predict = ((predict.tanh().cpu().numpy()[0, 0]+1)*127.5)>64
            _p,_r,_fs=get_perf(predict,gtt)
            PS.append(_p)
            RS.append(_r)
            FSS.append(_fs)
        print()
        print(f'Evaluation with precision:{np.mean(PS)}, Recall:{np.mean(RS)}, Dice:{np.mean(FSS)}')




