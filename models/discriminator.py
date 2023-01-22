import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainDiscriminator(nn.Module):
    def __init__(self, image_nc=1, ndf=64, num_domain=2, dimensions=2):
        super().__init__()
        assert dimensions in [2, 3]
        Conv = nn.Conv2d if dimensions == 2 else nn.Conv3d
        InsNrom=  nn.InstanceNorm2d  if dimensions == 2 else nn.InstanceNorm3d
        main= []
        main += [Conv(image_nc,ndf,4,2,1),nn.LeakyReLU(0.2, True)]
        main += [Conv(ndf,ndf*2,4,2,1),  InsNrom(ndf*2,affine=False) ,  nn.LeakyReLU(0.2, True)]
        main += [Conv(ndf*2,ndf*4,4,2,1),  InsNrom(ndf*4,affine=False) ,  nn.LeakyReLU(0.2, True)]
        main += [Conv(ndf*4,ndf*8,4,1,1),  InsNrom(ndf*8,affine=False) ,  nn.LeakyReLU(0.2, True)]
        main += [Conv(ndf * 8,num_domain, 4, 1, 1)]
        self.main = nn.Sequential(*main)
        self.gan_type = 'lsgan'

    def forward(self, x, y):
        out = self.main(x)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y, :, :]

        return out

    def calc_dis_loss(self, input0, input1, label):
        out0 = self.forward(input0,label)
        out1= self.forward(input1,label)

        if self.gan_type == 'lsgan':
            loss = torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
        elif self.gan_type == 'ralsgan':
            loss = torch.mean((out1 - torch.mean(out0) - 1) ** 2) + torch.mean((out0 - torch.mean(out1) + 1) ** 2)
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input0, input1, label):
        out0 = self.forward(input0,label)
        out1= self.forward(input1,label)

        if self.gan_type == 'lsgan':
            loss = torch.mean((out0 - 1)**2) + torch.mean((out1 - 0)**2)
        elif self.gan_type == 'ralsgan':
            loss = torch.mean((out0 - torch.mean(out1) - 1) ** 2) + torch.mean((out1 - torch.mean(out0) + 1) ** 2)
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss
