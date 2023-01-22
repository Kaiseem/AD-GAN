from .networks import *

class GeneratorDecoder(Generator_Base):
    def __init__(self, image_nc, ngf=64, num_domain=2, dimensions=2):
        super(GeneratorDecoder, self).__init__()
        Conv = nn.Conv2d if dimensions == 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if dimensions == 2 else nn.ConvTranspose3d
        pad = nn.ReflectionPad2d if dimensions == 2 else nn.ReflectionPad3d
        main = []
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ConvTranspose(ngf * 4, ngf * 2, 3, 2, 1, 1), AdaptiveInstanceNorm(ngf * 2), nn.ReLU()]
        main += [ConvTranspose(ngf * 2, ngf, 3, 2, 1, 1), AdaptiveInstanceNorm(ngf), nn.ReLU()]
        main += [pad(3), Conv(ngf, image_nc, 7)]
        self.main = nn.Sequential(*main)
        self.mlp = MLP(num_domain, self.get_num_adain_params(self.main), 64, 3)

    def forward(self, x, D_c,return_logits_only=False):
        ones = torch.sparse.torch.eye(2).cuda()
        adain_params = self.mlp(ones.index_select(0, D_c))
        self.assign_adain_params(adain_params, self.main)

        x = self.main(x)
        if return_logits_only:return x
        return x.tanh()

class GeneratorEncoder(Generator_Base):
    def __init__(self, image_nc, ngf=64, num_domain=2, dimensions=2):
        super(GeneratorEncoder, self).__init__()
        Conv = nn.Conv2d if dimensions == 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if dimensions == 2 else nn.ConvTranspose3d
        pad = nn.ReflectionPad2d if dimensions == 2 else nn.ReflectionPad3d

        main = []
        main += [pad(3), Conv(image_nc, ngf, 7), AdaptiveInstanceNorm(ngf), nn.ReLU()]
        main += [Conv(ngf, ngf * 2, 3, 2, 1), AdaptiveInstanceNorm(ngf * 2), nn.ReLU()]
        main += [Conv(ngf * 2, ngf * 4, 3, 2, 1), AdaptiveInstanceNorm(ngf * 4), nn.ReLU()]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        self.main = nn.Sequential(*main)
        self.mlp = MLP(num_domain, self.get_num_adain_params(self.main), 64, 3)

    def forward(self, x, E_c):
        ones = torch.sparse.torch.eye(2).cuda()
        adain_params = self.mlp(ones.index_select(0, E_c))
        self.assign_adain_params(adain_params, self.main)
        x = self.main(x)
        return x
