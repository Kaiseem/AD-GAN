import os.path
from .base_dataset import BaseDataset,make_power_2
from .image_folder import make_dataset
import torchvision.transforms as transforms
from PIL import Image
from .synthesisPatch import PatchSynthesis
import numpy as np
import albumentations as A
import random

# Datasets for 2D images
class NucleiDataset(BaseDataset):
    def __init__(self, opts, phase):
        BaseDataset.__init__(self, opts)
        assert phase in ['train','test']
        self.phase = phase
        self.dir_A = os.path.join(opts.dataroot, self.phase + 'A')
        self.A_paths = sorted(make_dataset(self.dir_A))

        if self.phase=='train':
            self.A_imgs=self.load_images()
            self.patchsyn=PatchSynthesis(ellipse_radius=[opts.ellipse_min_radius,opts.ellipse_max_radius],ellipse_number=[opts.ellipse_min_num,opts.ellipse_max_num],instance_mask=not opts.no_inst,patch_size=opts.crop_size)
        else:
            self.dir_B = os.path.join(opts.dataroot, self.phase + 'B')
            self.B_paths = sorted(make_dataset(self.dir_B))
            self.A_imgs, self.B_imgs=self.load_paired_images()

        self.img_aug = A.Compose([
            A.Rotate(30),
            A.RandomCrop(opts.crop_size,opts.crop_size),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),])

        self.ToTensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

    def load_paired_images(self):
        A_imgs=[]
        B_imgs=[]
        for A_path,B_path in zip(self.A_paths,self.B_paths):
            A_imgs.append(np.array(Image.open(A_path)))
            B_imgs.append(np.array(Image.open(B_path)))
        print(f'load images and masks with size of {len(A_imgs)} and {len(B_imgs)}')
        return A_imgs, B_imgs

    def load_images(self):
        A_imgs=[]
        for A_path in self.A_paths:
            A_img = np.array(Image.open(A_path))
            A_imgs.append(A_img)
        print(f'load images with size of {len(A_imgs)}')
        return A_imgs

    def __getitem__(self, index):
        if self.phase=='train':
            index = random.randint(0, len(self.A_imgs) - 1)
            A_img = self.A_imgs[index]
            A_img = self.img_aug(image=A_img)["image"]
            B_img = self.patchsyn.get_patch()
            # apply image transformation
        else:
            A_img = self.A_imgs[index]
            B_img = self.B_imgs[index]
            A_img=make_power_2(A_img,base=4)
            B_img=make_power_2(B_img,base=4,method=Image.NEAREST)
        A = self.ToTensor(A_img)
        B = self.ToTensor(B_img)
        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.A_imgs)
