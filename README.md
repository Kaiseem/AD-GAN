# AD-GAN

## Aligned Disentangling GAN: A Leap towards End-to-end Unsupervised Nuclei Segmentation for Fluorescence Microscopy Images. [ArXiv](https://arxiv.org/pdf/2107.11022.pdf)
Kai Yao, Kaizhu Huang, Jie Sun, Curran Jude \
Both University of Liverpool and Xi'an Jiaotong-liverpool University 


**Abstract**

Considering the common usage of supervised deep learning in biomedical images, we study unsupervised cell nuclei segmentation for fluorescence microscopy images in this paper. 
Exploiting the recently-proposed unpaired image-to-image translation between  cell nuclei images and randomly synthetic masks, existing approaches, e.g., CycleGAN, have achieved encouraging results. 
However, these methods usually take a two-stage pipeline and fail to learn the end-to-end mode appropriately in cell nuclei images. 
More seriously, they could lead to the lossy transformation problem, such as the content inconsistency between the original images and the corresponding segmentation output. 
To address these limitations, we propose a novel end-to-end unsupervised framework called Aligned Disentangling Generative Adversarial Network (AD-GAN).
Distinctively, AD-GAN introduces representation disentanglement to separate content representation (the underlying spatial structure) from style representation (the rendering of the structure). 
With this framework, spatial structure can be preserved explicitly, enabling a significant reduction on macro-level lossy transformation. 
We also propose a novel training algorithm which is able to align the disentangled content in the latent space to reduce micro-level lossy transformation. 
Evaluations on real-world 2D and  3D datasets show that AD-GAN substantially outperforms other comparative methods and the professional software quantitatively and qualitatively. 

## News:
\[2023/1/22\] We release the training and inference code for 2D unsupervised nuclei semantic/instance segmentation. We will release the code for 3D unsupervised nuclei segmentation ASAP.

## 1. Installation

Clone this repo.
```bash
git clone https://github.com/Kaiseem/AD-GAN.git
cd SLAug/
```

This code requires PyTorch 1.10 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

## 2. Data preparation
For 2D images, firstly, you should format the images like the following structure. 
```none
AD-GAN
├── ...
├── datasets
│   ├── YourDATA
│   │   ├── trainA
│   │   │   ├── 1.png
│   │   ├── testA
│   │   │   ├── 1.png
│   │   │   ├── 2.png
│   │   ├── testB
│   │   │   ├── 1.png
│   │   │   ├── 2.png
├── ...
```
The training images should be placed in trainA, while the test image and corresponding masks should be placed in testA and testB respectively.
The masks should be binary (0 and 255) with one channel.

Then you should manually estimate the number and the size of the nuclei. \
For the demostration of mask synthesis, run the command 
```bash
python demo_mask_synthesis.py --ellipse_min_radius=13 --ellipse_max_radius=18 --ellipse_min_num=5 --ellipse_max_num=40 --no_inst
```

For the demostration of instance mask synthesis, run the command 
```bash
python demo_mask_synthesis.py --ellipse_min_radius=13 --ellipse_max_radius=18 --ellipse_min_num=5 --ellipse_max_num=40
```

## 4. Training the model
To reproduce the performance, you need one GPU with more than 16 GB memory

For the training of AD-GAN for 2D images, run the command 
```bash
python train.py --name=ADGAN --dataroot=datasets/YourDATA --ellipse_min_radius=XX --ellipse_max_radius=XX --ellipse_min_num=XX --ellipse_max_num=XX --no_inst --dimension=2
```

For the training of AD-GAN-INS for 2D images, run the command 
```bash
python train.py --name=ADGAN --dataroot=datasets/YourDATA --ellipse_min_radius=XX --ellipse_max_radius=XX --ellipse_min_num=XX --ellipse_max_num=XX --dimension=2
```

## 5. Evaluation
For evaluation, run the command 
```bash
python test.py --resume=logs/xxx/latest.pth
```

## Acknowledgements

Our codes are built upon [MUNIT](https://github.com/NVlabs/MUNIT), thanks for theri contribution to the community and the development of researches!

## Citation
If our work or code helps you, please consider to cite our paper. Thank you!

```
@article{yao2021adgan,
  title={AD-GAN: End-to-end unsupervised nuclei segmentation with aligned disentangling training},
  author={Yao, Kai and Huang, Kaizhu and Sun, Jie and Jude, Curran},
  journal={arXiv preprint arXiv:2107.11022},
  year={2021}
}
```