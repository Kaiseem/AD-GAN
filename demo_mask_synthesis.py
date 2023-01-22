from data.synthesisPatch import PatchSynthesis
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no_inst', action='store_true', help='if specified, do not use instance segmentation')
parser.add_argument('--ellipse_min_radius', type=int, default=20)
parser.add_argument('--ellipse_max_radius', type=int, default=30)
parser.add_argument('--ellipse_min_num', type=int, default=5)
parser.add_argument('--ellipse_max_num', type=int, default=15)
parser.add_argument('--crop_size', type=int, default=256)
opts = parser.parse_args()

if __name__=='__main__':
    patchsyn=PatchSynthesis(ellipse_radius=[opts.ellipse_min_radius,opts.ellipse_max_radius],ellipse_number=[opts.ellipse_min_num,opts.ellipse_max_num],instance_mask=not opts.no_inst,patch_size=opts.crop_size)
    im=patchsyn.get_patch()
    from PIL import Image
    Image.fromarray(im).save('MASK.png')