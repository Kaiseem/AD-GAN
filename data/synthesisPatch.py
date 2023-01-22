import numpy as np
import random
import cv2
import scipy.ndimage as ndi

class PatchSynthesis(object):
    def __init__(self, ellipse_radius, ellipse_number, instance_mask=False, patch_size=256):
        self.patch_size=patch_size
        self.ellipse_radius=ellipse_radius
        self.ellipse_number=ellipse_number
        self.instance_mask=instance_mask
        self.eccentricity= [0.25, 0.75]
        print(f'2D Synthetic Maks Domain activate, with the radius of ellipse is from {ellipse_radius[0]} to {ellipse_radius[1]}, and the number is from {ellipse_number[0]} to {ellipse_number[1]}')
        if instance_mask:
            print('Instance Segmentation Activated')
        else:
            print('Semantic Segmentation Activated')

    def empty_mask(self):
        return np.zeros((self.patch_size,self.patch_size),dtype=np.uint8)

    def get_one_instance(self):
        tmp=self.empty_mask()

        # Random position
        center_x,center_y= int(random.uniform(0,self.patch_size-1)),int(random.uniform(0,self.patch_size-1))

        # Random size and shape
        radius_1=int(random.uniform(self.ellipse_radius[0],self.ellipse_radius[1]))
        eccentricity=random.uniform(self.eccentricity[0],self.eccentricity[1])
        radius_2=int((1-eccentricity**2)**0.5 * radius_1)

        # Random rotate angle
        rotate_angle = random.uniform(0, 359)
        cv2.ellipse(tmp, (center_x, center_y), (radius_1, radius_2), rotate_angle, 0, 360, 255, -1)
        if self.instance_mask:
            distance=ndi.distance_transform_edt(tmp)
            distance/=distance.max()
            tmp[np.logical_and(tmp>0,distance<0.5)]=127
        return tmp

    def get_patch(self):
        output=self.empty_mask()
        total_num=random.uniform(self.ellipse_number[0], self.ellipse_number[1])
        current_num=0
        exp=0
        while True:
            exp+=1
            one_inst=self.get_one_instance()
            if np.sum(np.logical_and(output>0,one_inst>0))==0:
                output+=one_inst
                exp=0
                current_num+=1
            if current_num>=total_num or exp>100:
                break
        return output

if __name__=='__main__':
    a=PatchSynthesis([30,50],[4,50],624)
    im=a.get_patch()
    from PIL import Image
    Image.fromarray(im).save('MASK.png')