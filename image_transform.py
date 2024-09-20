import cv2 
import numpy as np
import argparse 
import random, os , shutil

class ImageTrans:
    def __init__(self, height, width, borderValue=255):
        self.width = width
        self.height = height
        self.center = (self.width // 2, self.height // 2)
        self.borderValue =borderValue

    def __call__(self, image, angle, scale, delta):
        M = cv2.getRotationMatrix2D(self.center, angle, scale)
        dsize = (self.width + delta[0], self.height + delta[1])
        rotated_img = cv2.warpAffine(image, M, dsize, borderValue=self.borderValue)
        return rotated_img
    

def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--sample', type=int, default=100)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    spath = args.input
    dpath = args.output

    if os.path.exists(dpath):
        shutil.rmtree(dpath)
    os.makedirs(dpath, exist_ok=True)

    Imt = ImageTrans(height=256, width=256)

    for fi in os.listdir(spath):
        sf = os.path.join(spath, fi)
        img = cv2.imread(sf, 0)

        flag = random.randint(0, 1)
        if flag:
            img = cv2.flip(img, 1)

        for j in range(args.sample):
            angle = random.randint(-5, 5)
            scale = random.uniform(0.95, 1.05)
            delta = (0, 0)

            res_image = Imt(img, angle=angle, scale=scale, delta=delta)

            fname = fi.split('_')[0] + f'_{j}.png'
            df = os.path.join(dpath, fname)
            cv2.imwrite(df, res_image)

        print(fname)
