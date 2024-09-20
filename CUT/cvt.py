import os
import shutil
import cv2 as cv
import numpy as np
import random

dpath1 = './datasets/palm_cycle/train'
files = os.listdir(dpath1)

dpath2 = './datatsets/palm_cycle/trainA'
os.makedirs(dpath2, exist_ok=True)
dpath3 = './datatsets/palm_cycle/trainB'
os.makedirs(dpath3, exist_ok=True)

for fi in files:
    f1 = os.path.join(dpath1, fi)
    f2 = os.path.join(dpath2, fi)
    f3 = os.path.join(dpath3, fi)

    im = cv.imread(f1)

    im2 = im[:, :256]
    im3 = im[:, 256:]

    cv.imwrite(f2, im2)
    cv.imwrite(f3, im3)
