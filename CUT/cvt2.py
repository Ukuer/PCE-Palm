import os, shutil
import cv2 as cv
import numpy as np
import random

sp1 = r'C:\Users\JinJianlong\Desktop\code\contrastive-unpaired-translation-master\results/palm_ir/palm5_CUT\test_latest\images\fake_B'
sp2 = r'C:\Users\JinJianlong\Desktop\code\contrastive-unpaired-translation-master\results\palm_ir/palm5_CUT\test_latest\images\real_A'

files1 = os.listdir(sp1)
files2 = os.listdir(sp2)

# random.shuffle(files1)
# random.shuffle(files2)

dp = './palm_ir/train'
os.makedirs(dp, exist_ok=True)
shutil.rmtree(dp)
os.makedirs(dp, exist_ok=True)

for i in range(min(len(files1), len(files2))):
    f1 = os.path.join(sp1, files1[i])
    f2 = os.path.join(sp2, files2[i])
    f3 = os.path.join(dp, files1[i])

    im1 = cv.imread(f1)
    im2 = cv.imread(f2)

    im1 = cv.resize(im1, (256, 256))
    im2 = cv.resize(im2, (256, 256))

    im3 = np.hstack((im1, im2))
    cv.imwrite(f3, im3)

    if i % 10 == 0:
        print(i)
