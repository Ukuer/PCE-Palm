import os
import cv2 as cv
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    spath = args.input
    dpath = args.output

    if os.path.exists(dpath):
        shutil.rmtree(dpath)
    os.makedirs(dpath, exist_ok=True)

    for d in os.listdir(spath):
        if not os.path.isdir(os.path.join(spath, d)):
            continue

        fs = os.listdir(os.path.join(spath, d))

        for f in fs:
            sp = os.path.join(spath, d, f)
            dp = os.path.join(dpath, '{}_{}.{}'.format(int(d), int(f[0:3]), f.split('.')[-1]))
            os.rename(sp, dp)
            
        print(d)