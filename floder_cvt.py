import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--total_id', type=int, required=True)
    parser.add_argument('--sample', type=int, required=True)
    parser.add_argument('--spath', type=str, default='./results/palm_mpd/val/images')
    parser.add_argument('--dpath', type=str, default='./fakepalm')
    parser.add_argument('--batch_size', type=int, default=4)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    spath = os.path.join(args.spath, 'val', 'images')
    dpath = args.dpath 

    # os.makedirs(dpath, exist_ok=True)
    # shutil.rmtree(dpath)
    os.makedirs(dpath, exist_ok=True)

    sfiles = os.listdir(spath)

    dc = {}

    for fi in sfiles:
        lis = fi.split('_')
        

    # namelist = ['random_sample01.png', 'encoded.png', 'ground truth.png', 'input.png']

    for i in range(args.total_id):
        id_str = str(i)
        if not os.path.exists(os.path.join(dpath, id_str)):
            os.makedirs(os.path.join(dpath, id_str), exist_ok=True)
        
        for j in range(args.sample):
            # k = (j+1) % args.batch_size

            fi = f'{i}_{j}_random_sample01.png'
            
            sp = os.path.join(spath, fi)
            dp = os.path.join(dpath, id_str, fi)

            # shutil.copy(sp, dp)
            shutil.move(sp, dp)