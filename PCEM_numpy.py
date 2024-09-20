import os, shutil
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math
import multiprocessing


def get_ftype(ftype:str, ksize:int, sigma:float=0.01):
    
    if ftype == 'constant':
        value = np.ones(ksize)
    elif ftype == 'cosine':
        t = np.linspace(-np.pi/2, np.pi/2, ksize)
        value = np.cos(t)
    elif ftype == 'gaussian':
        t = np.linspace(-sigma*2, sigma*2, ksize)
        x = - t**2 / (2 * sigma**2)
        value = np.exp(x)
        
    mask_row = value[None, :]
    mask_col = value[:, None]

    return (mask_row, mask_col)


def single_filter(ksize:int, angle:float, width:int=1) -> np.ndarray:
    assert (ksize > 0)
    assert (0 <= angle <180)
    assert (width % 2 == 1)
    half = ksize // 2
    middle = half + 1 - 1
    filter = np.zeros(shape=(ksize,)*2, dtype=np.float32)
    half_width = (width - 1) // 2

    angle -= 180 if angle > 90 else 0
    radius = angle * math.pi / 180

    def in_box(x:int) -> bool:
        return 0 <= x < ksize

    if -45 < angle < 45:
        ratio = math.tan(radius)    
        for dx in range(-half, half+1):
            dy = round(dx * ratio)
            px = middle + dx
            py = middle + dy 
            filter[py, px] = 1.0
            if half_width > 0:
                for dw in range(1, half_width):
                    up = py+ dw 
                    down = py - dw 
                    if in_box(up):
                        filter[up, px] = 1.0
                    if in_box(down):
                        filter[down, px] = 1.0
    
    elif angle < -45 or angle > 45:
        ratio = math.cos(radius) / math.sin(radius)
        for dy in range(-half, half+1):
            dx = round(dy * ratio) 
            px = middle + dx
            py = middle + dy 
            filter[py, px] = 1.0
            if half_width > 0:
                for dw in range(1, half_width+1):
                    right = px + dw 
                    left = px - dw 
                    if in_box(right):
                        filter[py, right] = 1.0
                    if in_box(left):
                        filter[py, left] = 1.0

    elif angle == 45:
        for dx in range(-half, half+1):
            px = middle + dx
            py = middle + dx 
            filter[py, px] = 1.0
            if half_width > 0:
                for dw in range(1, half_width):
                    up = py+ dw 
                    down = py - dw 
                    if in_box(up):
                        filter[up, px] = 1.0
                    if in_box(down):
                        filter[down, px] = 1.0

    else:
        for dx in range(-half, half+1):
            px = middle + dx
            py = middle - dx 
            filter[py, px] = 1.0
            if half_width > 0:
                for dw in range(1, half_width):
                    up = py+ dw 
                    down = py - dw 
                    if in_box(up):
                        filter[up, px] = 1.0
                    if in_box(down):
                        filter[down, px] = 1.0 

    return filter


def get_filter(ksize:int, ftype:str='gaussian', norm:bool=False, angle0:int=15, width:int=1) -> np.ndarray:
    assert (ksize > 0)
    assert (ftype in ['constant', 'gaussian', 'cosine'] )

    mask_row, mask_col = get_ftype(ftype=ftype, ksize=ksize)
    anglelist = [i*angle0 for i in range(180//angle0)]
    result = None
    for angle in anglelist:
        f = single_filter(ksize, angle, width)
        if (angle <= 45) or (angle >= 135):
            f = f * mask_row
        else:
            f = f * mask_col

        if result is None:
            result = f[None, :, :]
        else:
            result = np.concatenate((result, f[None, :, :]), axis=0)

    if norm:
            result /= np.sum(result.reshape(-1, ksize*ksize), axis=1)[:, None, None]
    return result


def apply_gaussian_low_pass_filter(image):
    ksize=31
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0, borderType=cv2.BORDER_REFLECT)
    details = np.float32(image) - np.float32(blurred)
    return details


def apply_mean_low_pass_filter(image):
    ksize=11
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0, borderType=cv2.BORDER_REFLECT)
    details = np.float32(image) - np.float32(blurred)
    return details


def process_images(input_list, output_list):

    mfrat_kernels = get_filter(ksize=31, ftype='gaussian', norm=True, angle0=30, width=1)
    mfrat_kernels *= -1
    
    for sf, df in zip(input_list, output_list):
        image = cv2.imread(sf, 0)
        image = cv2.resize(image, (256, 256))
        image = apply_gaussian_low_pass_filter(image)

        filter_results =[]
        for kernel in mfrat_kernels:
            filter_results.append(cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)[None])
    
        filter_results = np.concatenate(filter_results, axis=0)
        response = np.max(filter_results, axis=0)

        hist, bins = np.histogram(response.flatten(), bins=256)
        cumulative_distribution = np.cumsum(hist) / np.sum(hist)
        threshold = np.interp(0.90, cumulative_distribution, bins[:-1])

        mask = response < threshold
        mask = np.uint8(mask * 255)
        cv2.imwrite(df, mask)


def main():

    spath = "xxx"
    dpath = "xxx-pce"
    os.makedirs(dpath, exist_ok=True)

    image_files = [os.path.join(spath, f) for f in os.listdir(spath)]
    save_files = [os.path.join(dpath, f) for f in os.listdir(spath)]

    num_process = 16
    process_list = []
    for i in range(num_process):
        process_list.append(multiprocessing.Process(target=process_images, args=(image_files[i::num_process], save_files[i::num_process])))

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

    print(f"{spath} --> {dpath} : Done !!!")


if __name__ == '__main__':
    main()