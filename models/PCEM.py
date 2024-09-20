import torch
import torchvision
from torchvision import transforms
import numpy as np
from torch import nn
import cv2
import math

IMAGE_TRANS = transforms.Compose([
    transforms.ToTensor()
])


def get_ftype(ftype:str, ksize:int, sigma:float=0.01):
    
    if ftype == 'linear':
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


def single_filter(ksize:int, angle:float) -> np.ndarray:
    assert (ksize > 0)
    assert (0 <= angle <180)
    half = ksize // 2
    middle = half + 1 - 1
    filter = np.zeros(shape=(ksize,)*2, dtype=np.float32)

    angle -= 180 if angle > 90 else 0
    radius = angle * math.pi / 180

    if -45 < angle < 45:
        ratio = math.tan(radius)    
        for dx in range(-half, half+1):
            dy = round(dx * ratio) 
            filter[middle + dy, middle + dx] = 1.0
    
    elif angle < -45 or angle > 45:
        ratio = math.cos(radius) / math.sin(radius)
        for dy in range(-half, half+1):
            dx = round(dy * ratio) 
            filter[middle + dy, middle + dx] = 1.0
    
    elif angle == 45:
        for dx in range(-half, half+1):
            filter[middle + dx, middle + dx] = 1.0

    else:
        for dx in range(-half, half+1):
            filter[middle + dx, middle - dx] = 1.0

    return filter


def get_filter(ksize:int, mode:int=0, ftype:str='constant', norm:bool=False, angle0:int=15) -> np.ndarray:
    assert (ksize > 0)
    assert (mode in [0, 1])
    assert (ftype in ['linear', 'gaussian', 'cosine'] )

    mask_row, mask_col = get_ftype(ftype=ftype, ksize=ksize)
    anglelist = [i*angle0 for i in range(180//angle0)]
    result = None
    for angle in anglelist:
        f = single_filter(ksize, angle)
        if (angle <= 45) or (angle >= 135):
            f = f * mask_row
        else:
            f = f * mask_col

        if result is None:
            result = f[None, :, :]
        else:
            result = np.concatenate((result, f[None, :, :]), axis=0)

    if mode == 1:
        # the folloeing is to make half line
        filter = result
        num, height, width = filter.shape
        filter_result = np.zeros((num*2, height, width), dtype=filter.dtype)
        mask_up = np.zeros((height, width), dtype=filter.dtype)
        mask_down = np.zeros((height, width), dtype=filter.dtype)
        mask_left = np.zeros((height, width), dtype=filter.dtype)
        mask_right = np.zeros((height, width), dtype=filter.dtype)
        
        mask_up[(height+1)//2:, :] =  1.0
        mask_down[:(height+1)//2+1, :] =  1.0
        mask_left[:, (width+1)//2:] =  1.0
        mask_right[:, :(width+1)//2+1] =  1.0

        for i in range(filter.shape[0]):
            img = filter[i]

            if anglelist[i] == 90:
                img1 = img * mask_down
                img2 = img * mask_up
                filter_result[i] = img1
                filter_result[i+num] = img2
            else:
                img1 = img * mask_right
                img2 = img * mask_left
                filter_result[i] = img1
                filter_result[i+num] = img2
        result = filter_result

    if norm:
            result /= np.sum(result.reshape(-1, ksize*ksize), axis=1)[:, None, None]

    return result



def get_mfrat(ksize:int, in_channel:int=1, mode:int=0, ftype:str='constant', 
              norm:bool=False, angle0:int=15, dilation:int=1, device:str='cpu')->torch.nn.Module:

    filter_rank = get_filter(ksize=ksize, ftype=ftype, mode=mode, norm=norm, angle0=angle0)
    out_channel = filter_rank.shape[0]
    filter_rank = filter_rank[:, None, :, :]
    filter_rank = torch.Tensor(filter_rank).to(device)
    filter_rank = filter_rank.repeat(in_channel, 1, 1, 1)

    mfrat_conv = nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel * in_channel,
        kernel_size=(ksize,) * 2,
        bias=False,
        stride=1,
        dilation=dilation,
        padding=(ksize//2)+(dilation-1),
        padding_mode='replicate',
        groups=in_channel
    ).to(device)

    assert mfrat_conv.weight.shape == filter_rank.shape
    mfrat_conv.weight = nn.Parameter(filter_rank)
    mfrat_conv.weight.requires_grad = False

    return mfrat_conv


def get_meanblur(ksize:int, in_channel:int=1, device:str='cpu')->torch.nn.Module:

    blur_conv = nn.Conv2d(
        in_channels=in_channel,
        out_channels=in_channel,
        kernel_size=(ksize,) * 2,
        bias=False,
        stride=1,
        padding=ksize // 2,
        padding_mode='replicate',
        groups=in_channel
    ).to(device)

    blur_kernel = np.ones(shape=(ksize,) * 2, dtype=np.float32)
    blur_kernel /= np.sum(blur_kernel)
    blur_kernel = blur_kernel[None, None, :, :]
    blur_kernel = torch.Tensor(blur_kernel)
    blur_kernel = blur_kernel.repeat(in_channel, 1, 1, 1)
    
    assert blur_conv.weight.shape == blur_kernel.shape
    blur_conv.weight = nn.Parameter(blur_kernel)
    blur_conv.weight.requires_grad = False
    return blur_conv


def show_filter():
    filter_mask = get_filter(ksize=25, norm=False, ftype='gaussian')
    filter_mask *= 255
    filter_mask = filter_mask.astype(np.uint8)

    for i in range(filter_mask.shape[0]):
        img = filter_mask[i]
        img = cv2.resize(img, dsize=(256, 256))
        cv2.imshow(f'filter_{i}', img)

    cv2.waitKey(0)


class MFRAT(nn.Module):
    def __init__(self, inchannel:int, mksize:int, bksize:int=9, blur:bool=True, 
                 use_arg:bool=True, scale:bool=False, add:bool=False, conv3:bool=False, 
                 mode:int=0, norm:bool=True, angle0:int=15, dilation:int=1,
                 ftype:str='gaussian', device:str='cpu') -> None:
        super(MFRAT, self).__init__()

        self.inchannel = inchannel
        self.add = add 
        self.use_arg = use_arg

        # define MFRAT
        self.mfrat_conv = get_mfrat(ksize=mksize, in_channel=inchannel, mode=mode, ftype=ftype, norm=norm,
                                    angle0=angle0, dilation=dilation, device=device)
        self.directions = 180 // angle0
        self.directions *= 2 if norm == 1 else 1

        # define blur
        self.blur = blur 
        if self.blur:
            self.const_blur = (bksize <= 0)
            if not self.const_blur:
                self.blur_conv = get_meanblur(ksize=bksize, in_channel=inchannel, device=device)

        # define conv3x3
        self.conv3 = conv3
        assert not ((not self.add) and self.conv3)
        if self.conv3:
            self.conv3_conv = nn.Conv2d(
                in_channels=inchannel,
                out_channels=inchannel,
                kernel_size=(3,) * 2,
                bias=True,
                stride=1,
                padding=1,
                padding_mode='replicate'
            ).to(device)

        self.scale = scale
        if self.scale:
            self.ratio = nn.Parameter(torch.ones(inchannel)[None, :, None, None], requires_grad=True).to(device)

    def mfrat(self, image:torch.Tensor):
        result = self.mfrat_conv(image)
        result = result.reshape((result.size(0), self.inchannel, -1,  result.size(2), result.size(3)))
        value_max, arg_max = torch.max(result, dim=2)
        if self.use_arg:
            self.arg_max = arg_max.to(value_max.dtype) / self.directions
        return value_max 

    def forward(self, image)->torch.Tensor:
        # image shape is N x C x H x W
        
        # first blur
        if self.blur:
            if self.const_blur:
                mean_image = torch.mean(image.reshape(image.size(0), image.size(1), -1), dim=2)
                image_diff = mean_image[:, :, None, None] - image
            else:   
                image_blur = self.blur_conv(image)
                image_diff = image_blur - image
        else:
            image_diff = image
        
        # MFRAT 
        output = self.mfrat(image_diff)

        # scale or not 
        if self.scale:
            output = self.ratio * output

        # add or not
        if self.add:
            if self.conv3:
                output = self.conv3_conv(image) - output
            else:
                output = image - output

        # use_arg or not 
        if self.use_arg:
            output = torch.concat((output, self.arg_max), dim=1)

        return output


class PCEM(nn.Module):
    def __init__(self, use_arg:bool=False, threshold:float=0.9, device:str='cpu') -> None:
        super().__init__()

        self.use_arg = use_arg
        self.threshold = threshold

        self.model = MFRAT(inchannel=1, mksize=35, bksize=11, blur=True, use_arg=use_arg,
                            add=False, scale=False, mode=0, norm=True, angle0=15,
                            dilation=1, ftype='gaussian', device=device)

    def get_label(self, image:torch.Tensor)->torch.Tensor:
        output_image = self.model(image)
        if self.use_arg:
            filter_image = output_image[:, 0:1]
        else:
            filter_image = output_image

        main_ratio = self.threshold
        diff_flatten = filter_image.detach().flatten()
        diff_sort_index = torch.argsort(diff_flatten, dim=-1, descending=False)
        main_threshold_index = diff_sort_index[int(diff_flatten.shape[0] * main_ratio)]
        main_threshold = diff_flatten[main_threshold_index].item()

        result = torch.where(filter_image < main_threshold, 0.0, 1.0)

        return result

    def forward(self, image:torch.Tensor)->torch.Tensor:
        output_image = self.model(image)
        if self.use_arg:
            filter_image = output_image[:, 0:1]
        else:
            filter_image = output_image

        filter_image = torch.clamp(filter_image, min=1e-4, max=1.0)
        ratio = 1.0 / torch.max(filter_image).item()
        filter_image = filter_image * ratio

        return filter_image
