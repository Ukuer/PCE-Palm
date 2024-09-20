import torch
import torchvision
from torchvision import transforms
import numpy as np
from torch import nn
import cv2
import math

from .utils2 import IMAGE_TRANS, get_mfrat, get_meanblur


class MFRAT(nn.Module):
    def __init__(self, inchannel:int, mksize:int, bksize:int=9, blur:bool=True, 
                 use_arg:bool=True, scale:bool=False, add:bool=False, conv3:bool=False, 
                 mode:int=0, norm:bool=True, angle0:int=15, dilation:int=1,
                 ftype:str='gaussian', mid:bool=False, device:str='cpu') -> None:
        super(MFRAT, self).__init__()

        self.inchannel = inchannel
        self.add = add 
        self.use_arg = use_arg
        self.mid = mid

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
            self.ratio = nn.Parameter(torch.ones(inchannel)[None, :, None, None], requires_grad=True)

    def mfrat(self, image:torch.Tensor):
        result = self.mfrat_conv(image)
        result = result.reshape((result.size(0), self.inchannel, -1,  result.size(2), result.size(3)))
        value_max, arg_max = torch.max(result, dim=2)
        if self.use_arg:
            self.arg_max = arg_max.to(value_max.dtype) / self.directions
        return value_max
    
    def get_label(self, image:torch.Tensor)->torch.Tensor:
        output_image = self.forward(image)
        if self.use_arg:
            filter_image = output_image[:, :output_image.size(1)//2]
        else:
            filter_image = output_image

        main_ratio = self.threshold
        diff_flatten = filter_image.detach().flatten()
        diff_sort_index = torch.argsort(diff_flatten, dim=-1, descending=False)
        main_threshold_index = diff_sort_index[int(diff_flatten.shape[0] * main_ratio)]
        main_threshold = diff_flatten[main_threshold_index].item()

        result = torch.where(filter_image < main_threshold, 0.0, 1.0)

        return result
    
    def get_orient(self, image:torch.Tensor)->torch.Tensor:
        output_image = self.forward(image)
        if self.use_arg:
            return output_image[:, output_image.size(1)//2:] 
        else:
            return None
        
    def get_label_orient(self, image:torch.Tensor)->torch.Tensor:
        output_image = self.forward(image)
        if self.use_arg:
            filter_image = output_image[:, :output_image.size(1)//2]
        else:
            filter_image = output_image

        main_ratio = self.threshold
        diff_flatten = filter_image.detach().flatten()
        diff_sort_index = torch.argsort(diff_flatten, dim=-1, descending=False)
        main_threshold_index = diff_sort_index[int(diff_flatten.shape[0] * main_ratio)]
        main_threshold = diff_flatten[main_threshold_index].item()

        result = torch.where(filter_image < main_threshold, 0.0, 1.0)

        return torch.concat([result, output_image[:, output_image.size(1)//2:]], dim=1)

    def forward(self, image):
        # image shape is N x C x H x W
        
        # first blur
        if self.blur:
            if self.const_blur:
                mean_image = torch.mean(image.reshape(image.size(0), image.size(1), -1), dim=2)
                if self.mid:
                    image_diff = image - mean_image[:, :, None, None]
                else:    
                    image_diff = mean_image[:, :, None, None] - image

            else:   
                image_blur = self.blur_conv(image)
                if self.mid:
                    image_diff = image - image_blur 
                else:
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
                if self.mid:
                    output = self.conv3_conv(image) + output
                else:
                    output = self.conv3_conv(image) - output
            else:
                if self.mid:
                    output = image + output
                else:
                    output = image - output

        # use_arg or not 
        if self.use_arg:
            output = torch.concat((output, self.arg_max), dim=1)

        return output


def demo():
    image1 = IMAGE_TRANS(cv2.imread('./bezierpalm/1.jpg', 0))
    image2 = IMAGE_TRANS(cv2.imread('./bezierpalm/2.jpg', 0))
    image = torch.concat([image1, image2], dim=0)[None, :, :, :]

    data = image

    model = MFRAT4(inchannel=2, mksize=35, bksize=11)

    output = model(data)

    # output = output.reshape(output.size(1), output.size(2), output.size(3))[:,None, :, :]
    output = output.permute(1, 0, 2, 3)

    torchvision.utils.save_image(output.detach(), './bezierpalm/mfrat5_label0_mean.jpg')

    max_value, _ = torch.max(output.reshape(output.size(0), -1), dim=-1)
    min_value, _ = torch.min(output.reshape(output.size(0), -1), dim=-1)

    # output = torch.clamp(output, min=0.0, max=1.0)
    # output *= (1.0 / max_value)

    output -= min_value[:, None, None, None]
    output *= (1 / (max_value[:, None, None, None]  - min_value[:, None, None, None]))

    output = output 

    torchvision.utils.save_image(output.detach(), './bezierpalm/mfrat5_label_mean.jpg')
     

def get_label(input_filename, label_filename=None, orient_filename=None):
    image = IMAGE_TRANS(cv2.imread(input_filename))
    image = image[None, :, :, :]

    data = image

    model = MFRAT4(inchannel=1, mksize=35, bksize=11, use_arg=True)

    output = model(data)

    if label_filename:
        torchvision.utils.save_image(output.cpu().detach()[:, :3], label_filename)
    
    if orient_filename:
        torchvision.utils.save_image(output.cpu().detach()[:, 3:], orient_filename)

    return output


def test():
    x = torch.randn((1, 3, 9, 9))
    model = nn.Conv2d(3, 6, 3, stride=1, padding=1+1, dilation=2)

    y = model(x)

    None


class LFEB(MFRAT):
    def __init__(self, inchannel:int, mksize:int, bksize:int=9, blur:bool=True, 
                use_arg:bool=True, scale:bool=False, add:bool=False, conv3:bool=False, 
                mode:int=0, norm:bool=True, angle0:int=15, dilation:int=1,
                ftype:str='gaussian', mid:bool=False, device:str='cpu') -> None:
        
        # initi MFRAT 
        super().__init__(inchannel=inchannel, mksize=mksize, bksize=bksize, blur=blur, use_arg=use_arg,
                         add=add, scale=scale, mode=mode, norm=norm, angle0=angle0, dilation=dilation, 
                         ftype=ftype,conv3=conv3,mid=mid, device=device)

    def forward(self, image):
        return super().forward(image)




def test():
    image = torch.randn((1, 3, 256, 256)).to('cuda')

    model = MFRAT(inchannel=3, mksize=35, bksize=-1, blur=True, use_arg=True,
                   scale=False, add=True, conv3=False, mode=0, norm=True, 
                   angle0=15, dilation=1, ftype='gaussian', device='cuda')
    

    output_image = model(image)

    None



if __name__ == '__main__':
    test()
