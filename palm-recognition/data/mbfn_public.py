import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import sys
from os.path import dirname, abspath

sys.path.insert(0, abspath(dirname(__file__)))
#from auto_augment import rand_augment_transform
#from timm.data.transforms import RandomResizedCropAndInterpolation

config = dict(

 
    ckpt_freq=5,
    print_freq=100,
    model_name="mobilefacenet_qat.MobileFaceNet",
    #model_name="IR_50",
    # model_args = dict(qat=False),
    model_args=dict(),
    head=dict(name="MyArcFace", m=0.5, s=48),
    # head = dict(name="MyArcFace", m=0.5, s=48, warmup_iters=10000),
    embed_dim=512,
    # optimizer related
    opt=torch.optim.SGD,
    opt_args=dict(lr=0, weight_decay=5e-4, momentum=0.9),
    scheduler_name="MultiStepScheduler",
    scheduler_args=dict(milestones=[14, 18, 22], gamma=0.1, base_lr=0.01, warmup_epochs=1, warmup_init_lr=1e-5),
    batchsize=32,
    epochs=26,

    test_data_list = './ef_val.list',
    test_transform = transforms.Compose([
        #transforms.CenterCrop(672), #margin0.0
        transforms.Resize([224, 224], interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [1, 1, 1])
    ])
)
