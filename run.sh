python3 train.py \
    --dataroot datasets/xxx \
    --name palm \
    --model pce \
    --load_size 262 \
    --crop_size 256 \
    --input_nc 1 \
    --output_nc 3 \
    --use_dropout \
    --netD basic_256_multi \
    --netD2 basic_256_multi \
    --netG unet_256 \
    --netE resnet_256 \
    --dataset_mode two \
    --niter 30 \
    --niter_decay 30 \
    --aug_p 0.3 \
    --lambda_cyc 0.1 \
    # --display_id -1 \