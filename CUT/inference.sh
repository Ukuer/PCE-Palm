python3 test.py \
    --results_dir ./results/ \
    --dataroot ../image_floder/palm \
    --name bezier_m$1 \
    --CUT_mode CUT \
    --input_nc 1 \
    --output_nc 1 \
    --load_size 256 \
    --crop_size 256 \
    --use_l1 \
    --lambda_l1 1.0 \
    --netG resnet_9blocks_lfeb \
    --batch_size $2


SPATH_CUT=./results/bezier_m$1/test_latest/images/fake_B
DPATH_CUT=../datasets/palm$1/val2

mkdir -p ${DPATH_CUT}

mv ${SPATH_CUT}/* ${DPATH_CUT}
