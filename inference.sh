set -ex

NAME_ID=1600

TOTAL_ID=40
SAMPLE=10

BATCH_SIZE=4

# First step: generate Bezier curves images.
# Noted that generate N IDs with 1 samples each ID.
python3 ./syn_bezier.py \
  --num_ids ${TOTAL_ID} \
  --samples 1 \
  --nproc 8 \
  --output './image_floder/bezierpalm/palm'

# move files into one floder
python3 ./bezier_cvt.py \
  --input ./image_floder/bezierpalm/palm \
  --output ./image_floder/palm/testA

cp -r ./image_floder/palm/testA ./image_floder/palm/testB
mkdir -p ./image_floder/palm/trainA 
mkdir -p ./image_floder/palm/trainB 

# Second step: convert Bezier curves images to PCE images.
cd ./CUT
sh ./CUT/inference.sh ${NAME_ID} ${BATCH_SIZE}
cd ../

# augmente PCE images through image transformation.
python3 image_transform.py \
  --input ./datasets/palm${NAME_ID}/val2 \
  --output ./datasets/palm${NAME_ID}/val \
  --sample ${SAMPLE}


# Third step: generate realistic palm images from PCE images.
GPU_ID=0
NAME=palm${NAME_ID}
RES_DIR=./results/${NAME}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./test.py \
  --dataroot ./datasets/palm${NAME_ID} \
  --results_dir ${RES_DIR} \
  --name ${NAME} \
  --model pce \
  --load_size 256 \
  --crop_size 256 \
  --input_nc 1 \
  --num_test 8000000000 \
  --n_samples 1 \
  --gpu_ids ${GPU_ID} \
  --single_test \
  --no_flip \
  --use_dropout 

DPATH=./image_floder/${NAME}

# move files to final folder
python3 ./floder_cvt.py \
  --total_id ${TOTAL_ID} \
  --sample ${SAMPLE} \
  --spath ${RES_DIR} \
  --dpath ${DPATH}


echo "generated datasets are in ${DPATH}"
