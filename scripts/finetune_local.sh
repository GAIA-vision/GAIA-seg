#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
WORK_DIR=$3
LOAD_FROM=$4
MODEL_SPACE_PATH=$5
CKPT_PATH=$6

PORT=${PORT:-29500}



PYTHONPATH="$(dirname $0)/../GAIA-cv-dev":"$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname $0)/../tools"/finetune_supernet.py \
    ${CONFIG} \
    --launcher pytorch \
    --work-dir ${WORK_DIR} \
    --eval mIoU \
    --model-space-path ${MODEL_SPACE_PATH} \
    --load-from ${LOAD_FROM}
