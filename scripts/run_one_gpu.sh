#!/bin/bash

# Navigate to the workspace folder (replace with your actual workspace path)
cd /path/to/your/workspace

export CUDA_VISIBLE_DEVICES=0
export NCLL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=200
export BATCH_SIZE=your_batch_size(int)
export EPOCHS=yout_epochs(int)
export LR=yout_lr(float)
export WEIGHT_DECAY=your_weight_decay(float)

# Run the Python script with the specified arguments
/your/python eraser/main.py \
  --batch_size $BATCH_SIZE \
  --base_dir /path/to/your/dataset \
  --pretrained_model vit-b-22k \
  --erasing_method lmeraser \
  --test_dataset cifar10 \
  --epochs $EPOCHS \
  --lr $LR \
  --weight_decay $WEIGHT_DECAY \
  --num_gpus 1
