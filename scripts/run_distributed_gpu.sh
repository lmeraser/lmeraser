#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=1234
export WORLD_SIZE=8
export NCLL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=200
# set your batch_size, epochs, lr, weight_decay
export BATCH_SIZE=your_batch_size(int)
# local batch size is $BATCH_SIZE / $WORLD_SIZE
export EPOCHS=your_epochs(int)
export LR=your_lr(float)
export WEIGHT_DECAY=your_weight_decay(float)
export OMP_NUM_THREADS=1

# Navigate to the workspace folder (replace with your actual workspace path)
cd /path/to/your/workspace

# Run the Python script with the specified arguments using distributed launch
/your/torchrun \
    --nproc_per_node 8 \
    eraser/main.py \
    --batch_size $BATCH_SIZE \
    --base_dir /path/to/your/dataset \
    --pretrained_model vit-b-22k \
    --erasing_method lmeraser \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --test_dataset cifar100 \
    --num_gpus 8 \
    --num_workers 4 \
    --distributed
