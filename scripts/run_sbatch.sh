#!/bin/bash
#SBATCH --job-name=your_job_name
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  
#SBATCH --output=your_job_name_%j.out
#SBATCH --time=12:00:00               
#SBATCH --mem=102400                     
#SBATCH --partition=your_partition_name

cd /path/to/your/workspace

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=1234
export WORLD_SIZE=8
export NCLL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=200
export BATCH_SIZE=yout_batch_size(int)
export EPOCHS=yout_epochs(int)
export LR=yout_lr(float)
export WEIGHT_DECAY=yout_weight_decay(float)
export OMP_NUM_THREADS=1

load module your_module_name

/your/torchrun \
    --nproc_per_node=8 \
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
    