#!/bin/bash
# Same recipe as 02 on 4 GPUs (16/GPU = global 64). Gives steady-state strong-scaling throughput
# AND a loss curve to overlay on 02 (same global batch => curves should match per step). ~5 GPU-h. Owner: Benjamin
#SBATCH -J bc4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH -t 01:15:00
#SBATCH -o logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/common.sh"
torchrun --standalone --nproc_per_node=4 train_scaling.py --run_name bc4_gen1ou $COMMON_ARGS --attn $ATTN --log \
  --batch_size_per_gpu 16 --dloader_workers 8 --steps_per_epoch 5000 --epochs 200 --ckpt_interval 1
