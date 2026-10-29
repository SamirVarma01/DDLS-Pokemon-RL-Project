#!/bin/bash
# DDP smoke test on one full node: strong (global 64) + weak (32/GPU) scaling points. ~1.7 GPU-h. Owner: Samir
#SBATCH -J prof4
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH -t 00:25:00
#SBATCH -o logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/common.sh"
# venv python, not bare `torchrun` (module script -> workers use module python, no metamon)
L="python -m torch.distributed.run --standalone --nproc_per_node=4 train_scaling.py"
# strong scaling: same global batch as the 1-GPU baseline (64 = 4 x 16)
$L --run_name prof4_strong64_$SLURM_JOB_ID $COMMON_ARGS --attn $ATTN \
   --batch_size_per_gpu 16 --dloader_workers 8 --epochs 1 --max_steps 400
# weak scaling: same per-GPU batch as the 1-GPU sweep (global 128 = 4 x 32)
$L --run_name prof4_weak32_$SLURM_JOB_ID $COMMON_ARGS --attn $ATTN \
   --batch_size_per_gpu 32 --dloader_workers 8 --epochs 1 --max_steps 400
cat $RESULTS_DIR/throughput_prof4_*_$SLURM_JOB_ID.csv
