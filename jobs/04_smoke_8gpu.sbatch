#!/bin/bash
# Optional before midterm: prove 2-node DDP works over Slingshot + first 8-GPU throughput point. ~3 GPU-h. Owner: Samir
#SBATCH -J prof8
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/common.sh"
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export NCCL_DEBUG=INFO   # log should mention the OFI/libfabric (cxi) transport, not plain Socket
srun -N 2 --ntasks-per-node=1 torchrun --nnodes=2 --nproc_per_node=4 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 --rdzv_id=$SLURM_JOB_ID \
  train_scaling.py --run_name prof8_strong64_$SLURM_JOB_ID $COMMON_ARGS --attn $ATTN \
  --batch_size_per_gpu 8 --dloader_workers 8 --epochs 1 --max_steps 400
cat $RESULTS_DIR/throughput_prof8_*_$SLURM_JOB_ID.csv
