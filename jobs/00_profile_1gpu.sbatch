#!/bin/bash
# Smoke test + reference point + per-GPU batch sweep. ~0.75 GPU-h. Owner: Samir
#SBATCH -J prof1
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/common.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv; echo "ATTN=$ATTN"

# 1) Released paper checkpoint on OUR gen1ou test split (all dates, then post-paper-cutoff only)
python heldout_eval.py --pretrained SmallIL --epochs 40 --attn $ATTN --batches 50
python heldout_eval.py --pretrained SmallIL --epochs 40 --attn $ATTN --batches 50 --min_date 2024-10-01

# 2) Throughput vs per-GPU batch size (400 steps each, first 50 ignored). Also finds the OOM ceiling.
for B in 16 32 64; do
  python train_scaling.py --run_name prof1_b${B}_$SLURM_JOB_ID $COMMON_ARGS --attn $ATTN \
    --batch_size_per_gpu $B --dloader_workers 8 --epochs 1 --max_steps 400 || echo "B=$B FAILED (OOM?)"
done
# 3) Same as B=32 with TF32 matmuls (free speed on A100?)
python train_scaling.py --run_name prof1_b32tf32_$SLURM_JOB_ID $COMMON_ARGS --attn $ATTN \
  --batch_size_per_gpu 32 --dloader_workers 8 --epochs 1 --max_steps 400 --tf32
cat $RESULTS_DIR/throughput_prof1_*_$SLURM_JOB_ID.csv
