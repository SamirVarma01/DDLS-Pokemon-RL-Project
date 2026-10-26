#!/bin/bash
# Midterm learning curve: Small BC on gen1ou, 1 GPU, global batch 64. Budget = the -t limit. Owner: Vicky
# Resubmitting the same script resumes from the last checkpoint.
#SBATCH -J bc1
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/common.sh"
RUN=bc1_gen1ou
B=${B:-64}; ACC=${ACC:-1}   # if B=64 OOMed in profiling: sbatch --export=ALL,B=32,ACC=2 ...
LAST=$(ls $CKPT_DIR/$RUN/ckpts/policy_weights 2>/dev/null | sed -E 's/policy_epoch_([0-9]+)\.pt/\1/' | sort -n | tail -1)
python train_scaling.py --run_name $RUN $COMMON_ARGS --attn $ATTN --log \
  --batch_size_per_gpu $B --grad_accum $ACC --dloader_workers 8 \
  --steps_per_epoch 5000 --epochs 200 --ckpt_interval 1 ${LAST:+--ckpt $LAST}
