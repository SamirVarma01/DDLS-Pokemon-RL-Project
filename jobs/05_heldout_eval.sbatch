#!/bin/bash
# Held-out top-1/top-2/NLL for every saved checkpoint of 02 and 03. ~0.5 GPU-h. Owner: Benjamin
#SBATCH -J heldout
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/common.sh"
for RUN in bc1_gen1ou bc4_gen1ou; do
  EPS=$(ls $CKPT_DIR/$RUN/ckpts/policy_weights 2>/dev/null | sed -E 's/policy_epoch_([0-9]+)\.pt/\1/' | sort -n | tr '\n' ' ')
  [ -n "$EPS" ] && python heldout_eval.py --run_name $RUN --epochs $EPS --attn $ATTN
  LAST=$(echo $EPS | awk '{print $NF}')
  [ -n "$LAST" ] && python heldout_eval.py --run_name $RUN --epochs $LAST --attn $ATTN --min_date 2024-10-01
done
