#!/bin/bash
# Held-out top-1/top-2/NLL for every saved checkpoint of 02 and 03. ~0.5 GPU-h. Owner: Benjamin
# Uses your own $CKPT_DIR for a run if it's there, else the copy teammates put in the shared
# folder with share_results.sh (02 and 03 are trained under different users' $SCRATCH).
#SBATCH -J heldout
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/common.sh"
SHARED_CKPTS=${SHARED_DIR:-/global/cfs/cdirs/${NERSC_PROJ%_g}/metamon-perlmutter}/ckpts
for RUN in bc1_gen1ou bc4_gen1ou; do
  D=$CKPT_DIR; [ -d $D/$RUN/ckpts/policy_weights ] || D=$SHARED_CKPTS
  EPS=$(ls $D/$RUN/ckpts/policy_weights 2>/dev/null | sed -E 's/policy_epoch_([0-9]+)\.pt/\1/' | sort -n | tr '\n' ' ')
  [ -z "$EPS" ] && { echo "no checkpoints for $RUN (own or shared), skipping"; continue; }
  echo "$RUN: epochs $EPS from $D"
  python heldout_eval.py --run_name $RUN --ckpt_dir $D --epochs $EPS --attn $ATTN
  LAST=$(echo $EPS | awk '{print $NF}')
  python heldout_eval.py --run_name $RUN --ckpt_dir $D --epochs $LAST --attn $ATTN --min_date 2024-10-01
done
