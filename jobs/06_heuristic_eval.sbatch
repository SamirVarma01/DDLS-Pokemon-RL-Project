#!/bin/bash
# Optional: win rate vs the 6 paper heuristics. Mostly CPU-bound; ~0.75 GPU-h. Owner: Benjamin
# One-time, on the login node:
#   module load conda && conda create -y -p $HOME/envs/node -c conda-forge nodejs=20
#   export PATH=$HOME/envs/node/bin:$PATH && cd $METAMON_REPO/server/pokemon-showdown && npm install
#SBATCH -J heur
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
source "$SLURM_SUBMIT_DIR/common.sh"
EP=${EP:?pass the checkpoint: sbatch --export=ALL,EP=8 ...}
export PATH=$HOME/envs/node/bin:$PATH
(cd $METAMON_REPO/server/pokemon-showdown && node pokemon-showdown start --no-security > $SCRATCH/showdown_$SLURM_JOB_ID.log 2>&1 &)
sleep 30
python heuristic_eval.py --run_name bc1_gen1ou --epoch $EP --battles 300
python heuristic_eval.py --pretrained SmallIL --epoch 40 --battles 300
