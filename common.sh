# Source this on the login node and at the top of every job.
# Edit the three lines marked EDIT.
export NERSC_PROJ=mXXXX                          # EDIT: class allocation account
export METAMON_REPO=$HOME/metamon                 # EDIT if you cloned elsewhere
export MP_DIR=$HOME/metamon-perlmutter            # EDIT: where this folder lives

export METAMON_CACHE_DIR=$SCRATCH/metamon_cache   # datasets + HF checkpoints
export HF_HOME=$SCRATCH/hf_home
export CKPT_DIR=$SCRATCH/metamon_ckpts            # training runs
export RESULTS_DIR=$MP_DIR/results                # small CSVs worth keeping (NOT on scratch)
export METAMON_WANDB_PROJECT=metamon-perlmutter
# export METAMON_WANDB_ENTITY=<your-wandb-team>
export WANDB_MODE=${WANDB_MODE:-online}           # set offline if wandb flakes; `wandb sync` later
mkdir -p "$METAMON_CACHE_DIR" "$HF_HOME" "$CKPT_DIR" "$RESULTS_DIR"

module load pytorch                               # NERSC build: NCCL + Slingshot plugin preconfigured
source $HOME/envs/metamon/bin/activate

# Paper-faithful Small IL settings (match the released SmallIL checkpoint exactly)
export COMMON_ARGS="--model_gin_config small_agent.gin --train_gin_config il.gin \
  --obs_space DefaultObservationSpace --tokenizer allreplays-v3 \
  --action_space MinimalActionSpace --reward_function DefaultShapedReward \
  --dataset_config $MP_DIR/configs/gen1ou_replays.yaml \
  --save_dir $CKPT_DIR --eval_gens"
# ^ trailing `--eval_gens` with no values disables in-training Showdown evals (no server needed)

# Attention backend: flash if installed, else vanilla. Keep it identical across every scaling run.
export ATTN=${ATTN:-$(python -c "import flash_attn" 2>/dev/null && echo flash || echo vanilla)}
export SLURM_CPU_BIND=cores
cd "$MP_DIR"
