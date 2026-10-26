# Metamon x Perlmutter — midterm kit

Small-scale Metamon reproduction: Small (15M) behavior cloning on Gen 1 OU human replays, plus 1 → 4 (→ 8) A100 DDP scaling.
Nothing here edits metamon/amago; `train_scaling.py` wraps `metamon.rl.train`.

## One-time setup (login node, no GPU charge)
```bash
cp -r metamon-perlmutter $HOME/ && cd $HOME/metamon-perlmutter
vim common.sh                      # set NERSC_PROJ (and paths if different)
bash setup_login_node.sh           # venv on top of NERSC torch, metamon install, gen1ou data
# optional, faster attention: pip install flash-attn --no-build-isolation
#   if it starts compiling from source instead of grabbing a wheel, Ctrl-C; ATTN falls back to vanilla
```

## Run order (always submit from this folder)
```bash
cd $HOME/metamon-perlmutter && source common.sh
sbatch -A $NERSC_PROJ jobs/00_profile_1gpu.sbatch     # smoke + SmallIL reference + batch sweep
sbatch -A $NERSC_PROJ jobs/01_profile_4gpu.sbatch     # DDP smoke + strong/weak points
# read results/throughput_*.csv, fix B/ACC and -t, then:
sbatch -A $NERSC_PROJ jobs/02_baseline_1gpu.sbatch    # learning curve (resubmit = resume)
sbatch -A $NERSC_PROJ jobs/03_ddp_4gpu.sbatch         # 4-GPU curve + steady-state throughput
sbatch -A $NERSC_PROJ jobs/05_heldout_eval.sbatch     # after 02/03 finish
# optional: jobs/04_smoke_8gpu.sbatch, jobs/06_heuristic_eval.sbatch (EP=<epoch>)
```
If `sbatch` rejects the account, use the GPU account name Iris shows (sometimes `mXXXX_g`).

## Budget (A100-hours; shared QOS bills only the GPUs you take, debug/regular bill the whole node = 4 GPUs)
| Job | Owner | GPU-h |
|---|---|---|
| 00 profile 1 GPU (shared, 45 min) | Samir | 0.75 |
| 01 profile 4 GPU (debug, 25 min) | Samir | 1.7 |
| 04 smoke 8 GPU (debug, 20 min) — optional | Samir | 2.7 |
| 02 baseline 1 GPU (shared, 10 h) | Vicky | 10 |
| 03 DDP 4 GPU (regular, 75 min) | Benjamin | 5 |
| 05 held-out eval (shared, 30 min) | Benjamin | 0.5 |
| 06 heuristic eval — optional | Benjamin | 0.75 |
| **Total** (+~50% retry buffer ≈ 32) | | **~21** |

Check spend: `sacct -X -u $USER -S 2026-10-01 -o JobID,JobName,QOS,NNodes,AllocTRES%40,Elapsed` or Iris.

## Outputs
- `results/throughput_<run>.csv` — steps/s, seqs/s, peak memory, per 100 steps (rank 0)
- `results/heldout_eval.csv` — top-1 / top-2 / NLL on the gen1ou test split (10%, seed 42)
- wandb (`--log` runs) — train loss vs step for 02 and 03
- `$CKPT_DIR/<run>/ckpts/policy_weights/policy_epoch_N.pt` (1 epoch = 5k steps here)

## Gotchas
- `--eval_gens` with no values (in `COMMON_ARGS`) disables in-training Showdown evals. Without it, training tries to battle on a server that isn't running.
- Keep `ATTN` (flash vs vanilla), dataloader workers, and global batch fixed across all scaling comparisons.
- amago DDP uses `find_unused_parameters=True`, which adds overhead at 4–8 GPUs. That's a finding for the report; don't treat it as a bug.
- Test split is per-trajectory, so the two players' views of one battle can land on opposite sides. That leaks a little, so mention it in the report.
- `$SCRATCH` purges files untouched for 8 weeks. Copy final checkpoints you care about to `$CFS` or `$HOME`.
