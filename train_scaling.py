"""
Thin wrapper around `python -m metamon.rl.train` for Perlmutter runs.
Does not modify metamon/amago. Adds:
  --split train         train only on the train split (held-out test split stays clean)
  --steps_per_epoch N   upstream hardcodes 25k; smaller = more checkpoints / shorter profiling jobs
  --attn flash|vanilla  vanilla avoids needing flash-attn installed (keep it FIXED across scaling runs)
  --tf32                allow TF32 matmuls on A100 (speed ablation; off = paper-faithful fp32)
  --max_steps N         stop after N optimizer steps on this rank (hard cap for profiling)
  throughput CSV        $RESULTS_DIR/throughput_<run_name>.csv, written by rank 0

Launch:
  1 GPU : python train_scaling.py <args>
  N GPU : torchrun --nproc_per_node=4 train_scaling.py <args>   (accelerate reads torchrun env vars)
"""
import csv
import os
import sys
import time
from argparse import ArgumentParser

import torch
import wandb

import metamon
from metamon.interface import (
    TokenizedObservationSpace,
    get_observation_space,
    get_reward_function,
    get_action_space,
)
from metamon.tokenizer import get_tokenizer
from metamon.rl.train import add_cli, create_offline_rl_trainer, WANDB_PROJECT, WANDB_ENTITY
from metamon.rl.dataset_config import load_dataset_config, build_dataset
from metamon.rl.metamon_to_amago import MetamonAMAGOExperiment

parser = ArgumentParser()
add_cli(parser)
parser.add_argument("--split", choices=["train", "none"], default="train")
parser.add_argument("--steps_per_epoch", type=int, default=25_000)
parser.add_argument("--attn", choices=["flash", "vanilla"], default="flash")
parser.add_argument("--tf32", action="store_true")
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--tput_warmup", type=int, default=50, help="steps ignored before timing")
parser.add_argument("--tput_window", type=int, default=100, help="steps per throughput sample")

# DataLoader workers can be started with 'spawn', which re-imports this file in each worker.
# Without this guard every worker re-runs training and crashes; only the launcher runs below.
if __name__ == "__main__":
    args = parser.parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------- throughput instrumentation (monkeypatch, rank 0 writes) ----------------
    _orig_train_step = MetamonAMAGOExperiment.train_step
    _T = {"n": 0, "last_n": None, "last_t": None, "csv": None}
    results_dir = os.environ.get("RESULTS_DIR", ".")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"throughput_{args.run_name}.csv")


    def _timed_train_step(self, batch, log_step):
        out = _orig_train_step(self, batch, log_step)
        _T["n"] += 1
        n = _T["n"]
        if n == args.tput_warmup:
            torch.cuda.synchronize()
            _T["last_n"], _T["last_t"] = n, time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
        elif _T["last_n"] is not None and (n - _T["last_n"]) >= args.tput_window:
            torch.cuda.synchronize()
            t = time.perf_counter()
            sps = (n - _T["last_n"]) / (t - _T["last_t"])
            world = self.accelerator.num_processes
            global_batch = self.batch_size * world * self.batches_per_update
            row = {
                "unix_time": round(time.time(), 1),
                "step": n,
                "n_gpus": world,
                "batch_per_gpu": self.batch_size,
                "global_batch": global_batch,
                "steps_per_sec": round(sps, 4),
                "seqs_per_sec": round(sps * global_batch, 2),
                "timesteps_per_sec": round(sps * global_batch * self.max_seq_len, 1),  # upper bound (pads counted)
                "peak_mem_gb_rank0": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "attn": args.attn,
                "tf32": args.tf32,
            }
            if self.accelerator.is_main_process:
                new = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"[tput] {row}", flush=True)
                self.log({k: v for k, v in row.items() if isinstance(v, (int, float))}, key="throughput")
            _T["last_n"], _T["last_t"] = n, t
        if args.max_steps is not None and n >= args.max_steps:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                print(f"[max_steps] reached {n} steps; exiting", flush=True)
            if self.log_to_wandb:
                wandb.finish()
            sys.exit(0)
        return out


    MetamonAMAGOExperiment.train_step = _timed_train_step

    # ---------------- same as metamon/rl/train.py __main__, plus split/steps/attn ----------------
    obs_space = TokenizedObservationSpace(
        get_observation_space(args.obs_space), get_tokenizer(args.tokenizer)
    )
    reward_function = get_reward_function(args.reward_function)
    action_space = get_action_space(args.action_space)

    amago_dataset = build_dataset(
        config=load_dataset_config(args.dataset_config),
        obs_space=obs_space,
        action_space=action_space,
        reward_function=reward_function,
        split=None if args.split == "none" else "train",
    )

    overrides = {}
    if args.attn == "vanilla":
        import amago.nets.transformer as T

        overrides["traj_encoders.TformerTrajEncoder.attention_type"] = T.VanillaAttention

    experiment = create_offline_rl_trainer(
        ckpt_dir=args.save_dir,
        run_name=args.run_name,
        model_gin_config=args.model_gin_config,
        train_gin_config=args.train_gin_config,
        obs_space=obs_space,
        action_space=action_space,
        reward_function=reward_function,
        amago_dataset=amago_dataset,
        eval_gens=args.eval_gens,
        async_env_mp_context=args.async_env_mp_context,
        dloader_workers=args.dloader_workers,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        steps_per_epoch=args.steps_per_epoch,
        batch_size_per_gpu=args.batch_size_per_gpu,
        log=args.log,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        manual_gin_overrides=overrides,
        ckpt_interval=args.ckpt_interval,
    )
    experiment.start()
    if args.ckpt is not None:
        experiment.load_checkpoint(args.ckpt)
    experiment.learn()
    if args.log:
        wandb.finish()
