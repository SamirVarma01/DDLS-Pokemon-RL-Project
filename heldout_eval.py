"""
Held-out action-prediction metrics on the gen1ou TEST split (same seeded split train_scaling.py trains on).
Reports top-1 / top-2 accuracy and NLL of the human action, ignoring padded and "missing" actions.

  # our run, several checkpoints
  python heldout_eval.py --run_name bc1_gen1ou --epochs 2 4 6 8
  # released paper checkpoint as a reference point (downloads from HF)
  python heldout_eval.py --pretrained SmallIL --epochs 40
  # paper-fair comparison: only battles after the paper's Sep-2024 data cutoff
  python heldout_eval.py --pretrained SmallIL --epochs 40 --min_date 2024-10-01

Sequences are sampled randomly (seeded) from the test split and sliced to max_seq_len=200,
so this is an estimate; --batches 100 x --batch_size 32 = 3,200 sequences is plenty for a midterm.
"""
import csv
import os
import random
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from amago.loading import RLData_pad_collate, MAGIC_PAD_VAL
from metamon.data import ParsedReplayDataset
from metamon.interface import get_observation_space, get_action_space, get_reward_function
from metamon.tokenizer import get_tokenizer
from metamon.rl.metamon_to_amago import MetamonAMAGODataset
from metamon.rl.pretrained import LocalPretrainedModel, get_pretrained_model

import py312_compat  # noqa: F401  (amago float traj_save_len vs Python 3.12 randint)

p = ArgumentParser()
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--run_name")
g.add_argument("--pretrained", help="e.g. SmallIL")
p.add_argument("--ckpt_dir", default=os.environ.get("CKPT_DIR"))
p.add_argument("--epochs", type=int, nargs="+", required=True)
p.add_argument("--format", default="gen1ou")
p.add_argument("--batches", type=int, default=100)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--min_date", default=None, help="YYYY-MM-DD; filters the test split AFTER splitting")
p.add_argument("--attn", choices=["flash", "vanilla"], default="flash")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--out", default=os.path.join(os.environ.get("RESULTS_DIR", "."), "heldout_eval.csv"))
args = p.parse_args()

# --- model (same obs/tokenizer/action space as the paper's Small models) ---
spaces = dict(
    tokenizer=get_tokenizer("allreplays-v3"),
    observation_space=get_observation_space("DefaultObservationSpace"),
    action_space=get_action_space("MinimalActionSpace"),
    reward_function=get_reward_function("DefaultShapedReward"),
)
if args.pretrained:
    model = get_pretrained_model(args.pretrained)
    label = args.pretrained
else:
    model = LocalPretrainedModel(
        amago_ckpt_dir=args.ckpt_dir,
        model_name=args.run_name,
        model_gin_config="small_agent.gin",
        train_gin_config="il.gin",
        default_checkpoint=args.epochs[0],
        **spaces,
    )
    label = args.run_name
if args.attn == "vanilla":
    import amago.nets.transformer as T

    model.gin_overrides = {"traj_encoders.TformerTrajEncoder.attention_type": T.VanillaAttention}

experiment = model.initialize_agent(checkpoint=args.epochs[0], log=False)
policy, device = experiment.policy, experiment.DEVICE
policy.eval()

# --- data: test split built exactly like training (no filters before the split) ---
dset = ParsedReplayDataset(
    observation_space=model.observation_space,  # already tokenized by the model wrapper
    action_space=model.action_space,
    reward_function=model.reward_function,
    formats=[args.format],
    split="test",
    verbose=True,
)
if args.min_date:
    cutoff = datetime.strptime(args.min_date, "%Y-%m-%d")

    def _date(fname):
        parts = os.path.basename(fname).split(".json")[0].split("_")
        if len(parts) not in (7, 8):
            return None
        try:
            return dset._parse_date(parts[-2])
        except ValueError:
            return None

    dset.filenames = [f for f in dset.filenames if (d := _date(f)) is not None and d >= cutoff]
    print(f"min_date {args.min_date}: {len(dset.filenames)} test battles remain")
assert len(dset.filenames) > 0, "empty test set"

amago_dset = MetamonAMAGODataset(dset_name="test", parsed_replay_dset=dset)
amago_dset.configure(
    items_per_epoch=args.batches * args.batch_size,
    max_seq_len=experiment.max_seq_len,
    padded_sampling="none",
    has_edit_rights=False,
)


@torch.no_grad()
def evaluate(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    loader = DataLoader(
        amago_dset, batch_size=args.batch_size, num_workers=args.workers,
        collate_fn=RLData_pad_collate,  # torch seeds each worker's `random` from manual_seed
    )
    n = top1 = top2 = 0
    nll = 0.0
    for batch in loader:
        batch = batch.to(device)
        o = policy.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
        s_rep, _ = policy.traj_encoder(seq=o, time_idxs=batch.time_idxs, hidden_state=None)
        dist = policy.actor(s_rep, straight_from_obs={"illegal_actions": batch.obs["illegal_actions"]})
        probs = dist.probs[:, :-1, -1, :]  # (B, L-1, A); last gamma head (only one for IL)
        target = batch.actions.argmax(-1)  # (B, L-1)
        not_pad = ~(batch.rl2s == MAGIC_PAD_VAL).all(-1)[:, 1:]
        not_missing = ~batch.obs["missing_action_mask"][:, :-1, 0].bool()
        mask = not_pad & not_missing
        top = probs.topk(2, dim=-1).indices
        top1 += ((top[..., 0] == target) & mask).sum().item()
        top2 += ((top == target.unsqueeze(-1)).any(-1) & mask).sum().item()
        nll += (-torch.log(probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)) * mask).sum().item()
        n += mask.sum().item()
    return dict(top1=top1 / n, top2=top2 / n, nll=nll / n, n_actions=n)


rows = []
for ep in args.epochs:
    if ep != args.epochs[0]:
        state = torch.load(model.get_path_to_checkpoint(ep), map_location="cpu")
        policy.load_state_dict(state, strict=True)
        policy.on_checkpoint_loaded(is_resume=False)
    m = evaluate(args.seed)
    row = dict(model=label, epoch=ep, format=args.format, min_date=args.min_date or "", **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
    print(row, flush=True)
    rows.append(row)

new = not os.path.exists(args.out)
with open(args.out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    if new:
        w.writeheader()
    w.writerows(rows)
print(f"appended {len(rows)} rows to {args.out}")
