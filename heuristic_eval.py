"""
Optional: win rate vs the paper's 6 heuristic baselines (the "Heuristic Composite Score", Fig. 6).
Needs a local Showdown server running (jobs/06_heuristic_eval.sbatch starts one).

  python heuristic_eval.py --run_name bc1_gen1ou --epoch 8 --battles 300
  python heuristic_eval.py --pretrained SmallIL --epoch 40 --battles 300
"""
import json
import os
from argparse import ArgumentParser

from metamon.env import get_metamon_teams
from metamon.interface import get_observation_space, get_action_space, get_reward_function
from metamon.tokenizer import get_tokenizer
from metamon.rl import LocalPretrainedModel, pretrained_vs_baselines
from metamon.rl.pretrained import get_pretrained_model

import py312_compat  # noqa: F401  (amago float traj_save_len vs Python 3.12 randint)

p = ArgumentParser()
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--run_name")
g.add_argument("--pretrained")
p.add_argument("--ckpt_dir", default=os.environ.get("CKPT_DIR"))
p.add_argument("--epoch", type=int, required=True)
p.add_argument("--battles", type=int, default=300)
p.add_argument("--format", default="gen1ou")
p.add_argument("--out_dir", default=os.environ.get("RESULTS_DIR", "."))
args = p.parse_args()

if args.pretrained:
    model, label = get_pretrained_model(args.pretrained), args.pretrained
else:
    model = LocalPretrainedModel(
        amago_ckpt_dir=args.ckpt_dir,
        model_name=args.run_name,
        model_gin_config="small_agent.gin",
        train_gin_config="il.gin",
        default_checkpoint=args.epoch,
        tokenizer=get_tokenizer("allreplays-v3"),
        observation_space=get_observation_space("DefaultObservationSpace"),
        action_space=get_action_space("MinimalActionSpace"),
        reward_function=get_reward_function("DefaultShapedReward"),
    )
    label = args.run_name

results = pretrained_vs_baselines(
    pretrained_model=model,
    battle_format=args.format,
    team_set=get_metamon_teams(args.format, "competitive"),
    checkpoint=args.epoch,
    total_battles=args.battles,
    battle_backend=model.battle_backend,
)
print(results)
out = os.path.join(args.out_dir, f"heuristic_{label}_ep{args.epoch}_{args.format}.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print("wrote", out)
