"""
`python -m metamon.rl.evaluate` with the two fixes this environment needs; takes the same flags.
  - vanilla attention: the pretrained models are configured for FlashAttention, which isn't installed
  - py312_compat: amago's float traj_save_len vs Python 3.12 randint

  # watch two copies battle on a local Showdown server (start the acceptor first)
  python run_evaluate.py --eval_type challenge --agent SmallIL --username BotA --opponent_username BotB --role acceptor
  python run_evaluate.py --eval_type challenge --agent SmallIL --username BotB --opponent_username BotA --role challenger
"""
from argparse import ArgumentParser

import amago.nets.transformer as T
from metamon.rl.evaluate import __main__ as ev  # safe to import: its CLI is behind a __main__ guard

import py312_compat  # noqa: F401  (amago float traj_save_len vs Python 3.12 randint)

_get_pretrained_model = ev.get_pretrained_model


def _vanilla_attention(name):
    model = _get_pretrained_model(name)
    # keep the model's own overrides (tokenizer etc.), only swap the attention type
    overrides = {k: v for k, v in (model.gin_overrides or {}).items() if not k.endswith("attention_type")}
    overrides["traj_encoders.TformerTrajEncoder.attention_type"] = T.VanillaAttention
    model.gin_overrides = overrides
    return model


if __name__ == "__main__":
    ev.get_pretrained_model = _vanilla_attention  # _run_default_evaluation looks it up in ev's namespace
    args = ev.add_cli(ArgumentParser()).parse_args()
    ev._run_default_evaluation(args)
