"""
Python 3.12 workaround for amago (pinned v3.4.0). Import before building any experiment.

amago's Experiment.traj_save_len defaults to 1e10 (a float), which flows into
SequenceWrapper.save_every and then random.randint(). Python <= 3.11 accepted integral
floats there; 3.12 raises "TypeError: 'float' object cannot be interpreted as an integer"
on the first env reset. Patching here (not in amago) keeps metamon/amago unmodified.

Import at module level, outside any __main__ guard, so spawned env/dataloader workers
(which re-import the launching script) get the patch too.
"""
import random

from amago.envs.amago_env import SequenceWrapper


def _random_traj_length(self):
    return random.randint(*(int(x) for x in self.save_every)) if self.save_every else None


SequenceWrapper.random_traj_length = _random_traj_length
