#!/bin/bash
# One-time setup. Run on a LOGIN node (free; no GPU hours). ~15-30 min mostly downloads.
set -euo pipefail
module load pytorch
python -c "import torch; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); assert v>=(2,6), f'need torch>=2.6, got {torch.__version__}; try module avail pytorch'; print('torch', torch.__version__)"

python -m venv --system-site-packages $HOME/envs/metamon   # inherits NERSC torch
source $HOME/envs/metamon/bin/activate
pip install --upgrade pip

[ -d $HOME/metamon ] || git clone --recursive https://github.com/UT-Austin-RPL/metamon.git $HOME/metamon
pip install -e $HOME/metamon
python -c "import torch; print('torch still', torch.__version__)"   # make sure pip didn't replace NERSC torch

source $(dirname "$0")/common.sh
python -m metamon.data.download parsed-replays --formats gen1ou
python -m metamon.data.download teams --formats gen1ou      # only needed for heuristic evals
echo "Setup done. Optional: pip install flash-attn --no-build-isolation (see README)."
