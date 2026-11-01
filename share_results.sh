#!/bin/bash
# Copy YOUR checkpoints + result CSVs into the project's shared folder, so one job 05 can evaluate
# everyone's runs and one person can make the combined charts. Login node, no GPU charge.
# Run it after your training job finishes, and again after any job 05 / profiling job you ran.
#   bash share_results.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
SHARED_DIR=${SHARED_DIR:-/global/cfs/cdirs/${NERSC_PROJ%_g}/metamon-perlmutter}   # m4431_g -> m4431
if ! mkdir -p "$SHARED_DIR/ckpts" "$SHARED_DIR/results" 2>/dev/null; then
  echo "Can't write to $SHARED_DIR. Check: ls -ld /global/cfs/cdirs/${NERSC_PROJ%_g}" >&2
  exit 1
fi
umask 002   # teammates need to read your files and write into the shared folders

# Only policy weights: that's all heldout_eval.py loads (training states are ~3x bigger)
for RUN in bc1_gen1ou bc4_gen1ou; do
  SRC=$CKPT_DIR/$RUN/ckpts/policy_weights
  [ -d "$SRC" ] || continue
  mkdir -p "$SHARED_DIR/ckpts/$RUN/ckpts"
  cp -r "$SRC" "$SHARED_DIR/ckpts/$RUN/ckpts/"
  chmod -R g+rwX "$SHARED_DIR/ckpts/$RUN"
  echo "shared $RUN: $(ls "$SRC" | wc -l) checkpoints"
done

# Throughput CSVs have unique run names; every heldout_eval.csv has the same name, so tag it
for f in "$RESULTS_DIR"/throughput_*.csv; do
  [ -e "$f" ] || continue
  cp "$f" "$SHARED_DIR/results/" && chmod g+rw "$SHARED_DIR/results/$(basename "$f")"
  echo "shared $(basename "$f")"
done
if [ -f "$RESULTS_DIR/heldout_eval.csv" ]; then
  cp "$RESULTS_DIR/heldout_eval.csv" "$SHARED_DIR/results/heldout_eval_$USER.csv"
  chmod g+rw "$SHARED_DIR/results/heldout_eval_$USER.csv"
  echo "shared heldout_eval.csv as heldout_eval_$USER.csv"
fi
chmod g+rwX "$SHARED_DIR" "$SHARED_DIR/ckpts" "$SHARED_DIR/results" 2>/dev/null || true
echo "Done: $SHARED_DIR"
