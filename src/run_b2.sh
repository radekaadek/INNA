#!/bin/bash
# Ball heatmap: 20-epoch control, and 20 epochs with prior channel (both with EMA weights). Safe to re-run (resumes). ~40 min each on the GTX 1650.
cd "$(dirname "$0")"; PY=.venv/bin/python; R=../results
step() { n=$1; shift; [ -f $R/$n.done ] && return; for t in 1 2 3; do "$@" >> $R/$n.log 2>&1 && { touch $R/$n.done; return; }; sleep 20; done; }
step b2_e20 $PY train_ball2.py --run b2_e20 --epochs 20 --max-min 70
step b2_prior $PY train_ball2.py --run b2_prior --prior --epochs 20 --max-min 70
echo done > $R/B2_DONE
