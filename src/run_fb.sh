#!/bin/bash
# FootAndBall-style joint ball+player networks (tiny from scratch; ResNet-18 backbone). Run from src/.
cd "$(dirname "$0")"; PY=.venv/bin/python; R=../results
step() { n=$1; shift; [ -f $R/$n.done ] && return; for t in 1 2; do "$@" >> $R/$n.log 2>&1 && { touch $R/$n.done; return; }; sleep 20; done; }
step fb_tiny $PY train_fb.py --run fb_tiny --backbone tiny --epochs 20 --max-min 30
step fb_r18 $PY train_fb.py --run fb_r18 --backbone r18 --epochs 12 --lr 5e-4 --max-min 35
touch $R/FB_TRAIN_DONE
