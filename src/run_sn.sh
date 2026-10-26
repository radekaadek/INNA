#!/bin/bash
# Ball + player modules on SN-Tracking-2023 (GTX 1650). Run from src/ after sn_build.py.  Finished runs are skipped.
cd "$(dirname "$0")"; PY=.venv/bin/python; R=../results
step() { n=$1; shift; [ -f $R/$n.done ] && return; for t in 1 2; do "$@" >> $R/$n.log 2>&1 && { touch $R/$n.done; return; }; sleep 20; done; }
step ball_r18 $PY train_ball.py --run ball_r18 --epochs 10 --per-epoch 6000 --max-min 45
step players_sn_r18 $PY train_sm.py --data sn --run players_sn_r18 --backbone resnet18 --head hist --epochs 20 --patience 8 --max-min 15
step yolo_sn_build $PY yolo_sn.py build
step yolo11n_sn $PY yolo_sn.py train 35
touch $R/SN_TRAIN_DONE
