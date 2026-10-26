#!/bin/bash
# YOLO11n player detector, started after the regressors finish (single GPU). Run from src/.
cd "$(dirname "$0")"; R=../results
until [ -f $R/REG_DONE ]; do sleep 20; done
for try in 1 2; do
  .venv/bin/python yolo_sm.py train 25 >> $R/yolo11n_sm.log 2>&1 && { touch $R/yolo11n_sm.done; break; }
  sleep 20
done
