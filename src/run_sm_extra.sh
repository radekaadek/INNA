#!/bin/bash
# Extra seeds of the best-structured model for a variance estimate; starts after the YOLO run. Run from src/.
cd "$(dirname "$0")"; R=../results
until [ -f $R/yolo11n_sm.done ]; do sleep 20; done
for s in 1 2; do
  [ -f $R/r18_hist_s$s/done ] && continue
  .venv/bin/python train_sm.py --run r18_hist_s$s --backbone resnet18 --head hist --seed $s --epochs 20 --patience 8 --max-min 20 >> $R/r18_hist_s$s.log 2>&1 && touch $R/r18_hist_s$s/done
done
touch $R/EXTRA_DONE
