#!/bin/bash
# Tile-heavy SoccerNet pool (+hard negatives, 2x zoom tiles), same per-epoch budget as run_sn.sh; then evaluate.
cd "$(dirname "$0")"; PY=.venv/bin/python; R=../results
until grep -q "^train " $R/sn_data2.log; do sleep 10; done
for try in 1 2; do
  $PY train.py --model cn_mnv3 --run cn_mnv3_sn2 --sn2 --epoch-cap 10300 --epochs 12 --neg-ratio 0.5 --patience 6 --threads 4 --strong-aug --ema 0.998 --wd 0.01 >> $R/cn_mnv3_sn2.log 2>&1 && break
done
$PY evaluate.py --run cn_mnv3_sn2 > $R/cn_mnv3_sn2.eval.log 2>&1
$PY eval_extra.py --sn cn_mnv3_sn2 > $R/cn_mnv3_sn2.sneval.log 2>&1
touch $R/DONE_sn2
