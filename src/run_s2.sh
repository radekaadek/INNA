#!/bin/bash
# Step 1: zoomed SoccerNet test for the earlier runs.  Step 2: stride-2 heatmap CenterNet-MNv3 (same wall-time budget).
cd "$(dirname "$0")"; PY=.venv/bin/python; R=../results
until grep -q "^testz " $R/sn_testz.log; do sleep 10; done
$PY eval_extra.py --snz cn_mnv3 cn_mnv3_sn cn_mnv3_sn2 > $R/snz_eval_step1.log 2>&1
touch $R/DONE_step1
for try in 1 2; do
  $PY train.py --model cn_mnv3_s2 --run cn_mnv3_s2 --sn2 --epoch-cap 8000 --bs 4 --epochs 12 --neg-ratio 0.5 --patience 6 --threads 4 --strong-aug --ema 0.998 --wd 0.01 >> $R/cn_mnv3_s2.log 2>&1 && break
done
$PY evaluate.py --run cn_mnv3_s2 > $R/cn_mnv3_s2.eval.log 2>&1
$PY eval_extra.py --sn cn_mnv3_s2 > $R/cn_mnv3_s2.sneval.log 2>&1
$PY eval_extra.py --snz cn_mnv3_s2 > $R/cn_mnv3_s2.snzeval.log 2>&1
touch $R/DONE_s2
