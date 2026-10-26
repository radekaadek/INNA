#!/bin/bash
# cn_mnv4m ran out of GPU memory with bs=8 on the 4 GB card; retrain with bs=4 and evaluate
cd "$(dirname "$0")"; PY=.venv/bin/python; R=../results
$PY train.py --model cn_mnv4m --run cn_mnv4m --epochs 20 --neg-ratio 0.5 --bs 4 --patience 6 --threads 4 --strong-aug --ema 0.998 --wd 0.01 >> $R/cn_mnv4m.log 2>&1 \
 && $PY evaluate.py --run cn_mnv4m > $R/cn_mnv4m.eval.log 2>&1 && $PY evaluate.py --run cn_mnv4m --tta > $R/cn_mnv4m_tta.eval.log 2>&1
echo done > $R/DONE_mnv4m
