#!/bin/bash
# Full experiment pipeline on GPU (GTX 1650 4GB). Run from src/: ./run_all.sh  (idempotent: finished runs are skipped)
# Same recipe for every model: pretrained weights, strong augmentation, AdamW (wd=1e-2), EMA weights,
# early stopping on the validation split. The test split is used only for the final evaluation.
cd "$(dirname "$0")"
PY=.venv/bin/python
R=../results
COMMON="--epochs 20 --neg-ratio 0.5 --bs 8 --patience 6 --threads 4 --strong-aug --ema 0.998 --wd 0.01"
MODELS="ssdlite centernet cn_mnv3 cn_mnv4s cn_mnv4m yolo11n yolo26n frcnn"

for m in $MODELS; do
  [ -f $R/$m/done ] && continue
  for try in 1 2; do   # the GPU occasionally hangs (launch failure); train.py resumes from last.pt
    if [[ $m == yolo* ]]; then $PY yolo_train.py $m >> $R/$m.log 2>&1
    else $PY train.py --model $m --run $m $COMMON >> $R/$m.log 2>&1; fi
    rc=$?
    if [ $rc -eq 0 ]; then mkdir -p $R/$m; touch $R/$m/done; break; fi
    tail -3 $R/$m.log | grep -q "CUDA not available" && { echo "GPU lost, stopping at $m" >> $R/pipeline.log; exit 1; }
    sleep 20
  done
done
for m in $MODELS; do [ -f $R/$m/test.json ] || $PY evaluate.py --run $m > $R/$m.eval.log 2>&1; done
for m in centernet cn_mnv3 cn_mnv4s cn_mnv4m; do
  [ -f $R/$m/test_tta.json ] || $PY evaluate.py --run $m --tta > $R/${m}_tta.eval.log 2>&1
done
echo ALL_DONE > $R/DONE
