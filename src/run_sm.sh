#!/bin/bash
# Player-driven crop models on SportsMOT football (GTX 1650, fp32). Run from src/: ./run_sm.sh  (finished runs are skipped)
cd "$(dirname "$0")"
PY=.venv/bin/python
R=../results
COMMON="--epochs 20 --patience 8 --max-min 20"
run() {  # run <name> <train_sm args...>
  n=$1; shift
  [ -f $R/$n/done ] && return
  for try in 1 2; do  # the GPU occasionally hangs
    $PY train_sm.py --run $n "$@" $COMMON >> $R/$n.log 2>&1 && { touch $R/$n/done; break; }
    sleep 20
  done
}
run r18_hist --backbone resnet18 --head hist
run r18_reg --backbone resnet18 --head reg
run r18_hist_2f --backbone resnet18 --head hist --two-frame
run mnv3_hist --backbone mobilenetv3 --head hist
run mnv4s_hist --backbone mobilenetv4s --head hist
echo REG_DONE > $R/REG_DONE
