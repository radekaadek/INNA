#!/bin/bash
# StreamBall (StreamYOLO-style) runs on train3/valid3, then evaluation. Safe to re-run (.done markers). Needs a working GPU; one job at a time.
cd "$(dirname "$0")"; PY=.venv/bin/python; R=../results
step() { n=$1; shift; [ -f $R/$n.done ] && return; for t in 1 2 3; do "$@" >> $R/$n.log 2>&1 && { touch $R/$n.done; return; }; sleep 20; done; }
nvidia-smi > /dev/null 2>&1 || { echo "GPU not available"; exit 1; }
for cfg in "sy_single --no-dfp" "sy_dfp_cur" "sy_dfp_cur_notal --no-tal" "sy_dfp_next --target next" "sy_single_next --no-dfp --target next"; do
  set -- $cfg; run=$1; shift
  step $run $PY streamball.py --run $run "$@"
  step eval_$run $PY exp_eval_stream.py $run
done
echo done > $R/STREAM_DONE
