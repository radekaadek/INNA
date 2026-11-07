#!/bin/bash
# StreamBall v2 (heatmap focal loss at stride 4) runs; waits for run_stream.sh (v1) to finish. Safe to re-run.
cd "$(dirname "$0")"; PY=.venv/bin/python; R=../results
step() { n=$1; shift; [ -f $R/$n.done ] && return; for t in 1 2 3; do "$@" >> $R/$n.log 2>&1 && { touch $R/$n.done; return; }; sleep 20; done; }
while [ ! -f $R/STREAM_DONE ]; do sleep 60; done
for cfg in "sy2_single --no-dfp" "sy2_dfp_cur" "sy2_dfp_next --target next" "sy2_single_next --no-dfp --target next"; do
  set -- $cfg; run=$1; shift
  step $run $PY streamball2.py --run $run "$@"
  step eval_$run $PY exp_eval_stream.py $run
done
echo done > $R/STREAM2_DONE
