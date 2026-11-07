#!/bin/bash
# BallHR stages on train3/valid3 (stride 3, 12 validation clips). Waits for c3_base (plain BallNet control), then smoke-tests and trains the variants. Safe to re-run.
cd "$(dirname "$0")"; PY=.venv/bin/python; R=../results
step() { n=$1; shift; [ -f $R/$n.done ] && return; for t in 1 2 3; do "$@" >> $R/$n.log 2>&1 && { touch $R/$n.done; return; }; sleep 20; done; }
while pgrep -f "train_ball2.py --run c3_base" > /dev/null; do sleep 30; done
step eval_c3_base $PY exp_eval_hr.py c3_base
$PY ball_hr.py --run smoke_t --temporal --motion --smoke >> $R/smoke.log 2>&1 || exit 1
rm -rf $R/smoke_t
step hr_t4 $PY ball_hr.py --run hr_t4 --temporal
step eval_hr_t4 $PY exp_eval_hr.py hr_t4
step hr_t4m $PY ball_hr.py --run hr_t4m --temporal --motion
step eval_hr_t4m $PY exp_eval_hr.py hr_t4m
echo done > $R/HR_DONE
