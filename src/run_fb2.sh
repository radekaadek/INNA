#!/bin/bash
# Everything for the FootAndBall-inspired ball/player experiments; safe to re-run (finished steps are skipped, training resumes).
# Run from src/ with the GPU free:   ./run_fb2.sh        (~2.5 h on the GTX 1650; log: ../results/run_fb2.log)
cd "$(dirname "$0")"; PY=.venv/bin/python; R=../results
step() { n=$1; shift; [ -f $R/$n.done ] && return; for t in 1 2 3; do "$@" >> $R/$n.log 2>&1 && { touch $R/$n.done; return; }; sleep 20; done; }
./run_fb.sh                                         # fb_tiny, fb_r18 (FootAndBall recipe; skipped when done)
step mine_neg $PY mine_negatives.py ball_r18 0.2    # false positives of the old ball network -> hard-negative crops
# v2: FootAndBall loss/aug + 2-frame input + mined negatives + backbone lr x0.2 + 30 epochs (~55 min)
step fb_r18_v2 $PY train_fb.py --run fb_r18_v2 --backbone r18 --two-frame --hardneg $R/sn_hardneg.npz --p-hn 0.25 \
     --epochs 30 --lr 1e-3 --bb-lr-mult 0.2 --max-min 75
step eval_fb $PY evaluate_sn_fb.py                  # appends fb_* systems to results/sn_eval.json
step report_sn $PY make_report_sn.py                # regenerates paper/tables/sn_*.tex and figures
echo FB2_DONE > $R/FB2_DONE
