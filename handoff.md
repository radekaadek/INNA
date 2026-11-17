# Handoff: ball/player network improvements (FootAndBall-inspired, Viterbi tracker, prior channel)

For the next model (or person). Everything is on disk in `/home/ard/prg/INNA`; **nothing is committed to git** (untracked: handoff.md, BACKLOG.md,
notes_viterbi.md, paper/figs, paper/main.pdf, src/exp_*.py, src/train_ball2.py, src/run_b2.sh, src/st_prepare.py, src/tt_*.py; modified: paper/main.tex,
paper/tables/sn_*.tex, src/sn_lib.py). Last updated 2026-10-26 ~16:00.

## Status in one paragraph
The five FootAndBall-style adjustments were run (`run_fb2.sh`, finished) and **did not beat** the original ResNet-18 heatmap network `ball_r18`.
A **Viterbi tracker over the top-K heatmap candidates did help** (test recall 77.6 -> 81.5% offline, 78.9% causal; ball-in-crop 88.9 -> 90.5%).
Candidate-fusion ensembles and flip/scale test-time augmentation gave nothing beyond it. The paper (`paper/main.tex`) is updated with all of this.
**`run_b2.sh` finished and was evaluated** (`results/exp/b2.json`): the prior-map input channel (`b2_prior`) improves the raw network (top-1 recall 74.1 -> 76.6 vs the 20-epoch control `b2_e20`) but gives no gain after Viterbi (79.1 vs 79.4 offline, 77.9 vs 78.2 causal); the 20-epoch control is itself ~2 points worse than the original `ball_r18` + Viterbi (81.5 offline / 78.9 causal). Nothing beats the original `ball_r18` + Viterbi so far. Table in `BACKLOG.md`.

**Update (later on 2026-10-26):** backlog items #2, #3, #5, #6, #7, #10, #12, #14, #15 were run: all null or within noise, see the end of `BACKLOG.md` (summary paragraph). A demo video of test clip SNMOT-149 with the offline Viterbi tracker, crop window and ball box is in `results/clip149/` (`src/render_clip_ball_vit.py`). New scripts: `src/exp_{vit2,reacq,yolo_cands,hires,hires_eval,players,clean,rerank,eval_net}.py`, `src/train_ball_clean.py`.

## Results so far (SN-Tracking-2023 test, 49 clips; hit = ball within 2% of frame width)
| Ball source | Params | GPU/CPU ms | Raw R/P | Tracked R/P (greedy) | Viterbi offline R/P | best val hit |
|---|---|---|---|---|---|---|
| `ball_r18` (heatmap, focal loss) | 11.3 M | 16 / 122 | 76.0 / 70.3 | 77.2 / 71.1 (77.6 / 71.4 with top-10 cands) | 81.5 / 78.8 | 81.0 |
| `fb_tiny` | 0.19 M | 6.5 / 67 | 66.1 / 76.5 | 72.4 / 67.5 | 75.0 | 79.1 |
| `fb_r18` (FB recipe, R18) | 11.3 M | 17 / 134 | 71.9 / 77.0 | 74.7 / 70.0 | 78.0 | 76.5 |
| `fb_r18_v2` (+2-frame, mined negatives, bb-lr x0.2, 30 ep) | 11.3 M | 18 / 144 | 71.7 / 75.5 | 75.3 / 70.3 | 77.1 | 80.8 |
Viterbi variants for `ball_r18`: causal 78.9 / 76.7, lag 4 (0.5 s) 80.9 / 78.3, coverage ~95% (long misses are not bridged).
Crop level (ball+players, EMA): ball-in-crop 88.9 greedy / 90.5 offline / 90.0 lag 4 / 89.0 causal; players in crop 43.1 -> 43.8.
No gain beyond Viterbi: candidate fusion of 2-4 nets (81.1-81.7 R, ~75 P), flip/1.25x-scale TTA (81.5-82.1 R, ~75.6 P, 2-4x compute).
Single seed everywhere; tracker tuned on 6 validation clips; differences of 1-2 points are within noise.
Error analysis (`notes_viterbi.md`): failures = wrong pick with ball in top-10 9.6%, ball not a candidate 6.0%, no estimate 3.0%; worse for small (<8 px),
fast (>30 px/sample), near/inside players; 68% of failed frames are in runs >= 8 samples (track locked on a wrong object).

## What to do next
1. Pick from **`BACKLOG.md`** (14 ideas, each with hypothesis, test and success criterion). Newest: #12 the MOT4MOT-style filter from arXiv 2308.16651
   (`src/exp_mot4mot.py` written, not run: needs scipy or a numpy polyfit), #13 full-resolution ball detector, #14 clean training labels. Others: re-acquisition on low
   confidence, speed-adaptive gate, multi-frame stack, player-aware scoring, tiled high-res pass, confidence-based dropping, stride-3 rebuild, bigger validation set, learned re-ranker.
2. Wire the Viterbi tracker into the real pipeline if it should be used: `evaluate_sn.py` / `evaluate_sn_fb.py` / `make_report_sn.py` still use the greedy
   `sn_lib.track_ball` (paper Tables `tab:ball`, `tab:sn` are greedy; Viterbi numbers: paper Table `tab:vit`, `results/exp/*.json`). The demo renderer
   (`src/render_clip_ball.py`) also uses greedy tracking. Causal Viterbi (lag 0) or fixed lag is needed for live use; offline uses look-ahead.
3. The paper mentions the prior-channel and longer-training results in one sentence; extend if more is tested (builds with `cd paper; tectonic main.tex`, copy to
   `results/paper_main.pdf`). Do not claim anything the results do not show. The FootAndBall bibliography entry (`komorowski2020`) was written from memory: verify it.
   The SoccerNet 2023 report (arXiv 2308.16651) is NOT cited in the paper yet; its ball AP numbers (0.95 / 0.71 on 200 hand-inspected frames, YOLOv8l at 1080p) are not comparable to our recall metric.

## How things were run / environment
`src/.venv/bin/python` (torch 2.6+cu124, ultralytics, timm, opencv). GTX 1650 4 GB: train in **fp32**. Do not run two GPU jobs at once. Do not use `pkill -f` with a pattern
that appears in your own command line. Datasets are research-only (SoccerNet terms; SportsMOT CC BY-NC); `data/` and `results/` are gitignored, do not commit data.
Sandboxed scratch: `/tmp/claude-1000/-home-ard-prg-INNA/<session>/scratchpad`.

Pipelines (all steps are skipped when `results/<step>.done` exists):
- `src/run_fb2.sh` (done): fb_tiny, fb_r18, mine negatives (`results/sn_hardneg.npz`, 5071 false positives of `ball_r18`), `fb_r18_v2`, `evaluate_sn_fb.py` (appends `fb_*`
  systems + latencies to `results/sn_eval.json`; backup of the earlier file: `sn_eval.json.bak`), `make_report_sn.py` (regenerates `paper/tables/sn_*.tex`, `paper/figs/sn_scatter.pdf`).
- `src/run_b2.sh` (done): `b2_e20` / `b2_prior`, evaluated by `src/exp_b2.py` (candidates `results/exp/cands_b2_*.pkl`, result `results/exp/b2.json`).
- Viterbi/analysis scripts (CPU unless noted), results in `results/exp/`: `exp_cands.py` (GPU, caches top-10 candidates `cands_<run>.pkl` for valid/test of ball_r18, fb_tiny,
  fb_r18, fb_r18_v2), `exp_topk.py` (oracle top-K, greedy vs Viterbi offline/lag/causal -> `topk.json`; `topk_offline.json` is the first offline-only run), `exp_ens.py` (fusion ->
  `ens.json`), `exp_policy.py` (crop level -> `policy.json`), `exp_tta.py` (GPU, `tta.json`, `cands_ball_r18_{flip,scale,flip+scale}.pkl`), `exp_errors.py` (-> `errors.json`).

## Key code
`src/sn_fb.py` (FootAndBall-style `FBNet`, `fb_loss`, `FBCrops`, `fb_outputs`), `src/train_fb.py`, `src/mine_negatives.py`, `src/evaluate_sn_fb.py`,
`src/train_ball2.py` (**new**: `BallNet` with optional 4th prior channel, `PriorCrops`, causal `prior_candidates`, EMA, resume via `last.pt`; flags `--prior --epochs --ema`),
`src/sn_lib.py` (`BallNet(pretrained, in_ch=3)` — `in_ch` is new, extra conv1 weights zero-initialised; `track_ball` greedy tracker, `filter_static`, `policy_centers`, metrics),
`src/exp_topk.py` (`viterbi(cands, seq, thr, lam, gate, miss, hold, topk, lag)`), `src/evaluate_sn.py` (`tune_tracker`, `evaluate_policies`), `src/sn_build.py` (dataset cache, `data/sn/`, STRIDE train 4 / test 3).
Data: `data/sn/{train,valid,test}.npz` + `data/sn/frames/*` (896x504 JPEGs).

## Verified vs not verified
Verified: all `run_fb2.sh` steps ran end to end; Viterbi, ensemble, TTA, policy and error scripts ran on the full valid/test sets; `train_ball2.py` smoke-tested with and
without `--prior` (1 epoch, 160 crops) and is training for real; `prior_candidates` runs end to end (used in validation each epoch).
Not verified: that the prior's crop-coordinate Gaussian matches inference (training draws it in crop
coordinates from GT/mined positions with jitter, inference draws it at full-frame coordinates from the model's own top-1 — the same function, but never compared on a
validation-set ablation); frame-gap mismatch for multi-frame/prior models (train stride 4 = 0.16 s, test stride 3 = 0.12 s); whether Viterbi gains hold across seeds.

## Caveats
- Validation = 6 clips (hit rate moves +-3 points between epochs): checkpoint and tracker selection is noisy. The test set was used only for reporting, once per configuration;
  the error analysis was run on test (descriptive, nothing tuned on it).
- Viterbi offline/lag modes use look-ahead (only lag 0 is causal); static-overlay filter (`sn_lib.filter_static`) is offline too and costs ~1 point of recall.
- Greedy baseline differs slightly between the paper's main tables (77.2) and the Viterbi table (77.6): top-5 vs top-10 candidates and re-tuned settings.

## Also unfinished (separate thread, not part of the ball work)
Static-camera evaluation of the existing models on fish-eye panoramas (6500x1000): TeamTrack soccer (3 clips, **done**: `results/tt_eval.json`, videos in `results/tt/`) and
SoccerTrack wide-view (3 clips in `data/soccertrack/`; `results/st/` has detections for 2 of 3 clips; the third, evaluation and rendering were stopped by the user).
Resume: `cd src; TT_DATA=soccertrack .venv/bin/python tt_extract.py F_20200220_1_0330_0360; TT_DATA=soccertrack .venv/bin/python tt_eval.py eval; TT_DATA=soccertrack .venv/bin/python tt_eval.py render yolo_sn`.
SoccerTrack has ball labels (`ball.txt` per clip from `src/st_prepare.py`), TeamTrack does not. A tighter pan-and-tilt window around the pitch band (the 9:16 window at full panorama
height is mostly sky/foreground) was offered but not built.

---
# UPDATE 2026-10-26 (evening): custom network "BallHR" work (backlog #17) — read this first

Plan file: `/home/ard/.claude/plans/please-read-handoff-and-polymorphic-peacock.md` (design, literature, staged ablations hr_a..hr_d). Nothing committed to git.

## Literature (short)
TrackNet V1-V4 (arXiv 1907.03698; frame stacking, V4 motion-attention maps), WASB-SBDT (Tarashima, BMVC 2023, arXiv 2311.05237, nttcom/WASB-SBDT: small HRNet with stride-free stem,
3-in/3-out frames, focal then quality-focal loss; known only from abstracts, PDF not parsed), FoT (Sci. Reports 2025), Kukleva 2019, SoccerNet 2023 MOT4MOT (arXiv 2308.16651, already reproduced, no gain).

## Code written
- `src/ball_hr.py`: `BallHR` (ResNet-18+FPN + full-res stem branch; flags `--out-stride 2|4`, `--temporal` (hr branch gets |gray diffs| to t-1, t-2), `--motion` (head predicting offset to ball at t-1, unit 32 px),
  `--scale-jit`, `--train-split/--valid-split` default train3/valid3, `--smoke`). `candidates()` returns (cands, extra) with predicted previous positions.
- `src/exp_eval_hr.py <run>`: candidates (cache `results/exp/cands_<run>.pkl`), oracle/greedy/Viterbi (tuned on valid3, test once), tunes motion weight `mw`; handles BallNet baselines from `train_ball2.py`. Output `results/exp/net_<run>.json`.
  WARNING: `cands_hr_a.pkl` / `net_hr_a.json` are in the OLD format (hr_a was evaluated with the first version of the script on old valid); do not re-run exp_eval_hr on hr_a without deleting that pkl.
- `src/exp_topk.py`: `viterbi(..., mw=0.0)`: optional motion-consistency link cost (candidates with 5 fields x,y,score,bx,by); default unchanged.
- `src/train_ball2.py`: new `--train-split/--valid-split`. `src/sn_build3.py`: built `data/sn/{train3,valid3}` = stride 3 (= test gap), train3 45 clips (11250 frames), valid3 12 clips (3000 frames; old 6 valid + 6 more). Old train/valid/test untouched.
- `src/run_hr.sh` (re-runnable, `.done` markers): waits for c3_base, eval, smoke test, trains `hr_t4` (temporal) and `hr_t4m` (temporal + motion), evaluates each. **It was running when the laptop was shut down** (hr_t4 at epoch ~2 of 10, ~3.5 min/epoch, ~35 min per run).
  Resume: `cd src; nohup ./run_hr.sh > ../results/run_hr.log 2>&1 &` (hr_t4 restarts from scratch: ball_hr.py has no resume; delete `results/hr_t4` first is not required).

## Results so far (test, 49 clips; Viterbi tuned on valid)
| run | train data | top-1 oracle | oracle top-10 | greedy R/P | Viterbi offline R/P | lag4 | causal |
|---|---|---|---|---|---|---|---|
| `ball_r18` (original) | train (51-6 clips, stride 4) | 76.0 | 90.5 | 77.6 / 71.4 | 81.5 / 78.8 | 80.9 / 78.3 | 78.9 / 76.7 |
| `hr_a` (stride-2 head, hi-res stem) | train | 73.3 | 90.2 | 75.0 / 69.0 | 78.4 / 72.3 | 78.2 / 72.1 | 76.6 / 70.6 |
| `c3_base` (plain BallNet control) | **train3** (45 clips, stride 3) | 73.2 | 88.1 | 73.5 / 67.6 | 79.0 / 72.7 | 78.7 / 72.4 | 77.1 / 71.0 |
- hr_a (val 83.9 vs 81.0) did NOT beat ball_r18 on test (gate failed; also in BACKLOG.md).
- **Important**: the control trained on train3 is itself ~2.5 points below the original `ball_r18` (79.0 vs 81.5 offline), so train3 (fewer clips, stride 3) is a worse training set; compare the new BallHR variants only against `c3_base`, not against the 81.5 number. Note the old val 6 clips are inside valid3, so valid3 numbers are not comparable with older valid numbers either.
- hr_t4 / hr_t4m: pending (training was in progress). Gate: beat c3_base by > 2 pts offline AND causal on test, gains in >30 px/sample bin; hr_t4m: also check `offline_mw0` vs tuned mw (does the motion link cost help).

## Next steps
1. Resume `run_hr.sh`; read `results/exp/net_hr_t4.json`, `net_hr_t4m.json`; compare with the c3_base row above. Append results to BACKLOG.md (#17 section) and `notes`/paper only with what they show.
2. If temporal helps: try `--out-stride 2`, `--scale-jit`, 2 seeds (noise is ~1-2 pts); consider also training the original recipe on train+valid3 old split to separate "data" effects from "model" effects.
3. If nothing beats c3_base: conclude that ball_r18+Viterbi stays best, document the BallHR nulls.

---
# UPDATE 2026-11-07: MOT/SOT backlog (#19-#22) — see BACKLOG.md sections "#19a", "#19b", "#20", "#21"
- #19a ByteTrack-style association on ball_r18 candidates (`src/exp_mot.py`): worse than greedy (test 71.8 / 69.5 R vs 77.6; Viterbi 81.5).
- #19b / #20 YOLO11n(4 ep) and YOLO12n(5 ep, 80 min) fine-tuned ball candidates (`src/yolo_sn2.py`, `src/exp_yolo_eval.py`, `results/exp/yolo2_*.json`): weak alone (oracle top-10 63-66 vs 90.5), merged with ball_r18 Viterbi 82.3-82.4 / 79.2-79.4 offline: within noise. Same-budget YOLO11n control (8 epochs): oracle top-10 69.6, merged 82.6 / 79.7 offline, also noise (BACKLOG.md). RT-DETR skipped (4 GB GPU).
- #21 SUTrack: environment in `/home/ard/prg/sot_ext`, `src/exp_sot.py`, `src/run_sot.sh`; smoke test only (GT-init pure SOT 36.9% recall on one valid clip). Full run needs the user's approval (see BACKLOG.md).
- GPU is back (driver fixed); a `systemd-inhibit` sleep blocker (48 h) was started on 2026-11-07, `pgrep systemd-inhibit`.
