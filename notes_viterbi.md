# How the Viterbi tracker improved the ball network's results

Date: 2026-10-26. Nothing was retrained for this: the same trained `ball_r18` heatmap network is used; only the step that turns its
per-frame candidate peaks into a ball track was replaced. Code: `src/exp_topk.py` (`viterbi`, `oracle`), `src/exp_policy.py`,
`src/exp_cands.py` (candidate cache), results in `results/exp/*.json`, paper text in `paper/main.tex` (paragraph "Viterbi tracker",
Table `tab:vit`).

## Problem
The greedy tracker (`sn_lib.track_ball`) takes, per sample, the best candidate (score minus distance penalty) and commits to it.
The strongest heatmap peak is often not the ball: on the test clips the ball is the top-1 peak in 76.0% of visible frames,
in the top 3 in 84.9% and in the top 10 in 90.5% (validation: 81.3 / 90.4 / 95.0%). So there was headroom in picking among candidates.

## Method (`exp_topk.viterbi`)
- Per frame keep up to 5 candidates with score >= `thr` (after static-overlay filtering) plus a MISS state.
- Maximise the sum over the clip of: candidate score (MISS: constant `miss`) minus a move penalty
  `lam * min(d / (gate * (1 + age/4)), 1)`, d = distance (frame-width units, y scaled by 0.5625) from the last real position,
  age = samples since it, gate = 0.25. Max-sum dynamic programming; MISS carries the last real position of its best path.
- A MISS run longer than `hold` outputs no estimate, shorter outputs the held position (confidence 0.01).
- Decoding modes: offline (whole clip), fixed lag L samples (0.12 s each), causal (L=0).
- Parameters (thr, lam, miss, hold; 108 combinations) tuned on the 6 validation clips only, objective = recall + 0.5 * ball-in-crop.
  Chosen for ball_r18: thr 0.1, lam 1.0, miss 0.2, hold 4. Test set was evaluated once per configuration.

## Results (SN-Tracking-2023 test, 49 clips, ball_r18 candidates, hit = within 2% of frame width)
| Tracker | Recall | Precision | Coverage |
|---|---|---|---|
| Greedy (thr 0.1, hold 16) | 77.6 | 71.4 | 100.0 |
| Viterbi causal | 78.9 | 76.7 | 94.6 |
| Viterbi lag 4 (0.5 s) | 80.9 | 78.3 | 95.1 |
| Viterbi offline | 81.5 | 78.8 | 95.1 |

Crop level (ball + players policy, EMA smoothing): ball-in-crop 88.9 -> 90.5 (offline), 90.0 (lag 4), 89.0 (causal);
players in crop 43.1 -> 43.8.
Other networks, offline Viterbi vs greedy tracked recall: fb_tiny 72.4 -> 75.0, fb_r18 74.8 -> 78.0, fb_r18_v2 75.4 -> 77.1.

## Caveats
- Single seed, tracker tuned on 6 validation clips, variance not measured. Gains of 1-4 recall points could partly be noise.
- Offline/lag modes use look-ahead; only the causal mode is usable for live streaming (its gain is smaller).
- Coverage drops to ~95%: long misses are not bridged.
- The greedy baseline here (77.6) differs a little from the earlier table (77.2) because top-10 candidates and re-tuned settings are used.
- No gain beyond the tracker from: candidate-level ensemble of 2-4 networks (`exp_ens.py`), flip/scale test-time augmentation (`exp_tta.py`).

## Reproduce
```
cd src
.venv/bin/python exp_cands.py     # GPU, caches top-10 candidates in results/exp/cands_*.pkl
.venv/bin/python exp_topk.py      # CPU, oracle top-K, greedy vs Viterbi (offline + lags) -> results/exp/topk.json
.venv/bin/python exp_policy.py    # crop-level effect -> results/exp/policy.json
```

## Error analysis (ball_r18 + offline Viterbi, test 11 268 ball-visible frames; `src/exp_errors.py`, `results/exp/errors.json`)
Descriptive only, nothing tuned on it. Outcomes: hit 81.5%; wrong pick although the ball is among the top-10 candidates 9.6%; ball not among the
candidates (network miss) 6.0%; no estimate 3.0%.
Hit rate by condition: ball <8 px wide 72% (8-18 px: 83-86%); ball moving >30 px/sample 62% (>60 px: 27%, n=118) vs 82-88% when slower;
ball within 20 px of a player's box centre 67% vs 87% at 20-50 px; ball inside a player box 71% vs 86% outside; frame rows 25-40% from the
top (far touchline / stands side) 74% vs 83-86% lower; centre columns 78% vs 82-87% at the sides. Per clip: median 86.5%, 3 of 49 clips <50% (worst 46%).
Failures are persistent, not flicker: 68% of failed frames lie in runs of >=8 samples (~1 s), median run 2 samples, 387 runs - the track locks on a wrong object.
Confidence of the chosen candidate: 0.69 when right vs 0.35 when wrong (a usable confidence signal). With no visible ball, 73% of those frames still get an estimate.

## Backlog (from the error analysis, untested) — now maintained in BACKLOG.md
Full list with rationale kept in memory `ball-tracker-ideas` (items 4-8): re-acquisition on low confidence, speed-adaptive gate / multi-frame stack for fast balls,
player-aware or occlusion-aware handling, tiled high-res pass for small balls, confidence-based dropping of uncertain/phantom estimates.

## Prior channel and longer training (`results/exp/b2.json`)
Prior-map input (b2_prior) vs 20-epoch control (b2_e20): raw top-1 recall 76.6 vs 74.1 on test, but after Viterbi 79.1 vs 79.4 offline (77.9 vs 78.2 causal): the tracker already
provides the temporal information. Both are below the original `ball_r18` + Viterbi (81.5 / 78.9 causal). Table in BACKLOG.md.
