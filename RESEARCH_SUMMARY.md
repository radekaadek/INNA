# Ball detection and tracking for 9:16 football cropping: research summary

Project: *Analiza metod kadrowania meczów piłkarskich z wykorzystaniem uczenia maszynowego*. Goal: automatically cut a vertical (9:16) window out of a horizontal (16:9) football broadcast, so that the video can be watched on a phone.

This document covers only the **ball**: how we find it in each frame, how we follow it over time, and how well that works. The ball matters because a vertical window shows only about a third of the picture, so it has to be placed where the ball is.

---
## 0. Summary in five sentences
1. A ResNet-18 network produces, for every frame, a "heatmap" in which bright peaks are possible ball locations; it finds the ball with its strongest peak in 76 % of the frames where the ball is visible.
2. The strongest peak is often a look-alike (a boot, a sock, a logo, a white line mark), but the true ball is among the 10 strongest peaks in 90.5 % of the frames.
3. A **Viterbi tracker** chooses one peak per frame so that the whole sequence is the most plausible ball path. This lifts the correct-ball rate from 77.6 % to 81.5 % (78.9 % if no future frames are allowed) without retraining anything.
4. With the ball-centred window, the ball is inside the 9:16 crop in about 89–90 % of the frames, against 53 % for a fixed centre crop.
5. Over 20 other ideas (extra inputs, other networks, other trackers, more resolution, more detectors) gave no improvement beyond this tracker that is larger than measurement noise.

---
## 1. Task, data, evaluation and hardware

### 1.1 What is being solved
- A broadcast frame is 16:9. The vertical window has the full frame height and 9/16 of that height as width, which is **31.6 % of the frame width**. The only free choice is the horizontal position of the window.
- For the ball experiments, the question is: *where in the frame is the ball in every frame of the video?* The crop position is then derived from it.
- The ball is tiny: in the 1080p frames it is often only 5–15 pixels wide. It moves fast (tens of pixels per sample), is hidden by players, and many things look like it.

### 1.2 Data: SoccerNet SN-Tracking-2023
- Public football tracking dataset (Cioppa et al., 2022, see References). Broadcast clips of 30 s, 1920x1080, 25 frames per second. The ball has a bounding box in every frame where it is visible.
- Split used: **51 clips for training, 6 for validation** (choosing settings and checkpoints), **49 official test clips** (used only for the reported numbers).
- All frames were shrunk to 896x504 pixels (the network's input size). The code keeps every 4th frame for training/validation and every 3rd frame for testing. A test "sample" is therefore 3 frames = 0.12 s apart, giving 12 250 test frames, of which 92 % show the ball. Everything below is computed **only on frames where the ball is visible** unless stated otherwise.
- Licence: research use only (SoccerNet terms). The data folder is not committed.

### 1.3 Metrics (all defined once here)
- **Hit**: the estimated ball position is within **2 % of the frame width** of the annotated ball centre (about 38 px in 1080p, 18 px in the 896-wide frame). A ball is only a few pixels wide, so the tolerance is a few ball diameters.
- **Recall**: of all frames where the ball is visible, the share with a *hit*. This is the main number: "how often do we know where the ball is".
- **Precision**: of all frames where the system *gave* an estimate, the share that are a hit. A system that sometimes says "no ball" can have higher precision than recall.
- **Coverage**: the share of frames for which the system gave any estimate.
- **Top-1 / top-3 / top-10 recall (oracle)**: a diagnostic that asks whether the correct ball is *among* the *k* strongest peaks of the heatmap, ignoring which one would be chosen. It is the upper limit for any method that chooses between peaks.
- **Ball in crop**: the share of ball-visible frames in which the true ball centre lies inside the final 9:16 window. This is the quantity that actually matters for viewers. The window position is smoothed over time (exponential moving average with a limit on how far it may move per sample) so the virtual camera does not jump.
- **Causal / offline / lag**: a method is *causal* if the estimate for frame *t* uses only frames up to *t* (needed for live use). *Offline* methods may look at the entire clip, i.e. at future frames. *Lag L* means the estimate for frame *t* is produced after seeing *L* more samples (L = 4 is 0.5 s of delay).
- **Raw**: the strongest heatmap peak taken directly, no tracking.

### 1.4 Hardware and how latency was measured
- **GPU**: NVIDIA GeForce GTX 1650 Mobile, 4 GB memory. All networks were trained in 32-bit floating point (16-bit mixed precision was about 4x *slower* on this card). Only one GPU job can run at a time. The driver was broken for a few days in early November and is reported by the system as unavailable at the time of writing; the documented numbers below were measured before.
- **CPU**: **AMD Ryzen 5 5600H** laptop processor (Zen 3 architecture, 7 nm class, 45 W class). 6 physical cores / **12 hardware threads**, boost clock up to 4.28 GHz, 3 MiB L2 cache, 16 MiB shared L3 cache. Supports AVX2 and FMA vector instructions; **no AVX-512**. It has an integrated Radeon graphics unit, which is not used. 27 GiB of RAM.
- **Software**: PyTorch 2.6 (CUDA 12.4 build), Ultralytics for YOLO, timm for backbones, OpenCV; Python virtual environment `src/.venv`.
- **How CPU latency was measured**: the code `src/evaluate_sn.py` sets `torch.set_num_threads(4)`, so CPU timings use **4 of the 12 threads**, batch size 1, one 504x896 frame, 32-bit floats. The reported value is the **median of 15 timed runs after 3 warm-up runs**. GPU latency: median of 40 runs after 8 warm-ups, with explicit synchronisation. These are timings of the neural network only, without video decoding or tracking.

---
## 2. Methods

### 2.1 The ball network (`ball_r18`)
- **Backbone**: ResNet-18 (He et al., 2016) pre-trained on ImageNet, 11.3 M parameters. A light "top-down" decoder merges the four feature stages (lateral 1x1 convolutions, upsampling by repetition, addition).
- **Output**: a **heatmap** at 1/4 of the input resolution (a 126x224 grid for a 504x896 frame). The value in each cell is the estimated probability that the ball centre is in that cell. One ball means one bright blob.
- **Training target**: a small Gaussian blob (sigma 1.5 cells) placed at the annotated ball position, trained with the **focal loss** used by CenterNet (Zhou et al., 2019; Lin et al., 2017), which handles the extreme imbalance between a few ball cells and tens of thousands of background cells.
- **Training recipe**: random 288x512 crops of the training frames (the ball is inside the crop with probability 0.75), horizontal flips, brightness/contrast jitter, AdamW optimiser, 10 epochs of 6 000 crops (about 18 minutes).
- **Candidates**: at test time the whole frame is processed and the strongest **local maxima** of the heatmap are extracted. Each candidate has a position and a **score** (the heatmap value at the peak, between 0 and 1). The experiments keep the 10 strongest; the tracker uses up to 5.
- **Static overlay filter**: peaks that fall into the same image cell in more than 40 % of a clip's samples (TV logos, score graphics) are removed. This uses statistics of the whole clip, so it is an offline step. It cost about 1 point of recall on the test set but removes false balls from logos.

### 2.2 Baseline tracker: greedy
For every sample, take the candidate above a score threshold that maximises *score minus a penalty proportional to the distance from the last accepted position*. Jumps farther than a gate (25 % of the frame width, growing the longer the ball has been lost) to candidates with a score below 0.8 are rejected. If nothing is accepted, hold the last position for a few samples.
Weakness: **each decision is final.** A wrong choice early on pulls all later choices towards the wrong object.

### 2.3 The Viterbi tracker (the main improvement)

**Idea.** Instead of deciding frame by frame, treat the whole clip as one problem: among all possible sequences "pick one candidate (or none) in every frame", find the sequence with the best total quality. A sequence is good when it uses high-scoring candidates and when consecutive positions are close to each other (a ball cannot teleport). The **Viterbi algorithm** (dynamic programming) finds that best sequence exactly and efficiently, without trying all combinations.

**Setup per frame (sample).**
- *States*: up to 5 heatmap candidates with score >= threshold `thr`, plus one special state **MISS** meaning "the ball is not found in this frame" (occluded, out of view, or no usable candidate).
- *Reward* for choosing a candidate: its heatmap score (between 0 and 1). Reward for MISS: a constant `miss` (0.2 for `ball_r18`). So MISS wins only when no candidate is better than 0.2, or when taking the candidates would require large jumps.
- *Cost of moving* between consecutive frames: `lam * min(d / (gate * (1 + age/4)), 1)`.
  - `d` = distance between the new candidate and the **last real (non-MISS) position** on that path, measured in units of the frame width (vertical distances are scaled by 0.5625, the aspect ratio).
  - `age` = number of samples since that last real position. A longer gap makes the allowed distance larger, because the ball could have travelled further.
  - `gate` = 0.25 (a quarter of the frame width). The cost grows linearly up to the full value `lam` (1.0) at that distance, and stays at `lam` beyond it. Small movements are almost free, a jump across a quarter of the picture costs a full point, which a single candidate score (<= 1) can hardly pay back.
- A path through MISS keeps remembering the last real position it came from, so the ball can reappear near where it vanished at no penalty beyond the elapsed time.

**Algorithm.**
1. For the first frame, each state's best total is just its reward.
2. For every later frame and every state *s*, look at all states *s'* of the previous frame and compute `best total(s') + reward(s) - move cost(s' -> s)`. Keep the maximum, and remember which *s'* it came from.
3. After the last frame, take the state with the highest total and follow the stored "came from" links backwards. This gives one state per frame: the chosen ball position, or MISS.
4. Post-processing: a run of MISS frames of up to `hold` (4) samples is filled with the last known position (marked as very low confidence). Longer runs produce **no estimate**, which is why coverage is about 95 %, not 100 %.

With about 6 states per frame the cost is 36 comparisons per frame, negligible next to running the network (not timed separately; it is plain arithmetic on a few numbers per frame).

**Why it is better than greedy: a worked toy example** (made-up numbers only to illustrate the mechanism; `lam` = 1, `miss` = 0.2).
- Five frames. The real ball B has score 0.7 and moves slowly in every frame. A look-alike L at the other side of the pitch has score 0.8 but is only visible in frames 1–3.
- *Greedy*: in frame 1 it picks L (0.8 > 0.7), since there is no earlier position to compare against. In frames 2 and 3 B is a far jump from L with a lower score, so it is rejected. The tracker stays locked on L until the gap and the age-widened gate finally allow a jump, several frames after L has disappeared.
- *Viterbi*, total over the five frames:
  - B in every frame: 0.7 x 5 = **3.5**
  - L in frames 1–3, then B: 2.4 + 1.4 - 1.0 (one big jump) = 2.8
  - L in frames 1–3, then MISS: 2.4 + 0.4 = 2.8

  The best total is B everywhere, so in frame 1 the algorithm does **not** choose the higher-scoring L, because the sequence as a whole is better with B. The decision for early frames is corrected using later evidence.
- Real data have exactly this structure: the ball is persistent and consistent, look-alikes flicker or jump.

**Three ways to run it** (the algorithm is the same; they differ in which future frames may be used):
| Mode | Which frames the estimate for frame *t* may use | Live use |
|---|---|---|
| **Offline** | the whole clip (the full backtrace at the end) | no |
| **Lag L** | frames up to *t + L*; the estimate for *t* is read from the best path as seen at time *t + L* (L = 4: 0.5 s delay) | yes, with delay |
| **Causal (lag 0)** | frames up to *t*; the estimate is the end state of the best path so far | yes |

In the toy example, the causal version still reports L in frames 1–3 (its path total is 2.4 against 2.1 for B at frame 3), but at frame 4 the path ending in B (2.8) beats all paths via L (2.6), so it switches immediately instead of waiting for a gate to open. Offline repairs frames 1–3 as well.

**Settings.** `thr`, `lam`, `miss` and `hold` (108 combinations) were chosen on the 6 validation clips by maximising recall plus half of the ball-in-crop value. For `ball_r18`: thr 0.1, lam 1.0, miss 0.2, hold 4, gate 0.25. The test set was evaluated once per configuration. Code: `src/exp_topk.py` (function `viterbi`).

### 2.4 Other ball detectors compared
- **FootAndBall-style networks** (inspired by Komorowski et al., VISAPP 2020; arXiv:1912.05445): a convolutional net that predicts a 2-class ball map (softmax, "ball / not ball" per cell) with extra weight on positives, hard-negative mining, and strong augmentation. Three versions were trained: (a) a *tiny* 0.19 M-parameter net trained from scratch, (b) the same recipe on a ResNet-18, and (c) "v2": a ResNet-18 that additionally sees the difference to the previous frame, trains 25 % of its crops on false positives of the heatmap net, and uses a lower learning rate for the backbone.
- **YOLO11n** (Ultralytics), fine-tuned for two classes, 8 ms on the GPU. A standard object detector; it outputs boxes with confidences.

---
## 3. Results (SoccerNet SN-Tracking-2023 test set, 49 clips)

### 3.1 Ball detectors, compared fairly (recall / precision, %)
Greedy tracker, no future frames except the offline static-overlay filter. "Raw" = strongest peak only.
| Ball source | Parameters | GPU / CPU ms per frame | Raw R / P | Tracked R / P | Coverage |
|---|---|---|---|---|---|
| **ResNet-18 heatmap (`ball_r18`)** | 11.3 M | 16 / 122 | 76.0 / 70.3 | 77.2 / 71.1 | 99.9 % |
| FootAndBall-style tiny | 0.19 M | 6.5 / 67 | 66.1 / 76.5 | 72.4 / 67.5 | 98.6 % |
| FootAndBall-style ResNet-18 | 11.3 M | 17 / 134 | 71.9 / 77.0 | 74.7 / 70.0 | 98.2 % |
| FootAndBall-style ResNet-18 v2 | 11.3 M | 18 / 144 | 71.7 / 75.5 | 75.3 / 70.3 | 98.5 % |
| YOLO11n fine-tuned | – | 8 / 34 | 50.4 / 88.8 | 60.5 / 69.6 | 80.0 % |

Reading it: the plain heatmap net with the focal loss is the best detector; the 0.19 M-parameter network costs about 5 points of tracked recall but is 2.5x faster on GPU and about 2x faster on CPU. YOLO11n is fast and precise but misses the ball in half of the frames (we did not investigate why; a 6 px ball is very small for a general detector).

### 3.2 Candidate ranks of the heatmap network
Ball among the strongest 1 / 3 / 10 peaks: **76.0 % / 84.9 % / 90.5 %** (test), 81.3 / 90.4 / 95.0 % (validation). The gap between 76.0 % and 90.5 % is what a better selection method can recover; the remaining 9.5 % are balls the network produced no peak for at all.

### 3.3 Greedy tracker vs Viterbi tracker (heatmap ResNet-18)
| Tracker | Recall % | Precision % | Coverage % |
|---|---|---|---|
| Greedy | 77.6 | 71.4 | 100.0 |
| Viterbi, causal | 78.9 | 76.7 | 94.6 |
| Viterbi, lag 4 (0.5 s) | 80.9 | 78.3 | 95.1 |
| **Viterbi, offline** | **81.5** | **78.8** | 95.1 |

- The same tracker helps every network (offline Viterbi vs greedy): tiny 72.4 -> 75.0, FootAndBall ResNet-18 74.8 -> 78.0, v2 75.4 -> 77.1. The heatmap network stays best.
- The greedy row here (77.6) differs a little from the table in 3.1 (77.2) because this analysis keeps 10 instead of 5 candidates and re-tuned settings.
- **Effect on the crop** (ball-centred window, smoothed): ball in crop 88.9 % (greedy) -> 90.5 % (offline), 90.0 % (lag 4), 89.0 % (causal). The gain is smaller than at the position level because a hit needs the ball only within a window 31.6 % of the frame wide, not within 2 %. (The window is always required to contain the estimated ball when possible.)

### 3.4 What the crop achieves
| Window position | Ball in crop % |
|---|---|
| Fixed centre of the frame | 53.0 |
| Centred on the tracked ball (`ball_r18`, greedy tracker) | 88.0 |
| Ball-priority policy, greedy tracker | 89.2 (88.1 after smoothing) |
| Same, FootAndBall-style tiny net | 86.4 |
| Same, YOLO11n | 81.6 |
| Ball-priority policy with the *true* ball position (upper bound) | 100 |

The fixed centre crop already contains the ball half the time because broadcast cameras keep the action near the middle. The ~11-point gap to 100 % is entirely due to errors of the ball estimate. Smoothing costs about 1 point of ball-in-crop (88.1 vs 89.2).

---
## 4. Other ideas tested: the few that matter
Test recall, offline Viterbi baseline 81.5 %. Every run is a single training run, so differences below about 2 points are treated as noise. The complete list with all numbers is in `BACKLOG.md`.

1. **Giving the network knowledge of the past** (extra input channel with the previous estimate, `b2_prior`; two-frame inputs; a StreamYOLO-style two-frame network, Yang et al. CVPR 2022). Raw top-1 recall rises (+2.5 points for the prior channel, +3 for the two-frame detector), but **after the Viterbi tracker the gain disappears** (79.1 % vs 79.4 % control). The tracker already supplies the temporal context. Forecasting the *next* frame's ball position did no better than copying the previous position or extrapolating it linearly (about 64 % raw for all three).
2. **Re-scoring candidates at higher resolution** (cutting a 192x192 window around each of the 10 candidates from the original 1080p frame and running the network again): 82.4 % / 80.1 % precision, +1 point. Validation did not show the same gain, so it is considered noise. Costs 10 extra network passes per frame.
3. **Adding detections of other detectors as extra candidates** (COCO-pretrained YOLO11n, and YOLO11n / YOLO12n fine-tuned on this data): 82.1–82.6 %. Alone they are weak (only 63–70 % of balls in their top-10, against 90.5 % for the heatmap net). No clear gain.
4. **Classic multi-object-tracking methods adapted to a single ball** (ByteTrack-style two-stage matching with a Kalman-like velocity model; and a polynomial smoother from the SoccerNet 2023 3rd-place report, arXiv:2308.16651): 71.8 % and 65.7 %, i.e. **worse than greedy**. They commit to a "nearest high-score detection" and cannot recover when the ball changes direction or is lost; a global search over candidates (Viterbi) is the better fit for one ball.
5. **Higher-resolution heatmap network** (stride-2 output, full-resolution stem; "BallHR"): validation 83.9 vs 81.0, but test 78.4 vs 81.5. The 6-clip validation set misled the selection; no gain.

Not a gain but informative: a **learned re-ranker** over candidate features improved the raw top-1 recall by 1.7 points (the clearest raw-detector gain), yet after Viterbi it was worse (80.3 %). The same story as in item 1.

---
## 4A. Two published approaches, adapted and tested
These two were picked from the literature as the most relevant. Both were **adapted** to our setting (not reproduced exactly) and the numbers below are **our** results on the same test set, from `BACKLOG.md` (sections #18 and #17). No new experiments were run for this summary.

### 4A.1 StreamYOLO (Yang, Liu, Li, Li, Sun, CVPR 2022, arXiv:2203.12338)
**What the paper does (as recorded in our notes; the abstract was re-checked online).** A real-time detector for *streaming perception*: by the time a detector finishes, the world has moved on, so the model should predict where objects will be in the **next** frame. It adds (a) **Dual-Flow Perception (DFP)**: a "dynamic" branch fed with the previous and current frame (motion) and a "static" branch (appearance), fused at feature-pyramid level; and (b) a **Trend-Aware Loss (TAL)** that weights each object by how fast it moves. On the Argoverse-HD driving benchmark the authors report +4.9 % AP over a strong baseline. The paper's results are for cars and pedestrians, not for tiny balls.

**Why it looked relevant.** The ball moves fast, so seeing two frames might let the network separate the ball from static look-alikes, and forecasting would give a camera-motion lead time.

**How we adapted it** (`src/streamball.py`, `src/streamball2.py`). Backbone ResNet-18 with a feature pyramid (strides 4 and 8) instead of YOLOX; DFP kept (fusing frames t-1 and t); a YOLOX-style box head; TAL kept (tau 0.3, nu 1.4); positives chosen near the ball centre instead of the paper's SimOTA assignment; one head shared across scales. Two versions: **v1** (hard 3x3 positives, sigmoid focal loss) and **v2** (the Gaussian-heatmap focal loss of `ball_r18` at stride 4, plus DFP, box head and TAL). Two targets: *current* frame (plain detection) and *next* frame (forecasting, as in the paper). All trained for 10 epochs with weight averaging, single seed, on the stride-3 training split (`train3`).

**Results** (test, 49 clips; recall %, Viterbi tuned on validation; the fair control is `c3_base` = plain heatmap net on the same split, since `ball_r18` was trained on a different, better split):
| Model | Target | Raw top-1 | Ball in top-10 | Viterbi offline R / P | Viterbi causal R / P |
|---|---|---|---|---|---|
| `ball_r18` (original split) | current | 76.0 | 90.5 | 81.5 / 78.8 | 78.9 / 76.7 |
| `c3_base` (control) | current | 73.2 | 88.1 | 79.0 / 72.7 | 77.1 / 71.0 |
| v1 single frame | current | 63.3 | 86.9 | 69.8 / 64.2 | 67.8 / 62.4 |
| v1 + DFP | current | 66.4 | 89.3 | 73.0 / 67.2 | 69.8 / 64.2 |
| v1 + DFP, no TAL | current | 62.4 | 88.1 | 66.5 / 61.2 | 63.5 / 58.4 |
| v2 single frame | current | 72.2 | 89.6 | 78.8 / 73.0 | 76.7 / 71.2 |
| v2 + DFP | current | 75.2 | 91.7 | 78.8 / 72.6 | 76.2 / 70.2 |
| v2 single frame | next | 60.5 | 79.9 | 65.5 / 61.5 | 61.5 / 58.0 |
| v2 + DFP | next | 64.1 | 84.3 | 68.7 / 64.4 | 64.4 / 60.8 |

Cost: 35 ms per frame pair on the GPU with DFP, against 20–21 ms for the single-frame versions (and 16 ms for `ball_r18`).

**What we conclude.**
- Two frames **do** improve the raw detector: with the v2 head, DFP gives +3 points in top-1 and +2 in top-10 over its single-frame counterpart (and +2.0 / +3.6 over the control).
- **After the Viterbi tracker the gain vanishes**: all three current-frame versions end at 78.8–79.0 % offline and 76–77 % causal, the same as the control (79.0 / 77.1), and below the original `ball_r18` (81.5 / 78.9). The tracker already supplies the temporal information. Same pattern as for the prior-channel experiment.
- **Trend-Aware Loss**: it helped the weak v1 head (+6.5 points offline). It was not tested on the better v2 head, so we do not know whether it matters there. (In our data the ball moves farther between two samples than its own size, so the loss may often reduce to a constant; this was an expectation noted before the run, not measured.)
- **Forecasting the next frame** reproduces the paper's trend (DFP clearly helps: +3–4 top-1, +4–9 top-10), but even the best forecaster (68.7 % offline) is worse than detecting the *current* frame, and its raw top-1 (64.1 %) only equals two trivial baselines built from `ball_r18`: copying the previous estimate (64.1 %, covers 88 % of frames) or extrapolating it at constant velocity (63.6 %, covers 83 %). No evidence that it beats simple extrapolation of a good detector.
- **Deviations** (ResNet-18 instead of YOLOX, simplified label assignment, shared head; no COCO-pretrained weights) mean this tests the idea, not the exact published model. Not run: multiple seeds, v2 without TAL.

### 4A.2 WASB-SBDT (Tarashima, Haq, Wang, Tagawa, BMVC 2023, arXiv:2311.05237)
**What the paper does.** A "widely applicable strong baseline" for sports ball detection and tracking, compared against 6 earlier methods on 5 datasets from different sports (it also adds two new datasets and new ball annotations on two existing ones). According to the abstract, its ingredients are **high-resolution feature extraction**, **position-aware model training** and **inference that accounts for temporal consistency**; the authors report substantially better results than the earlier methods. Code: github.com/nttcom/WASB-SBDT.
**Important limit:** only the abstract and listing were read. Details such as the exact network (our notes say a small HRNet with a stride-free stem, three input and output frames, focal then quality-focal loss) are **unverified** and must be checked in the PDF before they are written in a paper. The paper's numbers are not quoted here because we have not read them and its datasets and metrics differ from ours.

**Why it looked relevant.** Its main message, that tiny balls need high-resolution features, matches our error analysis: balls under 8 px are found only 72 % of the time and our heatmap is output at 1/4 of the input resolution.

**How we adapted it** (`src/ball_hr.py`, "BallHR", run `hr_a`). A ResNet-18 + feature-pyramid network as before, plus an extra branch that processes the image at **full input resolution** with no downsampling at the start, and a heatmap at **stride 2** instead of 4. Single frame, the same 10-epoch recipe and training split as `ball_r18`. This borrows the high-resolution idea only: it is **not** the WASB network, and the temporal-consistency part is covered by our Viterbi tracker instead.

**Result** (test, 49 clips):
| | Best validation top-1 | Raw top-1 | Ball in top-10 | Greedy R / P | Viterbi offline R / P | Viterbi lag 4 R / P | Viterbi causal R / P | GPU ms |
|---|---|---|---|---|---|---|---|---|
| `ball_r18` | 81.0 | 76.0 | 90.5 | 77.6 / 71.4 | 81.5 / 78.8 | 80.9 / 78.3 | 78.9 / 76.7 | 16 |
| BallHR `hr_a` | 83.9 | 73.3 | 90.2 | 75.0 / 69.0 | 78.4 / 72.3 | 78.2 / 72.1 | 76.6 / 70.6 | 23 |

**What we conclude.** The high-resolution variant looked better on the 6 validation clips (83.9 vs 81.0 %) but was about 3 points **worse** on the test set at every tracker setting, and slower and heavier (3.1 GB memory in training). In our setting the stride-2 heatmap did not help; validation again misled the choice. A related, much cheaper test supports this: re-scoring each candidate on a 192x192 window cut from the original 1080p frame gave only about +1 point (within noise). Two temporal BallHR variants (`hr_t4`: difference images as extra input; `hr_t4m`: plus a head predicting where the ball was in the previous frame) were started but interrupted and have **no results**, so the closest part of WASB (several input frames) is still untested here. Caveats: one seed, and our model is not the published network, so this says nothing about the quality of WASB-SBDT itself.

---
## 5. Challenges and failure analysis
Error analysis of `ball_r18` + offline Viterbi on 11 268 ball-visible test frames (`src/exp_errors.py`, descriptive only):
- **Outcome split**: correct 81.5 %. Wrong pick although the ball was among the 10 candidates: 9.6 %. The network produced no candidate near the ball: 6.0 %. No estimate (coverage gap): 3.0 %.
- **Small balls** (<8 px wide in the 896 px frame): 72 % correct; 8–18 px: 83–86 %.
- **Fast balls**: 82–88 % correct below 30 px per sample, 62 % above 30 px, 27 % above 60 px (118 frames only). Fast shots and long passes are where the distance-based cost penalises the *true* path.
- **Occlusion**: the ball within 20 px of a player: 67 % correct; inside a player's box: 71 %; otherwise 86–87 %.
- **Errors come in long runs, not as flicker**: 68 % of the failed frames lie in runs of 8+ samples (about 1 s). The tracker "locks on" to a wrong, consistent object. Widening the tracker's gate (idea #2 in the backlog) did not help, because the wrong object usually still scores well.
- **Confidence**: the chosen candidate's score averages 0.69 when right and 0.35 when wrong, a usable signal, but using it to drop uncertain estimates lowers recall about as fast as it raises precision, and lowers ball-in-crop.
- **Phantom balls**: in frames where the ball is not visible, 73 % still receive an estimate.
- **Clip-level**: median clip 86.5 % recall; 3 of 49 clips below 50 % (worst 46 %).
- **Other look-alikes**: TV overlays and logos (hence the static filter, which is not causal).
- **Cost**: 122 ms per frame on 4 CPU threads for the full 504x896 image. This is 8 fps at best, too slow for real-time on-device use without a smaller model, a tile-based approach, or lower inference rate (e.g. ball detection every 3rd frame).

---
## 6. Limits of the evidence (state these in the paper)
- **Single training run (one random seed)** for each network, so there is no measured run-to-run variance. Differences of 1–2 points should not be called improvements.
- **Small validation set**: 6 clips; accuracy on it moves by about 3 points between training epochs. Tracker settings and network checkpoints were chosen on it, which is why several "validation wins" did not transfer to the test set.
- The best result (81.5 %) is **offline**; the live-capable (causal) number is 78.9 %, only 1.3 points above the greedy tracker.
- The static-overlay filter and offline/lag modes use future information.
- The reference crop and ball positions are annotation-based; there is **no human camera operator** reference and no viewer study. Ball-in-crop says the ball is in the window, not that the video looks good.
- The Viterbi tracker has been evaluated in experiment scripts but is not yet connected to the main evaluation and demo scripts (`evaluate_sn.py`, `render_clip_ball.py` still use the greedy tracker). The paper's ball tables are greedy; Viterbi appears in a separate table.
- The stride-3 retraining split gives a model about 2.5 points worse than the original; only compare runs trained on the same split.
- The report's ball AP numbers (0.95 / 0.71) of the 3rd-place SoccerNet 2023 team (arXiv:2308.16651) use another metric (average precision on 200 hand-checked frames, a 1080p YOLOv8l detector) and cannot be compared with our recall.

---
## 7. Open items and directions for the cropping work
**Ball side**
- Untested: a 3–5-frame difference input, rebuilding training frames at stride 3, a larger validation set, higher-resolution training crops, bigger detectors (YOLO-s/m).
- The temporal variant of BallHR was started but interrupted; **no results exist**.
- Single-object tracker SUTrack (Chen et al., AAAI 2025): set up and smoke-tested, only 36.9 % recall when initialised from the ground truth on one clip. The full run requires approval to execute third-party code (`src/run_sot.sh`).
- Wire the Viterbi tracker (causal or lag 4) into the demo and evaluation scripts.

**Cropping side (for the co-authors)**
- Define the window path from ball estimates: today an exponential moving average with a per-sample speed limit costs about 1 point of ball-in-crop (88.1 vs 89.2 %), while cutting jitter. Optimising the whole camera path (e.g. minimising acceleration subject to keeping the ball inside) is the natural next step.
- Use the ball *confidence* (0.69 vs 0.35) and the MISS state to decide when to stop following (e.g. hold position or zoom out).
- Lower the cost: ball detection at lower frame rate with the tracker bridging between runs.

---
## 8. Where things are
- Documents: `handoff.md` (state and commands), `BACKLOG.md` (every experiment with numbers), `notes_viterbi.md` (tracker and error analysis), `paper/main.tex` (current draft).
- Code: `src/sn_lib.py` (network, greedy tracker, metrics), `src/exp_topk.py` (Viterbi), `src/exp_errors.py`, `src/train_ball*.py`, other `src/exp_*.py` (one per experiment). Results: `results/exp/*.json`.
- Build the paper: `cd paper; tectonic main.tex`.

---
## 9. References
Verified = authors, venue and id checked with a web search on 2026-11-17. ⚠ = from memory or secondary source, **check before citing**.

- ✔ A. Cioppa, S. Giancola, A. Deliège, L. Kang, X. Zhou, Z. Cheng, B. Ghanem, M. Van Droogenbroeck, "SoccerNet-Tracking: Multiple object tracking dataset and benchmark in soccer videos," *CVPR Workshops (CVsports)*, 2022, pp. 3491–3502, arXiv:2204.06918. *(dataset)*
- ✔ J. Komorowski, G. Kurzejamski, G. Sarwas, "FootAndBall: Integrated player and ball detector," *VISAPP 2020*, pp. 47–56, DOI 10.5220/0008916000470056, arXiv:1912.05445. *(model family 2.4)*
- ✔ G. Shitrit, I. Be'ery, I. Yerhushalmy, "SoccerNet 2023 Tracking Challenge – 3rd place MOT4MOT Team Technical Report," arXiv:2308.16651, 2023. *(YOLOv8l ball detector, polynomial smoothing)*
- ✔ J. Yang, S. Liu, Z. Li, X. Li, J. Sun, "Real-time object detection for streaming perception," *CVPR 2022*, arXiv:2203.12338. *(two-frame / forecasting network)*
- ✔ S. Tarashima, M. A. Haq, Y. Wang, N. Tagawa, "Widely applicable strong baseline for sports ball detection and tracking," *BMVC 2023*, arXiv:2311.05237, code github.com/nttcom/WASB-SBDT. *(only the abstract was read; high-resolution heatmap network with temporal consistency, close to our approach)*
- ✔ X. Chen, B. Kang, W. Geng, J. Zhu, Y. Liu, D. Wang, H. Lu, "SUTrack: Towards simple and unified single object tracking," *AAAI 2025*, vol. 39(2), pp. 2239–2247, arXiv:2412.19138. *(single-object tracker)*
- ⚠ X. Zhou, D. Wang, P. Krähenbühl, "Objects as points," arXiv:1904.07850, 2019. *(CenterNet: heatmap + focal loss)*
- ⚠ T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, "Focal loss for dense object detection," *ICCV 2017*.
- ⚠ Y.-C. Huang et al., "TrackNet: A deep learning network for tracking high-speed and tiny objects in sports applications," *AVSS 2019* (arXiv id given in notes: 1907.03698; later versions V2–V4 exist, look up before citing).
- ⚠ K. He, X. Zhang, S. Ren, J. Sun, "Deep residual learning for image recognition," *CVPR 2016*.
- ⚠ I. Loshchilov, F. Hutter, "Decoupled weight decay regularization," *ICLR 2019* (AdamW).
- ⚠ G. Jocher, J. Qiu, "Ultralytics YOLO11," 2024, github.com/ultralytics/ultralytics.
- ⚠ F. Liu, M. Gleicher, "Video retargeting: Automating pan and scan," *ACM Multimedia 2006* (background on cropping).
- ⚠ Mentioned in the notes but not read: ByteTrack, OC-SORT / DeepOC-SORT, BoT-SORT, YOLOv12, RT-DETR, EdgeDAM, Kukleva et al. 2019, FoT (Sci. Reports 2025).
