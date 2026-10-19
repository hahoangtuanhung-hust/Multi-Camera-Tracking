# Performance and identity optimization audit — 2026-10-19

## Decision

Keep `hybridsort` as the production default with YOLO26n at `imgsz=640`,
`conf_threshold=0.25`, `stride=1`, four PyTorch CPU threads, one OpenCV thread,
and tracklet matching every 0.25 s. This is the best measured Pareto point for
speed, IDF1, HOTA, ID switches, fragmentation, precision, and recall on the
current four-camera scene.

Do not promote the short-window tuning candidates, trajectory smoothing,
late binding, lower confidence floors, `imgsz=480`, or `stride=2`. CUDA FP16
and `torch.compile` are available as explicit A/B switches but remain disabled
by default because the external T4 comparison could not be authorized from
this environment.

The audit uses no neural ReID and no appearance-assisted candidate profiles.
Cross-camera identity remains geometry, event time, motion, and trajectory
based.

## Method

- Ran the complete unit suite before changes: 245/245 passed.
- Screened 38 no-appearance profiles on the same validated 300-tick detection
  cache (`YOLO26n`, `imgsz=640`, source floor `0.15`, four fixed cameras).
- Promoted the Pareto candidates to controlled 4,000-tick replay.
- Swept Hybrid-SORT alpha, primary/blended IoU gates, tracklet match interval,
  and tracklet resampling frequency one variable at a time.
- Benchmarked detector resolution, stride, and 1/2/4/8 PyTorch thread budgets
  end to end on real decoded frames.
- Added `--fast-metrics` to long replay. It skips repeated diagnostic reports,
  not aggregate MOT scoring; the 4,000-tick default run preserved the exact
  IDF1/HOTA/IDSW/fragmentation and prediction hash of the full report.
- Finished with 248/248 tests passing after the benchmark and CUDA precision
  controls were added.

## Long-sequence tracking evidence

The 4,000-tick production `hybridsort` baseline scored IDF1 0.316014, HOTA
0.214602, precision 0.802849, recall 0.667167, 252 ID switches, and 387
fragmentations.

The main alternatives behaved as follows:

- `hybridsort-late-binding-balanced`: IDF1 0.295004, HOTA 0.205235, 258
  switches, 374 fragmentations. It recovered recall (0.722513) but lost IDF1,
  HOTA, precision, and runtime through commit buffering.
- `hybridsort-trajectory`: IDF1 0.318374, HOTA 0.203310, 324 switches, 434
  fragmentations. The 0.75% IDF1 gain was not worth the HOTA, stability,
  recall, and throughput regressions.
- `identity-v1-safe`: IDF1 0.294181, HOTA 0.227803, 410 switches, 388
  fragmentations. High recall did not translate into stable identity.
- `hybridsort-balanced-low-floor` at confidence 0.15: IDF1 0.285359, HOTA
  0.193652, 259 switches, and recall 0.643663. Extra low-score boxes increased
  association/lifecycle noise instead of end-to-end recall.
- `tracklet_match_interval_s=1.0` plus blended IoU 0.30 won the 300-tick
  screen, then regressed at 4,000 ticks to IDF1 0.295488, HOTA 0.195468, and
  256 switches.
- Blended IoU 0.30 alone also failed the long gate: IDF1 0.305862, HOTA
  0.206402, and 265 switches.

The result is deliberately conservative: a candidate is rejected when it wins
on a short slice but loses the continuous 4,000-tick identity gate.

## Detector and runtime evidence

On the controlled 300-tick, four-camera CPU run:

- `640`, stride 1: 1.745 tick/s, IDF1 0.775074, HOTA 0.510834, 16 IDSW, 13
  fragmentations.
- `480`, stride 1: 2.197 tick/s (+26%), but IDF1 0.621319 (-19.8%), 25 IDSW
  (+56%), and 33 fragmentations (+154%). Rejected.
- `640`, stride 2: 1.327 processed tick/s and 2.655 source-frame-index/s
  (+52% source-time throughput), with IDF1 0.754226 (-2.7%). Retained only as
  an explicit deployment trade-off.

The 1/2/4/8-thread 60-tick sweep measured 1.892/1.434/2.041/1.447 tick/s with
identical identity metrics, selecting the existing four-thread default.

The final 100-tick profile attributed about 61% of runtime to YOLO inference,
10.6% to video decoding, 13.6% to fusion, 6.0% to local Hybrid-SORT, and 4.7%
to tracklet matching. Detector/backend work is therefore the next meaningful
optimization frontier; risky fusion micro-optimizations have a small ceiling.

## CUDA precision controls

`PipelineConfig.detector_fp16` maps to Ultralytics 8.4.96 `quantize=16` only
on CUDA. `detector_compile` maps to its `compile` option only on CUDA. CPU and
MPS stay on FP32. Modal evaluation and rendering expose both flags, including
`--detector-fp16`, `--no-detector-fp16`, `--detector-compile`, and
`--no-detector-compile`.

Both controls default to false. The intended release procedure is an FP32 vs
FP16 T4 A/B with identical source, cameras, seed, and 4,000-frame window,
followed by a compile A/B long enough to amortize graph warm-up. Promote only
when throughput improves without IDF1/HOTA regression or increased IDSW.

## Reproducible artifacts

The generated artifacts are under
`data/cpu-audit/optimization-20261019/`. Important files include:

- `fast-default-4000.json`
- `combo-interval-1-blended-0.3-4000.json`
- `blended-0.3-4000.json`
- `full-4000/hybridsort.json`
- `full-4000/hybridsort-trajectory.json`
- `full-4000/hybridsort-late-binding-balanced.json`
- `detector-runtime/640-s1.json`
- `detector-runtime/480-s1.json`
- `detector-runtime/640-s2.json`
- `cpu-threads/threads-{1,2,4,8}.json`
- `current-profile-100.profile.txt`
