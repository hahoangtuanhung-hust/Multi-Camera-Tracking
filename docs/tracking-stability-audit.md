# MTMC tracking stability audit (No-ReID)

Audit date: 2026-10-15. Dataset: `data/custom`; cameras `camera_0001`,
`camera_0007`, `camera_0008`, and `camera_0010`; `stride=1`; capture/event
time; fixed evaluator frame offset 0. The final runs replay one continuous
4,000-tick detector cache (16,000 camera frames), not concatenated clips.

Follow-up: the 2026-10-18 real-frame crowd recovery work, rejected ablations,
and current T4 release commands are documented in
[crowd-recovery-audit-2026-10-18.md](crowd-recovery-audit-2026-10-18.md).

## Outcome

**Do not promote a new production profile.** The correctness, observability,
determinism, physical-motion checks, and bounded-state work in this audit are
ready, but the measured tracking quality does not satisfy the release gates.
`hybridsort-balanced` is the corrected B0 reference, selected explicitly for
the final audit; the runtime default remains unchanged.

The main 4,000-tick failure is identity lifecycle, not detector recall alone.
The same 25 GT identities receive a median of 9 public GIDs (p95 12), and only
9 of 79 long-gap reappearances retain the same GID. Per-500-frame IDF1 remains
between 0.558 and 0.702, while cumulative IDF1 falls to 0.297 because identity
ownership does not persist across the full timeline.

No ReID, FaceID, appearance embedding, fixed population assumption, frame
skipping, or dataset-specific identity mapping was added.

## Correctness and instrumentation changes

- All runtime/evaluation/replay/render/Modal entry points now resolve profiles
  through `build_effective_config`; a profile owns its local tracker unless an
  explicit tracker override is supplied.
- CLEAR ID switches and fragmentation use one frame-level assignment stream.
  Offline alignment is fixed at frame offset 0; there is no best-offset search.
- Replay hashes canonical, sorted prediction positions and assignment state on
  every tick. Metrics-only equality is no longer accepted as deterministic.
- Run metadata records the dirty-worktree source fingerprint, model SHA-256,
  package versions, effective config, video/calibration inputs, cache trims,
  camera order, confidence floor, and stride.
- Stage diagnostics now measure bbox-GT detector, local, projection, fusion,
  public-affinity, and emitted-output coverage overall, per camera, and per
  500-frame segment. Detector misses are no longer inferred from fused FNs.
- Local Kalman state keeps raw and trusted observations separately. NIS
  outliers cannot contaminate the trusted observation used for recovery.
- Trajectory fitting uses confidence times recency weights with robust Huber
  reweighting. Raw segment speed, raw fitted speed, and raw acceleration are
  validated before accepting a point; output clipping cannot turn an
  impossible observation into a valid one.
- Rolling one-second displacement is enforced over accepted observations in a
  bounded deque. Camera-image boundaries are not treated as world exits unless
  all current camera evidence is at an exit boundary; explicit world exit
  zones remain authoritative.
- Regression coverage includes 0.3/0.5/1/2/3-second occlusions, merged-box
  crossings, same-direction overtaking with reversed detection order, clock
  learning off/offset-only, impossible raw trajectory motion, and replay hash
  sensitivity.

## Controlled 300-tick screening

All rows use the same cached YOLO output. B0 filters it at confidence 0.25.

| Profile | IDF1 | HOTA | AssA | IDSW | Frag | Decision |
|---|---:|---:|---:|---:|---:|---|
| B0 `hybridsort-balanced` | 0.7747 | 0.5111 | 0.6126 | 16 | 13 | Reference |
| A1 trusted local observation | 0.7384 | 0.4713 | - | 17 | 14 | Reject |
| A2 robust trajectory + smoother | 0.8015 | 0.5234 | 0.6186 | 10 | 18 | Long-run check |
| A3 common timestamp | 0.7749 | 0.5116 | - | 16 | 13 | No material gain |
| A3 fixed clock | 0.7770 | 0.5132 | 0.6105 | 16 | 13 | Reject: AssA regression |
| A3 offset-only clock | 0.7747 | 0.5111 | 0.6126 | 16 | 13 | Identical to B0 |
| A4 crossing guard | 0.7491 | 0.4944 | - | 15 | 10 | Reject: IDF1/HOTA loss |
| A5 lifecycle defer | 0.7797 | 0.5115 | - | 17 | 15 | Reject: IDSW/Frag |
| Detector floor 0.20 | 0.7425 | 0.4926 | 0.5893 | 19 | 16 | Reject |
| Detector floor 0.15 | 0.7732 | 0.5132 | 0.6163 | 16 | 15 | Reject: Frag |

A2 was split into motion-only and smoothing-only controls. Motion-only is
bit-for-bit equivalent to B0 at 300 ticks. Smoothing-only reproduces all A2
quality gains and all of its fragmentation regression; 644 raw observations
were rejected. This isolates the effect to the smoother rather than the robust
association-side motion estimate.

## Continuous 4,000-tick results

| Metric | B0 corrected | A2 trajectory | A2 vs B0 |
|---|---:|---:|---:|
| IDF1 | 0.2968 | 0.3271 | +0.0303 |
| IDP / IDR | 0.3268 / 0.2718 | 0.3459 / 0.3103 | +0.0191 / +0.0384 |
| HOTA / AssA | 0.1948 / 0.1202 | 0.2211 / 0.1481 | +0.0263 / +0.0279 |
| MOTA | 0.4593 | 0.4785 | +0.0192 |
| Precision / recall | 0.7781 / 0.6472 | 0.7689 / 0.6898 | -0.0092 / +0.0425 |
| IDSW / 1k ticks | 64.5 | 76.0 | +17.8% |
| Fragmentations | 355 | 447 | +25.9% |
| Position RMSE | 0.4352 m | 0.4332 m | -0.0020 m |
| Median / p95 GIDs per GT | 9 / 12 | 9 / 17 | worse p95 |
| Long-gap same-GID | 9 / 79 | 7 / 75 | worse rate |
| Uninstrumented replay | 631.6 s, 6.33 tick/s | 852.8 s, 4.69 tick/s | +35.0% time |

A2 improves aggregate IDF1/HOTA/recall but increases both ID switches and
fragmentation on the long run, so it is rejected and remains opt-in only.

### B0 cumulative stability

| End frame | IDF1 | IDR | HOTA | AssA | IDSW | Frag |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.7017 | 0.6871 | 0.4332 | 0.4661 | 31 | 39 |
| 1,000 | 0.5973 | 0.5633 | 0.3610 | 0.3557 | 62 | 80 |
| 1,500 | 0.4888 | 0.4508 | 0.3033 | 0.2633 | 88 | 115 |
| 2,000 | 0.4175 | 0.3823 | 0.2670 | 0.2130 | 128 | 187 |
| 4,000 | 0.2968 | 0.2718 | 0.1948 | 0.1202 | 258 | 355 |

## Stage funnel and root causes

The full B0 run contains 171,889 camera/GT instances.

| Stage | TP | Recall | Loss from previous stage |
|---|---:|---:|---:|
| Detector | 153,491 | 0.8930 | 18,398 detector misses |
| Local tracker | 139,293 | 0.8104 | 14,204 |
| Projection | 139,293 | 0.8104 | 0 |
| Fusion | 121,927 | 0.7093 | 17,364 |
| Public affinity | 80,519 | 0.4684 | 41,409 |
| Emitted output | 80,460 | 0.4681 | 59 |

The dominant measured loss is fusion-to-public identity publication. By
camera, that loss is 20,150 for `camera_0008`, 9,475 for `camera_0007`, 7,440
for `camera_0001`, and 4,344 for `camera_0010`. `camera_0008` also has the
largest detector-to-local loss (8,275). Projection RMSE is 0.305 m, so
projection is not the missing-coverage stage.

The 4,000-frame detector confidence curve is:

| Confidence | Precision | Recall |
|---:|---:|---:|
| 0.15 | 0.8614 | 0.9219 |
| 0.20 | 0.8970 | 0.9077 |
| 0.25 | 0.9211 | 0.8930 |
| 0.50 | 0.9794 | 0.7921 |

Lower detector floors provide more boxes, but the controlled end-to-end
screens regress fragmentation or identity metrics. The next work should target
public lifecycle/dormant reconnection and camera-specific local fragmentation,
then repeat the controlled replay. Detector changes are not the first priority.

## Determinism, runtime, and bounded state

Two independent B0 4,000-tick runs—one fast and one with stage diagnostics plus
`tracemalloc`—produced identical streams and identical metrics:

- prediction SHA-256:
  `a9b5d5ed3c5f5b4445ae1b8de0116b8fa11d86cfaee32420be408dc2a4c42fc9`
- assignment SHA-256:
  `e29ca59d602cf0523faf8e303e4ca59e044b95de91ab751120a8ee1c4c9df100`
- full instrumented run: 2,005.3 s inside replay, 1.99 tick/s, peak traced
  Python memory 86.51 MiB
- no ingest errors, late tracklets, or out-of-order packets
- peak retained global/local tracks: 48 / 87; final 39 / 48
- final tracklet state: 5,616 points, 269,568 bytes; accepted-position points:
  662; retained aliases/affinities: 45 / 80
- physical gates actively rejected 176,699 rolling-displacement candidates,
  1,106 speed candidates, and 198,824 trajectory matches; impossible raw
  segment/fit speed and acceleration paths are also covered by unit tests
- stage failure-event retention is bounded at 2,000 records; 89,434 additional
  events were counted as dropped instead of accumulating in memory
- source fingerprint:
  `e210c536ba279f37da5259cd6c1a71534b4e64671610a3bec83c6cce458bd788`
- model SHA-256:
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`

The cache contains 4,000 ticks with all four cameras, trims of 30 frames for
`camera_0001` and 60 for `camera_0007`, stride 1, YOLO image size 640, person
class 0, and capture floor 0.15. B0 deterministically filters it to 0.25.

## Release gates

| Gate | Required | Observed | Status |
|---|---:|---:|---|
| IDF1 at 2,000 | >= 0.60 | 0.4175 | FAIL |
| IDF1 2,000 vs 500 | no worse than -0.05 | -0.2842 | FAIL |
| IDF1 4,000 vs 2,000 | no worse than -0.05 | -0.1207 | FAIL |
| IDR at 4,000 | >= 0.60 | 0.2718 | FAIL |
| Recall at 4,000 | >= 0.72 | 0.6472 | FAIL |
| Precision at 4,000 | >= 0.78 | 0.7781 | FAIL |
| IDSW/1k reduction | >= 25% | A2 is 17.8% worse | FAIL |
| Long-gap changed-GID reduction | >= 50% | A2 is 2.9% better | FAIL |
| Median / p95 GIDs per GT | 1 / <= 2 | 9 / 12 | FAIL |
| HOTA / AssA / Frag no regression | all | A2 Frag +25.9% | FAIL |
| Physical speed/acceleration/rolling caps | enforced | raw-veto tests and runtime rejects | PASS |
| Exact deterministic replay | required | both stream hashes match | PASS |
| Bounded state / no ingest errors | required | bounded counters; zero errors | PASS |

## Reproduce

Build the continuous detector cache once:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe benchmarks\cpu_pipeline.py `
  --frames 4000 --tracking-profile hybridsort-low-floor `
  --capture-detections data\cpu-audit\detections-low-floor-4000.pkl `
  --out data\cpu-audit\detector-cache-build-4000.json
```

Fast controlled replay:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe benchmarks\replay_cpu.py `
  --detections data\cpu-audit\detections-low-floor-4000.pkl `
  --out data\cpu-audit\b0-balanced-corrected-4000-screen.json `
  --only-current --profile hybridsort-balanced `
  --skip-stage-diagnostics --skip-memory-profile
```

Full stage/memory audit:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe benchmarks\replay_cpu.py `
  --detections data\cpu-audit\detections-low-floor-4000.pkl `
  --out data\cpu-audit\b0-balanced-corrected-4000-full.json `
  --only-current --profile hybridsort-balanced
```

Regression tests:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q --ignore=tests/test_renderer.py
.\.venv\Scripts\python.exe tests\test_renderer.py
```

Expected result: 191 pytest tests plus 2 renderer tests pass. Benchmark/cache
artifacts stay under `data/cpu-audit/` and are intentionally not committed.

## Next bounded experiment and rollback

Prioritize a controlled lifecycle experiment that explains the 41,409
fusion-to-public losses and improves long-gap same-GID retention without
weakening the 3.2 m/s speed, 4.0 m/s^2 acceleration, or one-second rolling
displacement caps. Report per-camera results, especially `camera_0008` and
`camera_0007`, before considering any default change.

Rollback requires no code reversal: keep the current runtime default, do not
select `hybridsort-a2-trajectory`, and use explicit
`--profile hybridsort-balanced` only when reproducing this audit reference.
