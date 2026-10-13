# MTMC tracking stability audit (no ReID/FaceID)

Audit date: 2026-10-13. Dataset: `data/custom`, cameras `camera_0001`,
`camera_0007`, `camera_0008`, and `camera_0010`. All release runs use every
source frame (`stride=1`). The longest CPU comparison is intentionally capped
at 1,000 evaluated ticks.

Post-audit implementation note: the per-camera default was subsequently
changed from ByteTrack to motion-only OC-SORT. Historical measurements in this
document are ByteTrack measurements unless a result explicitly says OC-SORT;
use `--local-tracker bytetrack` to reproduce them.

## Outcome

`identity-v1-drift-guard` is the new CLI/runtime default. It preserves the
validated local tracker and v1 fusion policy, but rejects a camera-clock update
when independently matched people in the same camera-pair cycle imply time
shifts whose spread exceeds 0.40 seconds. This prevents an ambiguous crossing
or spatial calibration error from becoming a persistent clock error that
corrupts later associations.

The old behavior remains reproducible with `--profile identity-v1-baseline`.
No ReID, FaceID, appearance embedding, fixed population assumption, frame
skipping, or new heavy dependency is used.

## Audited pipeline

1. `video.py` decodes each camera independently into bounded lossless queues in
   live mode. Each packet carries source-frame index, sequence, capture time,
   arrival time, and camera-clock model. Offline evaluation/render uses the
   synchronized reader but the same downstream pipeline.
2. `detector.py` executes one dynamic YOLO batch for every emitted tick. The
   default model is `yolo26n.pt`, image size 640, confidence floor 0.25, person
   class only. There is no hidden detection interval and no tiled detection.
3. One persistent local tracker instance is created per camera. The current
   default is motion-only `OCSortTracker`; `ByteTracker` remains selectable
   for historical A/B. Primary association combines Hungarian IoU with
   observation-direction momentum; unmatched tracks may use low-confidence
   detections and last-observation recovery. After an occlusion, virtual
   observations re-update the Kalman state. New tracks require a high-score
   detection, become confirmed after three hits, and confirmed tracks survive
   two seconds using capture-time `dt`. A tentative track is removed on its
   first miss.
4. Confirmed local boxes are projected to world foot points. Fusion first uses
   same-tick geometry and local-to-global ownership, then metadata-only
   trajectory history for asynchronous association. History is five seconds;
   normal identity retention is three seconds.
5. Fusion advances on capture/event time. Arrival and processing monotonic time
   are diagnostic/queue signals and are not substituted for event time.

The current reader ends a camera stream on EOF/error; it does not yet implement
a reconnect protocol. Tracker instances are not reset or local IDs reused
during a normal run.

## Root causes found

### 1. Local fragmentation is the largest measured source

On the first 300-frame baseline, 19 of 39 evaluator-attributed switches were
classified as local fragmentation, versus two cases where the local ID stayed
stable but the global ID changed. On the 1,000-frame baseline the local layer
created 341 tracks, reactivated 963 tracks, removed 22 tentative tracks, and
removed 258 tracks after timeout. This is consistent with detector misses,
occlusion and IoU loss feeding fragmented tracklets to fusion.

An attempted local motion/Mahalanobis association was rejected: on the same
cached detections it increased ID switches from 39 to 43 and reduced IDF1 from
0.705 to 0.683. It remains experimental instead of being enabled on intuition.

### 2. Crossing ambiguity contaminates online clock learning

Twenty-seven of the 39 first-300-frame switches occurred while another ground
truth person was within 1.5 m. Several geometrically plausible people could
therefore supply incompatible time shifts to one camera-pair update. Once
learned, that bad correction shifts all subsequent trajectories and looks like
time drift even when the source clock is stable.

The 1,000-frame guard diagnostic recorded 71 accepted clock updates, 495
updates rejected specifically for inconsistent multi-person delta, and 221
rejected by the estimator's existing confidence/outlier rules. Rejections were
distributed across all camera pairs, not hard-coded to one camera.

### 3. Aggressive global safeguards can reduce switches but hurt identity

Publication delay, strict population slots, wider identity retention,
duplicate suppression, motion gates and confirmation hysteresis were evaluated
as explicit profiles. Some lowered ID switches, but also reduced IDF1/HOTA or
recall because a geometry-only system cannot safely decide every crossing or
long blind-zone re-entry. They remain opt-in profiles.

## Implemented changes

- Added a bounded streaming JSONL identity logger. It records local
  birth/match/miss/removal, local state and association evidence, global
  birth/publish/merge/reconnect/expiry, accepted/rejected clock learning, and
  evaluator-attributed switch classes. The writer never retains an unbounded
  event list in memory.
- Added the multi-person clock-delta consistency gate and counters. A single
  good path or several mutually consistent paths can still learn offsets up to
  the existing configured +/-3 seconds.
- Added detector-cache fingerprints (model SHA-256, detector settings, source
  videos, camera list, trims and stride) so A/B replay cannot silently use a
  stale or incomparable cache.
- Added IDP, IDR, IDF1, MOTA, MOTP, HOTA, AssA, windowed/cumulative reporting,
  ID-switch detail, and CLEAR trajectory fragmentation (`Frag`) to the
  evaluator/benchmark output.
- Added explicit local lifecycle counters and kept all diagnostic work disabled
  unless a diagnostics path is configured.
- Centralized `identity-v1-drift-guard` as the default local/evaluator/render
  profile; Modal uses the same default. Baseline remains a one-flag rollback.

## Reproducible results

### Same-scene CPU, 1,000 ticks, four cameras, stride 1

| Metric | v1 baseline | drift guard | Change |
|---|---:|---:|---:|
| IDF1 | 0.539 | 0.574 | +0.035 |
| IDP | 0.484 | 0.517 | +0.033 |
| IDR | 0.608 | 0.644 | +0.036 |
| HOTA | 0.346 | 0.359 | +0.013 |
| AssA | 0.318 | 0.340 | +0.022 |
| MOTA | 0.523 | 0.530 | +0.007 |
| ID switches | 181 | 161 | -20 (-11.0%) |
| TP / FP / FN | 18,095 / 7,318 / 2,151 | 18,052 / 7,169 / 2,194 | FP -149 |
| Position RMSE | 0.443 m | 0.440 m | -0.003 m |

The four 250-tick windows all improved IDF1 with the guard. The third window
had the same 56 switches and MOTA lower by 0.006, so the change reduces rather
than eliminates scene ambiguity.

### Same cached detector output, 300 ticks

| Metric | v1 baseline | drift guard |
|---|---:|---:|
| IDF1 | 0.7050 | 0.7094 |
| HOTA | 0.4803 | 0.4832 |
| MOTA | 0.5945 | 0.5946 |
| ID switches | 39 | 33 |
| Fragmentations | 54 | 53 |

### CPU and bounded-state checks

- Isolated 100-tick detector run: 1.32 tick/s, 400 camera frames, peak RSS
  520 MiB. Detection consumed 58.7 of 75.6 seconds, so YOLO CPU inference is
  the main bottleneck; the clock guard is not the dominant cost.
- A synthetic 3,600-tick identity-churn stress run stabilized at about 70.71
  MiB, 24 retained histories and 604 trajectory points.
- Synthetic tracklet matching (four cameras, five people, 1,000 stored points)
  took 30.3 ms steady-state for 15 accepted matches.
- Regression suite: 96 tests plus three subtests passed.

## Commands

Live CPU/Rerun, every frame:

```powershell
uv run demo --device cpu --stride 1 --profile identity-v1-drift-guard
```

Bounded evaluator with decision log:

```powershell
uv run evaluate --device cpu --max-frames 1000 `
  --profile identity-v1-drift-guard `
  --metric-window-frames 250 --cumulative-frames 300 600 1000 `
  --identity-events data/cpu-audit/identity-events.jsonl `
  --cameras camera_0001 camera_0007 camera_0008 camera_0010
```

Rollback A/B:

```powershell
uv run evaluate --device cpu --max-frames 1000 `
  --profile identity-v1-baseline `
  --cameras camera_0001 camera_0007 camera_0008 camera_0010
```

The two 60-frame renderer smoke artifacts are
`data/cpu-audit/id-drift-baseline-debug.mp4` and
`data/cpu-audit/id-drift-guard-debug.mp4`. Both are valid 1500x540, 30 fps,
60-frame MPEG-4 files. They are byte-identical because the policies do not
diverge in the first two seconds; the 300/1,000-tick metrics above, rather than
this short clip, are the meaningful before/after evidence.

## Remaining limitations and next experiment

Without appearance information, two people with similar motion crossing in the
same calibrated area are not always identifiable. The dominant remaining issue
is local fragmentation, but the tested generic motion association was harmful.
The next justified experiment is camera-specific local association calibration
using the diagnostic fields (IoU/cost, NIS, confidence, scale and boundary),
optimized against cumulative IDF1/HOTA on up to 1,000 CPU ticks. Do not increase
retention or suppress new IDs globally until that experiment demonstrates both
higher IDF1/HOTA and no unacceptable recall loss.
