# Crowd recovery and real-frame replay audit — 2026-10-18

## Decision

Do not change the runtime default yet. `hybridsort-crowd-recovery` is the
4-camera T4 release candidate, while `hybridsort-late-binding-balanced` remains
the reference. The candidate is quality-neutral on the 300-tick real-frame
screen and deterministic, but the screen contains no qualifying local
appearance reappearance. A continuous 4,000-tick T4 run and a separate
10-camera run are therefore required before promotion.

The stationary appearance reconnect experiment is not part of the candidate.
It remains available as `hybridsort-crowd-stationary-reconnect` for diagnosis
only because it regressed IDF1, ID switches, and fragmentation.

## Reproducible screen

All rows below use one continuous 300-tick, four-camera cache, frame offset 0,
stride 1, the current evaluator, YOLO26n at 640 px, and exact source frames
decoded from the cache's `source_frame_index`. The cache model SHA-256 is
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Appearance conclusions are not taken from the older 1x1 placeholder replay.

| Ablation | IDF1 | HOTA | AssA | Precision | Recall | IDSW | Frag | tick/s | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Late-binding balanced reference | 0.8474 | 0.5493 | 0.6391 | 0.8687 | 0.8587 | 6 | 13 | 10.08 | Reference |
| Local reappearance only | 0.8474 | 0.5493 | 0.6391 | 0.8687 | 0.8587 | 6 | 13 | 8.53 | Keep behind profile |
| Global ownership guard only | 0.8474 | 0.5493 | 0.6391 | 0.8687 | 0.8587 | 6 | 13 | 8.82 | Keep behind profile |
| Stationary reconnect | 0.8361 | 0.5493 | 0.6370 | 0.8716 | 0.8629 | 8 | 18 | 8.47 | Reject |
| Safe crowd composition | 0.8474 | 0.5493 | 0.6391 | 0.8687 | 0.8587 | 6 | 13 | 8.69 | T4 candidate |
| Safe crowd + floor 0.20 | 0.8266 | 0.5326 | 0.6294 | 0.8599 | 0.8232 | 7 | 16 | 8.37 | Reject |
| Safe crowd + floor 0.15 | 0.8392 | 0.5503 | 0.6405 | 0.8647 | 0.8662 | 7 | 16 | 7.88 | Reject |

The safe composition and reference have identical prediction and assignment
stream hashes. A repeated real-frame run also produced identical hashes and
metrics. This establishes non-regression and determinism for the screen, not a
quality improvement or long-video generalization.

In the instrumented, non-tracing CPU runs, the reference measured 10.05
tick/s (p50 73.4 ms, p95 281.6 ms) and the crowd candidate measured 7.99
tick/s (p50 97.7 ms, p95 300.3 ms). The appearance path therefore has a real
CPU cost even though p95 increased less than the median. Python `tracemalloc`
reported a 54.27 MiB peak for the candidate, but that excludes native OpenCV
allocations and slowed the run to 2.77 tick/s; its latency is not a production
measurement. T4 CUDA and RSS numbers remain a release gate.

Artifacts are under `data/cpu-audit/`:

- `realframes-late-binding-balanced-300.json`
- `realframes-crowd-global-guard-v2-300.json`
- `realframes-crowd-recovery-safe-300.json`
- `realframes-crowd-recovery-stage-300.json`
- `realframes-crowd-recovery-determinism-300.json`
- `realframes-late-binding-balanced-latency-300.json`
- `realframes-crowd-recovery-latency-300.json`
- `realframes-crowd-recovery-memory-300.json`
- `realframes-crowd-mid-floor-300.json`
- `realframes-crowd-low-floor-300.json`

## Stage diagnosis

The stage diagnostic uses bbox IoU 0.5, so its per-camera coverage is a
different measurement from the final world-plane MOT gate.

| Scope | Detector recall | Local recall | Fusion recall | Public recall | Detector precision | Local precision |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 0.8573 | 0.7332 | 0.6421 | 0.5276 | 0.9040 | 0.9804 |
| camera_0001 | 0.9232 | 0.8442 | 0.8033 | 0.6940 | 0.9386 | 0.9942 |
| camera_0007 | 0.9058 | 0.8382 | 0.6225 | 0.5242 | 0.9352 | 0.9825 |
| camera_0008 | 0.8340 | 0.6476 | 0.5816 | 0.4483 | 0.9081 | 0.9884 |
| camera_0010 | 0.7401 | 0.6367 | 0.6361 | 0.5411 | 0.7770 | 0.9260 |

This screen confirms that camera_0008 loses the most coverage between
detector and local association, while camera_0007 loses substantial coverage
between projection and fusion. camera_0010 has the weakest detector precision
and recall. People shorter than 64 px have detector recall 0.8122, versus
approximately 0.858 for larger boxes. These are priorities for the next
detector/resolution cache, not justification for lowering the current floor:
the end-to-end floor ablations above failed the release gate.

## Implemented safeguards

- Real-frame replay opens each camera video and decodes the exact monotonic
  `source_frame_index`. Appearance-enabled replay refuses placeholder frames
  unless the caller explicitly opts into a structural smoke test.
- Normal Hybrid-SORT association always runs before appearance fallback.
  Retained tracks older than normal `track_max_age_s` are recovery-only and
  cannot re-enter BYTE/OCR association by accident.
- Local reappearance requires one-to-one assignment, geometry, appearance,
  margin, and consecutive evidence. Votes reset when the candidate location
  changes discontinuously.
- Global ownership protection is separate from appearance reconnect. A
  confirmed takeover cannot detach an owner and immediately reconnect the
  same observation to that protected GID in the same tick.
- Moderate descriptor drift is accepted; only a strong, consecutive
  same-camera conflict can protect the old owner. HSV is never compared across
  cameras and is not treated as FaceID/ReID.
- Observed outputs, retained identities, and render-only `OCC` predictions
  remain separate. Only observed global tracks enter MOT evaluation.
- Modal evaluation now records pipeline tick latency p50/p95/max, peak CUDA
  allocated/reserved memory, and process peak RSS.

## Why the old full-camera files do not prove a cause

`mot-metrics-hybridsort-4000v1-fullcam.json` and
`mot-metrics-hybridsort-4000v3-fullcam.json` both contain 4,000 ticks, ten
cameras, frame offset 0, and a 1.0 m match gate. v1 has no run metadata, source
fingerprint, model hash, effective config, dependency versions, or input
fingerprints; v3 has them. Consequently they are not a controlled A/B pair.

The metric change is nevertheless mechanically clear: v3 has 9,434 fewer
IDTP, 9,434 more IDFN, and 9,959 more IDFP. Therefore IDF1 can fall from
0.4708 to 0.3770 even while IDSW falls from 250 to 190. IDSW counts transition
events; it does not measure how long a person is assigned to a wrong or split
identity. Lower recall (0.8712 to 0.8457), lower precision (0.8353 to 0.8068),
and lower AssA (0.2476 to 0.1947) are consistent with the IDF1 loss. No single
code change can be assigned as the cause without rebuilding both baselines
from the same source/data/evaluator contract.

## Release gates for the 4,000-tick T4 run

Compare candidate and reference from the current dirty source in separate
outputs. Promote only if IDF1, HOTA, and AssA improve; IDSW/1k, fragmentation,
GID churn, median/p95 GIDs per GT, and false ownership takeovers do not worsen;
precision/recall do not materially regress; cumulative IDF1 does not collapse;
and p95 latency plus CUDA/RSS stay within the deployment budget. Also require
at least one observed reappearance match before claiming that local appearance
recovery is effective.

## PowerShell commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

.\.venv\Scripts\python.exe benchmarks\replay_cpu.py `
  --detections data\cpu-audit\detections-low-floor-300.pkl `
  --out data\cpu-audit\realframes-crowd-recovery-300.json `
  --only-current --profile hybridsort-crowd-recovery `
  --video-frames --verify-determinism

# Current code is packaged on every Modal run. --upload is needed only when
# local data/custom or yolo26n.pt changed.
uv run --extra modal modal run modal_app.py `
  --evaluate --max-frames 4000 --frame-offset 0 `
  --tracking-profile hybridsort-late-binding-balanced `
  --cameras camera_0001,camera_0007,camera_0008,camera_0010 `
  --metric-window-frames 500 `
  --cumulative-frames 500,1000,2000,4000 `
  --metrics-out mot-metrics-late-binding-balanced-4000-4cam-20261018.json

uv run --extra modal modal run modal_app.py `
  --evaluate --max-frames 4000 --frame-offset 0 `
  --tracking-profile hybridsort-crowd-recovery `
  --cameras camera_0001,camera_0007,camera_0008,camera_0010 `
  --metric-window-frames 500 `
  --cumulative-frames 500,1000,2000,4000 `
  --metrics-out mot-metrics-crowd-recovery-4000-4cam-20261018.json

uv run --extra modal modal run modal_app.py `
  --evaluate --max-frames 4000 --frame-offset 0 `
  --tracking-profile hybridsort-crowd-recovery `
  --cameras camera_0001,camera_0002,camera_0003,camera_0004,camera_0005,camera_0006,camera_0007,camera_0008,camera_0009,camera_0010 `
  --metric-window-frames 500 `
  --cumulative-frames 500,1000,2000,4000 `
  --metrics-out mot-metrics-crowd-recovery-4000-10cam-20261018.json

uv run --extra modal modal run modal_app.py `
  --seconds 133.34 --start 0 --stride 1 `
  --tracking-profile hybridsort-crowd-recovery `
  --cameras camera_0001,camera_0007,camera_0008,camera_0010 `
  --out render-crowd-recovery-4000-4cam-20261018.mp4
```

Rollback is configuration-only: select
`hybridsort-late-binding-balanced`. No schema or stored identity migration is
required.
