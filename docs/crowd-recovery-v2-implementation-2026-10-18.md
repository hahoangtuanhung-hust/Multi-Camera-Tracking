# Crowd recovery v2 implementation — 2026-10-18

## Scope and release policy

Default remains `hybridsort`. Detector remains YOLO26n, 640, confidence 0.25.
Stationary reconnect is not re-enabled. No predicted/OCC bbox is added as a
MOT observation. Existing user changes are preserved.

This work adds experimental mechanisms, not a demonstrated tracking-quality
upgrade. Do not promote a candidate simply because IDSW or unresolved-drop
counters decrease. Recall, IDF1 and association accuracy are independent gates.

**Decision: reject the behavior-changing candidate.** After controlled screening,
`hybridsort-crowd-recovery-v2` is rolled back to
`hybridsort-late-binding-balanced` plus lifecycle observability only.
`hybridsort-crowd-terminal-batch-experimental` preserves the rejected terminal
experiment for debugging; `hybridsort-crowd-terminal` is its no-batch ablation.
`hybridsort-crowd-local-cascade` is also rejected and is not composed into v2.
No production-quality improvement is claimed. The 4,000-tick CPU replay and
T4 controlled A/B have **not** been run for this implementation: screening
already failed, so the behavior-changing variants were rolled back first.
The full requested optimization/benchmark programme is therefore not complete.

## Confirmed causes

- Legacy `_promote_ready_provisionals` increments timeout-release attempts on
  repeated ticks; 4,291 attempts are not 4,291 unique identities.
- Legacy commit delay can expire before an identity receives a decision.
  Negative hypotheses were counted as unresolved drops, including tracklets
  that disappeared before publication. End-of-stream previously could not
  request a final identity decision from fusion.
- A retained local identity could enter the ordinary high-confidence matcher
  when no appearance descriptor was available. The optional cascade excludes
  stale tracks regardless of descriptor availability.
- Strong trajectory validation is not equivalent to identifying a false
  person. Detector/homography jitter and contaminated member history can fail
  a real person's trajectory. A bounded reject policy can therefore suppress
  true observations and artificially improve IDSW.

## Implemented mechanisms

`identity_lifecycle.py` maintains unique original provisional records, canonical
internal IDs, source members, timestamps, candidate evidence, confirmations and
terminal outcomes. Private-private merges preserve the original records;
terminal decisions are idempotent. Conservation is checked against unique
births. The terminal report ring is bounded (10,000 by default); omitted-record
counts are explicit. Unique promoted records do not equal public allocations:
several original provisional IDs can resolve to one canonical public identity.

Optional terminal binding uses private -> bind-pending -> bound/promoted/rejected
states, a bounded grace window, and one-to-one Hungarian assignment. Evidence
includes distance, motion feasibility, NIS, velocity/direction, acceleration,
time gap, same-camera appearance and ownership conflict. Absolute cost,
alternative margin and consecutive distinct observation confirmations gate
binding. Repeated flushes do not manufacture confirmations. Existing trusted
reconnect/merge paths remain in place; this is not a replacement of all fusion
association paths.

The commit buffer consumes explicit decisions and aliases before committing.
The terminal profile requires a buffer covering the decision horizon. At EOF
fusion finalizes disappeared tracklets before the buffer flush. A missing
terminal decision raises an error in strict mode. Intentional rejection,
backfill and unresolved legacy drops have separate counters. Explicit reject
is an operational outcome, **not** ground-truth proof that a person is false.

Local cascade is a separate ablation: established visible tracks first, then
lost tracks using motion/NIS, trusted observation rescue, appearance recovery,
and remaining births. Merged ambiguous boxes can coast established tracks
without creating a third local ID. Retention and render windows remain separate.

Appearance adds a pluggable TorchScript backend with checkpoint SHA-256
verification, batch inference, event triggering, crop quality gates and bounded
same-camera galleries. HSV remains the default. No trained ReID checkpoint is
bundled or downloaded; no quality improvement is claimed for this adapter.
Contract: trained person-ReID encoder, RGB float32 NCHW 256x128,
ImageNet normalization, finite NxD embeddings. Set `appearance_backend`,
`appearance_model_path`, `appearance_model_sha256`, `appearance_event_triggered`
and `appearance_quality_filter_enabled` through `PipelineConfig` for a separate
calibrated experiment. This is not yet a packaged Modal ReID deployment.

## Files in this implementation

- `src/multicam_tracker/identity_lifecycle.py`: ledger and terminal matcher.
- `fusion.py`, `demo.py`: lifecycle hooks, finalizer and tick decision transport.
- `identity_commit.py`: strict decision contract, backfill and frame accounting.
- `config.py`: independent opt-in flags and ablation profiles.
- `hybridsort.py`, `appearance.py`: local cascade, encoder adapter and galleries.
- `evaluate.py`, `modal_app.py`, `benchmarks/replay_cpu.py`: lifecycle export.
- `reporting.py`: committed evaluator-stream hashes including EOF backfill.
- `tests/test_terminal_binding.py`, `tests/test_crowd_cascade.py`,
  `tests/test_appearance_backend.py`: new tests.

Other dirty files predate this implementation; do not treat the full working-tree
diff as exclusively this work.

## Validation status

Full unittest suite: **236 tests passed** after rollback, including a new
five-second dormant-owner recovery test and committed-stream hash regression.
New coverage includes true interior birth, false short track, disappeared valid
tracklet EOF promotion/backfill, stationary-fragment reconnection, timeout with
a plausible owner, competing provisional identities, ambiguous margins,
same-camera conflict, repeated-flush idempotence, alias conservation, missing
terminal decision, merged local detection, stale-track exclusion, crop rejection
and neural batch inference contract. Synthetic tests are not substitutes for
annotated camera_0008 crossing event slices.

Exploratory 300-tick terminal run was deterministic, but source changed during
execution; it is **not** controlled A/B evidence. It showed IDF1 0.681259,
recall 0.582811, IDSW 3, 2,996 intentionally rejected hypotheses, 620 backfilled
hypotheses, 143 original provisional identities and 18 public allocations.
Of 92 rejected original provisional identities, 88 failed trajectory/length
requirements and four had ambiguous owners at the deadline. This motivated
controlled screening, not promotion.

Controlled screening uses source SHA-256
`6c9e92af69d0c7ce8391729ce59157d2d8e700426e371f4128dfb138987fb458`,
the identical `detections-low-floor-300.pkl` and exact video frames, stride 1,
offset 0. Results are saved under `data/cpu-audit/v2-controlled-*-300.json`.

- Reference: IDF1 0.847419, HOTA 0.549253, AssA 0.639069, precision 0.868708,
  recall 0.858733, IDSW 6, Frag 13, public allocations 26.
- Lifecycle audit only: same metrics and both live/committed stream hashes as
  reference. Both runs passed exact determinism. Ledger: 96 unique births,
  47 promoted original records, 42 bound original records, seven unresolved
  errors, no pending original records. Those seven errors remain visible;
  observability alone does not repair them.
- Local cascade: IDF1 0.704075, HOTA 0.475545, AssA 0.556952, precision 0.831728,
  recall 0.826374, IDSW 20, Frag 20, public allocations 30. Both runs passed
  exact determinism. **Reject**: worse identity continuity and detection recall.
- Terminal + batch: IDF1 0.681259, HOTA 0.522998,
  AssA 0.688252, precision 0.892144, recall 0.582811, IDSW 3, Frag 4,
  public allocations 18. **Reject** despite better AssA/IDSW: IDF1 and recall
  regress severely. Both runs passed exact determinism. These files used the
  name `hybridsort-crowd-recovery-v2` **before rollback**; their effective config
  and source fingerprint identify the rejected policy, not current v2.
- Terminal without batch: IDF1 0.681259, HOTA 0.524149, AssA 0.689784,
  precision 0.892144, recall 0.582811, IDSW 3, Frag 4, public allocations 19.
  One run only, not a determinism claim. It also rejected 2,996 hypotheses;
  the recall regression is already present without Hungarian binding.

Reference/audit committed prediction hash:
`aabf8222fbb70cb5e82ed80a1724740e6747d9bc68112630666c3d99483adea6`.
Committed assignment hash:
`94c440aea45f21d1b5ed9d878c894fe036ed9f6ecadb332b248403997bff292a`.
The controlled audit proves output neutrality on this slice only, not a
long-run quality improvement.

Cache file SHA-256:
`edfd52b6513a83d9f073ba5d67d3dc48aec354aec818c1637f045fe813dd2480`.
Terminal + batch committed prediction hash:
`1b685363e73083736396723f67a3c13f1f0afcd3d2c30b5ed291099679cc4a00`;
committed assignment hash:
`54b3c78e6e9720157935fcad3947472c9c4f84504f1f5a484c2f92ac2770d88f`.

## Reproduce local replay

Post-rollback verification is saved as `v2-rollback-reference-300.json` and
`v2-rollback-candidate-300.json` under `data/cpu-audit/`, both using source hash
`3204f70b4ad2e0d6fc1a10464bcdb285dd9e7bbb9ebb46453aa6f7fb026d263b`.
Current v2 and reference have identical metrics and committed prediction and
assignment hashes. Current v2 passed its second deterministic run. Lifecycle
conservation is true with zero pending records; unresolved outcomes are still
reported honestly. This verifies rollback, **not** improved tracking quality.
Python compilation and `git diff --check` also passed.

From the repository root in PowerShell (trusted local pickle only):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q

.\.venv\Scripts\python.exe benchmarks/replay_cpu.py `
  --detections data/cpu-audit/detections-low-floor-300.pkl `
  --out data/cpu-audit/v2-reference-300.json --only-current `
  --profile hybridsort-late-binding-balanced --video-frames `
  --verify-determinism --skip-memory-profile --skip-stage-diagnostics
```

Run each candidate with a different output path and `--profile`. The cache's
0.15 storage floor is filtered to the selected profile's unchanged 0.25 floor.
Freeze source files during each comparison. Replay now checks start/end source
fingerprints and reports both live and committed output hashes. For 4,000 ticks,
use `detections-low-floor-4000.pkl`; do not interpret a 300-tick score as passing
the 4,000-tick release gate. Concurrent CPU screens are quality checks, not
comparable deployment throughput measurements.

## T4 commands (not evidence of execution)

Modal packages current local source on each invocation. Use `--upload` only if
the input videos/calibration/model on its volume need updating. The two profiles
must run from unchanged source and identical inputs. Existing Modal evaluator
reruns detection; check detector/input provenance and repeat committed hashes.
It does not currently replay a shared detection cache on T4.

```powershell
$env:PYTHONUTF8 = '1'
uv run --extra modal modal run modal_app.py `
  --evaluate --max-frames 4000 --frame-offset 0 `
  --tracking-profile hybridsort-late-binding-balanced `
  --cameras camera_0001,camera_0007,camera_0008,camera_0010 `
  --metric-window-frames 500 --cumulative-frames 500,1000,2000,4000 `
  --metrics-out mot-metrics-v2-reference-4000-t4.json

uv run --extra modal modal run modal_app.py `
  --evaluate --max-frames 4000 --frame-offset 0 `
  --tracking-profile hybridsort-crowd-recovery-v2 `
  --cameras camera_0001,camera_0007,camera_0008,camera_0010 `
  --metric-window-frames 500 --cumulative-frames 500,1000,2000,4000 `
  --metrics-out mot-metrics-v2-candidate-4000-t4.json

# Safe reference render, not the rejected terminal experiment:
uv run --extra modal modal run modal_app.py `
  --seconds 133.34 --start 0 --stride 1 `
  --tracking-profile hybridsort-late-binding-balanced `
  --cameras camera_0001,camera_0007,camera_0008,camera_0010 `
  --out render-reference-4000-4cam.mp4
```

Repeat each evaluation to a new output filename to test committed-stream
determinism. Render duration is approximately 4,000 ticks at 30 FPS, not an
exact frame-count contract. Rollback: select `hybridsort-late-binding-balanced`.

## Outstanding gates and risks

No T4 runtime, CUDA-memory or process-RSS claim may be inferred from CPU cache
replay. Long-run 4,000-tick and T4 gates need their own evidence. Do not run the
10-camera promotion test or detector variants until the 4-camera gate passes.
ReID calibration requires real trained weights and same-camera positives and
hard negatives; cross-camera ReID remains disabled. Full camera_0008 annotated
event-slice validation and strict end-to-end memory bounding remain outstanding:
the terminal report/gallery are bounded, but alias/commit-accounting dictionaries
retain history across a run.
