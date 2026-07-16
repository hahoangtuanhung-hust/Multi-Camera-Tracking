# Multi-Camera 3D People Tracking — on your laptop

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Runs on Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS-black.svg)](#performance)
[![YOLO26](https://img.shields.io/badge/detector-YOLO26-orange.svg)](https://docs.ultralytics.com/models/yolo26/)
[![Rerun](https://img.shields.io/badge/viewer-Rerun%200.34-purple.svg)](https://rerun.io)

**Watch multiple cameras agree on who is who.** A small, readable
implementation of the technique behind NVIDIA DeepStream's
[Multi-View 3D Tracking (MV3DT)](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html):
people are detected and tracked in every camera, projected onto a shared
ground plane, and **fused into one global identity per person** — rendered
live as a bird's-eye view. No NVIDIA GPU, no Linux, no SDK: Python on any
laptop (Apple Silicon included).

![Four warehouse cameras with color-coded tracked people and a bird's-eye map showing fused trajectories](docs/preview.gif)

*Same color = same person, in every camera and on the map. The data is
NVIDIA's synthetic warehouse ([PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces),
CC-BY-4.0) — the same scenery NVIDIA uses to demo MV3DT.*

## Table of contents

- [Why this exists](#why-this-exists)
- [Quickstart](#quickstart)
- [The live viewer](#the-live-viewer)
- [Architecture](#architecture)
- [Pipeline deep dive](#pipeline-deep-dive)
- [This repo vs. NVIDIA DeepStream MV3DT](#this-repo-vs-nvidia-deepstream-mv3dt)
- [Evaluation](#evaluation)
- [Performance](#performance)
- [Repository layout](#repository-layout)
- [Configuration reference](#configuration-reference)
- [Limitations & where to take it](#limitations--where-to-take-it)
- [Dataset & licenses](#dataset--licenses)
- [References](#references)

## Why this exists

Multi-target multi-camera (MTMC) tracking is a foundational problem in
real-time video analytics — warehouses, retail, traffic, sports. Production
stacks like DeepStream solve it impressively but run only on NVIDIA
hardware, and their pipelines are configuration-heavy, which makes the
*ideas* hard to see. This repo strips the problem to ~1,500 lines of
commented Python so you can read, run, and modify every stage yourself.

## Quickstart

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/) (or plain pip).

```bash
git clone https://github.com/alexandergg/multicam-3d-tracking && cd multicam-3d-tracking
uv sync                  # install pinned dependencies
uv run download-data     # ~1.1 GB: 4 cameras + calibration + ground truth
uv run demo              # opens the live Rerun viewer
```

Useful flags:

```bash
uv run download-data --full            # all 10 cameras (~2.6 GB)
uv run demo --stride 3 --device cpu    # slower machines
uv run demo --save session.rrd         # record instead of opening the viewer
uv run check-calibration --save-images # sanity-check the dataset geometry
uv run evaluate                        # metrics vs. ground truth
uv run render-preview --gif out.gif    # composite MP4/GIF like the one above
```

## The live viewer

One command gives you a fully interactive session — pause, scrub the
timeline, orbit the 3D world, and inspect any entity:

![Rerun viewer: 3D bird's-eye world with camera frusta and track trails, next to four synchronized camera feeds with global-ID-colored boxes](docs/rerun-viewer.png)

What you are looking at:

- **Left — world view.** The shared ground plane in meters. Each pyramid is
  a camera frustum placed from the dataset's intrinsics/extrinsics. Each
  dot is a *fused* person with their trail; the label `#67 (2 cams)` means
  global track 67 is currently corroborated by two cameras.
- **Right — camera grid.** The synchronized feeds with detection boxes
  painted in their **global** track color. When a person walks from one
  camera into another, the color and ID follow them.
- **Bottom — timeline.** Every entity is logged against the source frame
  index, so you can replay and debug any moment of the run.

## Architecture

### System overview

```mermaid
flowchart LR
    subgraph tick ["per tick — all cameras in lockstep"]
        V["<b>SyncedVideoReader</b><br/>video.py<br/><i>N BGR frames</i>"]
        D["<b>PersonDetector</b><br/>detector.py<br/><i>YOLO26n, one batched call</i>"]
        T["<b>ByteTracker × N</b><br/>bytetrack.py<br/><i>local IDs per camera</i>"]
        P["<b>CameraCalibration</b><br/>calibration.py<br/><i>foot pixel → world meters</i>"]
        F["<b>GlobalTrackManager</b><br/>fusion.py<br/><i>global IDs on ground plane</i>"]
        V --> D --> T --> P --> F
    end

    DS[("PhysicalAI-SmartSpaces<br/>4-10 synced MP4s +<br/>calibration + GT")] -->|download.py| V
    F --> RL["<b>RerunLogger</b><br/>viz.py"]
    RL --> RV(["Rerun viewer / .rrd"])
    F -.-> EV["evaluate.py<br/><i>metrics vs GT</i>"]
    F -.-> RP["render_preview.py<br/><i>MP4 / GIF</i>"]
```

### Data flow per tick

```mermaid
sequenceDiagram
    participant R as SyncedVideoReader
    participant Y as PersonDetector
    participant B as ByteTracker per camera
    participant C as CameraCalibration
    participant G as GlobalTrackManager
    participant Z as RerunLogger

    R->>Y: {cam: frame} — N synchronized BGR frames
    Y->>B: per-cam detections (M, 5) [x1 y1 x2 y2 conf]
    Note over B: KF predict → IoU Hungarian →<br/>low-conf rescue → lifecycle
    B->>C: confirmed Tracks (local_id, bbox)
    C->>G: CameraObservation(cam, local_id, world_xy)
    Note over G: sticky affinity → clustering →<br/>gated Hungarian → KF fusion
    G->>Z: GlobalTracks (gid, world_xy, trail, #cams)
    Z-->>Z: images, Boxes2D, Points3D,<br/>LineStrips3D, frusta
```

### Data contracts

| Type | Producer | Shape / fields | Meaning |
|---|---|---|---|
| detections | `PersonDetector` | `(M, 5)` float32 `[x1, y1, x2, y2, conf]` | people found in one frame |
| `Track` | `ByteTracker` | `track_id`, Kalman box, `hits`, `misses`, `confirmed` | one person, one camera, stable local ID |
| `CameraObservation` | `demo.run_pipeline` | `camera`, `local_id`, `world_xy` (m), `bbox_xyxy`, `conf` | a confirmed track projected to the floor |
| `GlobalTrack` | `GlobalTrackManager` | `gid`, fused Kalman `world_xy`, `trail`, supporting `cameras` | one *physical person* across all cameras |
| `affinity` | `GlobalTrackManager` | `(camera, local_id) → gid` | fusion output that lets the viewer color per-camera boxes globally |

### Key design decisions

| Decision | Choice | Rationale / trade-off |
|---|---|---|
| Fusion strategy | **Late fusion** (fuse tracks, not pixels) | Matches DeepStream MV3DT; modular, camera dropout degrades gracefully. Early fusion (BEV feature projection) is more accurate on dense scenes but monolithic. |
| Fusion topology | **Centralized** manager | One readable class. MV3DT distributes this over MQTT peers — great production property, unnecessary complexity for learning. |
| Association cost | **Geometry only** (world-plane distance) | The calibrated dataset makes geometry strong enough (RMSE 0.28 m). ReID embeddings are a documented extension, not a v1 dependency. |
| Per-camera tracker | **Vendored ~200-line ByteTrack** | The tracker *is* course material; a dependency would hide it. Also lets one batched detector feed N trackers (libraries typically tie tracker state to model instances). |
| Detection | **One batched YOLO26n call** for all cameras | Amortizes per-call overhead exactly like DeepStream's `nvinfer` stream batching; NMS-free architecture removes a postprocessing stage. |
| Time base | **Tick = synchronized frame index** | The dataset is frame-synced, so no timestamp alignment is needed; `stride` trades temporal resolution for throughput. |
| Visualization | **Rerun** | Camera grid + 3D world + timeline scrubbing in one dependency; a custom web UI would cost hundreds of lines and lose replay. |
| State estimation | **One generic constant-velocity Kalman class** | Reused at two altitudes: pixel boxes (`bytetrack.py`) and world meters (`fusion.py`) — one motion model to learn, two applications. |

### Coordinate systems

Three frames of reference, two conversions:

1. **Image plane** (pixels, per camera) — where detection and IoU tracking live.
2. **World ground plane** (meters, shared) — where fusion lives. Reached via
   each camera's 3×3 homography `H` (floor → pixels), inverted to map a
   person's *foot point* (bbox bottom-center) into meters.
3. **Full 3D world** (meters) — used only for visualization: camera frusta
   are placed from `K` (intrinsics) and `[R|t]` (extrinsics, world→camera,
   inverted to get camera pose).

## Pipeline deep dive

### 1. Synchronized ingest (`video.py`)

The dataset's cameras are frame-synchronized, so "time" advances by
reading frame *N* from every video at once. Decoding is the silent
bottleneck of multi-camera work — 4×1080p30 is ~250 Mpixels/s before any
AI runs — so the reader skips frames cheaply (`grab()`) to hold a
real-time budget. DeepStream solves this same problem with hardware
decoders (NVDEC) and its stream batcher (`nvstreammux`).

### 2. Detection (`detector.py`)

[YOLO26](https://docs.ultralytics.com/models/yolo26/)-nano detects people
in **all cameras with one batched inference call** per tick. YOLO26 is
end-to-end (no NMS postprocessing) and runs on Apple's GPU via PyTorch
MPS, or on CPU with `--device cpu`.

### 3. Single-camera tracking (`bytetrack.py`)

Detection says *where people are*; tracking says *who is who over time*.
Each camera runs a ~200-line simplified [ByteTrack](https://arxiv.org/abs/2110.06864):

- A **constant-velocity Kalman filter** per person predicts the next box,
  so matching survives motion and missed detections.
- **Hungarian matching** on IoU associates predictions with detections.
- **Two-stage association** (ByteTrack's key idea): unmatched tracks get a
  second chance against *low-confidence* detections — which is usually
  what an occluded person looks like — rescuing tracks that a single-pass
  tracker would drop.

Per-camera IDs still fragment under long occlusions (watch the racks in
camera_0001). That's expected — and it's exactly what the next stage fixes.

### 4. From pixels to meters (`calibration.py`)

The geometric heart of the demo. Each camera ships a 3×3 **homography** H
mapping world floor coordinates (meters) to image pixels; its inverse maps
a pixel *on the floor* back to world meters. A person's bounding-box
bottom-center — their **foot point** — is (approximately) on the floor, so:

```
world_xy = H⁻¹ @ [foot_u, foot_v, 1]   (then divide by the scale term)
```

Suddenly every camera speaks the same language: positions in meters on a
shared map. `uv run check-calibration` verifies this on the dataset's
ground truth (median error ≈ 0.2 m).

> DeepStream goes further here: SV3DT fits a 3D human cylinder model using
> the full 3×4 projection matrix, which stays accurate when feet are
> occluded. And when calibration doesn't exist, its AutoMagicCalib tool
> estimates it from footage of people walking around.

### 5. Cross-camera fusion (`fusion.py`) — the interesting part

Once all tracks live on one ground plane, "is camera A's track 7 the same
person as camera B's track 3?" becomes *"are they standing in the same
spot?"* Each tick:

1. **Sticky re-association** — a (camera, local ID) pair that already
   belongs to a global track keeps it while geometry agrees. Cheap and
   handles ~95% of observations. (MV3DT analog: peers negotiate a global
   ID, then keep it.)
2. **Clustering** — remaining observations near each other on the floor,
   from different cameras, are grouped: probably the same person.
3. **Hungarian assignment** with a distance gate matches clusters to
   predicted track positions. (MV3DT analog: tracklet matching by RMSE.)
4. **Kalman fusion + lifecycle** — matched tracks fuse their cameras'
   measurements; new clusters become tentative tracks (shown only after
   `fusion_min_hits` ticks); unseen tracks age out.

This is **late fusion** — fuse *tracks*, not pixels — which is what
DeepStream MV3DT does, and it's why the system degrades gracefully: any
single camera can lose a person (occlusion, FOV exit) and the global ID
survives on the other views. The academic frontier goes the other way
(*early fusion*: project image features to the ground plane and detect
once — see [MVDet](https://arxiv.org/abs/2007.07247),
[EarlyBird](https://arxiv.org/abs/2310.13350)), trading modularity and
scalability for accuracy on dense scenes.

### 6. Visualization (`viz.py`)

[Rerun](https://rerun.io) renders the whole state in one window: camera
grid with boxes colored by *global* ID, plus a 3D world view with camera
frusta (from the dataset's intrinsics/extrinsics) and track trails. You
get pause/scrub/inspect on the timeline for free — invaluable for
debugging trackers.

## This repo vs. NVIDIA DeepStream MV3DT

| Stage | DeepStream 9.1 MV3DT | This repo |
|---|---|---|
| Hardware | NVIDIA GPU (x86 Linux / Jetson) | Any laptop (MPS / CPU) |
| Ingest | GStreamer + NVDEC, `nvstreammux` | OpenCV `VideoCapture` |
| Detection | PeopleNet Transformer / RT-DETR (TensorRT) | YOLO26n (PyTorch, batched) |
| Per-camera tracking | NvMultiObjectTracker, SV3DT 3D human model | Simplified ByteTrack (this repo) |
| Cross-camera | Distributed peers over MQTT, RMSE tracklet matching, visibility-weighted Kalman fusion | Centralized: sticky affinity + clustering + gated Hungarian + Kalman |
| Calibration | AutoMagicCalib (no checkerboards) | Provided by the dataset |
| Output | Kafka protobuf + OSD + BEV visualizer | Rerun viewer / .rrd / MP4 |
| Scale | 10s–100s of cameras, production | 4–10 cameras, education |

## Evaluation

`uv run evaluate` Hungarian-matches fused tracks to ground truth (1 m gate)
per frame — the pipeline under test is byte-identical to the live demo
(both consume `demo.run_pipeline`). Default 4-camera subset, first 1,200
frames:

| Metric | Value |
|---|---|
| Precision | 0.94 |
| Recall | 0.80 |
| World-plane position RMSE | 0.28 m |
| Global ID switches | ~60 over 24 people / 40 s |

Recall losses concentrate at FOV edges and behind the racks — with only 4
of the 10 cameras there are blind spots. Download `--full` and re-run to
see coverage (and fusion quality) improve.

## Performance

Measured on an Apple Silicon laptop (M-series, MPS), 4×1080p streams,
`imgsz=640`, `stride=2`:

| Stage | Throughput |
|---|---|
| Detection only (batched, incl. decode) | ~78 ticks/s |
| Full pipeline incl. Rerun logging | ~37 ticks/s |
| Real-time requirement at stride 2 | 15 ticks/s |

Scaling behavior: decode cost grows linearly with camera count (CPU),
detection is one batched GPU call (sub-linear in N), fusion is
O(tracks × observations) Hungarian on a handful of entities — negligible.
The first knob to turn on slower machines is `--stride`.

## Repository layout

```
src/multicam_tracker/
├── config.py             # every tunable, one dataclass
├── download.py           # HF dataset subset fetcher
├── calibration.py        # homographies, K/[R|t], ground truth parsing
├── video.py              # synchronized N-stream reader
├── detector.py           # batched YOLO26 wrapper
├── bytetrack.py          # Kalman filter + simplified ByteTrack
├── fusion.py             # cross-camera association — the core idea
├── viz.py                # Rerun blueprint + logging
├── demo.py               # the pipeline as one readable loop + CLI
├── evaluate.py           # metrics vs ground truth
├── check_calibration.py  # dataset geometry sanity checks
└── render_preview.py     # composite MP4/GIF clips
```

## Configuration reference

All knobs live in `config.py` (`PipelineConfig`); the important ones:

| Knob | Default | Effect |
|---|---|---|
| `stride` | 2 | process every Nth frame; throughput ↔ temporal resolution |
| `conf_threshold` | 0.25 | detection floor; low values feed ByteTrack's rescue stage |
| `track_high_conf` | 0.5 | stage-1 association + track birth threshold |
| `track_max_age` | 30 ticks | how long a camera track survives occlusion (~2 s) |
| `fusion_gate_m` | 2.0 m | max world distance for cross-camera association |
| `fusion_merge_m` | 1.0 m | cluster radius when grouping new observations |
| `fusion_max_age` | 45 ticks | global track memory through total occlusion (~3 s) |
| `fusion_min_hits` | 3 | ticks of support before a global track is shown |

## Limitations & where to take it

- **No appearance features (ReID).** Fusion is purely geometric. Add a
  ReID embedding (e.g. OSNet, or Ultralytics' `yolo26n-reid` weights) to
  the association cost for crowded scenes where geometry is ambiguous.
- **Foot-point occlusion.** A cut-off bounding box projects wrongly;
  SV3DT's 3D human model is the principled fix.
- **Offline files, not live streams.** Swap `SyncedVideoReader` for RTSP
  readers with timestamp alignment to go live.
- **Centralized fusion.** MV3DT runs fusion distributed on the cameras
  themselves (MQTT peer-to-peer) — a great systems exercise.

## Dataset & licenses

- Code: MIT.
- Data: [NVIDIA PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)
  (CC-BY-4.0), downloaded at setup — not redistributed here. A synthetic
  (Isaac Sim) warehouse from the 2024 AI City Challenge MTMC track.
- Models & library: YOLO26 weights and the `ultralytics` package are
  [AGPL-3.0](https://github.com/ultralytics/ultralytics) — fine for learning
  and open source; swap the detector if you need a permissive-only stack.

## References

- NVIDIA, [Multi-View 3D Tracking in DeepStream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html) · [Build a Multi-Camera 3D Tracking Application with DeepStream 9.1 Skills](https://developer.nvidia.com/blog/build-a-multi-camera-3d-tracking-application-with-nvidia-deepstream-9-1-skills/)
- Zhang et al., [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864), ECCV 2022
- Aharon et al., [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651), 2022
- Zhou et al., [Omni-Scale Feature Learning for Person Re-Identification](https://arxiv.org/abs/1905.00953) (OSNet), ICCV 2019
- Chavdarova et al., [WILDTRACK: A Multi-camera HD Dataset for Dense Unscripted Pedestrian Detection](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.pdf), CVPR 2018
- Hou et al., [Multiview Detection with Feature Perspective Transformation](https://arxiv.org/abs/2007.07247) (MVDet), ECCV 2020
- Teepe et al., [EarlyBird: Early-Fusion for Multi-View Tracking in the Bird's Eye View](https://arxiv.org/abs/2310.13350), WACVW 2024

---

Built by [Alexander Gonzalez](https://github.com/alexandergg). If this
helped you understand multi-camera tracking, a ⭐ helps others find it.
