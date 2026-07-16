# Multi-Camera 3D People Tracking — on your laptop

**Watch multiple cameras agree on who is who.** This repo is a small,
readable implementation of the technique behind NVIDIA DeepStream's
[Multi-View 3D Tracking (MV3DT)](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html):
people are detected and tracked in every camera, projected onto a shared
ground plane, and **fused into one global identity per person** — rendered
live as a bird's-eye view. No NVIDIA GPU, no Linux, no SDK: Python on any
laptop (Apple Silicon included).

![Four warehouse cameras with color-coded tracked people and a bird's-eye map showing fused trajectories](docs/preview.gif)

*Same color = same person, in every camera and on the map. The data is
NVIDIA's synthetic warehouse ([PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces),
CC-BY-4.0) — the same scenery NVIDIA uses to demo MV3DT.*

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
git clone <this-repo> && cd <this-repo>
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

## How it works

```mermaid
flowchart LR
    subgraph "per-camera stages (run every tick, all cameras in lockstep)"
        V["video.py<br/>synced frame batch"] --> D["detector.py<br/>YOLO26, one batched call"]
        D --> T["bytetrack.py<br/>per-camera tracking<br/>(local IDs)"]
        T --> C["calibration.py<br/>foot pixel → world meters<br/>(homography)"]
    end
    C --> F["fusion.py<br/>cross-camera association<br/>(global IDs)"]
    F --> R["viz.py<br/>Rerun: camera grid + 3D bird's-eye view"]
```

### 1. Synchronized ingest (`video.py`)

The dataset's cameras are frame-synchronized, so "time" advances by
reading frame *N* from every video at once. Decoding is the silent
bottleneck of multi-camera work — 4×1080p30 is ~250 Mpixels/s before any
AI runs — so the reader skips frames cheaply (`grab()`) to hold a
real-time budget. DeepStream solves this same problem with hardware
decoders (NVDEC) and its stream batcher (`nvstreammux`).

### 2. Detection (`detector.py`)

[YOLO26](https://docs.ultralytics.com/models/yolo26/)-nano detects people
in **all cameras with one batched inference call** per tick — the same
trick DeepStream's `nvinfer` uses to amortize per-call overhead across
streams. YOLO26 is end-to-end (no NMS postprocessing), and runs on Apple's
GPU via PyTorch MPS. On an M-series laptop this pipeline runs ~2.5× faster
than real time.

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

## Does it actually work? (measured)

`uv run evaluate` Hungarian-matches fused tracks to ground truth (1 m gate)
per frame. On the default 4-camera subset, first 1,200 frames:

| Metric | Value |
|---|---|
| Precision | 0.94 |
| Recall | 0.80 |
| World-plane position RMSE | 0.28 m |
| Global ID switches | ~60 over 24 people / 40 s |

Recall losses concentrate at FOV edges and behind the racks — with only 4
of the 10 cameras there are blind spots. Download `--full` and re-run to
see coverage (and fusion quality) improve.

## Repo tour

| File | What it teaches |
|---|---|
| `config.py` | Every tunable in one place |
| `download.py` | Fetching a subset of a HF dataset |
| `calibration.py` | Homographies, projection matrices, world↔image |
| `video.py` | Synchronized multi-stream reading, stride tricks |
| `detector.py` | Batched multi-camera inference |
| `bytetrack.py` | Kalman filters + Hungarian matching + ByteTrack |
| `fusion.py` | Cross-camera association — the core idea |
| `viz.py` | Rerun blueprints, entity paths, consistent ID colors |
| `demo.py` | The whole pipeline as one readable loop |
| `evaluate.py` | Honest metrics against ground truth |
| `check_calibration.py` | Verifying dataset geometry before trusting it |
| `render_preview.py` | Compositing shareable MP4/GIF clips |

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
