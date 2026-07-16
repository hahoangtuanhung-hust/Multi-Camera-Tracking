"""Educational multi-camera 3D people tracking.

Replicates the core ideas of NVIDIA DeepStream's Multi-View 3D Tracking
(MV3DT) — per-camera detection + tracking, ground-plane projection via
homography, and cross-camera fusion into globally consistent track IDs —
using an open-source stack that runs on any laptop (no NVIDIA GPU needed).
"""

__version__ = "0.1.0"
