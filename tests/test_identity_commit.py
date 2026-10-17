import unittest
from types import SimpleNamespace

import numpy as np

from multicam_tracker.identity_commit import IdentityCommitBuffer


def _tick(
    frame_index: int,
    timestamp: float,
    hypotheses: dict[int, tuple[float, float]],
    *,
    aliases: dict[int, int] | None = None,
    affinity: dict[tuple[str, int], int] | None = None,
    observations: list[object] | None = None,
) -> dict[str, object]:
    if observations is None:
        observations = [
            SimpleNamespace(camera=camera, local_id=local_id)
            for camera, local_id in (affinity or {})
        ]
    return {
        "frame_index": frame_index,
        "capture_ts_s": timestamp,
        "identity_hypotheses": {
            gid: np.asarray(position, dtype=float)
            for gid, position in hypotheses.items()
        },
        "identity_aliases": aliases or {},
        "local_to_internal_gid": affinity or {},
        "observations": observations,
        "global_tracks": [],
    }


class IdentityCommitBufferTests(unittest.TestCase):
    def test_delayed_provisional_is_rewritten_to_public_gid(self) -> None:
        buffer = IdentityCommitBuffer(delay_s=1.0)
        self.assertEqual(
            buffer.push_tick(_tick(
                0, 0.0, {-1: (0.0, 0.0)},
                affinity={("a", 7): -1},
            )),
            [],
        )

        committed = buffer.push_tick(_tick(
            30, 1.0, {1: (1.0, 0.0)},
            aliases={-1: 1},
            affinity={("a", 7): 1},
        ))

        self.assertEqual(len(committed), 1)
        self.assertEqual(set(committed[0].predictions), {1})
        np.testing.assert_allclose(committed[0].predictions[1], (0.0, 0.0))
        self.assertEqual(committed[0].sources, {1: [("a", 7)]})

    def test_unresolved_private_gid_is_never_exposed(self) -> None:
        buffer = IdentityCommitBuffer(delay_s=1.0)
        buffer.push_tick(_tick(0, 0.0, {-1: (0.0, 0.0)}))

        committed = buffer.flush()

        self.assertEqual(committed[0].predictions, {})
        self.assertEqual(
            buffer.metrics["identity_commit_unresolved_hypotheses_dropped"], 1
        )

    def test_published_canonical_position_wins_when_aliases_collapse(self) -> None:
        buffer = IdentityCommitBuffer(delay_s=0.0)

        committed = buffer.push_tick(_tick(
            0,
            0.0,
            {1: (1.0, 0.0), 2: (9.0, 0.0)},
            aliases={2: 1},
        ))

        self.assertEqual(set(committed[0].predictions), {1})
        np.testing.assert_allclose(committed[0].predictions[1], (1.0, 0.0))
        self.assertEqual(
            buffer.metrics["identity_commit_duplicate_hypotheses_collapsed"], 1
        )

    def test_source_map_excludes_retained_local_affinity_without_observation(self) -> None:
        class Observation:
            camera = "a"
            local_id = 2

        buffer = IdentityCommitBuffer(delay_s=0.0)
        committed = buffer.push_tick(_tick(
            0,
            0.0,
            {1: (0.0, 0.0)},
            affinity={("a", 1): 1, ("a", 2): 1},
            observations=[Observation()],
        ))

        self.assertEqual(committed[0].sources, {1: [("a", 2)]})


if __name__ == "__main__":
    unittest.main()
