import unittest

import numpy as np

from multicam_tracker.config import PipelineConfig
from multicam_tracker.tracklets import TrackletMatcher, TrackletStore


class MatcherPruningRegressionTest(unittest.TestCase):
    def test_hungarian_sentinel_assignment_is_ignored(self) -> None:
        cfg = PipelineConfig(cameras=["a", "b"], tracklet_history_s=10.0)
        left_store, right_store = TrackletStore(cfg), TrackletStore(cfg)
        for timestamp in np.arange(0.0, 2.1, 0.1):
            for local_id, y in enumerate((0.0, 3.0, 6.0, 9.0, 12.0)):
                xy = np.array([timestamp, y])
                left_store.append("a", local_id, timestamp, xy, 0.9)
            # Deliberately fewer, far-away right rows forces Hungarian to
            # select sentinel cells that spatial pruning did not evaluate.
            for local_id, y in enumerate((0.0, 30.0, 60.0)):
                xy = np.array([timestamp, y])
                right_store.append("b", local_id, timestamp, xy, 0.9)
        matches = TrackletMatcher(cfg).assign(
            left_store.eligible(), right_store.eligible()
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].left_key, ("a", 0))
        self.assertEqual(matches[0].right_key, ("b", 0))


if __name__ == "__main__":
    unittest.main()
