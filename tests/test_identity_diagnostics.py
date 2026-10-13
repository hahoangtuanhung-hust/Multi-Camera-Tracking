import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multicam_tracker.diagnostics import IdentityEventWriter


class IdentityDiagnosticsTests(unittest.TestCase):
    def test_jsonl_writer_is_bounded_and_serializes_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.jsonl"
            writer = IdentityEventWriter(path, max_events=2, flush_every=1)
            writer.write("first", xy=np.array([1.0, 2.0]))
            writer.write("second", gid=np.int64(7), margin=float("inf"))
            writer.write("dropped", gid=8)
            writer.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual([row["event"] for row in records], [
            "first", "second", "diagnostic_summary",
        ])
        self.assertEqual(records[0]["xy"], [1.0, 2.0])
        self.assertEqual(records[1]["gid"], 7)
        self.assertIsNone(records[1]["margin"])
        self.assertEqual(records[-1]["events_written"], 2)
        self.assertEqual(records[-1]["events_dropped"], 1)

    def test_disabled_writer_does_not_create_a_file(self) -> None:
        writer = IdentityEventWriter(None)
        writer.write("ignored")
        writer.close()
        self.assertFalse(writer.enabled)


if __name__ == "__main__":
    unittest.main()
