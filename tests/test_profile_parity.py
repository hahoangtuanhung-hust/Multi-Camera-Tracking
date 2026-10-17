import ast
import unittest
from pathlib import Path

from multicam_tracker.config import build_effective_config


class ProfileParityTests(unittest.TestCase):
    def test_profile_owns_tracker_unless_override_is_explicit(self) -> None:
        self.assertEqual(
            build_effective_config(profile="hybridsort-balanced").local_tracker,
            "hybridsort",
        )
        self.assertEqual(
            build_effective_config(
                profile="hybridsort-balanced", local_tracker="ocsort"
            ).local_tracker,
            "ocsort",
        )

    def test_controlled_clock_and_detector_profiles_change_one_policy(self) -> None:
        baseline = build_effective_config(profile="hybridsort-balanced")
        low = build_effective_config(profile="hybridsort-balanced-low-floor")
        mid = build_effective_config(profile="hybridsort-balanced-mid-floor")
        fixed = build_effective_config(profile="hybridsort-a3-clock-fixed")
        offset = build_effective_config(
            profile="hybridsort-a3-clock-offset-only"
        )

        self.assertEqual((low.conf_threshold, low.imgsz), (0.15, 640))
        self.assertEqual((mid.conf_threshold, mid.imgsz), (0.20, 640))
        self.assertEqual(low.track_tentative_max_misses, baseline.track_tentative_max_misses)
        self.assertFalse(fixed.clock_learning_enabled)
        self.assertTrue(offset.clock_learning_enabled)
        self.assertEqual(offset.offset_max_drift_ppm, 0.0)

    def test_trajectory_ablation_separates_motion_from_output_smoothing(self) -> None:
        motion = build_effective_config(profile="hybridsort-a2-motion-only")
        smoothing = build_effective_config(
            profile="hybridsort-a2-smoothing-only"
        )

        self.assertTrue(motion.fusion_robust_trajectory_motion_enabled)
        self.assertFalse(motion.fusion_trajectory_smoothing_enabled)
        self.assertFalse(smoothing.fusion_robust_trajectory_motion_enabled)
        self.assertTrue(smoothing.fusion_trajectory_smoothing_enabled)

    def test_every_runtime_entrypoint_uses_effective_config_builder(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "src" / "multicam_tracker" / "demo.py",
            root / "src" / "multicam_tracker" / "evaluate.py",
            root / "src" / "multicam_tracker" / "render_preview.py",
            root / "benchmarks" / "cpu_pipeline.py",
            root / "benchmarks" / "replay_cpu.py",
            root / "modal_app.py",
        ]
        for path in paths:
            with self.subTest(entrypoint=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "build_effective_config"
                ]
                self.assertTrue(calls, f"{path} bypasses effective config builder")

    def test_tracker_cli_defaults_are_not_implicit_overrides(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "src" / "multicam_tracker" / "demo.py",
            root / "src" / "multicam_tracker" / "evaluate.py",
            root / "src" / "multicam_tracker" / "render_preview.py",
            root / "benchmarks" / "cpu_pipeline.py",
            root / "benchmarks" / "replay_cpu.py",
        ]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            tracker_calls = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                    continue
                if any(
                    isinstance(arg, ast.Constant) and arg.value == "--local-tracker"
                    for arg in node.args
                ):
                    tracker_calls.append(node)
            self.assertEqual(len(tracker_calls), 1, path)
            defaults = [
                keyword.value
                for keyword in tracker_calls[0].keywords
                if keyword.arg == "default"
            ]
            self.assertEqual(len(defaults), 1, path)
            self.assertIsNone(ast.literal_eval(defaults[0]), path)

    def test_offline_entrypoints_fix_alignment_and_use_event_output_time(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "src" / "multicam_tracker" / "evaluate.py",
            root / "src" / "multicam_tracker" / "render_preview.py",
            root / "benchmarks" / "cpu_pipeline.py",
            root / "benchmarks" / "replay_cpu.py",
            root / "benchmarks" / "evaluate_thresholds.py",
            root / "modal_app.py",
        ]
        for path in paths:
            with self.subTest(entrypoint=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("best_offset", source)
                self.assertIn("fusion_output_time_policy=\"event\"", source)

    def test_current_only_replay_does_not_require_a_baseline_tree(self) -> None:
        path = Path(__file__).resolve().parents[1] / "benchmarks" / "replay_cpu.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        baseline_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            if any(
                isinstance(arg, ast.Constant) and arg.value == "--baseline-source"
                for arg in node.args
            ):
                baseline_calls.append(node)
        self.assertEqual(len(baseline_calls), 1)
        required = [
            keyword.value
            for keyword in baseline_calls[0].keywords
            if keyword.arg == "required"
        ]
        self.assertFalse(required and ast.literal_eval(required[0]))


if __name__ == "__main__":
    unittest.main()
