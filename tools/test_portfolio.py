"""CPU tests. CUDA baseline and final ZIP inference are separately checked in Colab."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

import portfolio_lab as lab
from portfolio_runtime import (pf_video_features, pf_slope, pf_smooth, pf_signals,
    pf_classify, pf_event_features, pf_event_scores, pf_affine)


class PortfolioTests(unittest.TestCase):
    def test_centered_slope_boundaries(self):
        x = np.arange(11) * .2
        np.testing.assert_allclose(pf_slope(x, .1), 2)
        np.testing.assert_allclose(pf_slope(np.array([3.])), 0)
        np.testing.assert_allclose(pf_smooth(np.ones(11)), 1)

    def test_class_labels_and_texture_fallback(self):
        x = np.array([[10, 2, 4, 20], [10, -2, -4, 20], [0, 0, 0, 20], [0, 0, 0, 0]])
        a, s = pf_classify(x, [1, 1, 1])
        self.assertEqual(a.tolist(), ["ACCELERATING", "DECELERATING", "STOPPED", "CONSTANT"])
        self.assertEqual(s.tolist(), ["LEFT", "RIGHT", "STRAIGHT", "STRAIGHT"])

    def test_stop_mask_is_ground_truth(self):
        truth = pd.DataFrame({"accel_label": ["STOPPED", "CONSTANT"], "steer_label": ["LEFT", "RIGHT"]})
        pred = (np.array(["CONSTANT", "STOPPED"]), np.array(["RIGHT", "RIGHT"]))
        result = lab.head_metrics(truth, pred)
        self.assertEqual(result["steer"]["LEFT"]["support"], 0)
        self.assertEqual(result["steer"]["RIGHT"]["support"], 1)
        self.assertAlmostEqual(result["steer_macro_f1"], 1 / 3)

    def test_affine_sign(self):
        image = np.random.default_rng(4).integers(0, 255, (90, 160), dtype=np.uint8)
        shifted = cv2.warpAffine(image, np.array([[1., 0, 2], [0, 1, 0]]), (160, 90))
        affine = pf_affine(image, shifted)
        self.assertIsNotNone(affine)
        self.assertGreater(affine[0, 2], 1)

    def test_single_frame_and_real_frame_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / "frame_000042.jpg"), np.zeros((20, 20), np.uint8))
            numbers, features = pf_event_features(root)
            self.assertEqual(numbers.tolist(), [42])
            self.assertEqual(int(numbers[np.argmax(pf_event_scores(features, "B"))]), 42)
            cv2.imwrite(str(root / "frame_000051.jpg"), np.full((20, 20), 255, np.uint8))
            numbers, features = pf_event_features(root)
            self.assertEqual(int(numbers[np.argmax(pf_event_scores(features, "A"))]), 51)

    def test_video_contract_and_file_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.random.default_rng(3).integers(0, 255, (90, 160, 3), dtype=np.uint8)
            for fps in (10, 20):
                writer = cv2.VideoWriter(str(root / f"{fps}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
                self.assertTrue(writer.isOpened())
                for i in range(12):
                    writer.write(np.roll(image, i, axis=1))
                writer.release()
            first = pf_video_features(root / "10.mp4")
            with self.assertWarns(UserWarning):
                wrong_metadata = pf_video_features(root / "20.mp4")
            self.assertEqual(len(wrong_metadata), 12)  # metadata must not change output coverage
            second = pf_video_features(root / "10.mp4")
            np.testing.assert_allclose(first, second)
            np.testing.assert_allclose(first, wrong_metadata)
            self.assertEqual(first.shape, (12, 6))
            self.assertTrue(np.isfinite(first).all())
            self.assertEqual(pf_signals(first, "B").shape, (12, 4))

    def test_outer_labels_do_not_change_own_predictions(self):
        # Same test features, change only held-out labels: its fitted thresholds must not change.
        labels = pd.DataFrame({"ID": np.repeat(list("ABCDE"), 4),
            "accel_label": lab.ACCEL * 5, "steer_label": (["LEFT", "RIGHT", "STRAIGHT", "LEFT"] * 5)})
        rng = np.random.default_rng(1)
        x = rng.normal(size=(20, 4))
        x[:, 0], x[:, 3] = np.abs(x[:, 0]), 10
        signals = {"A": x, "B": x.copy()}
        before, report = lab.nested_cv(labels, signals, ("A",))
        changed = labels.copy()
        changed.loc[changed.ID == "A", "accel_label"] = "STOPPED"
        changed.loc[changed.ID == "A", "steer_label"] = "STRAIGHT"
        after, _ = lab.nested_cv(changed, signals, ("A",))
        np.testing.assert_array_equal(before[0][:4], after[0][:4])
        np.testing.assert_array_equal(before[1][:4], after[1][:4])
        self.assertNotIn("A", report["folds"][0]["train_groups"])

    def test_standalone_composition(self):
        source = lab.make_source({"stage1_slots": 5, "stage2_family": "B", "stage3": {"family": "A", "thresholds": [1, 2, 3]}})
        compile(source, "inference.py", "exec")
        tree = ast.parse(source)
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertNotIn("portfolio_runtime", imports)
        self.assertNotIn("baseline_inference", imports)
        self.assertIn("predict_stage3 = pf_predict_stage3", source)
        self.assertEqual(source.count("from __future__ import annotations"), 1)

    def test_no_package_means_no_go(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lab.report(root)
            value = json.loads((root / "reports/submission_gate.json").read_text())
            self.assertEqual(value["verdict"], "NO_GO")
            self.assertIsNone(value["improvement_probability"])

    def test_corrupt_zip_never_reuses_go(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lab.save(root / "reports/package_check.json", {"passed": True, "zip_sha256": "incorrect"})
            (root / "submission").mkdir()
            (root / "submission/submit.zip").write_bytes(b"not a tested zip")
            lab.report(root)
            value = lab.read(root / "reports/submission_gate.json")
            self.assertEqual(value["verdict"], "NO_GO")
            self.assertIn("ZIP hash differs from tested ZIP", value["failures"])


if __name__ == "__main__":
    unittest.main()
