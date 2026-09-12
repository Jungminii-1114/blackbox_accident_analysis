"""Dependency-free regression test for the offline scorer."""
from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import local_validate


def write(path: Path, columns, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dacon236753_validator_") as directory:
        root = Path(directory); labels = root / "labels"; predictions = root / "predictions"
        labels.mkdir(); predictions.mkdir()
        write(labels / "stage1.csv", ["ID", "label"], [{"ID": "a", "label": "ORIGINAL"}, {"ID": "b", "label": "RERECORDED"}])
        write(predictions / "stage1.csv", ["ID", "answer"], [{"ID": "a", "answer": "ORIGINAL"}, {"ID": "b", "answer": "RERECORDED"}])
        write(labels / "stage2.csv", ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"], [{"ID": "a", "collision_frame": 30, "entry_frame": 10, "evasion_space": 1, "entry_side": "LEFT"}])
        write(predictions / "stage2.csv", ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"], [{"ID": "a", "collision_frame": 33, "entry_frame": 7, "evasion_space": 1, "entry_side": "LEFT"}])
        write(labels / "stage3.csv", ["ID", "sample_index", "accel_label", "steer_label"], [{"ID": "a", "sample_index": 0, "accel_label": "STOPPED", "steer_label": "LEFT"}, {"ID": "a", "sample_index": 1, "accel_label": "ACCELERATING", "steer_label": "RIGHT"}])
        write(predictions / "stage3.csv", ["ID", "sample_index", "accel_label", "steer_label"], [{"ID": "a", "sample_index": 0, "accel_label": "STOPPED", "steer_label": "LEFT"}, {"ID": "a", "sample_index": 1, "accel_label": "ACCELERATING", "steer_label": "RIGHT"}])
        s1 = local_validate.score_stage1(labels / "stage1.csv", predictions / "stage1.csv")
        s2 = local_validate.score_stage2(labels / "stage2.csv", predictions / "stage2.csv", None, 10.0)
        s3 = local_validate.score_stage3(labels / "stage3.csv", predictions / "stage3.csv")
        assert s1["macro_f1"] == 1.0
        assert s2["collision"]["accuracy_at_0_3s"] == 1.0  # 3 frames at 10 fps
        assert math.isclose(s2["score"], 0.85)  # one-class binary labels => fixed Macro-F1 = 0.5
        assert s3["labeled_steer_samples_excluding_stopped"] == 1
        assert s3["score"] < 1.0  # fixed-class macro-F1 penalizes unseen classes
        checked = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("check_predictions.py")), "--predictions-dir", str(predictions)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert checked.returncode == 0, checked.stderr
    print("PASS: local scorer matches metric semantics and excludes STOPPED steering.")


if __name__ == "__main__":
    main()
