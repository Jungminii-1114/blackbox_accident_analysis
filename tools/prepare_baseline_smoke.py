"""Turn official example data into evaluator-shaped local smoke inputs.

This is only a regression fixture: the official checkpoints were trained from
these tiny examples, so its score must never be used as an estimate of hidden
leaderboard performance.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create local DACON-shaped smoke input and known labels from baseline data.")
    parser.add_argument("--baseline-data", type=Path, required=True, help="official Baseline/data directory")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        if not args.overwrite:
            parser.error(f"{args.out} already exists; use --overwrite to replace it")
        shutil.rmtree(args.out)
    try:
        import cv2
    except ImportError as error:
        raise SystemExit("OpenCV is required. Run this in the DACON/GPU environment.") from error

    source, out = args.baseline_data, args.out
    input_root, label_root = out / "input", out / "labels"

    stage1_labels = []
    for row in read_csv(source / "stage1/labels.csv"):
        original = source / "stage1" / row["path"]
        target = input_root / "stage1/videos" / f"{row['ID']}{original.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        stage1_labels.append({"ID": row["ID"], "label": row["label"]})
    write_csv(label_root / "stage1.csv", ("ID", "label"), stage1_labels)

    stage2_labels, frame_time = [], []
    for row in read_csv(source / "stage2/labels.csv"):
        video = source / "stage2" / row["path"]
        frame_dir = input_root / "stage2/images" / row["ID"]
        frame_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imwrite(str(frame_dir / f"frame_{index:06d}.jpg"), frame)
            frame_time.append({"ID": row["ID"], "frame": index, "time_seconds": capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0})
            index += 1
        capture.release()
        stage2_labels.append({
            "ID": row["ID"], "collision_frame": row["t_collision"], "entry_frame": row["t_entry"],
            "evasion_space": row["evasion_space"], "entry_side": row["entry_side"],
        })
    write_csv(label_root / "stage2.csv", ("ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"), stage2_labels)
    write_csv(label_root / "stage2_frame_time.csv", ("ID", "frame", "time_seconds"), frame_time)

    stage3_labels = []
    for row in read_csv(source / "stage3/labels.csv"):
        stage3_labels.append({key: row[key] for key in ("ID", "sample_index", "accel_label", "steer_label")})
    stage3_expected = []
    for video in (source / "stage3/videos").glob("*"):
        if video.is_file():
            target = input_root / "stage3/videos" / video.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, target)
            capture = cv2.VideoCapture(str(video))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            stage3_expected.extend({"ID": video.stem, "sample_index": index} for index in range(count))
    write_csv(label_root / "stage3.csv", ("ID", "sample_index", "accel_label", "steer_label"), stage3_labels)
    write_csv(input_root / "stage3_expected.csv", ("ID", "sample_index"), stage3_expected)
    print(f"Created smoke input: {input_root}")
    print(f"Created known labels: {label_root}")
    print("Warning: this data is for regression/interface checks only, not model selection.")


if __name__ == "__main__":
    main()
