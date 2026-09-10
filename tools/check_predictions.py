"""Fail fast on CSV mistakes before a DACON code submission."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
ACCEL = {"ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"}
STEER = {"LEFT", "STRAIGHT", "RIGHT"}
SIDES = {"LEFT", "RIGHT"}
EVASION = {"0", "1"}


def load(path: Path, columns: Sequence[str]) -> List[Dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Missing prediction file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = reader.fieldnames or []
        missing = [column for column in columns if column not in actual]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}; found {actual}")
        return list(reader)


def integer(value: str) -> int | None:
    try:
        number = float(str(value).strip())
        return int(number) if number.is_integer() else None
    except ValueError:
        return None


def duplicates(rows: Iterable[Dict[str, str]], keys: Sequence[str]) -> List[Tuple[str, ...]]:
    values = [tuple(row.get(key, "").strip() for key in keys) for row in rows]
    return [key for key, count in Counter(values).items() if count > 1]


def frame_number(path: Path) -> int | None:
    found = re.findall(r"\d+", path.stem)
    return int(found[-1]) if found else None


def report(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local Stage CSVs for DACON 236753 output constraints.")
    parser.add_argument("--predictions-dir", type=Path, required=True, help="stage1.csv, stage2.csv, stage3.csv")
    parser.add_argument("--stage1-videos", type=Path, help="optional actual input videos directory for ID coverage")
    parser.add_argument("--stage2-images", type=Path, help="optional actual Stage 2 images root for ID/frame-range checks")
    parser.add_argument("--stage3-expected", type=Path, help="optional CSV with ID,sample_index for exact Stage 3 coverage")
    args = parser.parse_args()
    root = args.predictions_dir
    errors: List[str] = []
    try:
        stage1 = load(root / "stage1.csv", ("ID", "answer"))
        stage2 = load(root / "stage2.csv", ("ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"))
        stage3 = load(root / "stage3.csv", ("ID", "sample_index", "accel_label", "steer_label"))
    except ValueError as error:
        parser.error(str(error))

    report(not duplicates(stage1, ("ID",)), "stage1.csv contains duplicate ID values", errors)
    report(not duplicates(stage2, ("ID",)), "stage2.csv contains duplicate ID values", errors)
    report(not duplicates(stage3, ("ID", "sample_index")), "stage3.csv contains duplicate (ID, sample_index) pairs", errors)
    for row in stage1:
        report(row["answer"].strip() in {"ORIGINAL", "RERECORDED"}, f"Stage 1 {row['ID']}: invalid answer", errors)
    for row in stage2:
        collision, entry = integer(row["collision_frame"]), integer(row["entry_frame"])
        report(collision is not None and collision >= 0, f"Stage 2 {row['ID']}: collision_frame must be a non-negative integer", errors)
        report(entry is not None and entry >= 0, f"Stage 2 {row['ID']}: entry_frame must be a non-negative integer", errors)
        report(row["evasion_space"].strip() in EVASION, f"Stage 2 {row['ID']}: evasion_space must be 0 or 1", errors)
        report(row["entry_side"].strip() in SIDES, f"Stage 2 {row['ID']}: entry_side must be LEFT or RIGHT", errors)
    for row in stage3:
        index = integer(row["sample_index"])
        report(index is not None and index >= 0, f"Stage 3 {row['ID']}: sample_index must be a non-negative integer", errors)
        report(row["accel_label"].strip() in ACCEL, f"Stage 3 {row['ID']}:{row['sample_index']}: invalid accel_label", errors)
        report(row["steer_label"].strip() in STEER, f"Stage 3 {row['ID']}:{row['sample_index']}: invalid steer_label", errors)

    if args.stage1_videos:
        expected = {path.stem for path in args.stage1_videos.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS}
        got = {row["ID"].strip() for row in stage1}
        report(expected == got, f"Stage 1 IDs differ: missing={sorted(expected-got)[:5]}, unexpected={sorted(got-expected)[:5]}", errors)
    if args.stage2_images:
        by_id = {row["ID"].strip(): row for row in stage2}
        expected = {folder.name for folder in args.stage2_images.iterdir() if folder.is_dir()}
        report(expected == set(by_id), f"Stage 2 IDs differ: missing={sorted(expected-set(by_id))[:5]}, unexpected={sorted(set(by_id)-expected)[:5]}", errors)
        for folder in (folder for folder in args.stage2_images.iterdir() if folder.is_dir()):
            frames = [frame_number(path) for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            frames = [frame for frame in frames if frame is not None]
            if not frames or folder.name not in by_id:
                continue
            for column in ("collision_frame", "entry_frame"):
                value = integer(by_id[folder.name][column])
                report(value is not None and min(frames) <= value <= max(frames), f"Stage 2 {folder.name}: {column}={value} is outside [{min(frames)}, {max(frames)}]", errors)
    if args.stage3_expected:
        expected_rows = load(args.stage3_expected, ("ID", "sample_index"))
        expected = {(row["ID"].strip(), row["sample_index"].strip()) for row in expected_rows}
        got = {(row["ID"].strip(), row["sample_index"].strip()) for row in stage3}
        report(expected == got, f"Stage 3 coverage differs: missing={len(expected-got)}, unexpected={len(got-expected)}", errors)

    if errors:
        print("FAIL: submission CSV checks found errors:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors[:30]), file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: schema and allowed values validated (S1={len(stage1)}, S2={len(stage2)}, S3={len(stage3)} rows).")


if __name__ == "__main__":
    main()
