"""Offline scorer for DACON 236753 predictions.

This tool deliberately separates *known local labels* from the hidden
leaderboard.  A missing target is reported as not evaluable; it is never
silently replaced with a guessed score.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


S1_CLASSES = ("ORIGINAL", "RERECORDED")
S2_SIDE_CLASSES = ("LEFT", "RIGHT")
S2_EVASION_CLASSES = ("0", "1")
S3_ACCEL_CLASSES = ("ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED")
S3_STEER_CLASSES = ("LEFT", "STRAIGHT", "RIGHT")


def _rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _number(value: Any) -> Optional[float]:
    try:
        parsed = float(_text(value))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> Optional[int]:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _index(rows: Iterable[Dict[str, str]], keys: Sequence[str]) -> Tuple[Dict[Tuple[str, ...], Dict[str, str]], List[Tuple[str, ...]]]:
    index: Dict[Tuple[str, ...], Dict[str, str]] = {}
    duplicates: List[Tuple[str, ...]] = []
    for row in rows:
        key = tuple(_text(row.get(name)) for name in keys)
        if not all(key):
            continue
        if key in index:
            duplicates.append(key)
        index[key] = row
    return index, duplicates


def macro_f1(truth: Sequence[str], prediction: Sequence[str], classes: Sequence[str]) -> Optional[float]:
    """Macro-F1 with the competition's fixed allowed class universe."""
    if not truth:
        return None
    values = []
    for label in classes:
        tp = sum(t == label and p == label for t, p in zip(truth, prediction))
        fp = sum(t != label and p == label for t, p in zip(truth, prediction))
        fn = sum(t == label and p != label for t, p in zip(truth, prediction))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else (2.0 * tp / denominator))
    return sum(values) / len(values)


def accuracy(truth: Sequence[bool]) -> Optional[float]:
    return None if not truth else sum(truth) / len(truth)


def _known(value: Any) -> bool:
    return _text(value) not in {"", "-1", "None", "nan", "NaN"}


def _alias(row: Dict[str, str], *columns: str) -> str:
    for column in columns:
        if column in row:
            return _text(row[column])
    return ""


def score_stage1(labels: Path, prediction: Path) -> Dict[str, Any]:
    label_rows, prediction_rows = _rows(labels), _rows(prediction)
    target, _ = _index(label_rows, ("ID",))
    pred, duplicates = _index(prediction_rows, ("ID",))
    truth, guess = [], []
    invalid = 0
    for key, row in target.items():
        actual, value = _alias(row, "label", "answer"), _alias(pred.get(key, {}), "answer")
        if value not in S1_CLASSES:
            invalid += 1
        truth.append(actual)
        guess.append(value)
    return {
        "samples": len(truth),
        "missing_predictions": sum(key not in pred for key in target),
        "unexpected_predictions": sum(key not in target for key in pred),
        "duplicate_predictions": len(duplicates),
        "invalid_predictions": invalid,
        "macro_f1": macro_f1(truth, guess, S1_CLASSES),
    }


def _load_frame_time_map(path: Optional[Path]) -> Dict[Tuple[str, int], float]:
    if path is None:
        return {}
    mapping: Dict[Tuple[str, int], float] = {}
    for row in _rows(path):
        frame, seconds = _integer(row.get("frame")), _number(row.get("time_seconds"))
        if frame is not None and seconds is not None and _text(row.get("ID")):
            mapping[(_text(row["ID"]), frame)] = seconds
    return mapping


def _as_time(video_id: str, frame: Optional[int], mapping: Dict[Tuple[str, int], float], fps: Optional[float]) -> Optional[float]:
    if frame is None or frame < 0:
        return None
    if (video_id, frame) in mapping:
        return mapping[(video_id, frame)]
    return (frame / fps) if fps and math.isfinite(fps) and fps > 0 else None


def _event_accuracy(target: Dict[Tuple[str, ...], Dict[str, str]], prediction: Dict[Tuple[str, ...], Dict[str, str]], label_columns: Sequence[str], prediction_column: str, mapping: Dict[Tuple[str, int], float], fps: Optional[float]) -> Dict[str, Any]:
    if not mapping and not (fps and math.isfinite(fps) and fps > 0):
        return {"labeled_samples": sum(_known(_alias(row, *label_columns)) for row in target.values()),
                "invalid_predictions": None, "accuracy_at_0_3s": None, "mae_seconds": None,
                "mae_frames": None, "mae_valid_samples": 0, "time_conversion_unavailable": True}
    correct: List[bool] = []
    time_errors: List[float] = []
    frame_errors: List[int] = []
    invalid = 0
    for key, row in target.items():
        raw_target = _alias(row, *label_columns)
        if not _known(raw_target):
            continue
        expected = _integer(raw_target)
        predicted = _integer(_alias(prediction.get(key, {}), prediction_column))
        expected_time = _as_time(key[0], expected, mapping, fps)
        predicted_time = _as_time(key[0], predicted, mapping, fps)
        if expected_time is None or predicted_time is None:
            invalid += 1
            correct.append(False)
        else:
            # A small epsilon prevents floating-point representations of 0.3
            # (for example 1.0 - 0.7) from becoming a false boundary error.
            correct.append(abs(expected_time - predicted_time) <= 0.300000001)
            time_errors.append(abs(expected_time - predicted_time))
            frame_errors.append(abs(expected - predicted))
    return {"labeled_samples": len(correct), "invalid_predictions": invalid, "accuracy_at_0_3s": accuracy(correct),
            "mae_seconds": sum(time_errors) / len(time_errors) if time_errors else None,
            "mae_frames": sum(frame_errors) / len(frame_errors) if frame_errors else None,
            "mae_valid_samples": len(time_errors)}


def _classification_score(target: Dict[Tuple[str, ...], Dict[str, str]], prediction: Dict[Tuple[str, ...], Dict[str, str]], label_columns: Sequence[str], prediction_column: str, classes: Sequence[str]) -> Dict[str, Any]:
    truth, guess = [], []
    invalid = 0
    for key, row in target.items():
        actual = _alias(row, *label_columns)
        if not _known(actual):
            continue
        value = _alias(prediction.get(key, {}), prediction_column)
        if value not in classes:
            invalid += 1
        truth.append(actual)
        guess.append(value)
    return {"labeled_samples": len(truth), "invalid_predictions": invalid, "macro_f1": macro_f1(truth, guess, classes)}


def score_stage2(labels: Path, prediction: Path, frame_time_map: Optional[Path], fps: Optional[float]) -> Dict[str, Any]:
    target, _ = _index(_rows(labels), ("ID",))
    pred, duplicates = _index(_rows(prediction), ("ID",))
    mapping = _load_frame_time_map(frame_time_map)
    collision = _event_accuracy(target, pred, ("collision_frame", "t_collision"), "collision_frame", mapping, fps)
    entry = _event_accuracy(target, pred, ("entry_frame", "t_entry"), "entry_frame", mapping, fps)
    evasion = _classification_score(target, pred, ("evasion_space",), "evasion_space", S2_EVASION_CLASSES)
    side = _classification_score(target, pred, ("entry_side",), "entry_side", S2_SIDE_CLASSES)
    components = (collision["accuracy_at_0_3s"], entry["accuracy_at_0_3s"], side["macro_f1"], evasion["macro_f1"])
    weighted = None if any(item is None for item in components) else (0.35 * components[0] + 0.35 * components[1] + 0.15 * components[2] + 0.15 * components[3])
    return {
        "prediction_rows": len(pred),
        "missing_predictions": sum(key not in pred for key in target),
        "unexpected_predictions": sum(key not in target for key in pred),
        "duplicate_predictions": len(duplicates),
        "time_conversion": "frame_time_map" if mapping else (f"fps={fps}" if fps else "unavailable (time metrics are N/A)"),
        "collision": collision,
        "entry": entry,
        "evasion_space": evasion,
        "entry_side": side,
        "score": weighted,
    }


def score_stage3(labels: Path, prediction: Path) -> Dict[str, Any]:
    target, _ = _index(_rows(labels), ("ID", "sample_index"))
    pred, duplicates = _index(_rows(prediction), ("ID", "sample_index"))
    accel_truth: List[str] = []
    accel_guess: List[str] = []
    steer_truth: List[str] = []
    steer_guess: List[str] = []
    invalid_accel = invalid_steer = 0
    for key, row in target.items():
        actual_accel, actual_steer = _alias(row, "accel_label"), _alias(row, "steer_label")
        predicted = pred.get(key, {})
        guessed_accel, guessed_steer = _alias(predicted, "accel_label"), _alias(predicted, "steer_label")
        if actual_accel in S3_ACCEL_CLASSES:
            accel_truth.append(actual_accel)
            accel_guess.append(guessed_accel)
            invalid_accel += int(guessed_accel not in S3_ACCEL_CLASSES)
        # Official rule: steering of ground-truth STOPPED frames is excluded.
        if actual_accel != "STOPPED" and actual_steer in S3_STEER_CLASSES:
            steer_truth.append(actual_steer)
            steer_guess.append(guessed_steer)
            invalid_steer += int(guessed_steer not in S3_STEER_CLASSES)
    accel = macro_f1(accel_truth, accel_guess, S3_ACCEL_CLASSES)
    steer = macro_f1(steer_truth, steer_guess, S3_STEER_CLASSES)
    known_ids = {key[0] for key in target}
    return {
        "labeled_accel_samples": len(accel_truth),
        "labeled_steer_samples_excluding_stopped": len(steer_truth),
        "missing_predictions": sum(key not in pred for key in target),
        "unexpected_predictions": sum(key[0] not in known_ids for key in pred),
        "unlabeled_predictions_same_video": sum(key not in target and key[0] in known_ids for key in pred),
        "duplicate_predictions": len(duplicates),
        "invalid_accel_predictions": invalid_accel,
        "invalid_steer_predictions": invalid_steer,
        "accel_macro_f1": accel,
        "steer_macro_f1": steer,
        "score": None if accel is None or steer is None else 0.7 * accel + 0.3 * steer,
    }


def _format(value: Any) -> str:
    return "N/A" if value is None else f"{value:.5f}" if isinstance(value, float) else str(value)


def _print_report(report: Dict[str, Any]) -> None:
    print("\nLocal validation (known labels only)")
    print(f"Stage 1  Macro-F1:                  {_format(report['stage1']['macro_f1'])}")
    stage2 = report["stage2"]
    print(f"Stage 2  collision Accuracy@0.3s:   {_format(stage2['collision']['accuracy_at_0_3s'])}")
    print(f"         entry Accuracy@0.3s:       {_format(stage2['entry']['accuracy_at_0_3s'])}")
    print(f"         direction Macro-F1:        {_format(stage2['entry_side']['macro_f1'])}")
    print(f"         evasion Macro-F1:          {_format(stage2['evasion_space']['macro_f1'])}")
    print(f"         weighted score:            {_format(stage2['score'])}")
    stage3 = report["stage3"]
    print(f"Stage 3  accel Macro-F1:            {_format(stage3['accel_macro_f1'])}")
    print(f"         steer Macro-F1:            {_format(stage3['steer_macro_f1'])}")
    print(f"         weighted score:            {_format(stage3['score'])}")
    print(f"Overall  0.2*S1 + 0.4*S2 + 0.4*S3: {_format(report['overall_score'])}")
    print("\nNote: N/A means this label is unavailable locally and is not estimated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score DACON 236753 predictions using only known local labels.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="contains stage1.csv, stage2.csv, stage3.csv")
    parser.add_argument("--predictions-dir", type=Path, required=True, help="contains stage1.csv, stage2.csv, stage3.csv")
    parser.add_argument("--stage2-frame-time-map", type=Path, help="optional CSV: ID,frame,time_seconds")
    parser.add_argument("--stage2-fps", type=float, help="fallback for Stage 2 frame-to-time conversion")
    parser.add_argument("--json-out", type=Path, help="write a machine-readable report")
    args = parser.parse_args()
    if args.stage2_fps is not None and args.stage2_fps <= 0:
        parser.error("--stage2-fps must be positive")
    labels, predictions = args.labels_dir, args.predictions_dir
    report = {
        "stage1": score_stage1(labels / "stage1.csv", predictions / "stage1.csv"),
        "stage2": score_stage2(labels / "stage2.csv", predictions / "stage2.csv", args.stage2_frame_time_map, args.stage2_fps),
        "stage3": score_stage3(labels / "stage3.csv", predictions / "stage3.csv"),
    }
    scores = (report["stage1"]["macro_f1"], report["stage2"]["score"], report["stage3"]["score"])
    report["overall_score"] = None if any(score is None for score in scores) else 0.2 * scores[0] + 0.4 * scores[1] + 0.4 * scores[2]
    _print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
