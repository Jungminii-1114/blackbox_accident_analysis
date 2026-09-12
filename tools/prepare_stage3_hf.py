"""Convert the official comma2k19 Hugging Face demo to a 10 Hz Stage-3 dataset."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


ACCEL_CLASSES = ("ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED")
STEER_CLASSES = ("LEFT", "STRAIGHT", "RIGHT")


def _array(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype == object:
        # Arrow represents scalar CAN values such as speed as an object array
        # whose elements are one-element ndarrays.
        result = np.asarray([np.asarray(item, dtype=np.float64).reshape(-1)[0] for item in raw], dtype=np.float64)
    else:
        result = raw.astype(np.float64, copy=False)
    return result.reshape(-1) if result.ndim <= 1 else result[:, 0]


def _signal(log: Dict[str, Any], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    times = _array(log[prefix + "__t"])
    values = _array(log[prefix + "__value"])
    valid = np.isfinite(times) & np.isfinite(values)
    times, values = times[valid], values[valid]
    order = np.argsort(times)
    return times[order], values[order]


def _safe_id(segment_id: str) -> str:
    route, segment = segment_id.strip("/").rsplit("/", 1)
    route = re.sub(r"[^A-Za-z0-9_-]+", "_", route)
    return f"{route}_seg{int(segment):03d}"


def _write_csv(path: Path, columns: Iterable[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _transcode(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 1_000_000:
        dimensions = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(target),
        ], text=True).strip().split(",")
        if len(dimensions) == 2 and all(int(value) % 2 == 0 for value in dimensions):
            return
    temporary = target.with_suffix(".part.mp4")
    if temporary.exists():
        temporary.unlink()
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "20",
        "-i", str(source), "-vf",
        "fps=10,scale=640:-2:force_original_aspect_ratio=decrease,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        if not temporary.is_file() or temporary.stat().st_size <= 1_000_000:
            raise RuntimeError(f"Transcode produced an empty/too-small file: {source}")
        dimensions = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(temporary),
        ], text=True).strip().split(",")
        if len(dimensions) != 2 or any(int(value) % 2 for value in dimensions):
            raise RuntimeError(f"Transcoded dimensions must be even: {dimensions}")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _frame_count(path: Path) -> int:
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path),
    ], text=True).strip()
    return int(output)


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare comma2k19 HF demo for Stage 3 training.")
    parser.add_argument("--raw-root", type=Path, required=True, help="contains data/*.parquet and compression_challenge/test_videos.zip")
    parser.add_argument("--data-root", type=Path, required=True, help="persistent Drive external_data/stage3")
    parser.add_argument("--work-dir", type=Path, default=Path("/content/comma2k19_hf_work"))
    parser.add_argument("--limit", type=int, default=0, help="0 means all 64 videos")
    parser.add_argument("--stop-speed", type=float, default=0.30)
    parser.add_argument("--accel-threshold", type=float, default=0.25)
    parser.add_argument("--steer-threshold", type=float, default=3.0)
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--steer-positive-is-left", dest="steer_positive_is_left", action="store_true")
    direction.add_argument("--steer-positive-is-right", dest="steer_positive_is_left", action="store_false")
    parser.set_defaults(steer_positive_is_left=True)
    args = parser.parse_args()

    parquet = sorted((args.raw_root / "data").glob("*.parquet"))
    video_zip = args.raw_root / "compression_challenge" / "test_videos.zip"
    if len(parquet) != 3 or not video_zip.is_file():
        parser.error("Expected 3 parquet files and compression_challenge/test_videos.zip")
    # Skip embedded preview images to keep Colab RAM usage predictable.
    table = pd.concat((pd.read_parquet(path, columns=["segment_id", "log"]) for path in parquet), ignore_index=True)
    if not {"segment_id", "log"}.issubset(table.columns):
        parser.error(f"Unexpected parquet columns: {list(table.columns)}")
    by_segment = {str(row.segment_id).strip("/"): row.log for row in table.itertuples(index=False)}

    extract_root = args.work_dir / "videos"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(video_zip) as archive:
        members = [name for name in archive.namelist() if name.endswith("/video.hevc")]
        if args.limit > 0:
            members = members[: args.limit]
        for member in members:
            target = extract_root / member
            if not target.is_file():
                archive.extract(member, extract_root)

    processed = args.data_root / "processed" / "comma2k19_hf_demo"
    manifest_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    route_names = sorted({member.rsplit("/", 2)[0] for member in members})
    validation_routes = set(route_names[::5])

    for position, member in enumerate(members, 1):
        segment_id = member[:-len("/video.hevc")].strip("/")
        if segment_id not in by_segment:
            raise KeyError(f"No parquet row for {segment_id}")
        log = by_segment[segment_id]
        if not isinstance(log, dict):
            log = dict(log)
        frame_times = _array(log["global_pose__frame_times"])
        speed_t, speed_v = _signal(log, "processed_log__CAN__speed")
        steer_t, steer_v = _signal(log, "processed_log__CAN__steering_angle")
        source = extract_root / member
        video_id = _safe_id(segment_id)
        video_relative = Path("processed/comma2k19_hf_demo/videos") / f"{video_id}.mp4"
        video = args.data_root / video_relative
        _transcode(source, video)
        count = _frame_count(video)
        sample_times = frame_times[::2][:count]
        count = len(sample_times)
        if count == 0:
            raise RuntimeError(f"No aligned frames: {segment_id}")
        coverage = float(np.mean((sample_times >= speed_t.min()) & (sample_times <= speed_t.max()) & (sample_times >= steer_t.min()) & (sample_times <= steer_t.max())))
        speed = _smooth(np.interp(sample_times, speed_t, speed_v))
        steering = _smooth(np.interp(sample_times, steer_t, steer_v))
        acceleration = _smooth(np.gradient(speed, sample_times))

        frame_path = processed / "frame_times" / f"{video_id}.csv"
        speed_path = processed / "speed" / f"{video_id}.csv"
        steer_path = processed / "steering" / f"{video_id}.csv"
        _write_csv(frame_path, ("sample_index", "time_seconds"), ({"sample_index": i, "time_seconds": t} for i, t in enumerate(sample_times)))
        _write_csv(speed_path, ("time_seconds", "value"), ({"time_seconds": t, "value": v} for t, v in zip(sample_times, speed)))
        _write_csv(steer_path, ("time_seconds", "value"), ({"time_seconds": t, "value": v} for t, v in zip(sample_times, steering)))

        accel_labels = np.full(count, "CONSTANT", dtype=object)
        accel_labels[acceleration > args.accel_threshold] = "ACCELERATING"
        accel_labels[acceleration < -args.accel_threshold] = "DECELERATING"
        accel_labels[speed <= args.stop_speed] = "STOPPED"
        signed = steering if args.steer_positive_is_left else -steering
        steer_labels = np.full(count, "STRAIGHT", dtype=object)
        steer_labels[signed > args.steer_threshold] = "LEFT"
        steer_labels[signed < -args.steer_threshold] = "RIGHT"
        for index, (accel, steer) in enumerate(zip(accel_labels, steer_labels)):
            label_rows.append({"ID": video_id, "sample_index": index, "accel_label": accel, "steer_label": steer})

        route = segment_id.rsplit("/", 1)[0]
        split = "validation" if route in validation_routes else "train"
        manifest_rows.append({
            "id": video_id, "source_segment_id": segment_id, "route_group": route, "split": split,
            "video_path": str(video_relative),
            "frame_times_path": str(frame_path.relative_to(args.data_root)),
            "speed_path": str(speed_path.relative_to(args.data_root)),
            "steering_path": str(steer_path.relative_to(args.data_root)),
            "labels_path": "processed/comma2k19_hf_demo/labels.csv", "samples": count,
        })
        audit_rows.append({"id": video_id, "route_group": route, "split": split, "samples": count, "time_coverage": coverage})
        print(f"[{position:02d}/{len(members):02d}] {video_id}: {count} samples, coverage={coverage:.3f}, {split}")

    _write_csv(args.data_root / "manifest.csv", manifest_rows[0].keys(), manifest_rows)
    _write_csv(processed / "labels.csv", ("ID", "sample_index", "accel_label", "steer_label"), label_rows)
    _write_csv(processed / "audit.csv", audit_rows[0].keys(), audit_rows)
    distributions = {
        "accel": pd.Series([row["accel_label"] for row in label_rows]).value_counts().to_dict(),
        "steer": pd.Series([row["steer_label"] for row in label_rows]).value_counts().to_dict(),
        "routes": len(route_names), "validation_routes": sorted(validation_routes),
        "thresholds": {"stop_speed": args.stop_speed, "accel": args.accel_threshold, "steer": args.steer_threshold, "steer_positive_is_left": args.steer_positive_is_left},
    }
    (processed / "label_audit.json").write_text(json.dumps(distributions, indent=2) + "\n", encoding="utf-8")
    if min(row["time_coverage"] for row in audit_rows) < 0.95:
        raise RuntimeError("At least one segment has <95% CAN/frame timestamp coverage; do not train.")
    print(f"PASS: wrote {len(manifest_rows)} videos and {len(label_rows)} 10 Hz labels")
    print(f"Manifest: {args.data_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
