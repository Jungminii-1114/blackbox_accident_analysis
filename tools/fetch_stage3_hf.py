"""Materialize the comma2k19 HF demo as real files on mounted Google Drive."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ID = "commaai/comma2k19"
ALLOW_PATTERNS = ("data/*.parquet", "compression_challenge/test_videos.zip")
MIN_VIDEO_ZIP_BYTES = 2_000_000_000


def _copy_real_file(source: Path, destination: Path) -> None:
    """Copy the resolved bytes, never an HF cache symlink, then replace atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        temporary.unlink()
    total = source.stat().st_size
    copied = 0
    next_report = 256 * 1024**2
    with source.open("rb") as input_file, temporary.open("wb") as output_file:
        while True:
            chunk = input_file.read(16 * 1024**2)
            if not chunk:
                break
            output_file.write(chunk)
            copied += len(chunk)
            if copied >= next_report or copied == total:
                print(f"  {source.name}: {copied / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
                next_report += 256 * 1024**2
        output_file.flush()
        os.fsync(output_file.fileno())
    if temporary.stat().st_size != source.stat().st_size:
        raise OSError(f"Size mismatch while copying {source.name}")
    os.replace(temporary, destination)


def _inventory(raw_root: Path) -> dict:
    parquet = sorted((raw_root / "data").glob("*.parquet"))
    video_zip = raw_root / "compression_challenge/test_videos.zip"
    files = parquet + ([video_zip] if video_zip.is_file() else [])
    return {
        "raw_root": str(raw_root),
        "parquet_count": len(parquet),
        "video_zip_bytes": video_zip.stat().st_size if video_zip.is_file() else 0,
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "is_symlink": path.is_symlink(),
            }
            for path in files
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/hf_stage3_cache"))
    args = parser.parse_args()

    drive_mount = Path("/content/drive")
    my_drive = drive_mount / "MyDrive"
    if not my_drive.is_dir():
        parser.error("Google Drive is not mounted at /content/drive/MyDrive")

    before = _inventory(args.raw_root)
    if before["parquet_count"] == 3 and before["video_zip_bytes"] > MIN_VIDEO_ZIP_BYTES:
        print("PASS: Stage 3 HF raw data already exists as Drive files.")
        print(json.dumps(before, ensure_ascii=False, indent=2))
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        parser.error(f"Install huggingface_hub first: {exc}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading the selected HF files to Colab local storage first...")
    snapshot_path = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=list(ALLOW_PATTERNS),
            local_dir=args.cache_dir,
        )
    )
    sources = sorted((snapshot_path / "data").glob("*.parquet"))
    source_zip = snapshot_path / "compression_challenge/test_videos.zip"
    if len(sources) != 3 or not source_zip.is_file() or source_zip.stat().st_size <= MIN_VIDEO_ZIP_BYTES:
        raise RuntimeError(
            f"Incomplete HF snapshot: parquet={len(sources)}, video_zip_bytes="
            f"{source_zip.stat().st_size if source_zip.is_file() else 0}"
        )

    for source in sources:
        destination = args.raw_root / "data" / source.name
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size or destination.is_symlink():
            print(f"Copying real bytes to Drive: {source.name}")
            _copy_real_file(source, destination)
    destination_zip = args.raw_root / "compression_challenge/test_videos.zip"
    if (
        not destination_zip.is_file()
        or destination_zip.stat().st_size != source_zip.stat().st_size
        or destination_zip.is_symlink()
    ):
        print("Copying real bytes to Drive: test_videos.zip (about 2.4 GB)")
        _copy_real_file(source_zip, destination_zip)

    os.sync()
    after = _inventory(args.raw_root)
    complete = after["parquet_count"] == 3 and after["video_zip_bytes"] > MIN_VIDEO_ZIP_BYTES
    real_files = all(not item["is_symlink"] for item in after["files"])
    if not complete or not real_files:
        raise RuntimeError("Drive materialization verification failed: " + json.dumps(after, ensure_ascii=False))

    marker = args.raw_root / "DRIVE_DATA_READY.json"
    marker.write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8")
    os.sync()
    print("PASS: 3 parquet files and the video ZIP are real files on mounted Drive.")
    print(json.dumps(after, ensure_ascii=False, indent=2))
    print(f"Drive visibility marker: {marker}")


if __name__ == "__main__":
    main()
