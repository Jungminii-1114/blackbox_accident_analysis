"""Build a tiny Colab code patch without checkpoints or video fixtures."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import zipfile


FILES = (
    "inference.py",
    "baseline_inference.py",
    "requirements.txt",
    "notebooks/dacon236753_colab_runner.ipynb",
    "notebooks/01_stage3_can_lab.ipynb",
    "notebooks/02_stage2_event_lab.ipynb",
    "notebooks/03_stage1_forensics_lab.ipynb",
)
DIRECTORIES = ("tools", "docs")
SKIP_NAMES = {".DS_Store", ".gitkeep", "__pycache__"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight Colab code patch.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/colab/dacon236753_patch.zip"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    missing = [path for path in FILES + DIRECTORIES if not (root / path).exists()]
    if missing:
        parser.error("Missing patch content: " + ", ".join(missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "dacon236753_colab_patch_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "excludes": ["model", "fixtures"],
    }
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("PATCH_INFO.json", json.dumps(metadata, indent=2) + "\n")
        for relative in FILES:
            archive.write(root / relative, relative)
        for relative in DIRECTORIES:
            for path in (root / relative).rglob("*"):
                if path.is_file() and not any(part in SKIP_NAMES for part in path.parts):
                    archive.write(path, path.relative_to(root))
    print(f"Created {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
