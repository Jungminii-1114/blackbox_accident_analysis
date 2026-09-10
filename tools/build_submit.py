"""Build the exact root-level submit.zip required by DACON 236753."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import zipfile


# The archive must have one standalone inference.py.  The workspace keeps the
# official source as baseline_inference.py for review, then flattens it here so
# the evaluator never depends on a sibling helper import.
ROOT_FILES = ("inference.py", "requirements.txt")
PROJECT_FILES = ("baseline_inference.py", "requirements.txt")
MODEL_FILES = (
    "model/stage1/best.pt",
    "model/stage2/best.pt",
    "model/stage2/resnet18-f37072fd.pth",
    "model/stage3/best.pt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a submission archive with no extra top-level directory.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/submit/submit.zip"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    required = PROJECT_FILES + MODEL_FILES
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        parser.error("Missing required submission files: " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(root / "baseline_inference.py", arcname="inference.py")
        archive.write(root / "requirements.txt", arcname="requirements.txt")
        for relative in MODEL_FILES:
            archive.write(root / relative, arcname=relative)
    size = args.output.stat().st_size
    if size > 10 * 1024**3:
        raise RuntimeError(f"ZIP is {size / 1024**3:.2f} GiB; maximum is 10 GiB")
    print(f"Created {args.output} ({size / 1024**2:.1f} MiB)")
    print(f"SHA256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
