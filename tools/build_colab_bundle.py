"""Create a self-contained Colab bundle: code, checkpoints, and smoke fixture."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


FILES = (
    "inference.py", "baseline_inference.py", "requirements.txt",
    "notebooks/dacon236753_colab_runner.ipynb",
    "notebooks/01_stage3_can_lab.ipynb",
    "notebooks/02_stage2_event_lab.ipynb",
    "notebooks/03_stage1_forensics_lab.ipynb",
)
DIRECTORIES = ("model", "tools", "docs", "fixtures/baseline_data")
SKIP_NAMES = {".DS_Store", ".gitkeep", "__pycache__"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an uploadable Colab bundle for DACON 236753.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/colab/dacon236753_colab_bundle.zip"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    missing = [path for path in FILES + DIRECTORIES if not (root / path).exists()]
    if missing:
        parser.error("Missing bundle content: " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in FILES:
            archive.write(root / relative, relative)
        for relative in DIRECTORIES:
            for path in (root / relative).rglob("*"):
                if path.is_file() and not any(part in SKIP_NAMES for part in path.parts):
                    archive.write(path, path.relative_to(root))
    print(f"Created {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
