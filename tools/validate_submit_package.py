"""Validate archive layout, syntax, required checkpoint files, and imports."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import zipfile

from build_submit import MODEL_FILES, ROOT_FILES


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a DACON 236753 submit.zip before upload.")
    parser.add_argument("--zip", dest="archive_path", type=Path, required=True)
    parser.add_argument("--import-check", action="store_true", help="also import inference.py in the extracted archive")
    args = parser.parse_args()
    if not zipfile.is_zipfile(args.archive_path):
        parser.error(f"Not a ZIP file: {args.archive_path}")
    with tempfile.TemporaryDirectory(prefix="dacon236753_submit_check_") as directory:
        root = Path(directory)
        with zipfile.ZipFile(args.archive_path) as archive:
            names = [info.filename for info in archive.infolist() if not info.is_dir()]
            invalid = [name for name in names if name.startswith("/") or ".." in Path(name).parts]
            if invalid:
                raise RuntimeError("Unsafe archive paths: " + ", ".join(invalid))
            archive.extractall(root)
        required = set(ROOT_FILES + MODEL_FILES)
        found = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        missing = sorted(required - found)
        if missing:
            raise RuntimeError("Missing required archive members: " + ", ".join(missing))
        unexpected_root = sorted({path.parts[0] for path in (Path(name) for name in found)} - {"model", *ROOT_FILES})
        if unexpected_root:
            raise RuntimeError("Unexpected top-level archive members: " + ", ".join(unexpected_root))
        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", "inference.py"],
            cwd=root, text=True, capture_output=True, check=False,
        )
        if compile_result.returncode:
            raise RuntimeError("Python syntax check failed:\n" + compile_result.stderr)
        if args.import_check:
            imported = subprocess.run(
                [sys.executable, "-c", "import inference; print('import OK')"],
                cwd=root, text=True, capture_output=True, check=False,
            )
            if imported.returncode:
                raise RuntimeError("Import check failed:\n" + imported.stdout + imported.stderr)
            print(imported.stdout.strip())
        print(f"PASS: {args.archive_path} has exact required files and valid Python syntax.")


if __name__ == "__main__":
    main()
