"""Install a compatible Stage-3 checkpoint, build submit.zip, and smoke it twice."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package and double-smoke a Stage-3 candidate.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    experiment = args.experiment_dir.resolve()
    if not args.checkpoint.is_file():
        parser.error(f"Missing candidate checkpoint: {args.checkpoint}")
    target_checkpoint = root / "model/stage3/best.pt"
    shutil.copy2(args.checkpoint, target_checkpoint)
    smoke = root / "artifacts/baseline_smoke"
    if not (smoke / "input").is_dir():
        run([sys.executable, "tools/prepare_baseline_smoke.py", "--baseline-data", "fixtures/baseline_data", "--out", str(smoke)], root)

    first = experiment / "predictions/smoke_workspace"
    second = experiment / "predictions/smoke_from_zip"
    submission = experiment / "submission/submit.zip"
    run([sys.executable, "tools/run_local_inference.py", "--input-root", str(smoke / "input"), "--model-dir", "model", "--out", str(first)], root)
    run([sys.executable, "tools/check_predictions.py", "--predictions-dir", str(first), "--stage1-videos", str(smoke / "input/stage1/videos"), "--stage2-images", str(smoke / "input/stage2/images"), "--stage3-expected", str(smoke / "input/stage3_expected.csv")], root)
    run([sys.executable, "tools/build_submit.py", "--output", str(submission)], root)
    run([sys.executable, "tools/validate_submit_package.py", "--zip", str(submission), "--import-check"], root)
    with tempfile.TemporaryDirectory(prefix="dacon236753_candidate_") as directory:
        extracted = Path(directory)
        shutil.unpack_archive(str(submission), extracted)
        run([sys.executable, str(root / "tools/run_local_inference.py"), "--inference", str(extracted / "inference.py"), "--input-root", str(smoke / "input"), "--model-dir", str(extracted / "model"), "--out", str(second)], root)
    run([sys.executable, "tools/check_predictions.py", "--predictions-dir", str(second), "--stage1-videos", str(smoke / "input/stage1/videos"), "--stage2-images", str(smoke / "input/stage2/images"), "--stage3-expected", str(smoke / "input/stage3_expected.csv")], root)
    marker = experiment / "reports/zip_smoke_pass.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"passed": True, "checks": 2, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "submission": str(submission)}, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: candidate ZIP completed two independent smoke runs: {submission}")


if __name__ == "__main__":
    main()
