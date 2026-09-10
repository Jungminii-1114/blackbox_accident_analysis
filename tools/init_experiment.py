"""Initialize one reproducible experiment under the persistent Google Drive root.

The tool deliberately does not copy model weights or data: those must be named in
config/experiment.json.  A candidate may be submitted only from this folder.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_COLUMNS = (
    "experiment", "stage", "parent", "status", "local_bss", "local_candidate",
    "local_delta", "official_stage_score", "official_weighted_score", "notes",
)


def append_registry(path: Path, row: dict[str, str]) -> None:
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if any(existing.get("experiment") == row["experiment"] for existing in rows):
            return
    rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one named DACON 236753 experiment folder on Google Drive.")
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--experiment", required=True, help="e.g. s3_can_v1; lowercase letters, digits, _ and - only")
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3"), required=True)
    parser.add_argument("--parent", default="baseline_official", help="frozen parent experiment name")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    if not args.experiment.replace("_", "").replace("-", "").isalnum() or args.experiment != args.experiment.lower():
        parser.error("--experiment must use lowercase letters, digits, _ or - only")

    root = args.drive_root / "experiments"
    experiment = root / args.experiment
    if experiment.exists() and any(experiment.iterdir()):
        parser.error(f"Refusing to overwrite an existing experiment: {experiment}")
    for name in ("checkpoints", "predictions", "reports", "submission", "logs"):
        (experiment / name).mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": args.experiment,
        "stage": args.stage,
        "parent": args.parent,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest": "",
        "split_definition": "external source/route group holdout; never random frame split",
        "baseline_metric_report": "",
        "candidate_metric_report": "reports/local_metrics.json",
        "zip_smoke_marker": "reports/zip_smoke_pass.json",
    }
    (experiment / "config" / "experiment.json").parent.mkdir(parents=True, exist_ok=True)
    (experiment / "config" / "experiment.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (experiment / "README.md").write_text(
        "# " + args.experiment + "\n\n"
        "1. Put the external data manifest path in `config/experiment.json`.\n"
        "2. Keep checkpoints in `checkpoints/`; predictions in `predictions/`.\n"
        "3. Save `tools/local_validate.py` output as `reports/local_metrics.json`.\n"
        "4. Create `reports/zip_smoke_pass.json` only after two extraction/re-inference checks pass.\n"
        "5. Run `tools/go_no_go.py` and submit only when it prints GO.\n",
        encoding="utf-8",
    )
    append_registry(root / "registry.csv", {
        "experiment": args.experiment, "stage": args.stage, "parent": args.parent,
        "status": "CREATED", "local_bss": "", "local_candidate": "", "local_delta": "",
        "official_stage_score": "", "official_weighted_score": "", "notes": args.notes,
    })
    print(f"Created: {experiment}")
    print(f"Registry: {root / 'registry.csv'}")


if __name__ == "__main__":
    main()
