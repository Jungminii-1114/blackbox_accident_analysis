"""Append one public-leaderboard observation to an experiment log."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


COLUMNS = ["timestamp_utc", "experiment", "submission_zip", "git_revision", "local_stage1", "local_stage2", "local_stage3", "public_stage1", "public_stage2", "public_stage3", "notes"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a Dacon public-leaderboard result.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--public-stage1", type=float, required=True)
    parser.add_argument("--public-stage2", type=float, required=True)
    parser.add_argument("--public-stage3", type=float, required=True)
    parser.add_argument("--log", type=Path, default=Path("experiments/leaderboard.csv"))
    parser.add_argument("--submission-zip", default="")
    parser.add_argument("--git-revision", default="")
    parser.add_argument("--local-stage1", type=float)
    parser.add_argument("--local-stage2", type=float)
    parser.add_argument("--local-stage3", type=float)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.log.exists()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "experiment": args.experiment,
        "submission_zip": args.submission_zip,
        "git_revision": args.git_revision,
        "local_stage1": args.local_stage1 if args.local_stage1 is not None else "",
        "local_stage2": args.local_stage2 if args.local_stage2 is not None else "",
        "local_stage3": args.local_stage3 if args.local_stage3 is not None else "",
        "public_stage1": args.public_stage1,
        "public_stage2": args.public_stage2,
        "public_stage3": args.public_stage3,
        "notes": args.notes,
    }
    with args.log.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    overall = 0.2 * args.public_stage1 + 0.4 * args.public_stage2 + 0.4 * args.public_stage3
    print(f"Recorded {args.experiment}; public weighted score = {overall:.5f}")


if __name__ == "__main__":
    main()
