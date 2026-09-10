"""Run the submission's three public entry points and persist Stage CSVs."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def load_module(path: Path):
    # A submitted inference.py imports its sibling baseline_inference.py.
    # Add the archive extraction directory before executing it.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("submission_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load inference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DACON 236753 inference module on local evaluator-shaped input.")
    parser.add_argument("--input-root", type=Path, required=True, help="contains stage1/, stage2/, stage3/")
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inference", type=Path, default=Path("inference.py"))
    args = parser.parse_args()
    module = load_module(args.inference.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = (("stage1", module.predict_stage1), ("stage2", module.predict_stage2), ("stage3", module.predict_stage3))
    for stage, function in jobs:
        prediction = function(args.input_root / stage, args.model_dir / stage)
        output = args.out / f"{stage}.csv"
        prediction.to_csv(output, index=False, encoding="utf-8")
        print(f"{stage}: {len(prediction)} rows -> {output}")


if __name__ == "__main__":
    main()
