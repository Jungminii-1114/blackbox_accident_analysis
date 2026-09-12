"""Fast, no-GPU preflight checks for the Dacon submission package."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    required = ("predict_stage1", "predict_stage2", "predict_stage3")
    entry = (ROOT / "inference.py").read_text(encoding="utf-8")
    assert "from baseline_inference import predict_stage1, predict_stage2, predict_stage3" in entry
    tree = ast.parse((ROOT / "baseline_inference.py").read_text(encoding="utf-8"))
    signatures = {
        node.name: [argument.arg for argument in node.args.args]
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in required
    }
    for name in required:
        assert signatures.get(name) == ["data_dir", "model_dir"], name
    for stage in ("stage1", "stage2", "stage3"):
        assert (ROOT / "model" / stage / "best.pt").is_file(), f"missing model/{stage}/best.pt"
    assert (ROOT / "model/stage2/resnet18-f37072fd.pth").is_file(), "missing Stage 2 backbone weights"
    assert (ROOT / "baseline_inference.py").is_file(), "missing official baseline inference implementation"
    print("PASS: entry-point wiring, three predictor signatures, and official checkpoint files are ready.")
    print("Run the three predictors on a CUDA host before spending a Dacon submission.")


if __name__ == "__main__":
    main()
