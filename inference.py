"""DACON 236753 submission entry point.

The official baseline implementation is kept verbatim in
``baseline_inference.py`` so its checkpoint architectures and preprocessing
remain exactly compatible with the supplied ``model/stage*/best.pt`` files.
Keep this small entry point at the ZIP root: the evaluator imports these three
functions directly.
"""

from baseline_inference import predict_stage1, predict_stage2, predict_stage3

__all__ = ["predict_stage1", "predict_stage2", "predict_stage3"]
