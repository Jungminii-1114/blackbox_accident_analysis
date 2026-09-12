"""Local delivery QA on real public videos; never a GPU submission certification."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd

import portfolio_lab as lab
from portfolio_runtime import pf_video_features, pf_signals, pf_classify, pf_event_features, pf_event_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = pd.read_csv(args.fixture / "labels/stage3.csv")
    signals = {name: np.zeros((len(labels), 4)) for name in ("A", "B")}
    dense = {}
    start = time.monotonic()
    for ident, group in labels.groupby("ID"):
        print("CPU QA flow:", ident, flush=True)
        features = pf_video_features(args.fixture / "input/stage3/videos" / f"{ident}.mp4")
        dense[ident] = features
        for family in signals:
            signals[family][group.index] = pf_signals(features, family)[group.sample_index.to_numpy(int)]
    _, cv_report = lab.nested_cv(labels, signals)
    _, family, quantile = lab.select_inner(labels, signals)
    configuration = {"stage1_slots": 5, "stage2_family": "B", "stage3": {
        "family": family, "thresholds": lab.thresholds(signals[family], quantile)}}
    event_rows = []
    stage2_labels = pd.read_csv(args.fixture / "labels/stage2.csv")
    times = pd.read_csv(args.fixture / "labels/stage2_frame_time.csv")
    for folder in sorted((args.fixture / "input/stage2/images").iterdir()):
        numbers, features = pf_event_features(folder)
        target_frame = int(stage2_labels.set_index("ID").loc[folder.name, "collision_frame"])
        mapping = times[times.ID == folder.name].set_index("frame").time_seconds.to_dict()
        for name in ("A", "B"):
            pred = int(numbers[np.argmax(pf_event_scores(features, name))])
            error = abs(mapping[pred] - mapping[target_frame])
            event_rows.append({"ID": folder.name, "candidate": name, "predicted_frame": pred,
                               "target_frame": target_frame, "error_seconds": error, "hit_0_3s": error <= .3})
    # Import generated inference, not just its source helpers; check checkpoint architectures on CPU.
    import torch
    from torch import nn
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="portfolio_import_qa_") as directory:
        root = Path(directory)
        source = lab.make_source(configuration)
        (root / "inference.py").write_text(source, encoding="utf-8")
        module = lab.load_module(root / "inference.py")
        s1 = torch.load(lab.ROOT / "model/stage1/best.pt", map_location="cpu", weights_only=False)
        model = module.mvit_v2_s(weights=None)
        model.head[1] = nn.Linear(model.head[1].in_features, 2)
        model.load_state_dict(s1["model"], strict=True)
        del s1, model
        backbone = module.resnet18(weights=None)
        backbone.load_state_dict(torch.load(lab.ROOT / "model/stage2/resnet18-f37072fd.pth", map_location="cpu", weights_only=True), strict=True)
        del backbone
        model = module._Stage2Temporal()
        model.load_state_dict(torch.load(lab.ROOT / "model/stage2/best.pt", map_location="cpu", weights_only=False)["model"], strict=True)
        del model
        model = module._Stage3MViT()
        model.load_state_dict(torch.load(lab.ROOT / "model/stage3/best.pt", map_location="cpu", weights_only=False)["model"], strict=True)
        del model
        print("All four checkpoint files load strictly; generated module imports", flush=True)
        actual = module.predict_stage3(args.fixture / "input/stage3", lab.ROOT / "model/stage3")
        for ident, group in actual.groupby("ID"):
            a, s = pf_classify(pf_signals(dense[ident], family), configuration["stage3"]["thresholds"])
            assert np.array_equal(a, group.accel_label.to_numpy())
            assert np.array_equal(s, group.steer_label.to_numpy())
        expected = pd.read_csv(args.fixture / "input/stage3_expected.csv")
        pd.testing.assert_frame_equal(actual[["ID", "sample_index"]], expected)
        # CPU smoke for composition's Stage2 field preservation without pretending to run GPU.
        original = stage2_labels.copy()
        original.entry_frame, original.entry_side, original.evasion_space = 0, "LEFT", 0
        module._PF_BASE_STAGE2 = lambda *unused: original.copy()
        composed = module.predict_stage2(args.fixture / "input/stage2", lab.ROOT / "model/stage2")
        pd.testing.assert_frame_equal(composed.drop(columns="collision_frame"), original.drop(columns="collision_frame"))
        archive_path = root / "submit.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
            archive.write(root / "inference.py", "inference.py")
            archive.write(lab.ROOT / "requirements.txt", "requirements.txt")
            for name in lab.MODEL_FILES:
                archive.write(lab.ROOT / name, name)
        lab.run([lab.sys.executable, lab.ROOT / "tools/validate_submit_package.py", "--zip", archive_path, "--import-check"])
    result = {"status": "CPU_QA_PASS", "gpu_submission_check": "NOT_RUN_LOCAL_NO_CUDA",
              "seconds": time.monotonic() - start, "stage3_rows": len(actual), "source_fingerprints": lab.source_fingerprint(),
              "stage3_nested_cv": cv_report, "stage2_collision_diagnostics": event_rows,
              "checks": ["real public flow + nested CV", "generated standalone import", "strict four checkpoint loads",
                         "CPU Stage3 standalone/runtime prediction equality", "exact sample coverage",
                         "Stage2 preserved fields (stub baseline)", "ZIP layout and subprocess import"]}
    lab.save(args.output, result)
    print("Saved", args.output, "Stage3 OOF:", cv_report["pooled_oof"]["score"], flush=True)


if __name__ == "__main__":
    main()
