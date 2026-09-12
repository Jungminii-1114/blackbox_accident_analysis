"""Train Stage-3 linear heads on frozen MViTv2 features from a route holdout."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_inference import ACCEL, STEER, S3_MEAN, S3_STD, _Stage3MViT, _stage3_frames


def macro_f1(truth: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    values = []
    for label in range(classes):
        tp = int(np.sum((truth == label) & (prediction == label)))
        fp = int(np.sum((truth != label) & (prediction == label)))
        fn = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return float(np.mean(values))


def metrics(accel_y: torch.Tensor, steer_y: torch.Tensor, accel_logits: torch.Tensor, steer_logits: torch.Tensor) -> Dict[str, float]:
    ay = accel_y.numpy()
    ap = accel_logits.argmax(1).numpy()
    accel_f1 = macro_f1(ay, ap, 4)
    moving = ay != ACCEL.index("STOPPED")
    sy = steer_y.numpy()[moving]
    sp = steer_logits.argmax(1).numpy()[moving]
    steer_f1 = macro_f1(sy, sp, 3) if len(sy) else 0.0
    return {"accel_macro_f1": accel_f1, "steer_macro_f1": steer_f1, "score": 0.7 * accel_f1 + 0.3 * steer_f1}


class ClipDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, labels: pd.DataFrame, data_root: Path, split: str, stride: int):
        selected = manifest[manifest["split"] == split].copy()
        label_map = {(str(r.ID), int(r.sample_index)): (ACCEL.index(r.accel_label), STEER.index(r.steer_label)) for r in labels.itertuples(index=False)}
        self.entries: List[Tuple[Path, str, int, int, int]] = []
        for row in selected.itertuples(index=False):
            for center in range(0, int(row.samples), stride):
                if (str(row.id), center) in label_map:
                    accel, steer = label_map[(str(row.id), center)]
                    self.entries.append((data_root / row.video_path, str(row.id), center, accel, steer))
        self._cache_path: Optional[Path] = None
        self._cache_frames: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        path, _, center, accel, steer = self.entries[index]
        if self._cache_path != path:
            self._cache_frames = _stage3_frames(path)
            self._cache_path = path
        assert self._cache_frames is not None
        count = len(self._cache_frames)
        indices = np.clip(center - 8 + np.arange(16), 0, count - 1)
        clip = self._cache_frames[torch.from_numpy(indices)].permute(1, 0, 2, 3).float() / 255.0
        clip = (clip - S3_MEAN[:, None, :, :]) / S3_STD[:, None, :, :]
        return clip, accel, steer


def extract(model: _Stage3MViT, dataset: ClipDataset, device: torch.device, batch_size: int) -> Dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    features, accel_y, steer_y, base_accel, base_steer = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for clips, accel, steer in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden = model.backbone(clips.to(device, non_blocking=True))
                al = model.accel(hidden)
                sl = model.steer(hidden)
            features.append(hidden.float().cpu())
            base_accel.append(al.float().cpu())
            base_steer.append(sl.float().cpu())
            accel_y.append(accel)
            steer_y.append(steer)
    return {"features": torch.cat(features), "accel_y": torch.cat(accel_y), "steer_y": torch.cat(steer_y), "baseline_accel": torch.cat(base_accel), "baseline_steer": torch.cat(base_steer)}


def weights(labels: torch.Tensor, classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=classes).float()
    result = torch.zeros_like(counts)
    present = counts > 0
    result[present] = counts[present].sum() / (present.sum() * counts[present])
    return result.to(device)


def report(stage3: Dict[str, float], samples: int) -> Dict[str, object]:
    return {
        "stage1": {"macro_f1": None}, "stage2": {"score": None},
        "stage3": {**stage3, "samples": samples}, "overall_score": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Stage-3 MViTv2 heads using comma2k19 HF demo.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--feature-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=236753)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    manifest = pd.read_csv(args.data_root / "manifest.csv")
    labels_path = args.data_root / manifest.iloc[0]["labels_path"]
    labels = pd.read_csv(labels_path)
    if set(manifest["split"]) != {"train", "validation"}:
        parser.error("manifest must contain train and validation splits")
    overlap = set(manifest.loc[manifest.split == "train", "route_group"]) & set(manifest.loc[manifest.split == "validation", "route_group"])
    if overlap:
        parser.error(f"route leakage: {sorted(overlap)}")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    reports = args.experiment_dir / "reports"; reports.mkdir(exist_ok=True)
    features_dir = args.experiment_dir / "features"; features_dir.mkdir(exist_ok=True)
    checkpoints = args.experiment_dir / "checkpoints"; checkpoints.mkdir(exist_ok=True)

    checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
    model = _Stage3MViT(); model.load_state_dict(checkpoint["model"]); model.to(device)
    cached: Dict[str, Dict[str, torch.Tensor]] = {}
    for split in ("train", "validation"):
        cache_path = features_dir / f"{split}_stride{args.feature_stride}.pt"
        if cache_path.is_file() and not args.rebuild_features:
            cached[split] = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            dataset = ClipDataset(manifest, labels, args.data_root, split, args.feature_stride)
            if not len(dataset):
                raise RuntimeError(f"No samples for {split}")
            cached[split] = extract(model, dataset, device, args.batch_size)
            torch.save(cached[split], cache_path)
        print(split, {"samples": len(cached[split]["accel_y"]), "accel": dict(Counter(cached[split]["accel_y"].tolist())), "steer": dict(Counter(cached[split]["steer_y"].tolist()))})

    baseline_metric = metrics(cached["validation"]["accel_y"], cached["validation"]["steer_y"], cached["validation"]["baseline_accel"], cached["validation"]["baseline_steer"])
    (reports / "baseline_local_metrics.json").write_text(json.dumps(report(baseline_metric, len(cached["validation"]["accel_y"])), indent=2) + "\n")
    print("baseline:", baseline_metric)

    accel_head = copy.deepcopy(model.accel).to(device)
    steer_head = copy.deepcopy(model.steer).to(device)
    train = cached["train"]
    train_set = torch.utils.data.TensorDataset(train["features"].float(), train["accel_y"], train["steer_y"])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=512, shuffle=True, generator=generator)
    criterion_accel = nn.CrossEntropyLoss(weight=weights(train["accel_y"], 4, device))
    criterion_steer = nn.CrossEntropyLoss(weight=weights(train["steer_y"], 3, device), reduction="none")
    optimizer = torch.optim.AdamW(list(accel_head.parameters()) + list(steer_head.parameters()), lr=2e-3, weight_decay=1e-3)
    best_score = -1.0; best_state = None
    val = cached["validation"]
    for epoch in range(1, args.epochs + 1):
        accel_head.train(); steer_head.train()
        for feature, accel, steer in loader:
            feature, accel, steer = feature.to(device), accel.to(device), steer.to(device)
            al, sl = accel_head(feature), steer_head(feature)
            moving = accel != ACCEL.index("STOPPED")
            steer_loss = criterion_steer(sl, steer)
            steer_loss = steer_loss[moving].mean() if moving.any() else steer_loss.mean() * 0
            loss = criterion_accel(al, accel) + 0.3 * steer_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        accel_head.eval(); steer_head.eval()
        with torch.inference_mode():
            vf = val["features"].float().to(device)
            candidate = metrics(val["accel_y"], val["steer_y"], accel_head(vf).cpu(), steer_head(vf).cpu())
        if candidate["score"] > best_score:
            best_score = candidate["score"]
            best_state = {"accel": copy.deepcopy(accel_head.state_dict()), "steer": copy.deepcopy(steer_head.state_dict()), "metrics": candidate, "epoch": epoch}
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} score={candidate['score']:.5f} best={best_score:.5f}")
    assert best_state is not None
    model.accel.load_state_dict(best_state["accel"]); model.steer.load_state_dict(best_state["steer"])
    output_checkpoint = checkpoints / "stage3_best.pt"
    torch.save({"model": model.state_dict(), "source": "comma2k19_hf_demo", "feature_stride": args.feature_stride, "best_epoch": best_state["epoch"]}, output_checkpoint)
    candidate_report = report(best_state["metrics"], len(val["accel_y"]))
    (reports / "local_metrics.json").write_text(json.dumps(candidate_report, indent=2) + "\n")
    summary = {"baseline": baseline_metric, "candidate": best_state["metrics"], "delta": best_state["metrics"]["score"] - baseline_metric["score"], "checkpoint": str(output_checkpoint)}
    (reports / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
