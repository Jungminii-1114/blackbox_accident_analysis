"""Executable three-stage experiment; no hidden-score extrapolation.

Run from project root: python tools/portfolio_lab.py COMMAND --experiment-dir PATH
Each command uses check=True semantics and logs progress without notebook shell magic.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import cv2
import numpy as np
import pandas as pd

from build_submit import MODEL_FILES, sha256
from local_validate import macro_f1, score_stage1, score_stage2, score_stage3
from portfolio_runtime import (pf_video_features, pf_signals, pf_classify,
                               pf_event_features, pf_event_scores)

ROOT = Path(__file__).resolve().parents[1]
ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]
OFFICIAL = {"stage1": .5339789355, "stage2": .11721, "stage3": .14581}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def load_module(path):
    spec = importlib.util.spec_from_file_location("portfolio_submission", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_source(config):
    """Flatten all helpers into the evaluator's one allowed inference module."""
    source = (ROOT / "baseline_inference.py").read_text(encoding="utf-8")
    assert source.count("    slots = 3") == 1, "Baseline layout changed: review composition"
    source = source.replace("    slots = 3", '    slots = PORTFOLIO_CONFIG.get("stage1_slots", 3)')
    needle = "    rows = []\n    for path, values in zip(videos, scores):"
    assert source.count(needle) == 1
    source = source.replace(needle, '    PF_S1_TRACE.clear()\n    PF_S1_TRACE.update({p.stem: v for p, v in zip(videos, scores)})\n' + needle)
    runtime = (ROOT / "tools/portfolio_runtime.py").read_text(encoding="utf-8")
    return (source + "\nPORTFOLIO_CONFIG = " + repr(config) + "\nPF_S1_TRACE = {}\n"
            "_PF_BASE_STAGE2 = predict_stage2\n_PF_BASE_STAGE3 = predict_stage3\n" + runtime +
            "\npredict_stage2 = pf_predict_stage2\npredict_stage3 = pf_predict_stage3\n")


def source_fingerprint():
    files = ["baseline_inference.py", "requirements.txt", "tools/portfolio_runtime.py",
             "tools/portfolio_lab.py", "tools/local_validate.py", "tools/check_predictions.py",
             "tools/validate_submit_package.py", "tools/portfolio_base_manifest.json"]
    return {name: sha256(ROOT / name) for name in files}


def decode_metadata(video):
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    times = []
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
    cap.release()
    if not times or not np.isfinite(times).all() or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError(f"Invalid decoded timestamps: {video}")
    return {"fps": fps, "frames": len(times), "pts": times}


def prepare(exp):
    # Never use mutable ROOT/model: the notebook extracts an immutable original base.
    model_manifest = read(ROOT / "tools/portfolio_base_manifest.json")
    for name, expected in model_manifest["model_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Original baseline checkpoint mismatch: {name}. Re-extract the original base ZIP.")
    source = ROOT / "fixtures/baseline_data"
    fingerprints = source_fingerprint()
    fingerprints.update({str(p.relative_to(ROOT)): sha256(p) for p in source.rglob("*") if p.is_file()})
    manifest_path = exp / "reports/provenance.json"
    if manifest_path.exists():
        prior = read(manifest_path)
        if prior["fingerprints"] != fingerprints:
            raise RuntimeError("Source/data changed. Use a NEW experiment ID; stale cache is not reusable.")
        old_audit = read(exp / "reports/time_audit.json")
        if all((exp / "fixture" / name).is_file() for name in old_audit["fixture_sha256"]):
            verify_fixture(exp)
            print("Verified existing fixture + immutable provenance", flush=True)
            return
        print("Restored reports without local fixture: regenerating identical fixture", flush=True)
    fixture = exp / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    label_root, inputs = fixture / "labels", fixture / "input"
    label_root.mkdir(exist_ok=True)
    for stage in (1, 2, 3):
        (inputs / f"stage{stage}" / ("images" if stage == 2 else "videos")).mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(source / "stage1/labels.csv")
    for row in labels.to_dict("records"):
        video = source / "stage1" / row["path"]
        shutil.copy2(video, inputs / "stage1/videos" / (row["ID"] + video.suffix))
    labels[["ID", "label"]].to_csv(label_root / "stage1.csv", index=False)
    labels = pd.read_csv(source / "stage2/labels.csv")
    time_rows = []
    for row in labels.to_dict("records"):
        folder = inputs / "stage2/images" / row["ID"]
        folder.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(source / "stage2" / row["path"]))
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not cv2.imwrite(str(folder / f"frame_{index:06d}.jpg"), frame):
                raise IOError("Frame write failed")
            time_rows.append({"ID": row["ID"], "frame": index, "time_seconds": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000})
            index += 1
        cap.release()
        if index == 0:
            raise ValueError("Stage2 decode failed")
    labels.rename(columns={"t_collision": "collision_frame", "t_entry": "entry_frame"})[
        ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]].to_csv(label_root / "stage2.csv", index=False)
    pd.DataFrame(time_rows).to_csv(label_root / "stage2_frame_time.csv", index=False)
    original_labels = pd.read_csv(source / "stage3/labels.csv")
    original_labels[["ID", "sample_index", "accel_label", "steer_label"]].to_csv(label_root / "stage3.csv", index=False)
    executable = shutil.which("ffmpeg")
    if not executable:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as error:
            raise RuntimeError("ffmpeg executable required (already included in Colab)") from error
    audits, expected_rows = [], []
    for video in sorted((source / "stage3/videos").glob("*.mp4")):
        print("Audit/10Hz conversion:", video.name, flush=True)
        original = decode_metadata(video)
        subset = original_labels[original_labels.ID == video.stem]
        positive = subset[subset.time_seconds > 0]
        rates = positive.frame_index.to_numpy() / positive.time_seconds.to_numpy()
        sample_rates = positive.sample_index.to_numpy() / positive.time_seconds.to_numpy()
        if not len(rates) or not np.allclose(rates, rates[0]) or not np.allclose(sample_rates, 10):
            raise ValueError("Public labels do not establish a consistent frame/time grid")
        physical_fps = float(rates[0])
        stride = int(round(physical_fps / 10))
        if stride < 1 or abs(physical_fps - 10 * stride) > 1e-6:
            raise ValueError("Non-integer public frame-to-sample ratio; manual audit required")
        if not np.array_equal(subset.frame_index.to_numpy(), subset.sample_index.to_numpy() * stride):
            raise ValueError("Public frame_index is not sample_index * stride")
        target = inputs / "stage3/videos" / video.name
        temporary = target.with_suffix(".part.mp4")
        run([executable, "-hide_banner", "-loglevel", "error", "-y", "-i", video,
             "-vf", f"select=not(mod(n\\,{stride})),setpts=N/(10*TB),pad=ceil(iw/2)*2:ceil(ih/2)*2", "-r", "10", "-an", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", temporary])
        temporary.replace(target)
        converted = decode_metadata(target)
        if subset.empty or abs(converted["fps"] - 10) > .1:
            raise ValueError("Missing labels or non-10Hz fixture")
        errors = []
        for row in subset.itertuples():
            if not (0 <= row.frame_index < original["frames"] and 0 <= row.sample_index < converted["frames"]):
                raise ValueError("Label index out of bounds")
            raw_error = abs(row.frame_index / physical_fps - row.time_seconds)
            container_error = abs(original["pts"][row.frame_index] - row.time_seconds)
            target_error = abs(converted["pts"][row.sample_index] - row.time_seconds)
            if raw_error > 1 / original["fps"] + .005 or target_error > .051:
                raise ValueError(f"Timestamp mismatch {video.name}: {raw_error}, {target_error}")
            errors.append({"sample_index": int(row.sample_index), "label_grid_error_s": raw_error,
                           "container_pts_error_s": container_error, "target_error_s": target_error})
        audits.append({"ID": video.stem, "raw_fps": original["fps"], "raw_frames": original["frames"],
                       "public_label_inferred_fps": physical_fps, "public_frame_stride": stride,
                       "warning": "public container timestamps are not authoritative; explicit label frame/time mapping is used",
                       "target_fps": converted["fps"], "target_frames": converted["frames"], "label_checks": errors})
        expected_rows.extend({"ID": video.stem, "sample_index": i} for i in range(converted["frames"]))
    pd.DataFrame(expected_rows).to_csv(inputs / "stage3_expected.csv", index=False)
    fixture_hashes = {str(p.relative_to(fixture)): sha256(p) for p in fixture.rglob("*") if p.is_file()}
    if manifest_path.exists() and fixture_hashes != old_audit["fixture_sha256"]:
        raise RuntimeError("Regenerated fixture differs (e.g. ffmpeg version changed). Use a new experiment ID.")
    save(exp / "reports/time_audit.json", {"passed": True, "contract": "official Baseline Inference markdown cell 7: 10Hz",
                                         "videos": audits, "fixture_sha256": fixture_hashes})
    save(manifest_path, {"fingerprints": fingerprints, "models": model_manifest["model_sha256"],
                         "label_exposure": "official baseline-trained public regression examples; NOT clean holdout"})


def verify_fixture(exp):
    audit = read(exp / "reports/time_audit.json")
    assert audit["passed"]
    for name, digest in audit["fixture_sha256"].items():
        if sha256(exp / "fixture" / name) != digest:
            raise RuntimeError(f"Fixture changed: {name}; use a new experiment ID")


def sparse_stage3_baseline(module, inputs, labels):
    import torch
    model = module._Stage3MViT()
    model.load_state_dict(torch.load(ROOT / "model/stage3/best.pt", map_location="cpu", weights_only=False)["model"])
    model.to(module._device()).eval()
    result = []
    with torch.inference_mode():
        for ident, subset in labels.groupby("ID"):
            print("Baseline S3 labeled centers:", ident, flush=True)
            frames = module._stage3_frames(inputs / "stage3/videos" / f"{ident}.mp4")
            centers = subset.sample_index.to_numpy(dtype=int)
            for offset in range(0, len(centers), 4):
                center = centers[offset:offset + 4]
                indexes = np.clip(center[:, None] - 8 + np.arange(16)[None], 0, len(frames) - 1)
                clips = frames[torch.from_numpy(indexes)].permute(0, 2, 1, 3, 4).float() / 255
                clips = (clips - module.S3_MEAN[None, :, None]) / module.S3_STD[None, :, None]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    a, s = model(clips.to(module._device()))
                result.extend({"ID": ident, "sample_index": int(i), "accel_label": ACCEL[x], "steer_label": STEER[y]}
                              for i, x, y in zip(center, a.argmax(1).tolist(), s.argmax(1).tolist()))
            del frames
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(result)


def baseline(exp):
    module = load_module(ROOT / "baseline_inference.py")
    inputs = exp / "fixture/input"
    out = exp / "predictions/baseline"
    out.mkdir(parents=True, exist_ok=True)
    for stage in (1, 2, 3):
        print(f"Original baseline Stage {stage}", flush=True)
        if stage == 3:
            prediction = sparse_stage3_baseline(module, inputs, pd.read_csv(exp / "fixture/labels/stage3.csv"))
        else:
            prediction = getattr(module, f"predict_stage{stage}")(inputs / f"stage{stage}", ROOT / f"model/stage{stage}")
        prediction.to_csv(out / f"stage{stage}.csv", index=False)
    save(exp / "reports/baseline.json", metrics(exp, out))


def metrics(exp, folder):
    labels = exp / "fixture/labels"
    return {"stage1": score_stage1(labels / "stage1.csv", folder / "stage1.csv"),
            "stage2": score_stage2(labels / "stage2.csv", folder / "stage2.csv", labels / "stage2_frame_time.csv", None),
            "stage3": score_stage3(labels / "stage3.csv", folder / "stage3.csv")}


def stage1(exp):
    source = exp / "config/s1_candidate.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(make_source({"stage1_slots": 5}), encoding="utf-8")
    module = load_module(source)
    prediction = module.predict_stage1(exp / "fixture/input/stage1", ROOT / "model/stage1")
    output = exp / "predictions/s1_A.csv"
    prediction.to_csv(output, index=False)
    median = prediction.copy()
    for index, row in median.iterrows():
        values = module.PF_S1_TRACE[row.ID]
        median.at[index, "answer"] = "RERECORDED" if not values or np.median(values) >= .5 else "ORIGINAL"
    median.to_csv(exp / "predictions/s1_B.csv", index=False)
    save(exp / "features/s1_clip_probabilities.json", module.PF_S1_TRACE)
    label = exp / "fixture/labels/stage1.csv"
    reports = {name: score_stage1(label, exp / f"predictions/s1_{name}.csv") for name in ("A", "B")}
    base = read(exp / "reports/baseline.json")["stage1"]["macro_f1"]
    valid = all(len(values) == 5 for values in module.PF_S1_TRACE.values())
    choice = "A" if valid and reports["A"]["macro_f1"] >= base else "baseline"
    save(exp / "reports/stage1.json", {"baseline": base, "candidates": reports, "selected": choice,
                                      "evidence": "EXPERIMENTAL", "all_clips_decoded": valid,
                                      "selection": "preselected A; median diagnostic only; public regression veto"})


def stage2(exp):
    base = pd.read_csv(exp / "predictions/baseline/stage2.csv")
    outputs = {name: base.copy() for name in ("A", "B")}
    diagnostics = []
    for row in base.itertuples():
        print("S2 event features:", row.ID, flush=True)
        numbers, features = pf_event_features(exp / "fixture/input/stage2/images" / row.ID)
        np.savez_compressed(exp / "features" / f"s2_{row.ID}.npz", numbers=numbers, features=features)
        for name in outputs:
            scores = pf_event_scores(features, name)
            top = np.argsort(-scores, kind="stable")[:3]
            outputs[name].loc[outputs[name].ID == row.ID, "collision_frame"] = int(numbers[top[0]])
            diagnostics.append({"ID": row.ID, "family": name, "top3_frames": numbers[top].tolist(),
                                "affine_valid_fraction": float(features[:, 2].mean())})
    reports = {}
    labels = exp / "fixture/labels"
    for name, output in outputs.items():
        path = exp / f"predictions/s2_{name}.csv"
        output.to_csv(path, index=False)
        reports[name] = score_stage2(labels / "stage2.csv", path, labels / "stage2_frame_time.csv", None)
        pd.testing.assert_frame_equal(output.drop(columns="collision_frame"), base.drop(columns="collision_frame"))
    base_score = read(exp / "reports/baseline.json")["stage2"]["collision"]["accuracy_at_0_3s"]
    choice = next((name for name in ("B", "A") if reports[name]["collision"]["accuracy_at_0_3s"] >= base_score), "baseline")
    save(exp / "reports/stage2.json", {"baseline_collision_accuracy": base_score, "candidates": reports,
                                      "selected": choice, "evidence": "EXPERIMENTAL", "diagnostics": diagnostics,
                                      "selection": "preordered B then A; public collision regression veto; full Stage2 score unavailable"})


def head_metrics(truth, prediction):
    acceleration, steering = prediction
    valid = truth.accel_label.to_numpy() != "STOPPED"
    a = macro_f1(truth.accel_label.tolist(), acceleration.tolist(), ACCEL)
    s = macro_f1(truth.steer_label.to_numpy()[valid].tolist(), steering[valid].tolist(), STEER)
    def per_class(actual, guessed, classes):
        return {label: {"support": int(np.sum(actual == label)), "f1": macro_f1(actual.tolist(), guessed.tolist(), [label])} for label in classes}
    return {"score": None if a is None or s is None else .7 * a + .3 * s, "accel_macro_f1": a, "steer_macro_f1": s,
            "accel": per_class(truth.accel_label.to_numpy(), acceleration, ACCEL),
            "steer": per_class(truth.steer_label.to_numpy()[valid], steering[valid], STEER),
            "predicted_accel_counts": {label: int(np.sum(acceleration == label)) for label in ACCEL}}


QUANTILES = list(itertools.product((.05, .10, .20), (.30, .50, .70), (.40, .60, .80)))


def thresholds(signals, quantiles):
    return [max(float(np.quantile(np.abs(signals[:, i]), q)), 1e-4) for i, q in enumerate(quantiles)]


def select_inner(labels, signals, families=("A", "B")):
    groups = sorted(labels.ID.unique())
    best = None
    for family in families:
        for quantile in QUANTILES:
            a = np.empty(len(labels), dtype=object)
            s = np.empty(len(labels), dtype=object)
            for group in groups:
                train = labels.ID.to_numpy() != group
                test = ~train
                cuts = thresholds(signals[family][train], quantile)
                a[test], s[test] = pf_classify(signals[family][test], cuts)
            value = head_metrics(labels, (a, s))["score"]
            if value is None:
                continue
            if best is None or value > best[0] + 1e-12:
                best = (value, family, quantile)
    if best is None:
        raise ValueError("No evaluable steering/acceleration labels in inner CV")
    return best


def nested_cv(labels, signals, families=("A", "B")):
    a, s = np.empty(len(labels), object), np.empty(len(labels), object)
    folds = []
    for group in sorted(labels.ID.unique()):
        train = labels.ID.to_numpy() != group
        test = ~train
        _, family, quantile = select_inner(labels[train].reset_index(drop=True), {k: v[train] for k, v in signals.items()}, families)
        cuts = thresholds(signals[family][train], quantile)
        a[test], s[test] = pf_classify(signals[family][test], cuts)
        folds.append({"held_out": group, "train_groups": sorted(labels[train].ID.unique()), "family": family,
                      "quantiles": quantile, "thresholds": cuts, "metrics": head_metrics(labels[test], (a[test], s[test]))})
    return (a, s), {"pooled_oof": head_metrics(labels, (a, s)), "folds": folds}


def stage3(exp):
    labels = pd.read_csv(exp / "fixture/labels/stage3.csv")
    signals = {family: np.zeros((len(labels), 4)) for family in ("A", "B")}
    for ident, subset in labels.groupby("ID"):
        print("S3 streaming flow:", ident, flush=True)
        features = pf_video_features(exp / "fixture/input/stage3/videos" / f"{ident}.mp4")
        np.savez_compressed(exp / "features" / f"s3_{ident}.npz", features=features)
        for family in signals:
            signals[family][subset.index] = pf_signals(features, family)[subset.sample_index.to_numpy(dtype=int)]
    predictions, report = nested_cv(labels, signals)
    report["family_ablations"] = {name: nested_cv(labels, signals, (name,))[1] for name in signals}
    output = labels[["ID", "sample_index"]].copy()
    output["accel_label"], output["steer_label"] = predictions
    output.to_csv(exp / "predictions/s3_nested_oof.csv", index=False)
    _, family, quantile = select_inner(labels, signals)
    config = {"family": family, "thresholds": thresholds(signals[family], quantile), "quantiles": quantile}
    base = read(exp / "reports/baseline.json")["stage3"]
    oof = report["pooled_oof"]
    delta = oof["score"] - base["score"]
    passed = (delta > 0 and oof["accel_macro_f1"] >= base["accel_macro_f1"] - .02
              and oof["steer_macro_f1"] >= base["steer_macro_f1"] - .05)
    report.update({"selected": config if passed else {"family": "baseline"}, "calibrated_candidate": config,
                   "baseline": base, "local_delta": delta, "evidence": "EXPERIMENTAL",
                   "selection": "positive pooled OOF delta; accel drop <=.02; steer drop <=.05; not a clean baseline holdout",
                   "optional_CAN_diagnostics": "not run: this experiment requires no external data"})
    save(exp / "reports/stage3.json", report)


def smoke_check(exp, module_path, model_root, output):
    run([sys.executable, ROOT / "tools/run_local_inference.py", "--input-root", exp / "fixture/input",
         "--inference", module_path, "--model-dir", model_root, "--out", output], timeout=3600)
    run([sys.executable, ROOT / "tools/check_predictions.py", "--predictions-dir", output,
         "--stage1-videos", exp / "fixture/input/stage1/videos", "--stage2-images", exp / "fixture/input/stage2/images",
         "--stage3-expected", exp / "fixture/input/stage3_expected.csv"], timeout=120)


def package(exp):
    # Invalidate old success FIRST. A failed rerun must not leave a valid GO marker.
    save(exp / "reports/package_check.json", {"passed": False, "reason": "packaging in progress"})
    reports = {f"stage{i}": read(exp / f"reports/stage{i}.json") for i in (1, 2, 3)}
    config = {"stage1_slots": 5 if reports["stage1"]["selected"] == "A" else 3,
              "stage2_family": reports["stage2"]["selected"], "stage3": reports["stage3"]["selected"]}
    changed = config["stage1_slots"] != 3 or config["stage2_family"] != "baseline" or config["stage3"]["family"] != "baseline"
    save(exp / "config/selected.json", config)
    if not changed:
        save(exp / "reports/package_check.json", {"passed": False, "reason": "all candidates rejected; do not resubmit baseline"})
        print("NO_GO: all candidates reverted; no submission ZIP generated", flush=True)
        return
    staging = exp / "submission/workspace"
    staging.mkdir(parents=True, exist_ok=True)
    source = make_source(config)
    compile(source, "inference.py", "exec")
    (staging / "inference.py").write_text(source, encoding="utf-8")
    shutil.copy2(ROOT / "requirements.txt", staging / "requirements.txt")
    for name in MODEL_FILES:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    start = time.monotonic()
    smoke_check(exp, staging / "inference.py", staging / "model", exp / "predictions/selected_workspace")
    first_seconds = time.monotonic() - start
    archive_path = exp / "submission/submit.zip"
    with zipfile.ZipFile(archive_path.with_suffix(".part.zip"), "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name in ("inference.py", "requirements.txt", *MODEL_FILES):
            archive.write(staging / name, name)
    archive_path.with_suffix(".part.zip").replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        if archive_path.stat().st_size > 10_000_000_000 or sum(info.file_size for info in archive.infolist()) > 32_000_000_000:
            raise ValueError("ZIP size limit exceeded")
    run([sys.executable, ROOT / "tools/validate_submit_package.py", "--zip", archive_path, "--import-check"], timeout=300)
    with tempfile.TemporaryDirectory(prefix="portfolio_zip_") as directory:
        extracted = Path(directory)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted)
        # Same Colab environment, new process and extracted directory; not a clean L40S server test.
        start = time.monotonic()
        smoke_check(exp, extracted / "inference.py", extracted / "model", exp / "predictions/selected_zip")
        second_seconds = time.monotonic() - start
    prediction_hashes = {}
    for stage in (1, 2, 3):
        name = f"stage{stage}.csv"
        first = pd.read_csv(exp / "predictions/selected_workspace" / name).sort_values(["ID"] + (["sample_index"] if stage == 3 else [])).reset_index(drop=True)
        second = pd.read_csv(exp / "predictions/selected_zip" / name).sort_values(["ID"] + (["sample_index"] if stage == 3 else [])).reset_index(drop=True)
        pd.testing.assert_frame_equal(first, second)
        prediction_hashes[name] = sha256(exp / "predictions/selected_zip" / name)
        if stage in (1, 2):
            selected = reports[f"stage{stage}"]["selected"]
            reference = exp / (f"predictions/baseline/{name}" if selected == "baseline" else f"predictions/s{stage}_{selected}.csv")
            expected = pd.read_csv(reference).sort_values("ID").reset_index(drop=True)
            pd.testing.assert_frame_equal(first, expected)
    # Stage2 preserved outputs must match original baseline, not a reconditioned scene head.
    columns = ["ID", "entry_frame", "entry_side", "evasion_space"]
    before = pd.read_csv(exp / "predictions/baseline/stage2.csv")[columns].sort_values("ID").reset_index(drop=True)
    after = pd.read_csv(exp / "predictions/selected_zip/stage2.csv")[columns].sort_values("ID").reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)
    # Check Stage3 exported classifier really matches the calibrated runtime/cache.
    actual = pd.read_csv(exp / "predictions/selected_zip/stage3.csv")
    labels = pd.read_csv(exp / "fixture/labels/stage3.csv")
    sparse = labels[["ID", "sample_index"]].merge(actual, on=["ID", "sample_index"], validate="one_to_one")
    if config["stage3"]["family"] == "baseline":
        expected = pd.read_csv(exp / "predictions/baseline/stage3.csv")
        pd.testing.assert_frame_equal(sparse.sort_values(["ID", "sample_index"]).reset_index(drop=True), expected.sort_values(["ID", "sample_index"]).reset_index(drop=True))
    else:
        for ident, group in actual.groupby("ID"):
            features = np.load(exp / "features" / f"s3_{ident}.npz")["features"]
            a, s = pf_classify(pf_signals(features, config["stage3"]["family"]), config["stage3"]["thresholds"])
            ordered = group.sort_values("sample_index")
            assert np.array_equal(a, ordered.accel_label.to_numpy()) and np.array_equal(s, ordered.steer_label.to_numpy())
    save(exp / "reports/selected_fit_metrics.json", metrics(exp, exp / "predictions/selected_zip"))
    evidence_files = ["reports/baseline.json", "reports/stage1.json", "reports/stage2.json", "reports/stage3.json",
                      "reports/provenance.json", "reports/time_audit.json", "config/selected.json"]
    save(exp / "reports/package_check.json", {"passed": True, "zip_sha256": sha256(archive_path),
        "source_sha256": sha256(staging / "inference.py"), "prediction_sha256": prediction_hashes,
        "evidence_sha256": {name: sha256(exp / name) for name in evidence_files},
        "fixture_seconds_workspace": first_seconds, "fixture_seconds_zip": second_seconds,
        "environment": "same Colab runtime; GPU model recorded in environment.json", "server_full_runtime_verified": False})


def report(exp):
    failures = []
    try:
        provenance = read(exp / "reports/provenance.json")
        if any(provenance["fingerprints"].get(k) != v for k, v in source_fingerprint().items()):
            failures.append("Source differs from experiment provenance")
        for name, expected in provenance["models"].items():
            if sha256(ROOT / name) != expected:
                failures.append(f"Model changed: {name}")
        verify_fixture(exp)
    except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as error:
        failures.append(f"Provenance/fixture not verified: {error}")
    check_path = exp / "reports/package_check.json"
    check = read(check_path) if check_path.exists() else {"passed": False, "reason": "package check not executed"}
    archive_path = exp / "submission/submit.zip"
    if not check.get("passed"):
        failures.append(check.get("reason", "package validation incomplete"))
    elif not archive_path.exists() or sha256(archive_path) != check["zip_sha256"]:
        failures.append("ZIP hash differs from tested ZIP")
    else:
        if not check.get("evidence_sha256"):
            failures.append("Package marker lacks bound evidence")
        for name, expected in check.get("evidence_sha256", {}).items():
            if not (exp / name).is_file() or sha256(exp / name) != expected:
                failures.append(f"Evidence changed after packaging: {name}")
        for name, expected in check.get("prediction_sha256", {}).items():
            path = exp / "predictions/selected_zip" / name
            if not path.is_file() or sha256(path) != expected:
                failures.append(f"Prediction changed after packaging: {name}")
    base = read(exp / "reports/baseline.json") if (exp / "reports/baseline.json").exists() else {}
    stages = {}
    for stage in (1, 2, 3):
        key = f"stage{stage}"
        path = exp / f"reports/{key}.json"
        if not path.exists():
            failures.append(f"{key} experiment incomplete")
            continue
        if key not in base:
            failures.append(f"{key} baseline metrics missing")
            continue
        value = read(path)
        if stage == 1:
            baseline_score = base[key]["macro_f1"]
            candidate_score = value["candidates"]["A"]["macro_f1"]
        elif stage == 2:
            baseline_score = candidate_score = None  # three unknown targets, no fake Stage score
        else:
            baseline_score = base[key]["score"]
            candidate_score = value["pooled_oof"]["score"]
        stages[key] = {"selected": value["selected"], "local_baseline": baseline_score,
                       "local_candidate": candidate_score,
                       "local_BSS_delta": None if baseline_score is None else candidate_score - baseline_score,
                       "official_baseline": OFFICIAL[key], "official_candidate": None, "official_BSS_delta": None,
                       "evidence": "EXPERIMENTAL"}
        if stage == 2:
            stages[key]["collision_only"] = {"baseline": value["baseline_collision_accuracy"],
                **{name: result["collision"]["accuracy_at_0_3s"] for name, result in value["candidates"].items()}}
    verdict = "NO_GO" if failures else "GO"
    result = {"verdict": verdict, "purpose": "exploration", "improvement_probability": None,
              "meaning": "GO = tested exploratory package; NOT a promise of score improvement",
              "stages": stages, "failures": failures, "package_check": check,
              "previous_observed_stage3": {"official": .1420426225, "official_delta_approx": -.0037673775},
              "warnings": ["Only five baseline-exposed public groups; OOF is not an independent baseline comparison",
                           "Stage2 full local score unavailable", "Hidden-server total runtime not measured",
                           "Fitted full-data metrics must not replace nested OOF"]}
    save(exp / "reports/submission_gate.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("\nVERDICT:", verdict, "(exploration; 공식 상승 확률은 미측정)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "baseline", "stage1", "stage2", "stage3", "package", "report"])
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    exp = args.experiment_dir.resolve()
    for folder in ("config", "features", "reports", "predictions", "submission", "logs"):
        (exp / folder).mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    np.random.seed(236753)
    if args.command == "prepare":
        save(exp / "reports/package_check.json", {"passed": False, "reason": "preparation/restart requires a fresh package check"})
    if args.command not in ("prepare", "report"):
        save(exp / "reports/package_check.json", {"passed": False, "reason": f"{args.command} rerun invalidates previous package"})
        manifest = read(exp / "reports/provenance.json")
        if any(manifest["fingerprints"][k] != v for k, v in source_fingerprint().items()):
            raise RuntimeError("Code changed; use a new experiment ID")
        for name, digest in manifest["models"].items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Baseline checkpoint changed")
        verify_fixture(exp)
    try:
        globals()[args.command](exp)
    except Exception as error:
        if args.command != "report":
            save(exp / "reports/package_check.json", {"passed": False, "reason": f"{args.command} failed: {error}"})
        raise


if __name__ == "__main__":
    main()
