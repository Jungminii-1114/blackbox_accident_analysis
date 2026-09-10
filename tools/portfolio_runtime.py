"""Self-contained CPU motion functions, also embedded into submit/inference.py.

No learning, cross-file aggregation, network access, or external helper imports.
"""
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def pf_affine(previous, current):
    cv2.setRNGSeed(236753)
    points = cv2.goodFeaturesToTrack(previous, 150, 0.01, 7)
    if points is None or len(points) < 8:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if tracked is None or status is None:
        return None
    valid = status.ravel().astype(bool) & np.isfinite(tracked.reshape(-1, 2)).all(1)
    if valid.sum() < 8:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(points[valid], tracked[valid], method=cv2.RANSAC)
    if matrix is None or not np.isfinite(matrix).all() or inliers.sum() < 6:
        return None
    return matrix


def pf_smooth(values, width=5):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    return np.median(np.lib.stride_tricks.sliding_window_view(
        np.pad(values, (width // 2, width // 2), mode="edge"), width), axis=-1)


def pf_slope(values, dt=0.1):
    idx = np.arange(len(values))
    lo, hi = np.maximum(idx - 3, 0), np.minimum(idx + 3, len(values) - 1)
    return (values[hi] - values[lo]) / np.maximum((hi - lo) * dt, dt)


def pf_video_features(path):
    """One row per decoded 10Hz frame. A/B use the exact same extraction at test time."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or abs(fps - 10.0) > 0.1:
        warnings.warn(f"Stage3 metadata FPS={fps}; use official decoded-frame-as-10Hz-sample contract: {path}")
    rows, previous = [], None
    yy, xx = np.mgrid[:90, :160]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        quality = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        if previous is None:
            rows.append([0.0, 0.0, 0.0, 0.0, quality, 0.0])
        else:
            flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            matrix = pf_affine(previous[:65], gray[:65])
            magnitude = np.linalg.norm(flow, axis=2)
            speed_a = float(np.median(magnitude[50:82, 24:136])) * 10
            steer_a = float(np.median(flow[25:50, 16:144, 0])) * 10
            speed_b, steer_b = speed_a, steer_a
            if matrix is not None:
                # Remove image-plane rotation, not full affine translation/forward motion.
                angle = np.arctan2(matrix[1, 0], matrix[0, 0])
                rotation = np.stack((-angle * (yy - 45), angle * (xx - 80)), axis=-1)
                compensated = flow - rotation
                speed_b = float(np.median(np.linalg.norm(compensated, axis=2)[50:82, 24:136])) * 10
                steer_b = float(matrix[0, 0] * 80 + matrix[0, 1] * 45 + matrix[0, 2] - 80) * 10
            rows.append([speed_a, steer_a, speed_b, steer_b, quality, float(matrix is not None)])
        previous = gray
    cap.release()
    if not rows:
        raise ValueError(f"Empty/unreadable Stage3 video: {path}")
    result = np.asarray(rows, dtype=np.float64)
    if len(result) > 1:
        result[0, :4] = result[1, :4]
    return result


def pf_signals(features, family):
    offset = 0 if family == "A" else 2
    speed = pf_smooth(features[:, offset])
    steer = pf_smooth(features[:, offset + 1])
    return np.column_stack((speed, pf_slope(speed), steer, features[:, 4]))


def pf_classify(signals, thresholds):
    stop, accel, steer = thresholds
    velocity, slope, turn, texture = signals.T
    acceleration = np.full(len(signals), "CONSTANT", dtype=object)
    acceleration[slope > accel] = "ACCELERATING"
    acceleration[slope < -accel] = "DECELERATING"
    acceleration[velocity <= stop] = "STOPPED"
    steering = np.full(len(signals), "STRAIGHT", dtype=object)
    steering[turn > steer] = "LEFT"
    steering[turn < -steer] = "RIGHT"
    # Textureless frames are uncertain, not evidence of a stopped vehicle.
    bad = texture < 2.0
    acceleration[bad], steering[bad] = "CONSTANT", "STRAIGHT"
    return acceleration, steering


def pf_frame_number(path):
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Frame filename has no numeric suffix: {path}")
    return int(match.group(1))


def pf_event_features(folder):
    paths = sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}), key=pf_frame_number)
    numbers = [pf_frame_number(p) for p in paths]
    if not paths or len(set(numbers)) != len(numbers):
        raise ValueError(f"Empty/ambiguous Stage2 frames: {folder}")
    rows, previous, last_translation = [], None, None
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Unreadable Stage2 frame: {path}")
        gray = cv2.resize(gray, (320, 180))
        difference, change, valid = 0.0, 0.0, 0.0
        if previous is not None:
            difference = float(np.mean(cv2.absdiff(previous, gray)))
            matrix = pf_affine(previous, gray)
            if matrix is not None:
                translation = matrix[:, 2] / np.hypot(320, 180)
                if last_translation is not None:
                    change, valid = float(np.linalg.norm(translation - last_translation)), 1.0
                last_translation = translation
            else:
                last_translation = None
        rows.append((difference, change, valid))
        previous = gray
    return np.asarray(numbers), np.asarray(rows)


def pf_robust_z(values):
    middle = np.median(values)
    return np.clip((values - middle) / max(1e-6, 1.4826 * np.median(np.abs(values - middle))), -8, 8)


def pf_event_scores(features, family):
    if family == "A":
        scores = features[:, 0].copy()
    else:
        a, b = pf_robust_z(features[:, 0]), pf_robust_z(features[:, 1])
        scores = np.where(features[:, 2] > 0, .5 * a + .5 * b, a)
    if len(scores) > 1:
        scores[0] = -np.inf  # no preceding frame; all real boundary pairs are kept
    return scores


def pf_predict_stage2(data_dir, model_dir):
    output = _PF_BASE_STAGE2(data_dir, model_dir)
    family = PORTFOLIO_CONFIG.get("stage2_family", "baseline")
    if family != "baseline":
        for index, row in output.iterrows():
            numbers, features = pf_event_features(Path(data_dir) / "images" / row["ID"])
            output.at[index, "collision_frame"] = int(numbers[np.argmax(pf_event_scores(features, family))])
    return output


def pf_predict_stage3(data_dir, model_dir):
    config = PORTFOLIO_CONFIG.get("stage3", {})
    if config.get("family", "baseline") == "baseline":
        return _PF_BASE_STAGE3(data_dir, model_dir)
    rows = []
    for video in _video_paths(Path(data_dir) / "videos"):
        features = pf_video_features(video)
        acceleration, steering = pf_classify(pf_signals(features, config["family"]), config["thresholds"])
        rows.extend({"ID": video.stem, "sample_index": i, "accel_label": a, "steer_label": s}
                    for i, (a, s) in enumerate(zip(acceleration, steering)))
    return pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])
