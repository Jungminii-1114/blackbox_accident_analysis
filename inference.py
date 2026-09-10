# BASELINE_INFERENCE_PART
"""3-Stage 영상 분석 베이스라인 추론 코드.

각 Stage의 평가 데이터를 예측하여 정해진 형식의 DataFrame을 반환한다.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import mvit_v2_s

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]
S1_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
S1_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
S3_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
S3_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None]
cv2.setNumThreads(1)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("이 제출물은 CUDA GPU 평가환경을 필요로 합니다.")
    return torch.device("cuda")


def _video_paths(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)

# BASELINE_INFERENCE_PART
# ---------------------------------------------------------------------------
# Stage 1: MViTv2-S 기반 재녹화 분류기
# ---------------------------------------------------------------------------
def _clip_ids(path: Path, n: int, slot: int, slots: int):
    cap = cv2.VideoCapture(str(path))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    center = (slot + 0.5) * total / slots
    start = max(0, min(total - n, round(center - n / 2)))
    return np.linspace(start, min(total - 1, start + n - 1), n).round().astype(int)


def _decode_stage1_clip(path: Path, size: int, frame_ids):
    cap = cv2.VideoCapture(str(path))
    out = []
    wanted = [int(x) for x in frame_ids]
    cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    pos = wanted[0]
    for idx in wanted:
        ok = False
        bgr = None
        while pos <= idx:
            ok, bgr = cap.read()
            pos += 1
            if not ok:
                break
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = size / min(h, w)
        nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        y, x = (nh - size) // 2, (nw - size) // 2
        out.append(rgb[y : y + size, x : x + size])
    cap.release()
    if not out:
        raise ValueError(f"cannot decode video: {path.name}")
    while len(out) < len(wanted):
        out.append(out[-1])
    x = torch.from_numpy(np.stack(out)).permute(3, 0, 1, 2).float() / 255.0
    return (x - S1_MEAN) / S1_STD


class _Stage1Clips(Dataset):
    def __init__(self, videos, slots, size, frames):
        self.videos, self.slots, self.size, self.frames = videos, slots, size, frames

    def __len__(self):
        return len(self.videos) * self.slots

    def __getitem__(self, index):
        video_index, slot = index // self.slots, index % self.slots
        path = self.videos[video_index]
        try:
            x = _decode_stage1_clip(path, self.size, _clip_ids(path, self.frames, slot, self.slots))
            valid = 1
        except Exception:
            x = torch.zeros(3, self.frames, self.size, self.size)
            valid = 0
        return x, video_index, valid


def predict_stage1(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    size, frames = int(checkpoint["size"]), int(checkpoint["frames"])
    model = mvit_v2_s(weights=None)
    model.head[1] = nn.Linear(model.head[1].in_features, 2)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    root = Path(data_dir) / "videos"
    videos = _video_paths(root)
    slots = PORTFOLIO_CONFIG.get("stage1_slots", 3)
    dataset = _Stage1Clips(videos, slots, size, frames)
    loader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)
    scores = [[] for _ in videos]
    with torch.inference_mode():
        for clips, video_indices, valid in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prob = torch.softmax(model(clips.to(device, non_blocking=True)), 1)[:, 1]
            for idx, value, ok in zip(video_indices.tolist(), prob.float().cpu().tolist(), valid.tolist()):
                if ok:
                    scores[idx].append(float(value))

    PF_S1_TRACE.clear()
    PF_S1_TRACE.update({p.stem: v for p, v in zip(videos, scores)})
    rows = []
    for path, values in zip(videos, scores):
        probability = float(np.mean(values)) if values else 1.0
        rows.append({"ID": path.stem, "answer": "RERECORDED" if probability >= 0.5 else "ORIGINAL"})
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "answer"])

# BASELINE_INFERENCE_PART
# ---------------------------------------------------------------------------
# Stage 2: ResNet18 + BiGRU 기반 네 과업 모델
# ---------------------------------------------------------------------------
class _Stage2Frames(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class _Stage2Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = nn.GRU(512, 192, 2, batch_first=True, bidirectional=True, dropout=0.15)
        self.tc = nn.Linear(384, 1)
        self.te = nn.Linear(384, 1)
        self.scene = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.2), nn.Linear(192, 4))

    def forward(self, x):
        h, _ = self.r(x)
        collision_logits = self.tc(h).squeeze(-1)
        entry_logits = self.te(h).squeeze(-1)
        collision_index = collision_logits.argmax(1)
        entry_index = entry_logits.argmax(1)
        batch = torch.arange(len(h), device=h.device)
        scene_input = torch.cat([h[batch, collision_index], h[batch, entry_index]], 1)
        return collision_index, entry_index, self.scene(scene_input)


def _frame_number(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def predict_stage2(data_dir, model_dir):
    device = _device()
    model_dir = Path(model_dir)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    backbone = resnet18(weights=None)
    backbone.load_state_dict(torch.load(model_dir / "resnet18-f37072fd.pth", map_location="cpu", weights_only=True))
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    temporal = _Stage2Temporal()
    temporal.load_state_dict(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)["model"])
    temporal.to(device).eval()

    image_root = Path(data_dir) / "images"
    folders = sorted(p for p in image_root.iterdir() if p.is_dir())
    rows = []
    with torch.inference_mode():
        for folder in folders:
            # 평가 정의상 모든 원본 프레임을 사용한다(stride=1).
            paths = sorted(
                (p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                key=_frame_number,
            )
            if not paths:
                continue
            loader = DataLoader(_Stage2Frames(paths, transform), batch_size=256, num_workers=6, pin_memory=True)
            features = []
            for images in loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features.append(backbone(images.to(device, non_blocking=True)).float().cpu())
            sequence = torch.cat(features)[None].to(device)
            collision_idx, entry_idx, scene = temporal(sequence)
            frame_numbers = [_frame_number(path) for path in paths]
            rows.append(
                {
                    "ID": folder.name,
                    "collision_frame": frame_numbers[int(collision_idx)],
                    "entry_frame": frame_numbers[int(entry_idx)],
                    "evasion_space": int(scene[:, :2].argmax(1)),
                    "entry_side": "RIGHT" if int(scene[:, 2:].argmax(1)) else "LEFT",
                }
            )
    del backbone, temporal
    torch.cuda.empty_cache()
    return pd.DataFrame(
        rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]
    )

# BASELINE_INFERENCE_PART
# ---------------------------------------------------------------------------
# Stage 3: MViTv2-S 기반 가감속/조향 다중헤드 모델
# ---------------------------------------------------------------------------
class _Stage3MViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mvit_v2_s(weights=None)
        dimension = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        self.accel = nn.Linear(dimension, 4)
        self.steer = nn.Linear(dimension, 3)

    def forward(self, x):
        features = self.backbone(x)
        return self.accel(features), self.steer(features)


def _stage3_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width, height = image.size
        scale = 256 / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)))
        width, height = image.size
        x, y = (width - 224) // 2, (height - 224) // 2
        image = image.crop((x, y, x + 224, y + 224))
        frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(torch.uint8))
    capture.release()
    if not frames:
        raise ValueError(f"cannot decode video: {path.name}")
    return torch.stack(frames)


def predict_stage3(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    model = _Stage3MViT()
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    videos = _video_paths(Path(data_dir) / "videos")
    rows = []
    with torch.inference_mode():
        for path in videos:
            frames = _stage3_frames(path)
            count = len(frames)
            centers = np.arange(count)
            accel_predictions, steer_predictions = [], []
            for start in range(0, count, 8):
                center = centers[start : start + 8]
                indices = np.clip(center[:, None] - 8 + np.arange(16)[None, :], 0, count - 1)
                clips = frames[torch.from_numpy(indices)].permute(0, 2, 1, 3, 4).float() / 255.0
                clips = (clips - S3_MEAN[None, :, None, :, :]) / S3_STD[None, :, None, :, :]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    accel_logits, steer_logits = model(clips.to(device, non_blocking=True))
                accel_predictions.extend(accel_logits.argmax(1).cpu().tolist())
                steer_predictions.extend(steer_logits.argmax(1).cpu().tolist())
            for sample_index, (accel, steer) in enumerate(zip(accel_predictions, steer_predictions)):
                rows.append(
                    {
                        "ID": path.stem,
                        "sample_index": sample_index,
                        "accel_label": ACCEL[accel],
                        "steer_label": STEER[steer],
                    }
                )
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])

PORTFOLIO_CONFIG = {'stage1_slots': 5, 'stage2_family': 'B', 'stage3': {'family': 'A', 'thresholds': [1.165121976286173, 2.930227120717365, 2.346569240093232], 'quantiles': [0.05, 0.7, 0.8]}}
PF_S1_TRACE = {}
_PF_BASE_STAGE2 = predict_stage2
_PF_BASE_STAGE3 = predict_stage3
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

predict_stage2 = pf_predict_stage2
predict_stage3 = pf_predict_stage3


# FOCUS12_APPEND_ONLY
_F12_PARENT_STAGE2 = predict_stage2
def _F12_BASE3(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    size, frames = int(checkpoint["size"]), int(checkpoint["frames"])
    model = mvit_v2_s(weights=None)
    model.head[1] = nn.Linear(model.head[1].in_features, 2)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    root = Path(data_dir) / "videos"
    videos = _video_paths(root)
    slots = 3
    dataset = _Stage1Clips(videos, slots, size, frames)
    loader = DataLoader(dataset, batch_size=4, num_workers=2, pin_memory=True)
    scores = [[] for _ in videos]
    with torch.inference_mode():
        for clips, video_indices, valid in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prob = torch.softmax(model(clips.to(device, non_blocking=True)), 1)[:, 1]
            for idx, value, ok in zip(video_indices.tolist(), prob.float().cpu().tolist(), valid.tolist()):
                if ok:
                    scores[idx].append(float(value))

    rows = []
    for path, values in zip(videos, scores):
        probability = float(np.mean(values)) if values else 1.0
        rows.append({"ID": path.stem, "answer": "RERECORDED" if probability >= 0.5 else "ORIGINAL"})
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "answer"])
FOCUS12_CONFIG = {'stage1': {'family': 'baseline3'}, 'stage2': 'parent_B'}
"""S1/S2 candidates appended to an immutable, actually submitted parent source.

The parent Stage3 code/config remains untouched. Training helpers are not imported here.
"""

def f12_frames(path, quality=None):
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        capture.release()
        raise ValueError(f"Cannot read frame count: {path}")
    wanted = sorted(set(np.linspace(0, count - 1, 8).round().astype(int).tolist()))
    images = []
    for index in wanted:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"Frame decode failed: {path}:{index}")
        if quality is not None:
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise ValueError("JPEG perturbation failed")
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return images


def f12_backbone(model_dir):
    model = resnet18(weights=None)
    weights = Path(model_dir).parent / 'stage2/resnet18-f37072fd.pth'
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.fc = nn.Identity()
    return model.to(_device()).eval()


def f12_features(path, model, quality=None):
    frames = f12_frames(path, quality)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    full = [(torch.from_numpy(np.asarray(im.resize((224, 224), Image.Resampling.BILINEAR)).copy()).permute(2, 0, 1).float() / 255 - mean) / std for im in frames]
    center = [transform(im) for im in frames]
    batch = torch.stack(full + center).to(_device())
    with torch.inference_mode():
        # Float32 in both calibration and export to reduce hardware/AMP drift.
        features = model(batch).float().cpu().numpy()
    a, b = features[:len(frames)].mean(0), features[len(frames):].mean(0)
    return {'A': a, 'B': np.concatenate([a, np.abs(a - b)])}


def f12_unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def f12_head_predict(x, head):
    return (f12_unit(x) - np.asarray(head['mean'])) @ np.asarray(head['coef']) + head['bias']


def f12_predict_stage1(data_dir, model_dir):
    config = FOCUS12_CONFIG['stage1']
    if config['family'] == 'baseline3':
        # A separate copy of the original function with slots=3, never mutate parent globals.
        return _F12_BASE3(data_dir, model_dir)
    model = f12_backbone(model_dir)
    rows = []
    for path in _video_paths(Path(data_dir) / 'videos'):
        feature = f12_features(path, model)[config['family']]
        margin = float(f12_head_predict(feature, config['head']))
        rows.append({'ID': path.stem, 'answer': 'RERECORDED' if margin >= 0 else 'ORIGINAL'})
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=['ID', 'answer'])


def f12_rank(values):
    # Average tied ranks. No arbitrary first-maximum tie introduced by clipping at z=8.
    x = np.asarray(values, float)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks / max(1, len(x) - 1)


def f12_event_features(folder, quality=None):
    paths = sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}), key=pf_frame_number)
    if not paths:
        raise ValueError(f'Empty frame folder: {folder}')
    rows, previous = [], None
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f'Unreadable frame: {path}')
        if quality is not None:
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise ValueError('JPEG perturbation failed')
            frame = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        frame = cv2.resize(frame, (160, 90))
        frame = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.float32)
        difference, residual, shift = 0., 0., 0.
        if previous is not None:
            difference = float(np.mean(np.abs(frame - previous)))
            # Remove uniform exposure jumps; preserve motion-related spatial changes.
            residual = float(np.mean(np.abs((frame - frame.mean()) - (previous - previous.mean()))))
            offset, response = cv2.phaseCorrelate(previous, frame)
            if response > .1 and np.isfinite(offset).all():
                shift = float(np.hypot(*offset))
        rows.append([difference, residual, shift])
        previous = frame
    return np.array([pf_frame_number(p) for p in paths]), np.asarray(rows)


def f12_event_index(features, family):
    if len(features) == 1:
        return 0
    r = f12_rank(features[:, 1])
    if family == 'C':
        scores = .75 * r + .25 * f12_rank(features[:, 2])
    else:
        # Temporal support penalizes isolated flashes; fixed coefficients, not a five-video weight search.
        supported = np.convolve(np.pad(r, (1, 1), mode='edge'), [.25, .5, .25], mode='valid')
        scores = .6 * r + .4 * supported
    scores[0] = -np.inf
    return int(np.argmax(scores))


def f12_parent_event_features(folder, quality=None):
    """Exact parent B input pathway, with optional validation-only JPEG perturbation."""
    paths = sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}),key=pf_frame_number)
    rows, previous, last_translation = [], None, None
    for path in paths:
        gray = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f'Cannot read {path}')
        if quality is not None:
            ok, blob = cv2.imencode('.jpg',gray,[cv2.IMWRITE_JPEG_QUALITY,quality])
            if not ok:
                raise ValueError('JPEG perturbation failed')
            gray = cv2.imdecode(blob,cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray,(320,180))
        difference, change, valid = 0.,0.,0.
        if previous is not None:
            difference = float(np.mean(cv2.absdiff(previous,gray)))
            matrix = pf_affine(previous,gray)
            if matrix is not None:
                translation = matrix[:,2]/np.hypot(320,180)
                if last_translation is not None:
                    change,valid = float(np.linalg.norm(translation-last_translation)),1.
                last_translation = translation
            else:
                last_translation = None
        rows.append([difference,change,valid]); previous=gray
    return np.array([pf_frame_number(p) for p in paths]),np.asarray(rows)


def f12_refine(numbers, features, anchor_frame):
    anchor = int(np.argmin(np.abs(numbers-anchor_frame)))
    lo, hi = max(1,anchor-2), min(len(numbers),anchor+3)
    if hi <= lo or np.max(features[lo:hi,1]) <= 1e-6:
        return anchor
    indices = np.arange(lo,hi)
    score = f12_rank(features[lo:hi,1]) - .01*np.abs(indices-anchor)
    return int(indices[np.argmax(score)])


def f12_predict_stage2(data_dir, model_dir):
    output = _F12_PARENT_STAGE2(data_dir, model_dir)
    family = FOCUS12_CONFIG['stage2']
    if family != 'parent_B':
        for index, row in output.iterrows():
            numbers, features = f12_event_features(Path(data_dir) / 'images' / row['ID'])
            selected = f12_refine(numbers,features,int(row['collision_frame'])) if family=='E' else f12_event_index(features,family)
            output.at[index, 'collision_frame'] = int(numbers[selected])
    return output

predict_stage1 = f12_predict_stage1
predict_stage2 = f12_predict_stage2
