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
