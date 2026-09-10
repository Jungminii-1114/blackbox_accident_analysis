"""Focus S1/S2 experiment, preserving the submitted Stage3 byte-for-byte prefix."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

import cv2
import numpy as np
import pandas as pd

import portfolio_lab as lab
from local_validate import macro_f1, score_stage1

ROOT = Path(__file__).resolve().parents[1]
HISTORY = json.loads((ROOT / 'tools/focus12_history.json').read_text())
PARENT = ROOT / 'parent'
CLASSES = ['ORIGINAL', 'RERECORDED']


def code_hashes():
    names = ['tools/focus12_lab.py', 'tools/focus12_runtime.py', 'tools/focus12_history.json']
    return {**lab.source_fingerprint(), **{n: lab.sha256(ROOT / n) for n in names}}


def verify_models():
    manifest=lab.read(ROOT/'tools/portfolio_base_manifest.json')['model_sha256']
    for name,expected in manifest.items():
        if lab.sha256(ROOT/name)!=expected or lab.sha256(PARENT/name)!=expected:
            raise RuntimeError('Baseline/parent checkpoint hash changed: '+name)


def source(config):
    parent = (PARENT / 'inference.py').read_text(encoding='utf-8')
    if lab.sha256(PARENT / 'inference.py') != HISTORY['parent_source_sha256']:
        raise ValueError('Scored parent source hash mismatch')
    original = (ROOT / 'baseline_inference.py').read_text()
    tree = ast.parse(original)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'predict_stage1')
    baseline3 = ast.get_source_segment(original, node).replace('def predict_stage1(', 'def _F12_BASE3(', 1)
    baseline3 = baseline3.replace('num_workers=4', 'num_workers=2')
    runtime = (ROOT / 'tools/focus12_runtime.py').read_text()
    return (parent + '\n\n# FOCUS12_APPEND_ONLY\n_F12_PARENT_STAGE2 = predict_stage2\n' + baseline3 +
            '\nFOCUS12_CONFIG = ' + repr(config) + '\n' + runtime +
            '\npredict_stage1 = f12_predict_stage1\npredict_stage2 = f12_predict_stage2\n')


def module(exp, config=None):
    config = config or {'stage1': {'family': 'baseline3'}, 'stage2': 'parent_B'}
    path = exp / 'config/working_inference.py'
    path.write_text(source(config), encoding='utf-8')
    return lab.load_module(path)


def prepare(exp):
    assert lab.sha256(PARENT / 'inference.py') == HISTORY['parent_source_sha256']
    verify_models()
    lab.prepare(exp)
    path = exp / 'reports/focus_provenance.json'
    identity = {'code': code_hashes(), 'parent_source': HISTORY['parent_source_sha256'],
                'parent_zip': HISTORY['parent_zip_sha256'], 'official_history': HISTORY}
    if path.exists() and lab.read(path) != identity:
        raise RuntimeError('Focus code/parent changed. Use a NEW experiment ID.')
    lab.save(path, identity)


def parent(exp):
    inputs = exp / 'fixture/input'
    out = exp / 'predictions/parent'
    lab.smoke_check(exp, PARENT / 'inference.py', PARENT / 'model', out)
    m = module(exp)
    m.predict_stage1(inputs / 'stage1', ROOT / 'model/stage1').to_csv(exp / 'predictions/s1_baseline3.csv', index=False)
    lab.save(exp / 'reports/parent_local.json', lab.metrics(exp, out))
    comparison = {'historical_outputs_checked': False}
    historical = ROOT / 'parent_review/predictions/selected_zip/stage3.csv'
    if historical.exists():
        a, b = pd.read_csv(historical), pd.read_csv(out / 'stage3.csv')
        merged = a.merge(b, on=['ID', 'sample_index'], suffixes=('_previous', '_current'), how='outer', indicator=True)
        same = ((merged._merge == 'both') & (merged.accel_label_previous == merged.accel_label_current) &
                (merged.steer_label_previous == merged.steer_label_current))
        comparison = {'historical_outputs_checked': True, 'matching_fraction': float(same.mean()),
                      'note': 'Differences may include fixture encoding/environment; candidate must still match current parent exactly.'}
    lab.save(exp / 'reports/parent_reproduction.json', comparison)


def ridge_fit(x, y, regularization):
    x = np.asarray(x, np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    mean, bias = x.mean(0), float(y.mean())
    centered = x - mean
    alpha = np.linalg.solve(centered @ centered.T + regularization * np.eye(len(x)), y - bias)
    return {'mean': mean.tolist(), 'coef': (centered.T @ alpha).tolist(), 'bias': bias}


def ridge_predict(x, head):
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    return (x - np.asarray(head['mean'])) @ np.asarray(head['coef']) + head['bias']


def as_labels(margins):
    return np.where(margins >= 0, 'RERECORDED', 'ORIGINAL')


def choose_head(x, y, groups, families=('A', 'B')):
    truth = np.where(y > 0, 'RERECORDED', 'ORIGINAL')
    best = None
    for family in families:
        for regularization in (.1, 1., 10.):
            margins = np.empty(len(y))
            for group in sorted(set(groups)):
                train, test = groups != group, groups == group
                head = ridge_fit(x[family][train], y[train], regularization)
                margins[test] = ridge_predict(x[family][test], head)
            value = macro_f1(truth.tolist(), as_labels(margins).tolist(), CLASSES)
            if best is None or value > best[0] + 1e-12:
                best = value, family, regularization
    return best


def s1_nested(x, perturbed, y, groups, families=('A', 'B')):
    margins, noise = np.zeros(len(y)), np.zeros(len(y))
    folds = []
    for group in sorted(set(groups)):
        train, test = groups != group, groups == group
        _, family, penalty = choose_head({k:v[train] for k,v in x.items()}, y[train], groups[train], families)
        head = ridge_fit(x[family][train], y[train], penalty)
        margins[test] = ridge_predict(x[family][test], head)
        noise[test] = ridge_predict(perturbed[family][test], head)
        folds.append({'held_out_source': group, 'train_sources': sorted(set(groups[train])),
                      'family': family, 'regularization': penalty})
    truth, pred = np.where(y > 0, 'RERECORDED', 'ORIGINAL'), as_labels(margins)
    return {'macro_f1': macro_f1(truth.tolist(), pred.tolist(), CLASSES),
            'jpeg90_agreement': float(np.mean(pred == as_labels(noise))), 'folds': folds,
            'class_prediction_counts': {c:int(np.sum(pred == c)) for c in CLASSES},
            'margins': margins.tolist(), 'perturbed_margins': noise.tolist()}


def stage1(exp):
    import torch
    m = module(exp)
    labels = pd.read_csv(ROOT / 'fixtures/baseline_data/stage1/labels.csv')
    groups = labels.path.map(lambda p: Path(p).stem).to_numpy()
    if len(set(groups)) < 5 or any(len(labels[groups == g]) != 2 for g in set(groups)):
        raise ValueError('Expected five complete original/rerecorded source pairs')
    y = np.where(labels.label.to_numpy() == 'RERECORDED', 1., -1.)
    features, noise = {'A':[], 'B':[]}, {'A':[], 'B':[]}
    backbone = m.f12_backbone(ROOT / 'model/stage1')
    for row in labels.itertuples():
        print('S1 frozen features + JPEG90:', row.ID, flush=True)
        path = exp / 'fixture/input/stage1/videos' / (row.ID + Path(row.path).suffix)
        clean = m.f12_features(path, backbone)
        altered = m.f12_features(path, backbone, quality=90)
        for name in features:
            features[name].append(clean[name]); noise[name].append(altered[name])
    del backbone
    torch.cuda.empty_cache()
    features = {k:np.stack(v) for k,v in features.items()}
    noise = {k:np.stack(v) for k,v in noise.items()}
    np.savez_compressed(exp / 'features/s1.npz', **features, A_jpeg90=noise['A'], B_jpeg90=noise['B'], groups=groups.astype(str))
    procedure = s1_nested(features, noise, y, groups)
    families = {k:s1_nested(features, noise, y, groups, (k,)) for k in features}
    _, family, penalty = choose_head(features, y, groups)
    fitted = {'family':family, 'head':ridge_fit(features[family], y, penalty), 'regularization':penalty}
    baseline = score_stage1(exp / 'fixture/labels/stage1.csv', exp / 'predictions/s1_baseline3.csv')['macro_f1']
    fixed = families[family]
    eligible = (procedure['macro_f1'] >= max(.6, baseline + .1) and fixed['macro_f1'] >= max(.6, baseline + .1)
                and min(procedure['jpeg90_agreement'], fixed['jpeg90_agreement']) >= .9
                and all(n > 0 for n in procedure['class_prediction_counts'].values()))
    config = fitted if eligible else {'family':'baseline3'}
    lab.save(exp / 'reports/stage1.json', {'selected':config, 'candidate':fitted, 'selection_procedure_oof':procedure,
        'fixed_family_oof':families, 'selected_family_oof':fixed, 'baseline3_local':baseline,
        'status':'EXPERIMENTAL' if eligible else 'RECOVERY_BASELINE3',
        'limitations':['five pairs only; target-like but tiny validation', 'reuse of baseline ResNet18; full pretraining provenance not re-established'],
        'gate':'procedure AND final-family F1 >= max(.6, baseline+.1); codec agreement >=.9; both classes predicted'})
    pred = labels[['ID']].copy()
    pred['answer'] = as_labels(ridge_predict(features[family], fitted['head']))
    pred.to_csv(exp / 'predictions/s1_candidate_fit.csv', index=False)
    pred['answer'] = as_labels(np.asarray(procedure['margins']))
    pred.to_csv(exp / 'predictions/s1_nested_oof.csv', index=False)


def event_metrics(predictions, labels, mapping):
    errors, frames = [], []
    for row in labels.itertuples():
        prediction = int(predictions[row.ID])
        errors.append(abs(mapping[(row.ID,prediction)] - mapping[(row.ID,int(row.collision_frame))]))
        frames.append(abs(prediction - int(row.collision_frame)))
    return {'accuracy_at_0_3s':float(np.mean(np.array(errors) <= .3 + 1e-9)),
            'mae_seconds':float(np.mean(errors)), 'mae_frames':float(np.mean(frames)), 'errors_seconds':errors}


def stage2(exp):
    m = module(exp)
    labels = pd.read_csv(exp / 'fixture/labels/stage2.csv')
    times = pd.read_csv(exp / 'fixture/labels/stage2_frame_time.csv')
    mapping = {(r.ID,int(r.frame)):r.time_seconds for r in times.itertuples()}
    parent_df = pd.read_csv(exp / 'predictions/parent/stage2.csv')
    parent_pred = parent_df.set_index('ID').collision_frame.to_dict()
    parent_metric = event_metrics(parent_pred, labels, mapping)
    variants = {k:{q:{} for q in ('clean', '95', '90')} for k in ('E','C','D')}
    for row in labels.itertuples():
        print('S2 robust motion / codec audit:',row.ID,flush=True)
        folder = exp / 'fixture/input/stage2/images' / row.ID
        for q in ('clean','95','90'):
            numbers, features = m.f12_event_features(folder, None if q == 'clean' else int(q))
            parent_numbers,parent_features=m.f12_parent_event_features(folder,None if q=='clean' else int(q))
            anchor=int(parent_numbers[np.argmax(m.pf_event_scores(parent_features,'B'))])
            if q=='clean' and anchor!=int(parent_pred[row.ID]):
                raise RuntimeError('Parent B collision pathway mismatch')
            for family in variants:
                index=m.f12_refine(numbers,features,anchor) if family=='E' else m.f12_event_index(features,family)
                variants[family][q][row.ID] = int(numbers[index])
    reports = {}
    for family, versions in variants.items():
        summary = {q:event_metrics(pred, labels, mapping) for q,pred in versions.items()}
        deviations = [abs(mapping[(i,p)] - mapping[(i,versions['clean'][i])]) for q in ('95','90') for i,p in versions[q].items()]
        robust = float(np.mean(np.array(deviations) <= .1 + 1e-9))
        clean = summary['clean']
        eligible = (clean['accuracy_at_0_3s'] >= parent_metric['accuracy_at_0_3s']
                    and clean['mae_seconds'] <= parent_metric['mae_seconds'] - .02
                    and min(summary[q]['accuracy_at_0_3s'] for q in summary) >= parent_metric['accuracy_at_0_3s']
                    and robust >= .9)
        reports[family] = {'metrics':summary, 'jpeg_stability_at_0_1s':robust, 'eligible':eligible, 'predictions':versions}
        output = parent_df.copy()
        output.collision_frame = output.ID.map(versions['clean'])
        output.to_csv(exp / f'predictions/s2_{family}.csv', index=False)
    selected = next((k for k in ('E','C','D') if reports[k]['eligible']), 'parent_B')
    lab.save(exp / 'reports/stage2.json', {'selected':selected, 'parent':parent_metric, 'candidates':reports,
        'full_stage_local_score':None, 'status':'EXPERIMENTAL' if selected != 'parent_B' else 'KEEP_OFFICIAL_CHAMPION',
        'gate':'no tolerance-accuracy loss; MAE improves >=.02s; JPEG95/90 accuracy no worse; stability >=.9',
        'limitations':'collision-only, five public videos; unknown entry/direction/evasion remain parent outputs'})


def package(exp):
    config = {'stage1':lab.read(exp / 'reports/stage1.json')['selected'], 'stage2':lab.read(exp / 'reports/stage2.json')['selected']}
    lab.save(exp / 'config/selected.json', config)
    workspace = exp / 'submission/workspace'
    workspace.mkdir(parents=True,exist_ok=True)
    (workspace / 'inference.py').write_text(source(config),encoding='utf-8')
    compile((workspace / 'inference.py').read_text(), 'inference.py', 'exec')
    shutil.copy2(PARENT / 'requirements.txt',workspace / 'requirements.txt')
    for name in lab.MODEL_FILES:
        target = workspace / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PARENT / name,target)
    start = time.monotonic()
    lab.smoke_check(exp,workspace / 'inference.py',workspace / 'model',exp / 'predictions/selected_workspace')
    path = exp / 'submission/submit.zip'
    with zipfile.ZipFile(path.with_suffix('.part.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
        for name in ('inference.py','requirements.txt',*lab.MODEL_FILES):
            archive.write(workspace / name,name)
    path.with_suffix('.part.zip').replace(path)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert path.stat().st_size <= 10_000_000_000 and sum(i.file_size for i in archive.infolist()) <= 32_000_000_000
    lab.run([sys.executable,ROOT / 'tools/validate_submit_package.py','--zip',path,'--import-check'])
    with tempfile.TemporaryDirectory(prefix='focus12_zip_') as directory:
        with zipfile.ZipFile(path) as archive:
            archive.extractall(directory)
        root = Path(directory)
        lab.smoke_check(exp,root / 'inference.py',root / 'model',exp / 'predictions/selected_zip')
    for stage in (1,2,3):
        keys = ['ID'] + (['sample_index'] if stage == 3 else [])
        first = pd.read_csv(exp / f'predictions/selected_workspace/stage{stage}.csv').sort_values(keys).reset_index(drop=True)
        second = pd.read_csv(exp / f'predictions/selected_zip/stage{stage}.csv').sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(first,second)
    parent3 = pd.read_csv(exp / 'predictions/parent/stage3.csv')
    exported3 = pd.read_csv(exp / 'predictions/selected_zip/stage3.csv')
    pd.testing.assert_frame_equal(parent3,exported3)
    parent2 = pd.read_csv(exp / 'predictions/parent/stage2.csv')
    exported2 = pd.read_csv(exp / 'predictions/selected_zip/stage2.csv')
    pd.testing.assert_frame_equal(parent2.drop(columns='collision_frame'), exported2.drop(columns='collision_frame'))
    expected2 = parent2 if config['stage2'] == 'parent_B' else pd.read_csv(exp / f"predictions/s2_{config['stage2']}.csv")
    pd.testing.assert_frame_equal(expected2,exported2)
    expected1 = pd.read_csv(exp / ('predictions/s1_baseline3.csv' if config['stage1']['family']=='baseline3' else 'predictions/s1_candidate_fit.csv'))
    exported1 = pd.read_csv(exp / 'predictions/selected_zip/stage1.csv')
    pd.testing.assert_frame_equal(expected1.sort_values('ID').reset_index(drop=True),exported1.sort_values('ID').reset_index(drop=True))
    files = ['reports/stage1.json','reports/stage2.json','reports/focus_provenance.json','config/selected.json']
    files += [f'predictions/selected_zip/stage{i}.csv' for i in (1,2,3)]
    lab.save(exp / 'reports/package_check.json', {'passed':True,'zip_sha256':lab.sha256(path),
        'bound_evidence':{name:lab.sha256(exp / name) for name in files}, 'parent_stage3_exact_match':True,
        'stage3_source_prefix_sha256':HISTORY['parent_source_sha256'],'total_local_smoke_seconds':time.monotonic()-start,
        'scope':'same Colab runtime, two extracted/process executions; NOT full hidden L40S test'})


def report(exp):
    failures = []
    check = lab.read(exp / 'reports/package_check.json') if (exp / 'reports/package_check.json').exists() else {'passed':False}
    if not check.get('passed'):
        failures.append(check.get('reason','Package check not complete'))
    else:
        path = exp / 'submission/submit.zip'
        if not path.exists() or lab.sha256(path) != check['zip_sha256']:
            failures.append('ZIP hash mismatch')
        for name,value in check['bound_evidence'].items():
            if not (exp / name).exists() or lab.sha256(exp / name) != value:
                failures.append('Evidence hash mismatch: '+name)
    try:
        provenance = lab.read(exp / 'reports/focus_provenance.json')
        if provenance['code'] != code_hashes() or lab.sha256(PARENT / 'inference.py') != HISTORY['parent_source_sha256']:
            failures.append('Code/parent changed after experiment')
        lab.verify_fixture(exp)
        verify_models()
    except (OSError,KeyError,RuntimeError) as error:
        failures.append(str(error))
    stages = {}
    for stage in (1,2):
        path = exp / f'reports/stage{stage}.json'
        if path.exists():
            stages[f'stage{stage}'] = lab.read(path)
        else:
            failures.append(f'Stage {stage} incomplete')
    if 'stage1' in stages:
        a = stages['stage1']
        stages['stage1']['local_BSS_delta'] = a['selected_family_oof']['macro_f1'] - a['baseline3_local'] if a['selected']['family']!='baseline3' else 0.
    if 'stage2' in stages:
        stages['stage2']['local_BSS_delta'] = None
    if 'stage1' in stages:
        for field in ('selected','candidate'):
            stages['stage1'][field] = {k:v for k,v in stages['stage1'][field].items() if k!='head'}
    recovery = 'stage1' in stages and stages['stage1']['selected']['family']=='baseline3'
    result = {'verdict':'NO_GO' if failures else 'GO', 'purpose':'best_known_recombination' if recovery and stages.get('stage2',{}).get('selected')=='parent_B' else 'exploration',
        'stages':stages, 'stage3':'FROZEN: byte-identical parent source prefix/config and identical current-fixture predictions',
        'official_parent_rounded':HISTORY['official_parent_rounded'], 'official_candidate':None, 'official_BSS_delta':None,
        'original_to_parent_official_delta_approx':{k:HISTORY['official_parent_rounded'][k]-HISTORY['official_original'][k] for k in HISTORY['official_original']},
        'recovery_s1_reference_delta':HISTORY['official_original']['stage1']-HISTORY['official_parent_rounded']['stage1'] if recovery else None,
        'improvement_probability':None, 'failures':failures, 'package_check':check,
        'warnings':['GO does not guarantee an official increase', 'Only five source pairs / five accidents; public reuse',
                    'S2 full local score is N/A; do not equate tolerance accuracy and timing precision',
                    'Displayed submission count was 3; do not assume another submission is available today']}
    lab.save(exp / 'reports/submission_gate.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    print('VERDICT:',result['verdict'],flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','parent','stage1','stage2','package','report'])
    p.add_argument('--experiment-dir',required=True,type=Path)
    args=p.parse_args(); exp=args.experiment_dir.resolve()
    for name in ('config','features','predictions','reports','logs','submission'):
        (exp/name).mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    if args.command!='report':
        lab.save(exp/'reports/package_check.json',{'passed':False,'reason':f'{args.command} in progress; previous marker invalidated'})
    try:
        if args.command not in ('prepare','report'):
            if lab.read(exp/'reports/focus_provenance.json')['code'] != code_hashes():
                raise RuntimeError('Code changed. New experiment ID required.')
            lab.verify_fixture(exp)
            verify_models()
        globals()[args.command](exp)
    except Exception as error:
        if args.command!='report':
            lab.save(exp/'reports/package_check.json',{'passed':False,'reason':str(error)})
        raise


if __name__=='__main__':
    main()
