"""Generate the Colab entry notebook and small code-only patch reproducibly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def notebook():
    cells = [cell("markdown", """
        # portfolio_v1 — 세 Stage 실험 → 검증 → submit.zip

        **이 노트북 하나만 GPU Colab에서 위에서 아래로 실행하세요.** 이전 01/02/03 노트북은 실행하지 않습니다.
        Drive 루트에 기존 `dacon236753_colab_bundle.zip`은 그대로 두고 새 `dacon236753_portfolio_patch.zip`만 올립니다.
        외부 데이터 다운로드나 80-epoch 학습은 필요 없습니다. S1은 clip 집계, S2는 충돌 신호,
        S3는 flow + nested video CV 실험입니다. 공식 점수 상승은 보장되지 않습니다.

        `GO`는 **검증된 탐색 제출물**이라는 뜻입니다. `NO_GO`이면 제출하지 마세요.
        공식 candidate 점수·BSS는 제출 전 N/A입니다. 모든 결과는 Drive의 새 `experiments/` 하위에 보존합니다.
        중간 셀 오류 시 이후 패키징을 진행하지 말고 마지막 판정 셀을 실행해 진단을 저장하세요.
    """), cell("code", '''
        # [1] Drive mount + 실제 mount 검사. 단순히 /content/drive 폴더가 있다는 것만 믿지 않습니다.
        from google.colab import drive
        drive.mount('/content/drive')
        from pathlib import Path
        import os, re, json, shutil, hashlib, zipfile, subprocess, sys, time, datetime
        mounts = Path('/proc/mounts').read_text()
        assert any(line.split()[1] == '/content/drive' and 'fuse' in line.split()[2] for line in mounts.splitlines()), 'Drive FUSE mount가 아닙니다. 중단하세요.'
        DRIVE_ROOT = Path('/content/drive/MyDrive/블랙박스 영상 기반 사고 분석')
        assert DRIVE_ROOT.is_dir(), f'Drive 경로 없음: {DRIVE_ROOT}'
        BASE_ZIP = DRIVE_ROOT / 'dacon236753_colab_bundle.zip'
        PATCH_ZIP = DRIVE_ROOT / 'dacon236753_portfolio_patch.zip'
        assert BASE_ZIP.is_file() and PATCH_ZIP.is_file(), '기존 base ZIP과 새 portfolio patch ZIP을 Drive 루트에 두세요.'
        EXPERIMENT_ID = 'portfolio_v1_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        # 재개하려면 위 값을 이전 experiment ID 문자열로 바꾸세요. 코드가 바뀌면 반드시 새 ID를 쓰세요.
        assert re.fullmatch(r'[A-Za-z0-9_-]+', EXPERIMENT_ID)
        DRIVE_EXP = DRIVE_ROOT / 'experiments' / EXPERIMENT_ID
        DRIVE_EXP.mkdir(parents=True, exist_ok=True)
        print('Drive 저장 위치:', DRIVE_EXP)
    '''), cell("code", '''
        # [2] 로컬 디스크에 원본 base + 소형 patch 설치. 기존 /content/dacon236753는 건드리지 않습니다.
        WORK = Path('/content/dacon236753_portfolio')
        WORK.mkdir(parents=True, exist_ok=True)
        def digest(path):
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            return h.hexdigest()
        def extract_safe(path, target, only_assets=False):
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    p = Path(item.filename)
                    assert not p.is_absolute() and '..' not in p.parts, f'Unsafe ZIP entry: {p}'
                    if not only_assets or item.filename.startswith(('model/', 'fixtures/baseline_data/')):
                        archive.extract(item, target)
        # 새 런타임에서는 약 480MB base를 Drive에서 한 번 읽습니다. 사용자가 재업로드하는 것은 아닙니다.
        if not (WORK / 'model/stage3/best.pt').exists():
            extract_safe(BASE_ZIP, WORK, only_assets=True)
        extract_safe(PATCH_ZIP, WORK)
        manifest = json.loads((WORK / 'tools/portfolio_base_manifest.json').read_text())
        for name, expected in manifest['model_sha256'].items():
            assert digest(WORK / name) == expected, f'원본 모델 hash 불일치: {name}. 이전 후보를 원본으로 사용하지 마세요.'
        patch_info = json.loads((WORK / 'PORTFOLIO_PATCH.json').read_text())
        for name, expected in patch_info['sha256'].items():
            assert digest(WORK / name) == expected, f'Patch hash 불일치: {name}'
        EXP = WORK / 'runs' / EXPERIMENT_ID
        EXP.mkdir(parents=True, exist_ok=True)
        # 런타임 재시작 후 작은 보고서/캐시만 복원합니다. 영상 fixture는 다시 만들 수 있습니다.
        for folder in ('reports', 'features', 'predictions', 'config', 'logs'):
            saved = DRIVE_EXP / folder
            if saved.exists() and not (EXP / folder).exists():
                shutil.copytree(saved, EXP / folder)
        os.chdir(WORK)
        print('Runtime work:', WORK, '\\nExperiment:', EXP)
    '''), cell("code", '''
        # [3] 설치 대신 기존 환경 우선 검사. 자동 source build / 대형 CUDA 재설치 없음.
        probe = r"""
        import json, torch, torchvision, cv2, numpy, pandas
        from torchvision.ops import nms
        from torchvision.models.video import mvit_v2_s
        assert torch.cuda.is_available(), 'GPU 런타임이 아닙니다'
        nms(torch.tensor([[0.,0.,1.,1.]], device='cuda'), torch.ones(1, device='cuda'), 0.5)
        print(json.dumps({'torch':torch.__version__, 'torchvision':torchvision.__version__,
                          'opencv':cv2.__version__, 'numpy':numpy.__version__, 'pandas':pandas.__version__,
                          'gpu':torch.cuda.get_device_name(0)}, indent=2))
        """
        import textwrap
        probe = textwrap.dedent(probe)
        tested = subprocess.run([sys.executable, '-c', probe], text=True, capture_output=True)
        print(tested.stdout, tested.stderr)
        assert tested.returncode == 0, '환경 검사 실패. 런타임 연결 해제 및 삭제 → 새 GPU 런타임에서 다시 실행하세요. 오류 로그를 공유해 주세요.'
        assert shutil.which('ffmpeg'), 'ffmpeg 없음'
        env = json.loads(tested.stdout)
        env['python'] = sys.version
        env['patch_sha256'] = digest(PATCH_ZIP)
        (EXP / 'reports').mkdir(exist_ok=True)
        environment_path = EXP / 'reports/environment.json'
        if environment_path.exists():
            previous_env = json.loads(environment_path.read_text())
            for key in ('torch', 'torchvision', 'opencv', 'numpy', 'pandas', 'python', 'patch_sha256'):
                assert previous_env.get(key) == env[key], f'실험 환경이 바뀌었습니다({key}). 새 EXPERIMENT_ID로 시작하세요.'
        (EXP / 'reports/environment.json').write_text(json.dumps(env, indent=2))
        print('PASS: GPU/import/NMS/ffmpeg')
    '''), cell("code", '''
        # [4] 공통 실행/저장 함수. 각 단계가 끝나면 작은 결과를 Drive에 복사합니다.
        # 프로세스는 실제 진행 로그를 출력하며, 장시간 계산 시에도 경과시간을 보여줍니다.
        import threading
        def sync_results(include_submit=False):
            for folder in ('reports', 'features', 'predictions', 'config', 'logs'):
                source = EXP / folder
                if source.exists():
                    shutil.copytree(source, DRIVE_EXP / folder, dirs_exist_ok=True)
            if include_submit:
                source = EXP / 'submission/submit.zip'
                target = DRIVE_EXP / 'submission/submit.zip'
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.part.zip')
                shutil.copy2(source, temporary)
                assert digest(source) == digest(temporary), 'Drive ZIP 복사 hash 불일치'
                temporary.replace(target)
            os.sync()
        def execute(stage):
            logs = EXP / 'logs'
            logs.mkdir(exist_ok=True)
            started = time.monotonic()
            done = threading.Event()
            def heartbeat():
                while not done.wait(30):
                    print(f'[{stage}] 실행 중: {(time.monotonic()-started)/60:.1f}분', flush=True)
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                command = [sys.executable, '-u', 'tools/portfolio_lab.py', stage, '--experiment-dir', str(EXP)]
                with open(logs / f'{stage}.log', 'w') as log:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    try:
                        for line in process.stdout:
                            print(line, end='')
                            log.write(line)
                            log.flush()
                        code = process.wait()
                    except BaseException:
                        process.terminate()
                        process.wait()
                        raise
                    if code:
                        raise RuntimeError(f'{stage} failed (exit {code}). 이후 셀을 진행하지 마세요.')
            finally:
                done.set()
                sync_results()
            print(f'[{stage}] 완료 / Drive 동기화: {DRIVE_EXP}')
        # 재개 시 provenance는 남아 있지만 fixture가 없으면 안전하게 재생성하도록 처리합니다.
        execute('prepare')
    '''), cell("code", '''
        # [5] 원본 baseline 재평가 — GPU. Stage3는 라벨이 있는 50개 중심 window만 계산해 시간을 줄입니다.
        execute('baseline')
    '''), cell("code", '''
        # [6] Stage 3: flow A/B + nested 영상별 CV. 외부 데이터나 팀원 가중치가 필요 없습니다.
        execute('stage3')
    '''), cell("code", '''
        # [7] Stage 2: 차영상 / 전역 운동 변화. 충돌 이외 출력은 유지합니다.
        execute('stage2')
    '''), cell("code", '''
        # [8] Stage 1: 5-clip 확률 평균, median 보조 비교. GPU.
        execute('stage1')
    '''), cell("code", '''
        # [9] 제출 ZIP 생성 및 2회 실제 추론 비교 + ZIP/import/schema/hash 검사.
        # NO_GO이면 이전 실험의 submit.zip을 대신 제출하지 마세요.
        execute('package')
    '''), cell("code", '''
        # [10] 최종 셀: GO / NO_GO, Local/Official BSS, Drive 최종 산출물.
        # 중간 단계 오류 뒤에도 이 셀을 실행하면 NO_GO 진단을 저장할 수 있습니다([4]까지 실행된 경우).
        execute('report')
        gate = json.loads((EXP / 'reports/submission_gate.json').read_text())
        if gate['verdict'] == 'GO':
            try:
                sync_results(include_submit=True)
                print('제출 파일:', DRIVE_EXP / 'submission/submit.zip')
                print('주의: GO는 탐색 제출 가능. 공식 점수 상승 확률은 아직 알 수 없습니다.')
            except Exception as error:
                gate['verdict'] = 'NO_GO'
                gate['failures'].append('Drive 제출 ZIP 저장/검증 실패: ' + str(error))
                (EXP / 'reports/submission_gate.json').write_text(json.dumps(gate, ensure_ascii=False, indent=2))
                sync_results()
                print('NO_GO:', gate['failures'])
        else:
            print('제출하지 마세요:', gate['failures'])
        # 검토용 작은 결과 ZIP: 모델/영상/submit.zip은 넣지 않습니다.
        review = EXP / 'review_results.zip'
        with zipfile.ZipFile(review, 'w', zipfile.ZIP_DEFLATED) as archive:
            for folder in ('reports', 'predictions', 'config', 'logs'):
                for path in (EXP / folder).rglob('*'):
                    if path.is_file():
                        archive.write(path, path.relative_to(EXP))
        shutil.copy2(review, DRIVE_EXP / review.name)
        os.sync()
        assert digest(review) == digest(DRIVE_EXP / review.name)
        print('나에게 전달할 파일:', DRIVE_EXP / 'review_results.zip')
        print('VERDICT:', gate['verdict'])
    ''')]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "04_portfolio_v1.ipynb"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def main():
    path = ROOT / "notebooks/04_portfolio_v1.ipynb"
    value = notebook()
    for index, item in enumerate(value["cells"]):
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"cell_{index}", "exec")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = ["baseline_inference.py", "requirements.txt", "notebooks/04_portfolio_v1.ipynb",
             "tools/portfolio_runtime.py", "tools/portfolio_lab.py", "tools/portfolio_base_manifest.json",
             "tools/local_validate.py", "tools/check_predictions.py", "tools/build_submit.py",
             "tools/run_local_inference.py", "tools/validate_submit_package.py", "tools/test_portfolio.py",
             "docs/portfolio_v1_run_guide.md"]
    metadata = {"format": "portfolio_v1", "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}}
    output = ROOT / "artifacts/colab/dacon236753_portfolio_patch.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("PORTFOLIO_PATCH.json", json.dumps(metadata, indent=2))
        for name in files:
            archive.write(ROOT / name, name)
    print(f"Notebook: {path}\nPatch: {output} ({output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
