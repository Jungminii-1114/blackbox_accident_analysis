"""Build a code-only update and the next single-entry Colab notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from build_portfolio_delivery import notebook, cell

ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    old = notebook()['cells']
    first = ''.join(old[1]['source']).replace('dacon236753_portfolio_patch.zip','dacon236753_focus12_patch.zip').replace("'portfolio_v1_'", "'focus12_v2_'")
    first += "\nPARENT_EXP = DRIVE_ROOT / 'experiments/portfolio_v1_20260910T063010Z'\nPARENT_ZIP = PARENT_EXP / 'submission/submit.zip'\nassert PARENT_ZIP.is_file(), f'직전 제출 ZIP이 필요합니다: {PARENT_ZIP}'\n"
    bootstrap = ''.join(old[2]['source']).replace('/content/dacon236753_portfolio','/content/dacon236753_focus12').replace('PORTFOLIO_PATCH.json','FOCUS12_PATCH.json')
    bootstrap += '''
history = json.loads((WORK / 'tools/focus12_history.json').read_text())
assert digest(PARENT_ZIP) == history['parent_zip_sha256'], '직전 실험의 실제 제출 ZIP hash가 다릅니다. 다른 ZIP을 임의로 사용하지 마세요.'
extract_safe(PARENT_ZIP, WORK / 'parent')
assert digest(WORK / 'parent/inference.py') == history['parent_source_sha256']
saved = PARENT_EXP / 'predictions/selected_zip/stage3.csv'
if saved.exists():
    target = WORK / 'parent_review/predictions/selected_zip/stage3.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(saved, target)
print('직전 공식 제출 코드를 고정했습니다. Stage3 재학습 없음.')
'''
    execute = ''.join(old[4]['source']).replace('tools/portfolio_lab.py','tools/focus12_lab.py')
    final = ''.join(old[10]['source']).replace('# [10]', '# [9]')
    cells = [cell('markdown', '''
        # focus12_v2 — Stage 1·2 집중 실험 / Stage 3 공식 0.57137 구성 고정

        이 노트북 하나를 GPU Colab에서 순서대로 실행합니다. 기존 base ZIP과 직전 `submit.zip`을 재사용합니다.
        Drive 루트에 `dacon236753_focus12_patch.zip`만 새로 업로드하세요. 외부 데이터 다운로드는 없습니다.

        - S1: 기존 ResNet18을 고정한 특징 + 작은 ridge 분류 head, 원본/재촬영 5쌍 nested CV.
        - S2: 부모 B의 국소 정밀화 E, 독립 충돌 신호 C·D와 JPEG95/90 민감도 비교.
        - 후보가 기준에 못 미치면 S1은 기존 공식 점수가 더 좋았던 3-clip으로 복구하고, S2는 직전 B를 유지합니다.
        - S3: 직전 ZIP의 코드·임계값 그대로 보존. 부모와 새 ZIP 예측이 한 행이라도 다르면 패키징 실패.

        GO는 탐색 또는 기존 우수 Stage 조합의 제출 가능 판정이며 공식 상승 보장은 아닙니다.
        리더보드 제출수 3은 당일 사용량이라고 단정할 수 없습니다. 제출탭에서 오늘 남은 횟수를 확인하세요. 실행과 제출은 별개입니다.
    '''), cell('code',first),cell('code',bootstrap),old[3],cell('code',execute),
        cell('code',"# [5] 직전 제출 모델 세 Stage 재추론 + 원래 S1 3-clip 비교군\nexecute('parent')"),
        cell('code',"# [6] S1: frozen ResNet18 + nested source-pair CV, JPEG90 안정성\nexecute('stage1')"),
        cell('code',"# [7] S2: E/C/D vs 공식 개선된 B; 시간 MAE + JPEG95/90 검증\nexecute('stage2')"),
        cell('code',"# [8] 제출 ZIP 생성 + 두 번 추론 + Stage3 부모 동등성\nexecute('package')"),cell('code',final)]
    result = notebook()
    result['cells'] = cells
    result['metadata']['colab']['name'] = '05_focus12_v2.ipynb'
    return result


def main():
    value = build_notebook()
    for index,c in enumerate(value['cells']):
        if c['cell_type']=='code':
            compile(''.join(c['source']),f'cell_{index}','exec')
    path = ROOT / 'notebooks/05_focus12_v2.ipynb'
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files = ['baseline_inference.py','requirements.txt','notebooks/05_focus12_v2.ipynb',
             'tools/focus12_runtime.py','tools/focus12_lab.py','tools/focus12_history.json','tools/test_focus12.py',
             'tools/portfolio_runtime.py','tools/portfolio_lab.py','tools/portfolio_base_manifest.json',
             'tools/local_validate.py','tools/check_predictions.py','tools/run_local_inference.py',
             'tools/build_submit.py','tools/validate_submit_package.py','docs/focus12_v2_run_guide.md',
             'docs/github_issue_portfolio_v1_result.md']
    manifest = {'format':'focus12_v2','sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}}
    output = ROOT / 'artifacts/colab/dacon236753_focus12_patch.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        archive.writestr('FOCUS12_PATCH.json',json.dumps(manifest,indent=2))
        for name in files:
            archive.write(ROOT/name,name)
    print('Notebook:',path,'\nPatch:',output,round(output.stat().st_size/1024,1),'KiB')


if __name__=='__main__':
    main()
