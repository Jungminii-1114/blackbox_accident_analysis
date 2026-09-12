# Drive 누적 실험 운영법

> 2026-09-10: 아래는 이전 단일 Stage 운영 기록입니다. 세 Stage 동시 탐색은
> [portfolio_v1 실행 안내](portfolio_v1_run_guide.md)를 따릅니다. 대형 base는 그대로 두고
> `dacon236753_portfolio_patch.zip`과 `04_portfolio_v1.ipynb`만 새로 사용합니다.

Google Drive의 `experiments/registry.csv`가 모든 실험과 제출의 단일 기록이다. 실험 하나는
한 Stage, 한 가설, 한 고정 외부 group/route holdout만 다룬다. 나머지 Stage는 부모 실험의
제출 코드를 그대로 유지한다.

## Colab 업로드 구조

- `dacon236753_colab_bundle.zip`은 체크포인트와 smoke 영상을 포함한 최초 1회용 base다.
- `dacon236753_patch.zip`은 코드·노트북·도구·문서만 포함한다. 일반 수정 때는 이 파일만 교체한다.
- 새 Colab 런타임에서는 base를 먼저, patch를 나중에 `/content/dacon236753`에 푼다.
- 이미 `model/`이 준비된 실행 중 런타임에서는 patch만 같은 경로에 덮어쓴다.
- Colab에서 학습한 후보 체크포인트는 Drive의 `experiments/<name>/checkpoints/`에 유지하므로 base를 다시 올리지 않는다.

## 첫 실험 만들기

Colab에서 번들을 `/content/dacon236753`에 푼 뒤 다음을 한 번 실행한다.

```bash
cd /content/dacon236753
python tools/init_experiment.py \
  --drive-root '/content/drive/MyDrive/블랙박스 영상 기반 사고 분석' \
  --experiment s3_hf_demo_v1 --stage stage3 \
  --notes 'comma2k19 HF demo, frozen MViT backbone and trained heads'
```

생성된 폴더는 다음과 같다.

```text
experiments/s3_hf_demo_v1/
├── config/experiment.json
├── checkpoints/
├── predictions/
├── reports/local_metrics.json
├── reports/zip_smoke_pass.json
├── submission/submit.zip
└── README.md
```

`registry.csv`에는 생성 시점과 Stage·부모 실험이 기록된다. `GO`로 제출한 뒤에는 동일
행에 로컬 BSS, 공식 Stage 점수, 공식 종합점수, 제출 날짜와 핵심 변경을 채운다. Public
점수는 학습에 쓰지 않고 실험 이력으로만 저장한다.

## 실행 순서

1. `dacon236753_colab_runner.ipynb`로 코드·모델·제출 ZIP smoke를 먼저 확인한다.
2. `01_stage3_can_lab.ipynb`가 `s3_hf_demo_v1` 폴더와 표준 하위 폴더를 만든다.
3. 같은 노트북에서 HF 원본 변환 → route split → baseline 평가 → head 학습을 수행한다.
4. 제출 ZIP을 두 번 추론해 확인한 경우에만 smoke marker를 남긴다.
5. 해당 실험 노트북의 마지막 `go_no_go.py` 셀을 실행한다.
6. `GO`면 그 폴더의 `submission/submit.zip`만 제출하고 공식 결과를 registry에 기록한다.
7. 다음 실험은 `s3_hf_demo_v2`처럼 새 폴더에서 시작한다. 기존 폴더·점수는 수정하지 않는다.
