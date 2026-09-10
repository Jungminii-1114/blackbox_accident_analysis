# 로컬 검증과 2회 남은 제출의 사용법

## Colab에서 바로 실행

`notebooks/dacon236753_colab_runner.ipynb`와
`artifacts/colab/dacon236753_colab_bundle.zip`을 Colab/Google Drive에 올린다. 노트북은
제출 파일을 만든 뒤, ZIP 내부 코드를 다시 풀어 같은 smoke 입력에서 **두 번째 GPU
추론**까지 수행한다. 이 두 실행과 `check_predictions.py`가 모두 PASS일 때만 업로드한다.

로컬 검증 점수는 비공개 리더보드 점수의 재현값이 아니다. 반드시 라벨이 있는
학습/검증 분할에서만 비교하고, Public 점수는 별도의 일반화 관측값으로 기록한다.

## 공식 예제 smoke 검증

공식 예제는 먼저 아래처럼 평가 입력 형태로 변환한다. 이 단계는 OpenCV가 있는
CUDA/DACON 환경에서 한 번만 실행하면 된다.

```bash
python tools/prepare_baseline_smoke.py \
  --baseline-data /path/to/Baseline/data \
  --out artifacts/baseline_smoke
```

`artifacts/baseline_smoke/labels`은 즉시 `local_validate.py`에 전달할 수 있고,
`stage2_frame_time.csv`은 영상별 프레임-시간 대응표다. 이 예제는 공식 가중치의
학습 샘플이므로 **회귀·형식 검사에만** 사용한다. 하이퍼파라미터 선택이나 Public
점수 예측에는 사용하지 않는다.

GPU 환경에서 현재 제출 모듈을 실행해 Stage CSV를 만든다.

```bash
python tools/run_local_inference.py \
  --input-root artifacts/baseline_smoke/input \
  --model-dir model \
  --out artifacts/baseline_smoke/predictions
```

## 예측 파일

모델 실행 결과를 다음처럼 분리한다.

```text
predictions/
├── stage1.csv  # ID,answer
├── stage2.csv  # ID,collision_frame,entry_frame,evasion_space,entry_side
└── stage3.csv  # ID,sample_index,accel_label,steer_label
```

라벨 디렉터리도 동일한 이름의 CSV 세 개를 사용한다. Stage 2 프레임 번호가 실제
시간으로 불규칙하게 대응하는 경우 `ID,frame,time_seconds` 형태의 대응표를 전달한다.

```bash
python tools/local_validate.py \
  --labels-dir artifacts/baseline_smoke/labels \
  --predictions-dir artifacts/baseline_smoke/predictions \
  --stage2-frame-time-map artifacts/baseline_smoke/labels/stage2_frame_time.csv \
  --json-out artifacts/exp01/local_metrics.json
```

고정 FPS 영상만 있다면 대응표 대신 `--stage2-fps 30`을 쓴다. 대응표가 없으면 도구는
프레임 번호 자체를 시간으로 간주하고 경고성 `time_conversion`을 결과 JSON에 남긴다.

## 제출 전 순서

1. 변경마다 고정 검증 분할에서 `local_validate.py`를 실행한다.
2. Stage별 점수와 예측 CSV를 `artifacts/<실험명>/`에 보관한다.
3. 제출 직전에 범주·중복·프레임 범위를 검사한다.

```bash
python tools/check_predictions.py \
  --predictions-dir artifacts/exp01/predictions \
  --stage1-videos data/eval_input/stage1/videos \
  --stage2-images data/eval_input/stage2/images \
  --stage3-expected data/eval_input/stage3_expected.csv
```

4. 오늘은 **현재 기준선 1회 + 검증에서 개선된 단 하나의 변경 1회**만 Public에 낸다.
5. 결과가 표시되면 즉시 기록한다.

## 필수 GO / NO_GO 셀

모든 실험 노트북의 마지막 셀은 `tools/go_no_go.py`를 호출한다. BSS는 Baseline Score를
뜻한다. `Local BSS`는 반드시 외부 route/group holdout의 기준선과 후보 점수이고,
`Official BSS`는 현재 관측된 Public 기준선(S1=.5339789355, S2=.11721, S3=.14581)이다.

```bash
python tools/go_no_go.py \
  --stage stage3 \
  --baseline-local artifacts/baseline_external/local_metrics.json \
  --candidate-local artifacts/s3_can_v1/local_metrics.json \
  --evaluation-kind external_group_holdout \
  --smoke-pass \
  --json-out artifacts/s3_can_v1/go_no_go.json
```

`smoke` 또는 학습셋 점수는 자동으로 `NO_GO`다. 처음에는 local→Public 관계를 알 수 없으므로,
도구는 local 향상의 25–75%만 Public 종합점수에 전달된다는 넓은 범위를 표시한다. 이 범위는
점수 예측이 아니라 제출 결정을 보수적으로 만드는 가정이다.

```bash
python tools/record_submission.py \
  --experiment baseline-official \
  --log '/content/drive/MyDrive/블랙박스 영상 기반 사고 분석/experiments/leaderboard.csv' \
  --submission-zip artifacts/baseline-official/submit.zip \
  --public-stage1 0.0 --public-stage2 0.0 --public-stage3 0.0 \
  --notes "Replace values with the Dacon result; never fabricate them."
```

공개 베이스라인 라벨은 Stage 2의 충돌시점 외 세 목표가 `-1`이고 Stage 3도 희소하다.
따라서 해당 파일만으로 Stage 2 전체 점수와 종합점수가 `N/A`인 것은 정상이다.
