# [Stage 3] comma2k19 CAN 신호 기반 차량 가감속·조향 모델 개선 실험

## 1. 실험 목적

블랙박스 영상만을 입력으로 받아 0.1초 단위로 다음 차량 거동을 예측하는 Stage 3 모델을 개선한다.

- 가감속: `ACCELERATING`, `DECELERATING`, `CONSTANT`, `STOPPED`
- 조향: `LEFT`, `STRAIGHT`, `RIGHT`

대회 링크: https://dacon.io/competitions/official/236753/overview/description

## 2. 평가 방식과 현재 기준점

```text
Stage3 Score = 0.7 × Acceleration Macro-F1
             + 0.3 × Steering Macro-F1
```

- 전체 대회 점수에서 Stage 3 가중치: `0.4`
- `STOPPED` 프레임은 조향 Macro-F1 계산에서 제외
- 제출 형식을 맞추기 위해 `STOPPED` 프레임에도 `steer_label`을 출력해야 함

현재 공식 baseline 점수:

```text
Stage 1 = 0.5339789355
Stage 2 = 0.11721
Stage 3 = 0.14581
Weighted overall ≈ 0.21200
```

## 3. 핵심 가설

공개 주행 데이터의 CAN 신호를 이용하면 영상과 실제 차량 거동 사이의 관계를 추가로 학습할 수 있다.

> 실제 속도와 조향각이 포함된 comma2k19 데이터로 Stage 3 head를 추가 학습하면, 기존 MViTv2 backbone의 영상 표현을 유지하면서 가감속·조향 분류 성능을 개선할 수 있다.

comma2k19와 대회 데이터 사이에는 domain shift가 존재할 수 있다. 첫 실험에서는 전체 backbone을 미세조정하지 않고 backbone을 고정한 채 분류 head만 학습한다.

## 4. 사용 데이터

공개 데이터 `commaai/comma2k19`의 Hugging Face demo subset을 사용한다.

- HEVC 주행 영상 64개
- 영상 ZIP 약 2.4GB
- CAN 및 프레임 정보 Parquet 3개
- 주요 신호: 차량 속도, 조향각, 영상 프레임 timestamp
- 데이터 분할 단위: 영상 프레임이 아닌 `route`
- 같은 route가 train과 validation에 동시에 포함되지 않도록 분리

Drive 저장 구조:

```text
블랙박스 영상 기반 사고 분석/
└── external_data/stage3/
    ├── raw/comma2k19_hf_demo/
    │   ├── data/*.parquet
    │   └── compression_challenge/test_videos.zip
    ├── processed/comma2k19_hf_demo/
    │   ├── videos/
    │   ├── frame_times/
    │   ├── speed/
    │   ├── steering/
    │   ├── labels.csv
    │   ├── audit.csv
    │   └── label_audit.json
    └── manifest.csv
```

## 5. 전처리 방법

### 영상

- HEVC를 MP4/H.264로 변환
- 원본 20 FPS를 대회 출력 주기에 맞춰 10 FPS로 변환
- 최대 가로 해상도 640px, 종횡비 유지
- `yuv420p/libx264` 호환성을 위해 가로·세로를 짝수 크기로 보정
- `.part.mp4`에 먼저 출력하고 크기와 해상도를 확인한 후 최종 파일로 교체
- 이미 정상 변환된 영상은 재실행 시 건너뜀

### CAN 신호 정렬

1. 영상 프레임 timestamp를 10Hz 기준으로 선택한다.
2. 속도와 조향각을 각 프레임 timestamp에 보간한다.
3. median smoothing을 적용한다.
4. 속도의 시간 미분으로 가속도를 계산한다.
5. 각 프레임에 가감속·조향 pseudo-label을 부여한다.

프레임과 CAN 신호의 timestamp coverage가 95% 미만인 영상이 있으면 학습을 중단한다.

## 6. 초기 라벨 생성 규칙

다음 threshold는 대회의 공식 기준이 아니라 첫 실험을 위한 가정이다.

### 가감속

```text
speed ≤ 0.30 m/s           → STOPPED
acceleration > 0.25 m/s²   → ACCELERATING
acceleration < -0.25 m/s²  → DECELERATING
그 외                       → CONSTANT
```

### 조향

```text
steering angle > 3°   → LEFT
steering angle < -3°  → RIGHT
그 외                  → STRAIGHT
```

현재는 comma2k19 조향각의 양수를 `LEFT`로 가정한다. 실제 부호 정의가 반대라면 LEFT/RIGHT가 뒤집히므로 영상 표본을 이용한 확인이 필요하다.

## 7. 모델 구조

공식 baseline Stage 3 구조를 유지한다.

```text
16-frame video clip
        ↓
MViTv2-S backbone (frozen)
        ↓
cached video feature
        ├── acceleration linear head → 4 classes
        └── steering linear head     → 3 classes
```

학습 전략:

- 공식 baseline MViTv2-S backbone 고정
- backbone feature를 한 번 계산해 Drive에 캐싱
- acceleration/steering linear head만 학습
- class-weighted loss로 클래스 불균형 보정
- `STOPPED` 샘플은 steering loss에서 제외
- 모델 선택은 대회와 같은 `0.7 × accel F1 + 0.3 × steer F1` 사용

## 8. 로컬 검증 설계

랜덤 프레임 분할은 인접 프레임 누수로 성능이 과대평가될 가능성이 높으므로 사용하지 않는다.

```text
comma2k19 routes
    ├── train routes
    └── validation routes
```

동일한 validation routes에서 다음 두 모델을 비교한다.

1. 공식 baseline checkpoint
2. comma2k19로 Stage 3 head를 학습한 candidate checkpoint

```text
Local BSS = Candidate local Stage 3 score
          - Baseline local Stage 3 score

Official BSS = Candidate official Stage 3 score
             - 0.14581
```

현재 제출 gate의 최소 기준은 `Local BSS ≥ +0.030`이다.

기존 smoke Stage 3 점수 `0.14014`는 실행 회귀 검사 값이며 실제 성능 추정에는 사용하지 않는다. 제출 전 공식 점수 예상 범위는 local improvement의 25~75%가 전이된다는 보수적 heuristic을 사용하지만, 이는 통계적으로 보장된 값이 아니다.

## 9. GO / NO_GO 기준

다음 조건을 모두 만족한 경우에만 `GO`로 판정한다.

- route-group holdout에서 `Local BSS ≥ +0.030`
- baseline보다 acceleration Macro-F1이 개선됨
- steering 성능이 비정상적으로 붕괴하지 않음
- route leakage가 없음
- 모든 Stage의 출력 CSV schema 검사 통과
- 후보 checkpoint를 포함한 제출 ZIP 생성 성공
- 제출 ZIP을 서로 다른 임시 환경에서 두 번 smoke 추론해 모두 통과
- Stage 1·2 결과가 baseline과 동일하게 유지됨

하나라도 만족하지 못하면 `NO_GO`이며 공식 제출 횟수를 사용하지 않는다.

## 10. 제출 패키징 전략

이번 실험에서는 Stage 3만 변경한다.

```text
Stage 1 → 공식 baseline 유지
Stage 2 → 공식 baseline 유지
Stage 3 → candidate checkpoint로 교체
```

최종 `submit.zip` 구조:

```text
submit.zip
├── inference.py
├── requirements.txt
└── model/
    ├── stage1/best.pt
    ├── stage2/best.pt
    ├── stage2/resnet18-f37072fd.pth
    └── stage3/best.pt
```

공식 평가 데이터는 파일별로 독립 추론하며 평가 데이터 간 통계 공유, 튜닝 또는 pseudo-labeling은 수행하지 않는다.

## 11. Colab 운영 방식

업로드 시간을 줄이기 위해 base와 code patch를 분리한다.

### Base bundle

```text
dacon236753_colab_bundle.zip
```

- 약 480MB
- 모델 checkpoint와 smoke fixture 포함
- 최초 한 번만 Drive에 업로드
- 일반 코드 수정에서는 교체하지 않음

### Code patch

```text
dacon236753_patch.zip
```

- 약 54KB
- Python 코드, 도구, 노트북 및 문서만 포함
- 코드 수정 시 patch만 Drive에서 교체
- Colab에서는 base 위에 patch를 덮어씀

Colab 학습 결과는 다음 경로에 누적한다.

```text
experiments/s3_hf_demo_v1/
├── checkpoints/
│   ├── parent_baseline.pt
│   └── stage3_best.pt
├── features/
├── reports/
│   ├── baseline_local_metrics.json
│   ├── local_metrics.json
│   ├── training_summary.json
│   ├── zip_smoke_pass.json
│   └── submission_gate.json
└── submission/
    └── submit.zip
```

## 12. 현재 진행 상황

- [x] 공식 baseline 구조 분석
- [x] 공식 Stage 3 점수 기록: `0.14581`
- [x] comma2k19 HF demo 다운로드 로직 구현
- [x] Drive 실제 파일 저장 및 symlink 방지
- [x] Parquet schema 확인
- [x] CAN timestamp와 영상 프레임 정렬 로직 구현
- [x] 20 FPS → 10 FPS 변환 구현
- [x] route 단위 train/validation 분리 구현
- [x] CAN 기반 pseudo-label 생성 구현
- [x] frozen-backbone feature caching 구현
- [x] acceleration/steering head 학습 구현
- [x] Local BSS 및 Official BSS gate 구현
- [x] 제출 ZIP 이중 smoke 검증 구현
- [x] H.264 홀수 해상도 오류 수정
- [x] base/patch 업로드 구조로 변경
- [ ] Colab에서 64개 영상 전체 변환 완료 확인
- [ ] 조향각 부호를 영상으로 표본 검증
- [ ] baseline validation score 측정
- [ ] candidate head 학습
- [ ] candidate validation score 측정
- [ ] 최종 `GO/NO_GO` 확인
- [ ] `GO`인 경우에만 공식 제출

## 13. 확인된 오류와 조치

일부 영상을 종횡비 유지 방식으로 축소했을 때 `639×480`이 생성되어 H.264 인코딩이 실패했다.

```text
width not divisible by 2 (639x480)
```

수정한 FFmpeg filter:

```text
scale=640:-2:force_original_aspect_ratio=decrease,
pad=ceil(iw/2)*2:ceil(ih/2)*2
```

실패한 0바이트 MP4는 자동으로 재처리하고, 변환 결과는 검증 후에만 최종 경로로 이동한다.

## 14. 주요 위험 요소

### Domain shift

comma2k19는 국내 블랙박스 사고영상과 촬영 위치, 카메라 및 주행 상황이 다르다. 로컬 개선이 공식 평가에 그대로 전이되지 않을 수 있다.

### Threshold 불확실성

가감속 및 조향 threshold는 실험적 가정이다. 대회의 실제 라벨링 기준과 다르면 label noise가 커질 수 있다.

### 조향각 부호

조향각 양수 방향이 LEFT인지 RIGHT인지 영상으로 확인해야 한다.

### 프레임 시간 정렬

decoder의 실제 출력 프레임 수와 CAN `frame_times[::2]`가 정확히 대응하는지 audit이 필요하다.

### 데이터 규모

현재 demo subset은 64개 영상이므로 다양한 사고, 도로 및 날씨 조건을 충분히 포함하지 못한다.

### 공식 점수 과적합

공식 점수를 이용해 threshold를 반복 조정하면 제한된 제출 횟수와 Public leaderboard에 과적합될 수 있다. 공식 점수는 최종 확인과 실험 기록 용도로만 사용한다.

## 15. 다음 작업

1. 수정된 transcoding patch 적용
2. 64개 영상 전체 변환 완료
3. 변환 영상의 길이, FPS, 해상도 및 프레임 수 audit
4. 조향각 부호와 threshold 표본 시각 검증
5. baseline route-holdout 점수 측정
6. candidate head 학습
7. Local BSS 계산
8. 제출 ZIP 이중 smoke 검증
9. 마지막 셀의 `GO/NO_GO` 확인
10. `GO`인 경우에만 공식 제출

## 16. 완료 조건

- [ ] `external_data/stage3/manifest.csv`
- [ ] `processed/comma2k19_hf_demo/label_audit.json`
- [ ] `reports/baseline_local_metrics.json`
- [ ] `reports/local_metrics.json`
- [ ] `reports/training_summary.json`
- [ ] `reports/zip_smoke_pass.json`
- [ ] `reports/submission_gate.json`
- [ ] `submission/submit.zip`
- [ ] 최종 gate가 `GO`, 또는 `NO_GO` 원인이 명확히 기록됨
