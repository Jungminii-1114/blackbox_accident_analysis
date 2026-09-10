# portfolio_v1 — 세 Stage 동시 탐색 실험 설계

작성일: 2026-09-10. 후속 구현: `04_portfolio_v1.ipynb`와 소형 portfolio patch 제공.
실제 실행 방식과 설계에서 좁혀진 범위는 [실행 안내](portfolio_v1_run_guide.md)를 따른다.
아래는 최초 설계 기록이다. 이후 공개 MP4의 약 480fps 컨테이너 오류가 확인되어,
공개 라벨 frame/time 대응으로만 fixture 시간축을 복원하도록 구현했다. 비공개 추론은 공식 10Hz frame/sample 계약을 따른다.
이 문서는 기존 `model_design_research.md`의 제출 운영과 성공 확률 추정을 대체한다.
GPU 전체 실행과 실제 제출 패키지 검증은 Colab에서 완료해야 한다.

## 1. 최종 재검토 결과

| 기존 제안 | 재검토 | 이번 결정 |
|---|---|---|
| Stage 1 logit 평균은 극단값에 강함 | 일반적으로 성립하지 않음. 확률 [0.99, 0.30, 0.30]의 확률 평균은 0.530, logit 평균 후 확률은 0.724 | 5-clip 확률 평균을 우선 후보로 하고 집계 방식은 별도 ablation |
| Stage 2 여러 충격 신호를 합치면 개선 | 유망하지만 5개 예제로 네 가중치를 최적화하면 과적합 위험이 큼 | 단순 차영상 기준과 2신호 고정 가중치 후보만 비교 |
| 시작·끝 10%를 제외하면 안전 | 실제 충돌도 제외할 수 있음 | 경계는 제외하지 않고 장면 전환·읽기 오류를 별도 진단 |
| flow는 실제 속도·조향각을 직접 측정 | 2차원 영상 운동의 대리 신호. 깊이, 회전, 독립 객체 운동의 영향을 받음 | 절대 속도 추정이라고 부르지 않고 운동 대리 특징으로 검증 |
| 공개 라벨 group CV이면 깨끗한 검증 | 이미 baseline 학습에 쓰인 예제는 backbone 수준에서 독립 아님 | 학습 노출 이력과 sparse calibration 검증을 별도 표시 |
| pseudo-label holdout 개선이면 공식 상승 | 이번 관측에서 실패. 다만 pseudo-label 학습 자체가 무효인 것은 아님 | 보조 검증으로 유지하고 단독 성능 GO 금지 |
| 매일 세 번째 제출은 최고 조합 확정 | 결합 필요성과 남은 시간에 따라 달라짐 | 마지막 제출을 기계적으로 소비하지 않고 결과 확인 후 배정 |

실패 원인은 아직 확정되지 않았다. 라벨 정의·시간 정렬·도메인·표현력·모델 선택 편향을 각각 확인한다.
근거가 없는 성공 확률(예: 65~80%)과 공식 예상 점수 범위는 제공하지 않는다.

## 2. 기준점과 실험 목표

- 유지할 공식 baseline: S1=0.5339789355, S2≈0.11721, S3≈0.14581.
- 실패한 `s3_hf_demo_v1`: 공식 S3=0.1420426225. 다음 실험의 부모 Stage 3 가중치로 사용하지 않는다.
- Yejun 이슈 보고값: S2≈0.267, S3≈0.521. 비교 참고값이며 이번 독립 구현의 재현 보장이 아니다.
- 목표: Stage별 후보를 로컬에서 비교하고, 준비된 후보를 하나의 ZIP에 결합해 Stage별 공식 변화량을 관측한다.
- 한 Stage 안에서는 하나의 고정 후보를 제출한다. 파일 간 정보를 공유하거나 평가 도중 모델·임계값을 갱신하지 않는다.

## 3. 공통 선행 검증: 시간·라벨·산출물 계약

### Stage 3 sample 대응

공개 `labels.csv`에는 `sample_index=60`, `frame_index=120`, `time_seconds=6.0`인 행이 있다.
반면 기존 `prepare_baseline_smoke.py`는 원본 영상을 그대로 복사하고 디코딩 프레임 수만큼
sample을 생성한다. 기존 smoke 점수를 신뢰하기 전에 이 차이를 해결해야 한다.

1. 각 공개 영상의 디코딩 프레임 수, PTS, 길이, FPS 메타데이터를 대조한다.
2. 공개 라벨 평가는 라벨의 `frame_index`와 `time_seconds`를 사용해 원본 영상 시점에 정렬한다.
3. 공식 평가 입력이 사전에 10Hz로 변환되는지 공식 baseline/안내의 입력 계약으로 확인한다.
4. 평가 입력이 이미 10Hz라면 sample k를 frame k에 대응시킨다. 원본 20Hz 공개 영상을 10Hz로 변환한 fixture와 구별한다.
5. 확인되지 않은 FPS를 추측해 출력 행 수를 바꾸지 않는다. 계약을 확인할 때까지 최종 ZIP 생성은 NO_GO.
6. 모든 모듈은 동일 시간 그리드를 공유한다. flow의 gap, smoothing, slope 구간도 초 단위로 명시한다.

Stage 2는 실제 파일명의 프레임 번호를 보존하고 frame→time 표로 Accuracy@0.3초를 계산한다.
프레임 MAE만으로 대회 시점 점수를 대체하지 않는다.

### 데이터의 실제 규모

공개 Stage 3 라벨: 5영상, 50표본.

| 범주 | 표본 수 |
|---|---:|
| CONSTANT | 30 |
| ACCELERATING | 9 |
| DECELERATING | 8 |
| STOPPED | 3 |
| STRAIGHT | 39 |
| LEFT | 6 |
| RIGHT | 5 |

조향 수치는 전체 50표본 기준이며 평가는 정답 STOPPED를 제외한 뒤 다시 집계한다.
Stage 2는 충돌 라벨 5건만 있고 나머지 목표는 -1이다. Stage 1은 원본·파생 영상 5쌍이다.
표본 수보다 프레임 수를 강조해 검증 규모를 부풀리지 않는다.

## 4. Stage 1: 시간 표본 수 증가

가설: 기존 세 시점에서 놓친 단서를 추가 시점이 제공할 수 있다. 추가 clip이 같은 오분류를
반복하거나 희소한 재녹화 단서를 희석할 수도 있으므로 개선은 미확정이다.

| ID | 구성 | 목적 |
|---|---|---|
| S1-B0 | 3 clip + 확률 평균 | 제출 baseline 재현 |
| S1-A | 5 clip + 확률 평균 | 시간 표본 수만 변경한 우선 후보 |
| S1-B | S1-A의 동일 5 clip + 확률 median | 집계 방식만 분리해 보는 보조 실험 |

- checkpoint, 입력 크기, clip 길이, threshold=0.5는 고정한다.
- clip별 logit·확률·디코딩 성공률·최종 예측을 캐시한다. S1-B는 GPU 재추론이 필요 없다.
- S1-A 추론량은 clip 수 기준 약 5/3배이며 전체 소요시간은 실제 측정한다.
- 원본과 대응 파생 영상은 같은 source group으로 취급한다.
- 기존 checkpoint가 본 예제에서 계산한 Macro-F1은 회귀 검사다. LOOV라고 이름을 바꿔 독립 holdout으로 취급하지 않는다.
- jitter/압축 변형에 대한 예측 안정성은 보조 지표이고 정답률을 대체하지 않는다.
- 실제 미노출 재촬영 검증셋이 없으면 상태는 EXPERIMENTAL. 기본 탐색 후보는 S1-A로 사전 고정하며 public 점수로 median을 반복 선택하지 않는다.

현실적 판단: 세 Stage 중 근거가 가장 약하다. 탐색은 가능하지만 높은 성공 가능성을 주장할 수 없다.

## 5. Stage 2: 충돌 출력만 교체

가설: 접촉 직후 화면 변화와 전역 운동 변화가 충돌 후보를 찾는 단서가 된다.
요철·급제동·노출 변화·장면 전환도 같은 신호를 만들 수 있다.

| ID | 충돌 후보 점수 | 역할 |
|---|---|---|
| S2-B0 | 기존 BiGRU | baseline |
| S2-A | 320×180 grayscale 프레임 차이 평균의 peak | 단순 비교군 |
| S2-B | 0.5×차영상 robust z + 0.5×전역 이동 변화 robust z | 우선 새 가설 |

- 전역 이동은 sparse LK + RANSAC affine의 translation으로 계산한다. 이미지 대각선으로 정규화한다.
- robust z는 같은 파일의 median/MAD로 계산한다. MAD 하한을 두고 극단값을 고정 범위로 clip한다.
- affine 적합 실패 시 해당 프레임은 차영상 항만 사용하며 유효 마스크를 기록한다.
- 시작·끝 10%를 일괄 제거하지 않는다. 길이 0/1/2프레임, 검은 프레임, 컷, 점멸 사례를 별도 테스트한다.
- 모든 프레임에서 가벼운 신호를 계산하고 top-3 후보 주변을 검토한다. 번호는 실제 파일명으로 반환한다.
- entry_frame/entry_side/evasion_space는 B0가 낸 값을 그대로 유지한다. 새 collision index를 baseline scene head에 주입하지 않는다.

검증: 5건의 Accuracy@0.3초, 프레임·초 MAE, 영상별 오차, top-3 recall을 모두 보고한다.
기존 smoke에서 collision accuracy=1.0이었던 기록도 있으므로 이 다섯 영상에서 개선이 안 보일 수 있다.
추가 라벨을 확보하지 못하면 두 후보의 고정 비교만 하고 네 신호 가중치 grid search는 하지 않는다.
bbox IoU를 collision/entry 정답 점수로 대신하지 않는다.

현실적 판단: 물리적 근거와 팀원의 보고 사례는 있으나 이번 충돌 단독 후보가 S2=0.267을
재현한다고 볼 수 없다. Yejun의 수치는 탐지·다른 출력도 함께 변경한 Stage 전체 점수다.

## 6. Stage 3: 시간 정렬된 optical flow와 회전 보조 특징

가설: 외형 특징만 재학습하는 것보다 영상 운동의 시간 변화가 가감속과 조향 구분에 유용하다.
새 구현과 자체 calibration을 사용하며 팀원의 checkpoint 다운로드를 필수 조건으로 두지 않는다.

| ID | 특징 | 목적 |
|---|---|---|
| S3-B0 | 원래 MViT | 공식 baseline |
| S3-B1 | 이번 CAN head | 실패 모델의 진단용, 기본 제출 제외 |
| S3-A | 도로 ROI median flow magnitude + horizon horizontal flow | 단순 motion 비교군 |
| S3-B | S3-A + 전역 horizontal translation/rotation 보조 | 회전과 전진 혼입 완화 실험 |

공통 구현:

- grayscale 160×90, Farneback, 파일별 streaming.
- 시간 계약이 확인된 10Hz 그리드에서 주 특징은 0.1초 간격. 0.2초 간격은 후속 ablation으로 남긴다.
- flow를 시간 간격으로 나눠 pixels/second로 표현한다. 단위 변경 시 기존 임계값을 그대로 재사용하지 않는다.
- 하단 ROI와 horizon ROI를 사전 고정하고 텍스처 부족·흔들림 품질 지표를 기록한다.
- speed proxy는 edge-preserving padding을 쓴 0.5초 중앙값 필터로 처리한다.
- accel proxy는 ±0.3초의 실제 시간 간격으로 나눈 차분이다. 경계에서는 사용 가능한 시간 폭으로 나눈다.
- 조향은 배경 flow 양수→좌회전이라는 기본 관계를 CAN/영상 표본에서 검증한다. CAN 조향각의 양수 정의와 혼동하지 않는다.
- STOPPED는 도로와 배경 운동이 함께 낮다는 증거를 사용한다. texture가 없어 flow가 0인 경우와 구분한다.
- 품질 부족 시 사용되는 fallback도 calibration과 평가에서 동일하게 실행한다. 숨은 평가 파일 통계를 모으지 않는다.

급하게 추가하지 않을 항목: FOE/divergence 전면 대체, 대형 flow 모델, MViT-flow 학습형 ensemble,
세 gap 동시 ensemble. 각각 별도 실험이 필요하다. monocular flow로 절대 속도가 식별된다고 가정하지 않는다.

### 임계값 검증

1. calibration과 inference가 동일 함수·시간 창을 사용하도록 한다.
2. 5영상 outer LOOV를 구성한다. 각 outer test 영상은 임계값 선택에서 제외한다.
3. outer train 4영상 내부에서 inner grouped CV로 S3-A/B와 임계값을 선택한다.
4. 정지·가감속·조향의 세 임계값 축에 각 3개 quantile 후보를 두어 총 27조합 이내로 제한한다. 각 inner split에서는 inner train만으로 후보 수치를 산출하고, 선택된 quantile 설정을 outer train에 다시 적합한다. inner validation/outer test의 특징 분포로 후보 경계를 만들지 않는다.
5. 각 outer test 예측을 모아 고정 클래스 Macro-F1과 0.7/0.3 가중 점수를 계산한다.
6. STOPPED 정답을 steering 평가에서 제외한다. fold에 클래스가 없으면 support와 정의를 명시한다.
7. 5개 fold 점수와 pooled OOF 점수를 함께 보고한다. 희귀 클래스 3개로 안정적인 신뢰구간이 나온다고 주장하지 않는다.
8. 최종 calibration은 전체 공개 5영상으로 수행하되 최종 학습 점수를 OOF 점수와 혼동하지 않는다.
9. comma2k19는 별도 route별 CAN 상관관계·부호·장면 변화 검증에 사용한다. threshold 선택 근거의 전부로 삼지 않는다.

공개 라벨을 이미 본 baseline의 점수와 새 후보의 OOF 점수 비교에는 학습 노출 차이가 있다.
이 점수를 깨끗한 모델 간 외부 holdout 비교라고 부르지 않는다.

현실적 판단: 세 Stage 중 새 접근의 근거는 가장 강하다. 다만 optical-flow 성공 사례가 새로운
feature 조합의 우월성까지 입증하지 않으므로 공식 예상 점수는 N/A다.

## 7. 제출 선택과 결과 판독

세 Stage가 각각 준비되면 S1-A / 선택된 S2 후보 / 선택된 S3 후보를 하나의 ZIP으로 묶는다.
준비 안 된 Stage는 검증된 baseline을 포함한다. 세 후보를 반드시 모두 바꾸기 위해 실패 gate를 무시하지 않는다.

- Stage별 공식 점수가 있으므로 동시 변경의 효과는 Stage 단위로 확인할 수 있다.
- 다만 각 Stage 내부 여러 변경의 기여는 로컬 ablation으로 확인한다.
- 공개 점수는 실험 관측값이다. 반복적인 임계값 탐색용 학습셋으로 사용하지 않는다.
- 다음 제출의 가치는 결과에 따라 판단한다. 하루 3회는 상한이며 세 번째 제출을 항상 최종 조합으로 소비하지 않는다.
- 최고 Stage 조합의 계산상 점수와 실제 제출 관측 점수는 구분한다.

### 판정 출력 계약

각 Stage에 baseline/candidate Local Score, Delta(기존 명칭 Local BSS), 라벨 출처, 그룹 수,
학습 노출 여부, 클래스별 F1/support, 증거 상태를 출력한다.

- SUPPORTED: 독립적인 목표 대응 라벨 검증이 사전 기준을 충족함. 공식 상승 보장을 뜻하지 않음.
- EXPERIMENTAL: 실행 검증은 통과했고 검토할 근거는 있으나 일반화 검증이 부족함.
- REJECTED: 계약/실행 오류, 과도한 퇴행, 근거 부재 등으로 제외.

최종 셀은 `GO` 또는 `NO_GO`를 반드시 출력하고 `purpose=exploration|best_combination`을 함께 표시한다.
EXPERIMENTAL 후보의 탐색 제출은 사용자가 승인한 이번 탐색 목적 아래 허용하되,
`GO`를 점수 상승 확률로 표현하지 않는다. 공식 candidate 점수는 제출 전 N/A다.
25~75% 전이 예상은 삭제한다. 한 번의 음의 전이율도 다음 모델 예측 공식으로 쓰지 않는다.

## 8. 구현·운영 명세

예정 노트북: `04_portfolio_v1.ipynb`. 다음 순서의 단일 진입점으로 구현한다.

개발 우선순위는 공통 시간 계약 audit → Stage 3 → Stage 2 → Stage 1 → 결합 패키징이다.
이는 근거와 병목을 고려한 작업 순서이며, 아래 노트북의 실행 셀 순서와는 구별한다.

1. Drive/base/patch 및 의존성 확인
2. 새 experiment ID와 원본 checkpoint·source hash 고정
3. 시간·라벨 계약 audit
4. S1 clip 예측 캐시와 후보 비교
5. S2 전역 운동/차영상 캐시와 후보 비교
6. S3 flow 캐시, nested video CV와 CAN 진단
7. Stage별 후보 선택과 실패 시 baseline 선택
8. 선택한 모듈로 standalone inference.py 생성
9. 동일 후보 workspace 추론과 ZIP 내부 추론 비교
10. 최종 GO/NO_GO 및 실험 정보 출력

새 후보별 코드와 가중치는 실험 전용 경로에 둔다. 기존 전역 `model/stage3/best.pt`를 부모 모델로
신뢰하지 않는다. 과거 패키징 과정에서 이미 후보로 덮어써졌을 수 있다.
기존 `build_submit.py`는 `baseline_inference.py`만 ZIP에 넣으므로 새 composition builder가 필요하다.
새 모델을 `inference.py`에만 추가하고 ZIP에는 baseline이 들어가는 오류를 차단한다.

각 보고서와 캐시는 source/checkpoint/data/label/split/config hash를 저장한다.
smoke marker는 ZIP SHA-256에 결합하고 새 패키징 시 이전 marker를 무효화한다.
ZIP 내부와 workspace의 세 CSV를 정렬 후 비교하고, 보존하기로 한 Stage/필드도 부모 예측과 비교한다.
두 추론은 같은 런타임의 별도 프로세스·추출 경로이며 독립적인 평가 서버 재현이라고 부르지 않는다.
오프라인 의존성·ZIP 크기·설치 및 추론 제한을 점검한다. 데이터 일부 smoke 시간은 전체 서버 시간 보장이 아니다.

코드 배포는 소형 patch만 사용한다. 기존 대형 base와 raw 데이터는 재업로드하지 않는다.
작은 노트북 변경은 셀 위치와 교체 코드를 함께 안내한다.

## 9. 근거

- 공개 라벨: `fixtures/baseline_data/stage1/labels.csv`, `stage2/labels.csv`, `stage3/labels.csv`.
- 구현 검토: `baseline_inference.py`, `tools/prepare_baseline_smoke.py`, `tools/train_stage3_hf.py`, `tools/go_no_go.py`.
- [Yejun Stage 2 보고](https://github.com/Jungminii-1114/blackbox_accident_analysis/issues/2), [Stage 3 보고](https://github.com/Jungminii-1114/blackbox_accident_analysis/issues/3): 팀원 관측값이며 독립 재현하지 않음.
- [OpenCV optical flow 설명](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html): 영상 내 운동과 카메라 운동이 모두 flow에 기여한다는 해석 근거.
- [comma2k19 원본 설명](https://github.com/commaai/comma2k19): 센서·영상 데이터 구조의 출처. 대회 범주 임계값의 출처는 아님.
