# 블랙박스 사고 분석 모델 설계와 실험 우선순위

## 결정 요약

가장 먼저 검증할 대상은 **Stage 3: 전방 영상→가감속·조향**이다. 공개 데이터에서
영상과 CAN 신호를 동시에 확보할 수 있고, 두 범주를 프로그램으로 일관되게 생성해
학습·검증을 분리할 수 있기 때문이다. 첫 Public 제출은 Stage 3만 바꾸고 Stage 1·2는
기준선을 고정한다. 그 다음 제출에서 Stage 2 충돌시점 모델만 바꾼다.

현재 기준선은 S1=0.53397, S2=0.11721, S3=0.14581이다. 종합 기여는 각각 0.10679,
0.04688, 0.05832이다. S2와 S3은 동일하게 0.4의 가중치를 가지므로, 둘 중 데이터
라벨을 더 신뢰성 있게 만들 수 있는 Stage 3을 먼저 선택한다.

## 고정 가정과 검증 항목

| ID | 가정 | 위험 | 검증 또는 완화 |
|---|---|---|---|
| A1 | Private 영상의 장비·해상도·도로 환경은 외부 데이터와 다르다. | 도메인 갭 | 비율 유지 crop, 압축·야간·비·흔들림 증강, 출발지/경로 단위 validation |
| A2 | Stage 3의 숨은 범주는 CAN 차속·종가속도·조향 신호로 생성된다. | 임계값은 비공개 | 속도 미분·가속도·조향각 세 채널로 후보 라벨을 만들고 validation에서만 선택 |
| A3 | Stage 3 입력은 10Hz이며 모든 sample을 출력해야 한다. | 누락 시 제출 실패 | 매 영상의 예상 sample 목록으로 CSV completeness 검사 |
| A4 | Stage 2의 충돌은 실제 접촉이며, 위험·근접 시점이 아니다. | 조기 이벤트 예측 | 충돌 전후 temporal window와 객체 거리 변화량을 함께 학습 |
| A5 | Stage 2 진입은 차량의 첫 등장 대신 바퀴의 차선 침범이다. | 공개 데이터 직접 라벨 부족 | detector/tracker/lane model pretrain + 소량 정밀 라벨링 |
| A6 | Stage 1의 재녹화는 실제 화면 재촬영 artefact를 포함한다. | 합성 artefact 과적합 | 원본 영상 단위 그룹 분리, 실제 재촬영 소량 확보, 강한 합성 다양화 |
| A7 | 제출은 인터넷 없이 L40S에서 60분 내 실행해야 한다. | 가중치 누락·시간 초과 | 모든 모델을 ZIP에 포함, smoke 2회 실행, Stage별 시간 로그 |
| A8 | Public 제출 두 번은 실험 데이터가 아니라 검증 관측값이다. | Public 과적합 | 한 제출당 한 Stage·한 변경, local holdout을 모델 선택 기준으로 유지 |

## 데이터 후보와 사용 판단

| 목적 | 후보 | 제공 정보 | 적합도 | 사용 전 조건 |
|---|---|---|---|---|
| Stage 3 주 학습 | comma2k19 | 전방 video timestamp, CAN 차속·조향각·휠속도, IMU | 높음 | 100GB 규모이므로 우선 1개 chunk/route group만 받고 데이터·라이선스 기록 |
| Stage 3 보강 | A2D2 | 동기화 camera·vehicle bus, speed/accel/steering 등 | 높음 | CC BY-ND 4.0 해석과 대회 목적 적합성을 별도 확인하고 출처 기록 |
| Stage 2 지각 pretrain | BDD100K | 영상, 차선·주행가능 영역·탐지·추적 라벨 | 중간 | 데이터 자체의 별도 라이선스 동의·기록 필요 |
| Stage 2 충돌 event | CCD, DADA-2000 | 실제 dashcam 사고와 사고 이벤트/상황 annotation | 중간 | 각 데이터의 download/license를 직접 확인, 충돌 프레임 정의를 재검수 |
| Stage 2 희귀 시나리오 | CARLA | camera, lane invasion, collision event, actor 상태 완전 관측 | 보조 | 합성→실사 도메인 갭을 인정하고 real 데이터 fine-tune 전제로만 사용 |
| Stage 1 | 자체 생성 screen recapture | 화면 반사·moiré·rolling banding·keystone·compression | 중간 | 사용 허용 원본으로 물리 재촬영하고 원본 clip 단위 분리 |

comma2k19는 약 33시간의 주행, 전방 camera frame timestamp 및 CAN 차속·조향각을
제공한다.^1 A2D2는 camera와 자동차 버스 데이터를 동기화해 제공한다.^2 BDD100K는
lane/drivable area/detection/tracking을 포함하므로 Stage 2 지각 모듈의 pretrain에는
유용하지만, 사고 네 라벨을 직접 주지는 않는다.^3 CARLA는 충돌 이벤트와 차선 침범
sensor를 제공해 target label을 완전하게 만들 수 있다.^4

## Stage별 설계 후보

### Stage 3 — 우선순위 1

**V1: 영상 temporal encoder + dual head**

- 입력: 현재 sample에서 끝나는 16프레임(1.6초), 전방 도로 crop. 화면 하단 timestamp,
  로고, 속도 오버레이가 있다면 별도 실험으로 mask한다.
- backbone: MViTv2-S 또는 경량 2D encoder+TCN. 초기 비교는 현재 제출과 가중치 호환성이
  좋은 MViTv2-S로 한다.
- heads: accel 4-class, steer 3-class. class-balanced CE 및 label smoothing 0.02만 사용한다.
- split: frame 단위가 아니라 route/drive/date 단위 group split. 동일 연속 주행이 train과
  validation에 섞이면 성능이 과대평가된다.
- postprocess: validation에서만 3~5 sample median/transition smoothing 유무를 비교한다.
  Public 점수를 보고 smoothing 길이를 고르지 않는다.

**라벨 생성 후보**

1. STOPPED: smoothed speed ≤ `v_stop`.
2. ACCELERATING/DECELERATING: smoothed longitudinal acceleration 또는 speed derivative의
   부호와 절대값이 임계치 초과.
3. CONSTANT: 나머지 주행 sample.
4. LEFT/RIGHT/STRAIGHT: steering angle의 부호·절대값 또는 yaw-rate로 결정.

임계값은 숨은 정답과 다를 수 있으므로 단일 상수가 아니라 YAML experiment matrix로
관리한다. comma2k19/A2D2 간 label agreement를 먼저 측정하고, agreement가 낮으면
dataset별 calibration 또는 soft target을 시도한다.

**성공 가능성: 높음(55–70%)**

여기서 성공은 “기준선보다 Stage 3 Public 점수가 의미 있게 상승”할 가능성의 전문가
추정치다. CAN-supervision 자체는 강하지만, 국가·카메라 위치·비공개 임계값의 도메인 갭이
상한을 제한한다. 목표는 첫 실험에서 Stage 3 절대 +0.10 이상이며, 종합점수로는 +0.04다.

### Stage 2 — 우선순위 2

**V1: event proposal + temporal refinement**

- 1단계: 모든 프레임의 scene-change, object-track relative motion, detector confidence로
  충돌 후보 top-K를 만든다.
- 2단계: 후보 전후 2~4초 clip을 temporal model로 재순위화한다.
- 출력 collision frame은 반드시 실제 파일명 frame number 중 하나로 snap한다.

**V2: 피해차량 track과 차선 침범**

- vehicle detector+tracker로 충돌 상대 track을 역추적한다.
- lane/drivable-area model로 ego lane boundary를 추정한다.
- 피해차량 wheel/bottom-center가 경계에 최초 닿는 frame을 entry로 정의한다.
- entry_side는 track의 최초 ego-lane 침범 전 screen x 위치로 계산한다.
- evasion_space는 충돌 frame에서 drivable mask, road edge, 주변 object occupancy를 결합한
  별도 binary head로 둔다.

CCD에는 실제 dashcam 충돌 영상과 사고 관련 annotation이 있고,^5 DADA-2000은 2,000 clips,
54 사고 유형의 사고·driver attention annotation을 제공한다.^6 그러나 둘 다 대회 정의의
entry/evasion을 완전히 해결하지 못한다. 따라서 최소 150–300개 clip의 정밀 라벨링 또는
CARLA scenario pretrain이 필요하다.

**성공 가능성: 중간(40–55%; 충돌만) / 중간 이상(50–65%; 정밀 라벨 확보 시)**

충돌시점만 개선하는 V1은 빠르지만 Stage 2의 35%에만 직접 기여한다. 네 과업을 동시에
올리려면 라벨링 시간이 핵심 비용이다.

### Stage 1 — 우선순위 3

**V1: multi-clip forensic classifier**

- 전체 영상에서 시간적으로 분리한 3~8 clip을 추출한다.
- RGB temporal encoder와 high-frequency forensic branch(FFT residual, demosaic/moire cue)를
  late fusion한다.
- 실제 재촬영: 다른 display/밝기/거리/각도/셔터로 촬영한 소량 data.
- 합성: homography, scanline/banding, screen glare, rolling shutter, moiré, resampling,
  compression을 조합하고 원본 scene 단위 group split.

**성공 가능성: 중간 이하(35–50%)**

현재 점수는 가장 높고 가중치는 0.2다. 실제 재촬영 data 없이 합성만 늘리면 생성 방식에
과적합될 위험이 크므로, S2/S3보다 뒤로 둔다.

## 피해야 할 설계

- Public 제출의 파일 순서·ID·다른 영상 통계를 사용하는 규칙: 대회 규칙 위반 위험.
- 비공개 영상을 pseudo-labeling하거나 제출 후 모델을 업데이트하는 방식: 금지.
- 한 제출에서 모든 Stage와 backbone을 동시에 교체: 원인 식별 불가.
- 대형 VLM API/모델을 inference에 의존: 인터넷 차단, ZIP 용량·60분 제한과 충돌.
- CCTV 사고 데이터만으로 dashcam Stage 2를 학습: ego-motion 부재로 domain gap이 큼.

## 제출 운영

| 순서 | 제출 내용 | 판단 기준 |
|---|---|---|
| 기준선 | 이미 완료 | S1=.53397, S2=.11721, S3=.14581 기록 |
| 제출 A | Stage 3 V1만 교체; S1/S2 기준선 유지 | local route-holdout 개선, ZIP smoke 2회 PASS |
| 제출 B | A의 Stage 3을 유지하고 Stage 2 collision V1만 교체 | local event-holdout 개선, CSV/frame 범위 PASS |
| 이후 | 정밀 Stage 2 labels 또는 Stage 1 recapture | 각 Stage별 holdout 개선 후만 제출 |

## 실험 노트북 설계

`01_stage3_can_lab.ipynb`은 comma2k19 공식 HF demo의 Parquet와 64개 영상을 직접 연결해
10Hz manifest/라벨을 생성한다. route-group 누수를 검사한 뒤 기존 MViTv2 backbone 특징에서
두 분류 head를 학습하고, 동일 holdout의 baseline/candidate BSS를 기록한다. 후보 checkpoint는
Stage 1·2 baseline을 유지한 제출 ZIP으로 패키징하며 두 번의 smoke 추론이 통과해야 마지막
`GO/NO_GO` 셀이 제출을 허용한다.

## Sources

1. comma.ai. [comma2k19](https://github.com/commaai/comma2k19). Camera timestamps, CAN car speed and steering angle, dataset structure, MIT repository license.
2. Audi. [A2D2: Audi Autonomous Driving Dataset](https://a2d2-dataset.github.io/). Synchronized cameras and automotive bus data; CC BY-ND 4.0 terms.
3. BDD100K. [Dataset repository](https://github.com/bdd100k/bdd100k). Video, lane, drivable-area, detection and tracking tasks. Dataset license must be separately reviewed.
4. CARLA. [Sensor reference](https://carla.readthedocs.io/en/latest/ref_sensors/). Collision event frame/timestamp and lane-invasion sensor; [license](https://carla-driving-simulator-client.readthedocs.io/en/stable/license.html).
5. Cogito Lab. [Car Crash Dataset](https://github.com/Cogito2012/CarCrashDataset). Real dashcam crash corpus and annotations; verify data terms before use.
6. Fang et al. [DADA-2000](https://arxiv.org/abs/1904.12634). 2,000 accident clips and 54 accident categories; verify download terms before use.
