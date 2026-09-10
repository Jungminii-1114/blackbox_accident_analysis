# [Review] s3_hf_demo_v1 — Local +0.20542, Official -0.00377: 검증과 제출 gate 재설계

## 결과와 결정

comma2k19 CAN 파생 라벨로 MViTv2-S 분류 head를 학습한 결과, 외부 route validation에서는
개선됐지만 공식 Stage 3 점수는 소폭 하락했다. 이 후보를 새 기준 모델로 승격하지 않는다.
다음 실험은 함께 작성한 `docs/portfolio_v1_design.md`에 따라 세 Stage별 후보를 준비한다.

한 실험의 결과만으로 CAN supervision 전체가 실패했다거나 특정 도메인 차이가 원인이라고 확정하지 않는다.
이번 결과가 반박한 것은 이 후보의 공식 개선과 기존 GO 판정의 신뢰성이다.

## 1. 제출 관측값

근거: 사용자 제공 공식 제출 화면 및 Colab 실행 출력. 전체 평가 파일이나 공식 정답은 보유하지 않는다.

- 실험: `s3_hf_demo_v1`
- 제출 화면 번호: `86699`
- 제출 표시 시각: `2026-09-10 14:05:41` (화면 표기)
- 제출 파일: `submit.zip`
- 표시 소요시간: `14분 27초` (설치·대기·추론 세부 구분은 확인하지 못함)

| 지표 | 기준 | 후보 | 변화 |
|---|---:|---:|---:|
| 공식 Stage 1 | 0.5339789355 | 0.5339789355 | 0 |
| 공식 Stage 2 | 약 0.11721 | 0.1172182028 | 기존 표시 정밀도 차이로 정확한 변화 미확정 |
| 공식 Stage 3 | 약 0.14581 | 0.1420426225 | 약 -0.00376738 |
| 공식 가중 종합 | 약 0.21200 | 0.21050011722 | 약 -0.00150 |
| comma2k19 local Stage 3 | 0.14047706256921272 | 0.3458997366214684 | +0.20542267405225567 |

S3 상대 변화는 약 -2.58%다. S1 점수 동일과 S2 유사값은 변경 범위가 의도와 부합함을
뒷받침하지만 코드·가중치 hash 동등성까지 증명하지는 않는다.

용어: 이전 대화의 Local BSS는 local candidate-minus-baseline 차이였다. 코드·문서 일부에서는
BSS를 baseline score 의미로도 써서 혼동이 있었다. 이 리뷰에서는 Local Score / Local Delta,
Official Score / Official Delta로 구분한다. 사용자 출력에 BSS를 병기할 때에도 뜻을 명시한다.

## 2. 실행한 실험

- HF comma2k19 demo: 영상 64개, CAN Parquet 3개.
- HEVC 20Hz를 10Hz MP4로 변환하고 프레임 timestamp에 CAN 신호를 보간.
- 가감속 파생 라벨: 정지 ≤0.30 m/s, 가속도 경계 ±0.25 m/s².
- 조향 파생 라벨: 조향각 경계 ±3°, 양수 LEFT 가정.
- 원래 MViTv2-S checkpoint의 backbone 고정; acceleration/steering linear head만 학습.
- `--epochs 80 --feature-stride 2 --batch-size 6`.
- route 문자열 기준 train/validation 분리.
- 동일 validation 점수로 80 epoch 중 최고 checkpoint 선택.
- Stage 1·2는 baseline 유지 의도로 패키징.

`weights=None`으로 네트워크를 생성한 뒤 기존 checkpoint를 로드한다. checkpoint의 전체 사전학습
이력과 전이 가능한 motion 표현 수준은 이 결과만으로 입증되지 않았다. 모델 이름만으로
충분히 학습된 영상 backbone이라고 가정하면 안 된다.

## 3. 실행 검증과 발생했던 오류

Colab 기록상 첫 smoke의 출력은 S1=10행, S2=5행, S3=5992행이었다.
출력 schema/범주 검사와 ZIP의 문법·import 검사를 통과했다.

수정한 오류:

1. 원본 다운로드 경로의 파일 부재 → 다운로드 및 Drive 파일 저장 확인 로직 추가.
2. H.264 `639×480` 홀수 폭 → 짝수 크기 패딩과 임시 MP4 저장 추가.
3. ZIP 재추론의 `--inference-module` 옵션 오타 → `--inference`로 수정.
4. 이후 사용자가 최종 gate `GO`, `Failures: []`를 보고.

gate는 smoke marker 파일이 존재하면 `--smoke-pass`를 추가하는 구조다. 원격 파일 내용을
직접 재검증한 것은 아니다. 또한 두 smoke는 같은 Colab 환경의 별도 프로세스·추출 경로다.
설치된 패키지가 전혀 없는 별도 평가 서버 두 곳에서 실행한 것은 아니다.

로그에 남은 ZIP: 289.5 MiB, SHA-256
`1d16fb2643761094d5f35f4dbeb129d50013f0ebf04e0fe2647db9c205795dda`.
이 hash는 옵션 오류가 난 첫 패키징 로그의 값이다. 재패키징 후 실제 제출 ZIP과 동일한지는
최종 hash 기록이 없어 확인되지 않았다. 공식 제출 artifact의 확정 식별자로 사용하지 않는다.

## 4. 원인 분석: 확인된 사실과 가설

### 확인된 사실

- local 학습·평가 라벨은 CAN에서 자체 threshold로 생성했다. 대회의 실제 라벨 규칙과 동일하다는 증거가 없다.
- validation route는 head 학습 데이터와 분리했지만 최고 epoch 선택에 반복 사용했다. 최종 독립 test가 아니다.
- 특징 캐시는 split 이름과 stride로만 식별된다. checkpoint·라벨·데이터 변경 여부를 hash로 검사하지 않는다.
- CAN timestamp coverage ≥95% 검사는 시간 범위 겹침을 확인한다. frame-to-CAN 정렬 정확도나 부호를 보장하지 않는다.
- 공개 원본 라벨에서 sample 60은 frame 120에 대응한다. 기존 smoke의 raw-frame-as-sample 방식과 대조가 필요하다.
- 이번 공식 결과만으로 클래스별 퇴행, constant 예측 붕괴, 조향 부호 오류 여부를 판별할 수 없다.

### 우선 확인할 가설

| 후보 원인 | 근거 | 구분할 검사 |
|---|---|---|
| 자체 라벨과 공식 범주 불일치 | stop/accel/steer 경계와 부호를 자체 설정 | 실제 정의·단위·부호 검토, 공개 라벨 시각 대조 |
| 시간 축 불일치 | frame/sample 매핑 차이, 변환 후 frame_times[::2] 사용 | PTS·frame count·라벨 frame_index와 학습/추론 윈도우 대조 |
| domain shift | 해외 주행영상으로만 head 학습 | 국내 별도 영상과 comma2k19의 클래스별 오류 비교 |
| 영상 표현 부족 또는 클래스 편향 | backbone 고정, 적은 외부 경로 | head별 F1/support/confusion matrix, 예측 분포·학습 curve |
| validation 선택 편향 | 같은 split으로 80회 epoch 선택 | route별 held-out test 또는 outer grouped CV |
| 캐시/산출물 provenance 문제 | 캐시 hash 검사와 ZIP-marker 결합 부재 | 최종 checkpoint·cache·ZIP hash 및 생성 시각 감사 |

route holdout에서 pseudo-label을 예측하는 것은 유효한 보조 과제다. 라벨 생성 규칙이 같다는
이유만으로 데이터 누수나 무의미한 자기참조 학습이라고 규정하지 않는다. 문제는 그 점수를
숨은 공식 성능으로 직접 환산한 데 있다.

## 5. GO 판정에 대한 리뷰

기존 `tools/go_no_go.py`는 다음을 확인했다.

- `evaluation-kind=external_group_holdout`
- `--smoke-pass` 존재
- baseline/candidate Stage score 존재
- Stage 3 local delta ≥0.030

하지만 이전 이슈에는 구현보다 더 강한 검증이 적혀 있었다.

| 이전 설명 | 실제 구현 상태 |
|---|---|
| 가감속 개별 개선, 조향 붕괴 방지 | gate는 가중 score만 비교 |
| 독립 외부 holdout 검증 | CLI 문자열을 신뢰; 학습 노출·label provenance를 검사하지 않음 |
| Stage 1·2 동일성 확인 | schema 검사는 있지만 baseline 예측과 동등성 비교는 없음 |
| 현재 ZIP의 smoke 통과 | marker 존재 여부에 의존, ZIP hash 결합 없음 |
| 공식 점수 상승 가능성 수치화 | local delta의 25~75% 전이를 임의 가정 |

보고된 예상 종합 범위는 0.232546~0.273631이었고 실제는 0.210500이었다.
이 범위는 통계적 신뢰구간이 아니며 앞으로 사용하지 않는다.
관측된 S3 delta/local delta 비율은 약 -0.01834이지만, 한 쌍의 결과로 전이율 모델을 만들지 않는다.

어시스턴트가 local delta의 크기와 이 gate를 근거로 제출을 강하게 권장한 것은 과도했다.
일반화 불확실성을 명시하는 데 그치지 않고 실제 판정 로직과 검증 데이터 설계에 반영해야 했다.

## 6. 후속 결정

- 이 후보는 공식 baseline 대비 실패한 실험으로 보존하고 재제출하지 않는다.
- CAN 데이터·특징 캐시는 삭제하지 않는다. hash 감사 후 물리 신호 및 후속 비교에 활용한다.
- Stage 3는 시간 정렬된 flow 비교군과 회전 보조 특징을 독립 구현한다.
- Stage 1은 clip 수 증가, Stage 2는 collision-only 물리 신호로 별도 후보를 구성한다.
- 준비된 세 Stage를 한 ZIP에 결합한다. 공식 Stage별 점수로 각 후보를 판독한다.
- 실패한 Stage는 원래 baseline으로 되돌릴 수 있게 source/checkpoint/설정을 별도 보존한다.
- 함께 작성한 `docs/portfolio_v1_design.md`에 선택 기준과 실행 노트북 명세를 고정했다.

## 7. 추적할 작업

- [x] Local/Official score와 하락 사실 기록
- [x] GO 조건과 문서의 불일치 코드 검토
- [x] 다음 세 Stage 실험 설계 작성
- [ ] 최종 제출 ZIP/checkpoint/캐시 hash 확정
- [ ] `training_summary.json`의 head별 F1, epoch와 예측 분포 확인
- [ ] frame/sample/CAN timestamp audit
- [ ] threshold 단위·조향 부호 표본 검증
- [ ] 새 판정기 구현: 증거 수준과 실행 통과 분리, projected 공식 점수 제거
- [ ] ZIP hash에 smoke 결과 결합, 이전 marker 무효화
- [ ] 새 포트폴리오 노트북 구현 및 Colab 실행
- [ ] 다음 공식 결과를 Stage별로 기록

리뷰 문서 작성과 다음 설계는 완료됐다. 원인 규명과 새 구현은 위 미완료 작업으로 남긴다.

## 관련 자료

- [기존 실험 이슈 #5](https://github.com/Jungminii-1114/blackbox_accident_analysis/issues/5)
- [Yejun Stage 3 결과 #3](https://github.com/Jungminii-1114/blackbox_accident_analysis/issues/3): 팀원 보고 참고값; 이 실험의 직접 비교 holdout은 아님.
- 로컬 코드: `tools/train_stage3_hf.py`, `tools/prepare_stage3_hf.py`, `tools/prepare_baseline_smoke.py`, `tools/go_no_go.py`, `tools/package_stage3_candidate.py`.
