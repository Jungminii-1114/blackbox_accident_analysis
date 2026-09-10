# focus12_v2 — 바로 실행하는 방법

## 새 파일 두 개만 사용

1. `dacon236753_focus12_patch.zip`을 Drive의 **블랙박스 영상 기반 사고 분석** 루트에 업로드한다.
2. `05_focus12_v2.ipynb`를 GPU Colab에서 열고 위→아래로 실행한다.

기존 base ZIP과 다음 부모 파일은 그대로 유지한다. 재업로드할 필요 없다.

```text
블랙박스 영상 기반 사고 분석/
├── dacon236753_colab_bundle.zip       # 기존 base
├── dacon236753_focus12_patch.zip      # 새 소형 코드 ZIP
└── experiments/
    ├── portfolio_v1_20260910T063010Z/
    │   └── submission/submit.zip     # 이번 공식 성적을 낸 부모; 삭제/교체 금지
    └── focus12_v2_<UTC시각>/
        ├── reports/
        ├── features/
        ├── predictions/
        ├── config/
        ├── logs/
        ├── submission/submit.zip     # 최종 GO + 복사 hash 검사 통과 시 저장
        └── review_results.zip        # 완료 후 어시스턴트에게 전달
```

원본 `04_portfolio_v1` 및 이전 Stage별 노트북은 다시 실행하지 않는다.
새 작업은 `/content/dacon236753_focus12`에서 처리하므로 기존 실행 디렉터리를 덮어쓰지 않는다.

## 셀 순서

| 셀 | 하는 일 |
|---|---|
| 1 | Drive FUSE mount 확인, 부모 ZIP 경로, 새 experiment ID |
| 2 | base/새 patch 추출, 부모 ZIP·코드 hash 고정 |
| 3 | 기존 GPU/Torch/TorchVision/OpenCV 환경 검사. 자동 대형 재설치 없음 |
| 4 | 공개 fixture 시간 대응 검증, 단계별 로그·Drive 저장 함수 |
| 5 | 부모 전체 재추론 + 원래 S1 3-clip 비교군 |
| 6 | S1 frozen ResNet18 + ridge A/B + source-pair nested CV + JPEG90 |
| 7 | S2 E/C/D vs 부모 B + 정확도/MAE/JPEG95·90 안정성 |
| 8 | 선택 모델 ZIP, import/스키마/2회 추론/부모 S3 동등성 검사 |
| 9 | GO/NO_GO, Local/Official 수치, Drive submit.zip 및 review_results.zip |

새 학습은 S1의 매우 작은 ridge head만 수행한다. S3 flow/임계값 재학습 및 CAN 다운로드는 없다.
30초마다 경과시간이 출력된다. 시간은 런타임 상태에 따라 달라지며 보장하지 않는다.

## 가설과 현실적인 기대

- **S1 A:** 전체 프레임 ResNet18 평균 특징으로 원본/재촬영을 분류한다.
- **S1 B:** 전체/중앙 crop 특징 차이도 사용해 화면 주변 단서를 보강한다.
  기존 모델이 공개 영상 전부 ORIGINAL로 예측한 한계를 직접 겨냥한다.
  기존 baseline ResNet18 가중치를 재사용하며 새로운 외부 모델 다운로드가 없다.
  모델 생성의 `weights=None` 자체를 사전학습의 증거로 보지 않는다. 기존 가중치의 완전한 이력은 별도 확인 사항이다.
- **S2 C:** 전역 밝기 차이를 제거한 프레임 변화 + phase shift 순위 점수. RANSAC 의존과 z-score clipping 동률을 피하는 가설.
- **S2 E(우선 후보):** 공식 B가 찾은 충돌 구간을 유지하고 앞뒤 2개 프레임 안에서 노출 보정 차영상으로 정밀화한다.
  JPEG 민감도 검사에서는 B의 기준 위치부터 다시 계산해, 기준 위치를 고정해서 안정성을 부풀리지 않는다.
- **S2 D:** 한 프레임만 튀는 변화보다 이웃 프레임의 지지를 사용하는 고정 점수.
- **S3:** 공식 0.57137을 낸 부모의 source prefix/설정을 보존한다. 새 후보의 S3 출력이 부모와 다르면 NO_GO.

S1은 target-like 5쌍밖에 없고 S2도 충돌 라벨 5건뿐이다. 높은 성공 확률을 숫자로 단정할 근거는 없다.
S1은 명확한 OOF 개선 없으면 3-clip 복구, S2는 공식 개선된 B를 이기는 후보가 없으면 B 유지가 기본이다.
로컬 CPU 사전 검사에서 첫 S1 가설의 절차 OOF는 약 0.495로 채택 기준 0.60에 미달했다.
실행 성공을 성능 성공으로 포장하지 않으며, Colab에서도 미달하면 자동 복구한다.
이번 S2 모델들은 충돌시점만 개선 대상으로 삼으며, 정답 없는 진입/방향/회피공간은 변경하지 않는다.

참고: [ResNet18 공식 구조·전처리](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
[OpenCV 영상 운동의 의미와 한계](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html).

## 검증기에서 달라진 점

- Stage 2 Accuracy@0.3초와 시간/프레임 MAE를 같이 출력한다. 시간표/FPS가 없으면 시간 점수는 N/A다.
- Stage 3 희소 라벨에 없는 같은 영상의 정상 예측 행은 `unlabeled_predictions_same_video`로 구분한다.
  “라벨에 없는 2,948행”을 출력 오류라고 해석하지 않는다. 실제 행 수는 별도의 expected schema로 검사한다.
- S1 검증 절차 OOF와 최종 선택 계열 OOF를 분리해 표시한다. 전체 적합 점수로 GO를 결정하지 않는다.
- Local BSS는 candidate-minus-reference score다. S2 전체 점수는 정답 부족으로 N/A이며 충돌 지표만 표시한다.
- Official 부모는 S1 0.52121 / S2 0.16320 / S3 0.57137(화면 반올림값)이다.
  새로운 Official candidate/BSS/상승 확률은 제출 전 N/A다.
- 부모를 포함해 파일별 독립 추론을 유지한다. 비공개 파일을 모아 임계값을 다시 맞추지 않는다.
- 후보가 모두 미달해도 S1의 기존 우수 3-clip 복구 + 이번 S2/S3 유지 조합을 생성할 수 있다.
  이때 GO는 `best_known_recombination`이며 실제 조합 점수는 아직 관측되지 않았다.

## 오류 또는 재개

- 부모 ZIP hash 불일치: 마지막에 제출한 파일과 지정 경로가 맞는지 확인. 검사를 지우지 않는다.
- 런타임 초기화: 셀 1의 `EXPERIMENT_ID`를 이전 폴더명으로 지정하고 위부터 실행한다.
- 코드/패키지 버전이 바뀌면 새 experiment ID 사용. 다른 실험 캐시를 섞지 않는다.
- 중간 실패 시 이후 제출 셀을 진행하지 말고 마지막 셀로 NO_GO 보고서를 만든다([4] 실행 이후).
- 실행 후 `review_results.zip`을 공유한다. 모델/영상 전체를 다시 전송할 필요 없다.

이번 노트북은 전체 실험 흐름이 달라진 **새 노트북**이므로 기존 셀 1~2개를 교체하는 방식이 아니다.
경로만 바꾸려면 셀 1의 `DRIVE_ROOT`, `PARENT_EXP`만 수정하면 된다.

GPU 전체 추론과 평가 서버 인터넷 비활성화·L40S 전체 실행시간을 Mac CPU 테스트로 보장하지 않는다.
Colab에서 실제 ZIP 추론을 두 번 검사하고, 서버 전체 시간은 별도 미검증 항목으로 남긴다.
자동 제출은 하지 않는다. 리더보드의 제출수 3이 당일 사용량이라는 뜻인지는 확인되지 않았다.
하루 3회 한도 내 남은 횟수는 제출탭에서 확인한다.
