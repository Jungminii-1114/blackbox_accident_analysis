# portfolio_v1 실행 안내

## 지금 필요한 파일 두 개

1. `dacon236753_portfolio_patch.zip`: Google Drive 대회 루트에 업로드. 압축을 직접 풀지 않는다.
2. `04_portfolio_v1.ipynb`: Colab에서 열어서 GPU 런타임을 선택하고 위에서 아래로 실행.

기존 `dacon236753_colab_bundle.zip`은 Drive에 그대로 둔다. 다시 업로드할 필요 없다.
이전 `dacon236753_patch.zip`과 새 portfolio patch는 다른 파일이다. 이번 노트북은 새 파일만 읽는다.
기존 `01_stage3_can_lab`, `02_stage2_event_lab`, `03_stage1_forensics_lab`은 이번 실행에 필요 없다.

```text
/content/drive/MyDrive/블랙박스 영상 기반 사고 분석/
├── dacon236753_colab_bundle.zip       # 기존 대형 base, 그대로 유지
├── dacon236753_portfolio_patch.zip    # 이번 소형 코드 업데이트
├── 04_portfolio_v1.ipynb              # Colab에서 열 파일; 저장 위치 자체는 자유
└── experiments/
    └── portfolio_v1_<UTC시각>/
        ├── reports/                  # baseline / Stage별 / 시간 audit / 최종 판정
        ├── features/                 # 작은 flow/충돌 특징과 clip 확률
        ├── predictions/              # baseline, 후보, OOF, ZIP 검증 예측
        ├── config/                   # 선택된 Stage 구성
        ├── logs/                     # 셀별 실행 로그
        ├── submission/submit.zip     # GO + Drive 복사 hash 통과 시 생성
        └── review_results.zip        # 실행 후 어시스턴트에게 전달할 작은 결과 파일
```

## 셀 순서

| 셀 | 동작 | 실행 위치/계산 |
|---|---|---|
| 1 | Drive mount 확인, 새 experiment ID | Colab |
| 2 | 기존 base와 새 patch를 로컬 디스크로 추출, hash 검사 | Colab `/content/dacon236753_portfolio` |
| 3 | GPU·Torch/TorchVision·NMS·FFmpeg 검사 | 자동 대형 재설치 없음 |
| 4 | 실행·Drive 저장 함수, 공개 영상 10Hz 변환과 라벨 시간 검증 | CPU |
| 5 | 원래 세 Stage baseline 평가 | GPU; S3는 라벨 시점 50개만 |
| 6 | S3 A/B flow + nested video CV | CPU, 외부 데이터 불필요 |
| 7 | S2 차영상/운동 변화 후보 | CPU |
| 8 | S1 5-clip 평균/median 후보 | GPU |
| 9 | 통과 후보 조합, ZIP 생성, 2회 전체 예제 추론 비교 | GPU + CPU |
| 10 | GO/NO_GO, Drive 최종 ZIP 저장, 리뷰 ZIP 생성 | Colab → Drive |

시간은 GPU·스토리지 상태에 따라 달라 실제 측정값을 기록한다. 30초마다 경과시간을 출력한다.
Stage3 flow 자체는 GPU를 사용하지 않는다. 멈춘 것처럼 보여도 진행 로그/경과시간으로 확인한다.
코드와 영상은 `/content`에서 처리하고 작은 결과를 단계마다 Drive에 저장하여 작은 파일 I/O를 줄인다.
이 운영은 [Colab 공식 Drive I/O 안내](https://research.google.com/colaboratory/faq.html)를 따른다.

## 이번 실험에서 달라지는 것

- S1: 기존 checkpoint 고정, 3→5 clip. 평균이 주 후보, median은 진단용.
- S2: 충돌시점만 교체. 진입시점·방향·회피공간은 baseline과 동등성 검사.
- S3: 원본을 10Hz로 변환한 공개 예제에서 flow A/B를 비교. 정지·가감속·조향 threshold를
  inner group CV로 선택하고 outer 영상별 OOF를 생성한다. 특징·윈도우 함수는 제출 추론과 동일하다.
- 실제 공개 MP4의 컨테이너에는 약 480fps/2.5초가 기록돼 있다. 공개 라벨의
  `frame_index / time_seconds = 20`, `sample_index / time_seconds = 10` 관계를 모두 검사하고
  짝수 원본 프레임을 선택해 10Hz timestamp를 부여한다. 해당 복원은 공개 fixture에만 적용한다.
  평가 추론은 공식 10Hz decoded-frame 규약으로 모든 프레임을 출력하며 메타데이터 FPS만으로
  프레임을 삭제하지 않는다. FPS 불일치는 warning을 출력한다.
- pseudo-label 80-epoch head 학습이나 comma2k19 재다운로드는 하지 않는다. CAN 부호/상관 진단은
  이번 최소 독립 실험의 필수 셀에서 제외했다. 실제 조향 부호 일반화는 여전히 검증 과제다.
- 원래 Stage3 checkpoint hash를 고정하므로 앞선 실패 후보로 잘못 출발하지 않는다.

## 점수/판정 읽기

- `local_BSS_delta` = 후보 local score − baseline local score. Brier Skill Score가 아니다.
- S1은 public regression, S3 후보는 nested OOF이나 baseline은 이 예제에 학습 노출됨. 깨끗한 외부 비교가 아니다.
- S2는 세 목표 정답이 없어 전체 Local Score/BSS는 null(N/A). 충돌 Accuracy@0.3초를 별도 출력.
- Official baseline: S1 0.5339789355, S2 약 0.11721, S3 약 0.14581.
- 새 실험 Official candidate/BSS는 제출 전 null(N/A). 임의 전이율·상승 확률을 출력하지 않는다.
- `selected_fit_metrics.json`은 최종 공개 데이터 적합/회귀 점수. S3 OOF보다 우선해 모델을 선택하면 안 된다.
- 후보가 공개 회귀 조건을 통과하지 못하면 해당 Stage를 baseline으로 되돌린다. 전부 baseline이면 NO_GO.
- `GO, purpose=exploration`은 검증된 탐색 제출물이다. 점수가 오른다는 보장이 아니다.
- 오류/누락/hash 불일치가 있으면 NO_GO. `submission_gate.json`과 `review_results.zip`을 공유한다.

## 재개·작은 수정

- 같은 런타임에서 에러를 고쳤고 코드가 안 바뀌었다면 실패 셀부터 재실행할 수 있다.
- 런타임이 초기화됐다면 셀 1의 `EXPERIMENT_ID`를 이전 폴더명으로 지정하고 1부터 실행한다.
  보존된 보고서를 복원하고 fixture를 재생성한다. 인코더 변경 등으로 데이터 hash가 바뀌면 새 ID를 사용한다.
- patch/code가 바뀌면 새 experiment ID를 쓴다. 이전 캐시를 섞지 않는다.
- 다른 Drive 경로를 사용할 때는 셀 1의 `DRIVE_ROOT` 한 줄만 수정한다.
- 대형 base는 수정하지 않는다. 차후 코드 변경은 이 소형 patch 교체로 배포한다.

## 검증 범위

배포 전 로컬에서는 CPU 단위검사·실제 공개 영상 시간 정렬·flow·CV·노트북 문법을 확인한다.
Colab에서는 GPU import/NMS, 원본 checkpoint 로드, workspace/ZIP 재추론, 결과 동등성,
범주·ID·프레임 범위·10Hz sample 전수 검사, ZIP CRC/크기/hash를 실행한다.
동일 Colab의 두 프로세스는 서로 다른 평가 서버가 아니다. L40S 전체 비공개 입력의 실행시간과
완전히 깨끗한 오프라인 서버 설치까지 로컬에서 보장할 수 없으므로 그 한계를 최종 보고서에 남긴다.
