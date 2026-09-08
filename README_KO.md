# 작업자 이상 행동 감지 시스템

포즈 추정과 객체 추적을 활용한 실시간 규칙 기반 이상 행동 감지 시스템입니다. 모델 훈련 없이 YOLO11s-pose 키포인트에 생체역학적 규칙을 적용합니다.

**[English](README.md) | [O'zbek](README_UZ.md)**

---

## 감지 대상 행동

| 행동 | 설명 | 정확도 |
|---|---|---|
| **낙상 감지** | 작업자가 갑자기 넘어지는 경우 감지 | 92.4% |
| **위험한 달리기** | 제한 구역에서의 달리기 감지 | 91.0% |
| **장시간 무활동** | 5분 이상 움직이지 않는 작업자 감지 | 95.8% |

> 피험자별 교차 검증(LOOCV) 프로토콜로 평가

---

## 프로젝트 구조

```
worker-abnormal-behavior-detection/
│
├── fall_detection/               # 낙상 감지 모듈
│   ├── detector.py               # 감지 로직 (규칙)
│   └── evaluate.py               # 평가 스크립트
│
├── running_detection/            # 위험 달리기 모듈
│   ├── detector.py               # 감지 로직 (규칙)
│   └── evaluate.py               # 평가 스크립트
│
├── inactivity_detection/         # 장시간 무활동 모듈
│   ├── detector.py               # 감지 로직 (규칙)
│   └── evaluate.py               # 평가 스크립트
│
├── src/                          # 공유 핵심 모듈
│   ├── config.py                 # 임계값 및 설정
│   ├── pose_extractor.py         # YOLO11s-pose + ByteTracker
│   ├── feature_extractor.py      # 생체역학적 특징 추출
│   └── behavior_monitor.py       # 세 감지기 통합 관리
│
├── datasets/                     # 데이터셋 유틸리티
│   ├── npy_loader.py             # 사전 추출 키포인트 로드 (X.npy)
│   └── download_running.py       # KTH 데이터셋 다운로드
│
├── evaluation/
│   └── feature_utils.py          # 공유 특징 추출 도우미
│
├── main.py                       # 실시간 데모 진입점
├── requirements.txt
├── REPORT.md                     # 상세 기술 보고서
└── README.md / README_UZ.md / README_KO.md
```

---

## 시스템 동작 원리

```
CCTV / 카메라
      ↓
YOLO11s-pose  →  인물당 17개 관절 추출
      ↓
ByteTracker   →  작업자별 고유 ID 부여
      ↓
┌──────────────────┬──────────────────┬──────────────────┐
│ fall_detection/  │running_detection/│inactivity_       │
│ detector.py      │ detector.py      │detection/        │
│                  │                  │ detector.py      │
└──────────────────┴──────────────────┴──────────────────┘
      ↓
경보 (FALL | RUNNING | INACTIVITY)
```

### 낙상 감지 로직

**1단계 — 기울기 각도 측정:** 매 프레임마다 척추가 수직에서 얼마나 기울었는지 계산합니다 (0° = 직립, 90° = 수평).

**2단계 — 기울기 속도 측정:** 해당 각도가 얼마나 빠르게 변하는지 측정합니다 (도/초).

**판정:** 신체가 70° 이상 기울고 65°/초보다 빠르게 발생했다면 → **낙상 감지**

> 속도가 중요한 이유: 실제 낙상은 0.3~0.5초 내에 발생합니다 (매우 빠름). 의도적으로 눕는 동작은 3~5초가 걸립니다 (느림). 속도로 두 경우를 구분합니다.

---

### 위험한 달리기 로직

**1단계 — 수평 이동 추적:** 매 프레임마다 무게중심이 카메라 프레임 내에서 수평으로 이동하는 속도를 측정합니다.

**2단계 — 환경별 보정:** 임계값은 훈련 피험자 데이터로 카메라 환경에 맞게 보정됩니다.

**판정:** 수평 속도가 보정된 임계값을 초과하면 → **달리기 감지**

> 달리기는 카메라 프레임 내에서 걷기보다 약 2배 빠르게 이동합니다.

---

### 무활동 감지 로직

**1단계 — 정지 프레임 비율 계산:** 최근 프레임 중 신체 관절이 거의 움직이지 않은 비율을 계산합니다.

**2단계 — 자세 안정성 확인:** 신체 기울기 각도가 안정적인지도 확인합니다 (물건을 줍는 동작은 굽혔다 펴기 때문에 제외됩니다).

**3단계 — 타이머 시작:** 두 조건이 모두 충족되면 5분 카운트다운이 시작됩니다. 어떤 움직임이든 타이머를 초기화합니다.

**판정:** 작업자가 5분 연속으로 움직이지 않으면 → **무활동 경보 발생**

---

## 데이터셋

### UP-Fall Detection Dataset
- **출처:** Martinez-Velasco 외, *Data* 2019 — [https://sites.google.com/up.edu.mx/har-up/](https://sites.google.com/up.edu.mx/har-up/)
- **사용 피험자:** 4명 (Subject 1–4, 전체 17명 중)
- **카메라:** RGB, ~17fps, 실내 환경
- **총 윈도우:** 4,479개 (각 30프레임, 스트라이드 15)

| 활동 | 레이블 | 윈도우 | 용도 |
|---|---|---|---|
| Act 1–5 (낙상 5가지) | 낙상 | 629 | 낙상 평가 |
| Act 6 (걷기) | 활동 | 854 | 무활동 negative |
| Act 7 (서기) | 무활동 | 844 | 무활동 positive |
| Act 8 (앉기) | 무활동 | 834 | 무활동 positive |
| Act 9 (물건 줍기) | 활동 | 120 | 무활동 negative |

### KTH Action Dataset
- **출처:** Schuldt 외, *ICPR* 2004 — [https://www.csc.kth.se/cvap/actions/](https://www.csc.kth.se/cvap/actions/)
- **사용 피험자:** 25명 (전체)
- **카메라:** 측면 뷰, 25fps, 실내외 환경
- **총 클립:** 200개 (달리기 100개 + 걷기 100개)
- **클립 길이:** 약 15초 (150프레임 사용)
- **용도:** 달리기 감지 평가

### Train / Test 분할 (LOOCV)

LOOCV는 고정된 분할이 없습니다 — 각 피험자가 번갈아 테스트(홀드아웃) 대상이 되고, 나머지는 threshold를 fitting하는 데 사용됩니다. 아래 수치는 fold별 평균입니다. 낙상과 무활동은 **프레임** 시퀀스(포즈 keypoint 윈도우)로 평가되고, 달리기는 전체 **비디오 클립** 단위로 평가됩니다.

| 감지기 | 총 프레임 수 | Train (fold당 평균) | Test (fold당 평균) | Fold 수 |
|---|---|---|---|---|
| 낙상 | 69,150 프레임 (낙상: 10,320 / 정상: 58,830), 프레임 시퀀스 131개 기준 (낙상 59 + 정상 72) | ~51,863 프레임 | ~17,288 프레임 | 4 — UP-Fall 피험자별 1개 |
| 무활동 | 40,500 프레임 (무활동: 25,530 / 활동: 14,970), 프레임 시퀀스 48개 기준 (무활동 24 + 활동 24), 포즈 추정 윈도우 2,652개에서 그룹화됨 | ~30,375 프레임 | ~10,125 프레임 | 4 — UP-Fall 피험자별 1개 |

| 감지기 | 총 비디오 클립 수 | 처리된 프레임 수 | Train (fold당 평균) | Test (fold당 평균) | Fold 수 |
|---|---|---|---|---|---|
| 달리기 | 200개 비디오 클립 (달리기 100 + 걷기 100) | 30,000 프레임 (비디오 클립당 150프레임 상한; 클립 평균 521프레임) | ~100개 비디오 클립 | ~100개 비디오 클립 | 2 — 홀수/짝수 KTH 피험자 |

### 예시 테스트 데이터 폴더

위의 pre-extracted LOOCV 데이터 풀과 별도로, `data/`에는 **완전한, 단일 fold 테스트 세트**도 있습니다 — 실제 원본(raw) 데이터로 구성된, 하나의 LOOCV fold 전체의 실제 held-out(홀드아웃) 데이터이며, 단순 샘플이 아닙니다. 낙상과 무활동은 프레임 데이터입니다 — held-out된 Subject1의 원본 **프레임** 시퀀스(전체 11개 activity, Camera1만); 달리기는 held-out된 홀수 피험자 그룹을 전체 **비디오 클립**으로 사용합니다.

| 폴더 | 내용 | 대상 행동 | 이 fold의 결과 확인 |
|---|---|---|---|
| `data/test_upfall/` | Subject1의 완전한 fold — 프레임 시퀀스 33개에 걸친 **17,932 프레임** (낙상 activity 1–5: 2,832 프레임; 정상 activity 6–11: 15,100 프레임) | 낙상 | `python -m fall_detection.evaluate` — Subject1 행: **90.9%** (TN=15 FP=3 FN=0 TP=15) |
| `data/test_inactivity/` | Subject1의 완전한 fold — 프레임 시퀀스 12개에 걸친 **10,431 프레임** (무활동 Act7+8: 6,527 프레임; 활동 Act6+9: 3,904 프레임) | 무활동 | `python -m inactivity_detection.evaluate` — Subject1 행: **100.0%** (TN=6 FP=0 FN=0 TP=6) |
| `data/test_running/` | 홀수 피험자 그룹의 완전한 fold — **104개 비디오 클립** (KTH 피험자 13명) | 달리기 | `python main.py --source data/test_running/person01_running_d1_uncomp.avi` (시각적 확인), 또는 `python -m running_detection.evaluate`의 Fold 1 |

> 이 원본 폴더들은 수치 뒤에 실제 파일이 있다는 것을 눈으로 확인시켜 주기 위한 것입니다 — 정확도 자체는 위의 플레인 `evaluate` 명령(추가 플래그 없이)으로 산출되며, README의 수치를 재현하려면 바로 그 명령을 실행하면 됩니다.

> **git에 커밋되지 않음.** 이 폴더들은 총 ~10GB(원본 PNG 프레임이 용량이 큼)로, git 저장소가 감당할 범위를 훨씬 넘습니다. 그래서 `data/`는 계속 `.gitignore`에 남습니다. git이 아닌 **클라우드 드라이브**를 통해 공유되므로, `git clone` 후 자동으로 받아지지 않습니다 — 최신 링크를 요청하세요.

---

## 결과

| 감지기 | 정확도 | 데이터셋 | 평가 방법 |
|---|---|---|---|
| 낙상 | **92.40%** ± 3.4% | UP-Fall (피험자 4명) | LOOCV |
| 달리기 | **90.99%** ± 0.4% | KTH Action (피험자 25명) | LOOCV |
| 무활동 | **95.83%** ± 4.2% | UP-Fall (피험자 4명) | LOOCV |
| **평균** | **93.07%** | | |

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 평가

`data/upfall_npy/`(사전 추출된 keypoint, ~27MB)는 이 저장소에 커밋되어 있어 클론 후 낙상/무활동 평가가 바로 동작합니다. 달리기는 KTH 비디오 클립이 필요하며, 이는 git에 **커밋되어 있지 않습니다**(용량이 너무 큼) — 아래의 공개 원클릭 스크립트로 먼저 다운로드하세요.

```bash
# 낙상 감지  →  92.40%  (data/upfall_npy는 저장소에 이미 포함됨)
python -m fall_detection.evaluate

# 장시간 무활동  →  95.83%  (동일한 data/upfall_npy 사용)
python -m inactivity_detection.evaluate

# 위험한 달리기  →  90.99%
python -m datasets.download_running     # 1회성: kth.se에서 data/running_dataset (~290MB) 다운로드
python -m running_detection.evaluate    # ~30-60분 — YOLO가 200개 비디오 클립에서 실행됨
```

달리기 평가 전 KTH 데이터셋 다운로드:
```bash
python -m datasets.download_running
```

---

## 실시간 데모

```bash
# 웹캠
python main.py

# 영상 파일
python main.py --source video.mp4

# RTSP 스트림
python main.py --source rtsp://192.168.1.10/stream
```

---

## 핵심 기술

- **YOLO11s-pose** — 실시간 17관절 포즈 추정
- **ByteTracker** — 다중 인물 지속 ID 추적
- **Butterworth Filter** — 낙상 운동학 신호 평활화
- **규칙 기반 로직** — 모델 훈련 불필요, 완전한 해석 가능성
