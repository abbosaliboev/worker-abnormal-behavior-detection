# 작업자 이상 행동 감지 시스템

포즈 추정과 객체 추적을 활용한 실시간 규칙 기반 이상 행동 감지 시스템입니다. 모델 훈련 없이 YOLO11n-pose 키포인트에 생체역학적 규칙을 적용합니다.

**[English](README.md) | [O'zbek](README_UZ.md)**

---

## 감지 대상 행동

| 행동 | 설명 | 정확도 |
|---|---|---|
| **낙상 감지** | 작업자가 갑자기 넘어지는 경우 감지 | 92.4% |
| **위험한 달리기** | 제한 구역에서의 달리기 감지 | 90.4% |
| **장시간 무활동** | 5분 이상 움직이지 않는 작업자 감지 | 95.8% |

> 피험자별 교차 검증(LOOCV) 프로토콜로 평가

---

## 시스템 동작 원리

```
CCTV / 카메라
      ↓
YOLO11n-pose  →  인물당 17개 관절 추출
      ↓
ByteTracker   →  작업자별 고유 ID 부여
      ↓
┌──────────────┬─────────────────┬───────────────────┐
│ 낙상         │ 달리기          │ 무활동            │
│ 감지기       │ 감지기          │ 감지기            │
└──────────────┴─────────────────┴───────────────────┘
      ↓
경보 (FALL | RUNNING | INACTIVITY)
```

### 낙상 감지 로직
- **신체 기울기 각도** (수직 기준) 및 **각속도** (°/초) 측정
- 규칙: `신체_각도 > 70° AND 각속도 > 65°/초`
- 핵심: 낙상은 빠름(74–140°/초), 의도적 눕기는 느림(2–5°/초)

### 위험한 달리기 로직
- 프레임별 **무게중심의 수평 이동 속도** 추적
- 규칙: `수평_속도 > 보정된_임계값`
- 달리기는 걷기보다 약 2배 빠른 수평 이동

### 무활동 감지 로직
- **정지 프레임 비율** 및 **자세 안정성** 측정
- 규칙: `정지_비율 > 0.70 AND 각도_표준편차 < 3.5°`
- 타이머: 5분 연속 정지 시 경보 발생

---

## 결과

| 감지기 | 정확도 | 데이터셋 | 평가 방법 |
|---|---|---|---|
| 낙상 | **92.40%** ± 3.4% | UP-Fall (피험자 4명) | LOOCV |
| 달리기 | **90.42%** ± 1.9% | KTH Action (피험자 25명) | LOOCV |
| 무활동 | **95.83%** ± 4.2% | UP-Fall (피험자 4명) | LOOCV |
| **평균** | **92.88%** | | |

---

## 데이터셋

### UP-Fall Detection Dataset
- Martinez-Velasco 외, *Data* 2019
- 활동: 낙상(4가지 유형), 걷기, 서기, 앉기, 물건 줍기
- 용도: 낙상 및 무활동 평가

### KTH Action Dataset
- Schuldt 외, *ICPR* 2004
- 피험자 25명, 영상 200개 (달리기 100개 + 걷기 100개)
- 용도: 달리기 감지 평가

---

## 설치

```bash
pip install -r requirements.txt
```

**요구 사항:** Python 3.10+, PyTorch, Ultralytics YOLO, OpenCV, SciPy

---

## 사용법

### 실시간 데모
```bash
# 웹캠
python main.py

# 영상 파일
python main.py --source video.mp4

# RTSP 스트림
python main.py --source rtsp://192.168.1.10/stream

# 추적 없이 (단일 인물)
python main.py --no-tracking
```

### 평가
```bash
# 낙상 + 무활동 평가 (UP-Fall)
python -m evaluation.evaluate

# 달리기 평가 (KTH)
python -m evaluation.eval_running_kth_calibrated

# 무활동 전체 평가
python -m evaluation.eval_inactivity_full
```

### KTH 데이터셋 다운로드
```bash
python -m datasets.download_running_dataset
```

---

## 프로젝트 구조

```
├── src/
│   ├── config.py               # 임계값 및 설정
│   ├── pose_extractor.py       # YOLO11n-pose + ByteTracker
│   ├── feature_extractor.py    # 생체역학적 특징 추출
│   ├── fall_detector.py        # 낙상 감지 규칙
│   ├── running_detector.py     # 달리기 감지 규칙
│   ├── inactivity_detector.py  # 인물별 무활동 타이머
│   └── behavior_monitor.py     # 전체 감지기 조율
├── evaluation/
│   ├── evaluate.py                      # 낙상 + 무활동 LOOCV
│   ├── eval_running_kth_calibrated.py   # 달리기 LOOCV (KTH)
│   └── eval_inactivity_full.py          # 무활동 전체 평가
├── datasets/
│   ├── npy_loader.py                    # 사전 추출 X.npy 로드
│   └── download_running_dataset.py      # KTH 데이터셋 다운로더
├── main.py                              # 실시간 데모
├── requirements.txt
└── REPORT.md                            # 상세 기술 보고서
```

---

## 핵심 기술

- **YOLO11n-pose** — 실시간 17관절 포즈 추정
- **ByteTracker** — 다중 인물 지속 ID 추적
- **Butterworth Filter** — 낙상 운동학 신호 평활화
- **규칙 기반 로직** — 모델 훈련 불필요, 완전한 해석 가능성

---

## 주요 설계 결정

- **훈련 불필요** — 규칙은 생체역학에서 도출되고 데이터셋 통계로 보정
- **다중 인물** — ByteTracker로 각 작업자에게 독립적인 타이머 부여
- **캐스케이드 구조** — 낙상 → 달리기 → 무활동 순서로 검사
- **임계값 보정** — LOOCV 내 폴드별 보정으로 피험자 간 일반화 보장
