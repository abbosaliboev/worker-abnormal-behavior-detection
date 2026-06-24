# 작업자 이상 행동 감지 시스템

포즈 추정과 객체 추적을 활용한 실시간 규칙 기반 이상 행동 감지 시스템입니다. 모델 훈련 없이 YOLO11n-pose 키포인트에 생체역학적 규칙을 적용합니다.

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
│   ├── pose_extractor.py         # YOLO11n-pose + ByteTracker
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
YOLO11n-pose  →  인물당 17개 관절 추출
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
- **신체 기울기 각도** 및 **각속도** (°/초) 측정
- 규칙: `신체_각도 > 70° AND 각속도 > 65°/초`
- 낙상은 빠름(74–140°/초), 의도적 눕기는 느림(2–5°/초)

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

각 감지기를 개별적으로 평가:

```bash
# 낙상 감지  →  92.4%
python -m fall_detection.evaluate

# 위험한 달리기  →  90.4%
python -m running_detection.evaluate

# 장시간 무활동  →  95.8%
python -m inactivity_detection.evaluate
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

- **YOLO11n-pose** — 실시간 17관절 포즈 추정
- **ByteTracker** — 다중 인물 지속 ID 추적
- **Butterworth Filter** — 낙상 운동학 신호 평활화
- **규칙 기반 로직** — 모델 훈련 불필요, 완전한 해석 가능성
