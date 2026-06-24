# Ishchilarning Anormal Xatti-Harakatini Aniqlash

Ishchilarning anormal xatti-harakatini real vaqtda aniqlash tizimi. Pose estimation va ob'ektlarni kuzatish asosida ishlaydi. Model train qilish talab etilmaydi.

**[English](README.md) | [한국어](README_KO.md)**

---

## Aniqlanadigan Xatti-Harakatlar

| Xatti-Harakat | Ta'rif | Aniqlik |
|---|---|---|
| **Yiqilishni Aniqlash** | Ishchi to'satdan yiqilganda bildiradi | 92.4% |
| **Xavfli Yugurish** | Cheklangan zonada yugurish | 91.0% |
| **Uzoq Harakatsizlik** | 5+ daqiqa harakatsiz qolish | 95.8% |

> LOOCV (Leave-One-Out Cross-Validation) protokoli bilan baholangan.

---

## Loyiha Tuzilmasi

```
worker-abnormal-behavior-detection/
│
├── fall_detection/               # Yiqilish aniqlash moduli
│   ├── detector.py               # Aniqlash mantiqi (qoidalar)
│   └── evaluate.py               # Baholash skripti
│
├── running_detection/            # Xavfli yugurish moduli
│   ├── detector.py               # Aniqlash mantiqi (qoidalar)
│   └── evaluate.py               # Baholash skripti
│
├── inactivity_detection/         # Uzoq harakatsizlik moduli
│   ├── detector.py               # Aniqlash mantiqi (qoidalar)
│   └── evaluate.py               # Baholash skripti
│
├── src/                          # Umumiy modullar
│   ├── config.py                 # Barcha threshold va sozlamalar
│   ├── pose_extractor.py         # YOLO11n-pose + ByteTracker
│   ├── feature_extractor.py      # Biomexanik feature hisoblash
│   └── behavior_monitor.py       # 3 ta detektori boshqaradi
│
├── datasets/                     # Dataset yordamchilari
│   ├── npy_loader.py             # X.npy yuklash
│   └── download_running.py       # KTH datasetni yuklab olish
│
├── evaluation/
│   └── feature_utils.py          # Umumiy feature yordamchisi
│
├── main.py                       # Real-vaqt demo
├── requirements.txt
├── REPORT.md                     # Texnik hisobot
└── README.md / README_UZ.md / README_KO.md
```

---

## Tizim Qanday Ishlaydi

```
CCTV / Kamera
      ↓
YOLO11n-pose  →  Har bir odamdan 17 ta bo'g'im
      ↓
ByteTracker   →  Har ishchiga alohida ID
      ↓
┌──────────────────┬──────────────────┬──────────────────┐
│ fall_detection/  │running_detection/│inactivity_       │
│ detector.py      │ detector.py      │detection/        │
│                  │                  │ detector.py      │
└──────────────────┴──────────────────┴──────────────────┘
      ↓
Ogohlantirish (FALL | RUNNING | INACTIVITY)
```

### Yiqilish Aniqlash Mantiq
- **Tana burchagi** va **burchak o'zgarish tezligi** (°/sek) o'lchanadi
- Qoida: `tana_burchagi > 70° VA burchak_tezligi > 65°/sek`
- Yiqilish tez (74–140°/sek), ataylab yotish esa sekin (2–5°/sek)

### Xavfli Yugurish Mantiq
- **Markaziy massaning gorizontal tezligi** har kadrda kuzatiladi
- Qoida: `gorizontal_tezlik > kalibrlangan_chegara`
- Yugurish yurgandan ~2x tezroq harakatlanadi

### Harakatsizlik Mantiq
- **Sokin oynalar ulushi** va **poza barqarorligi** o'lchanadi
- Qoida: `sokin_ulush > 0.70 VA burchak_std < 3.5°`
- Taymer: 5 daqiqa uzluksiz harakatsizlikdan so'ng ogohlantirish

---

## Natijalar

| Detektor | Aniqlik | Dataset | Protokol |
|---|---|---|---|
| Yiqilish | **92.40%** ± 3.4% | UP-Fall (4 subject) | LOOCV |
| Yugurish | **90.99%** ± 0.4% | KTH Action (25 subject) | LOOCV |
| Harakatsizlik | **95.83%** ± 4.2% | UP-Fall (4 subject) | LOOCV |
| **O'rtacha** | **93.07%** | | |

---

## O'rnatish

```bash
pip install -r requirements.txt
```

---

## Baholash

Har bir detektor uchun alohida baholash:

```bash
# Yiqilish aniqlash  →  92.4%
python -m fall_detection.evaluate

# Xavfli yugurish  →  90.4%
python -m running_detection.evaluate

# Uzoq harakatsizlik  →  95.8%
python -m inactivity_detection.evaluate
```

KTH datasetni yuklab olish (yugurish baholashdan oldin):
```bash
python -m datasets.download_running
```

---

## Real-vaqt Demo

```bash
# Webcam
python main.py

# Video fayl
python main.py --source video.mp4

# RTSP oqim
python main.py --source rtsp://192.168.1.10/stream
```

---

## Asosiy Texnologiyalar

- **YOLO11n-pose** — Real vaqtda 17 ta bo'g'im aniqlash
- **ByteTracker** — Ko'p kishilik doimiy ID kuzatish
- **Butterworth Filter** — Yiqilish kinematikasi uchun signal tekislash
- **Rule-based Logic** — Model train qilinmagan, to'liq tushuntiriladi
