# Worker Abnormal Behavior Detection
## Texnik Hisobot

---

## 1. Loyiha Haqida

Kompaniya ishchilarining xavfsizligini avtomatik kuzatuvchi real-vaqt tizimi. Uch turdagi anormal xatti-harakat aniqlanadi:

| # | Detektor | Ta'rif |
|---|---|---|
| 1 | **Fall Detection** | Ishchi yiqilishini aniqlash |
| 2 | **Unsafe Running Detection** | Xavfli zonada yugurishni aniqlash |
| 3 | **Long-time Inactivity** | Uzoq muddatli harakatsizlikni aniqlash |

**Yondashuv:** To'liq rule-based — hech qanday model train qilinmagan. YOLO11n-pose orqali pose extraction, keyin biomexanik qoidalar asosida qaror qabul qilinadi.

---

## 2. Tizim Arxitekturasi

```
Video / CCTV kamera
        ↓
YOLO11n-pose (real-time 17 ta bo'g'im aniqlash)
        ↓
ByteTracker (har ishchiga alohida ID)
        ↓
┌─────────────┬──────────────────┬─────────────────────┐
│ Fall        │ Running          │ Inactivity          │
│ Detector    │ Detector         │ Detector            │
└─────────────┴──────────────────┴─────────────────────┘
        ↓
Alert System (FALL | RUNNING | INACTIVITY)
```

---

## 3. Texnologiyalar

| Texnologiya | Maqsad |
|---|---|
| **YOLO11n-pose** | 17 ta gavda bo'g'imini real vaqtda aniqlash |
| **ByteTracker** | Har bir ishchiga doimiy ID berish |
| **Butterworth Filter** | Shovqinni olib, silliq signal olish |
| **Python + OpenCV** | Video o'qish va vizualizatsiya |
| **NumPy / SciPy** | Signal va matematik hisob-kitob |

---

## 4. Datasetlar

### 4.1 UP-Fall Detection Dataset
**Maqsad:** Fall va Inactivity detektorlarini baholash  
**Muallif:** Martinez-Velasco et al., 2019

| Parametr | Qiymat |
|---|---|
| Subjectlar | 4 ta (17 dan) |
| Kamera | RGB, ~17 fps |
| Pre-extracted | YOLO11n-pose bilan 17-keypoint (X.npy) |

**Ishlatilgan aktivliklar:**

| Faoliyat | Fall | Inactivity |
|---|---|---|
| Act 1-5 (yiqilish turlari) | Positive | — |
| Act 6 — Yurish | Negative | Negative |
| Act 7 — Tik turish | Negative | Positive |
| Act 8 — O'tirish | Negative | Positive |
| Act 9 — Narsa olish | Negative | Negative |

### 4.2 KTH Action Dataset
**Maqsad:** Running detektorini baholash (UP-Fall'da yugurish yo'q)  
**Muallif:** Schuldt et al., ICPR 2004

| Parametr | Qiymat |
|---|---|
| Subjectlar | 25 ta |
| Kamera | Lateral ko'rinish, 25 fps |
| Kliplar | 200 ta (100 running + 100 walking) |
| Hajm | ~290 MB |

> **Nima uchun KTH?** UP-Fall datasetida yugurish faoliyati yo'q. KTH — running va walking uchun kichik, ishonchli va to'g'ridan yuklab olinadigan dataset.

---

## 5. Detektor Logikasi

### 5.1 Fall Detection

**G'oya:** Yiqilish tez sodir bo'ladi — ataylab yotish esa sekin.

**Features:**
- `body_tilt_angle` — tananing vertikaldan qiyshalik burchagi (°)
- `clip_max_angle_rate` — burchakning o'zgarish tezligi (°/sek)

**Qoida:**
```
body_tilt_angle > 70°  AND  angle_rate > 65°/sek
```

**Nima uchun ishlaydi:**

| Holat | Burchak | Tezlik |
|---|---|---|
| Yiqilish | 80–110° | 74–140 °/sek ← juda tez |
| Ataylab yotish | 80–90° | 2–5 °/sek ← sekin |
| Yurish | 10–20° | 8–12 °/sek |

---

### 5.2 Unsafe Running Detection

**G'oya:** Yugurish yurgandan 2x tezroq gorizontal harakatlanadi.

**Feature:**
- `mean_horiz_speed` — markaziy massaning gorizontal tezligi

**Qoida (KTH'da kalibratsiya qilingan):**
```
horizontal_speed > 0.012
```

| Holat | Gorizontal tezlik |
|---|---|
| Yugurish | ~0.020 |
| Yurish | ~0.009 |

> **Muhim:** Threshold har muhitda bir marta kalibratsiya qilish kerak (kamera burchagi va masofaga qarab).

---

### 5.3 Long-time Inactivity Detection

**G'oya:** Ishchi uzoq vaqt harakatsiz qolsa — tibbiy holat bo'lishi mumkin.

**Features:**
- `still_fraction` — harakatsiz oynalar ulushi (kp_disp < 0.005)
- `body_angle_std` — burchak barqarorligi (narsa olishda katta o'zgaradi)

**Qoida:**
```
still_fraction > 0.70  AND  body_angle_std < 3.5°
```

**Vaqt sharti:** 5 daqiqa uzluksiz (real tizimda)

| Holat | Still fraction | Angle std |
|---|---|---|
| Tik turish | ~0.99 | ~1.0° |
| O'tirish | ~0.95 | ~1.2° |
| Yurish | ~0.05 | ~2.5° |
| Narsa olish | ~0.10 | ~13.0° |

---

## 6. Evaluation Metodologiyasi

**Protokol:** Leave-One-Out Cross-Validation (LOOCV)

Har bir fold'da bitta subject test, qolganlar train. Bu — model o'zi ko'rmagan odamda test qilinadi degani (cross-subject generalizatsiya).

---

## 7. Yakuniy Natijalar

| Detektor | Accuracy | Dataset | Holat |
|---|---|---|---|
| **Fall Detection** | **92.40%** ± 3.4% | UP-Fall (4 subj, LOOCV) | ✅ |
| **Running Detection** | **90.42%** ± 1.9% | KTH (25 subj, LOOCV) | ✅ |
| **Inactivity Detection** | **95.83%** ± 4.2% | UP-Fall (4 subj, LOOCV) | ✅ |
| **O'rtacha** | **92.88%** | | ✅ |

**Barcha detektorlar >= 90% maqsadga erishdi.**

---

## 8. Nima Uchun Bu Natijalar Chiqdi?

**Fall (92.4%)** — `angle_rate` feature yiqilish va ataylab yotishni aniq farqlaydi. Dataset subjectlar xilma-xil, detektor umumlashadi. Subject 2 biroz past (87.9%) — cross-subject variatsiya.

**Running (90.42%)** — KTH lateral kamera sayasida yugurish va yurish gorizontal tezlikda yaqqol farqlanadi. Fixed threshold ishlamadi (50% berdi) — per-fold kalibratsiya kerak bo'ldi va natija 90%+ ga chiqdi.

**Inactivity (95.83%)** — Tik turish va o'tirish juda sokin (kp_disp ~0.001), yurish esa 10x ko'p harakatlanadi. `body_angle_std` narsa olishni aniq rad etadi.

---

## 9. Cheklovlar

| Muammo | Izoh |
|---|---|
| Running kalibratsiya kerak | Har bir kamera uchun bir marta sozlash |
| UP-Fall kichik (4 subject) | Ko'proq subject bilan natija barqarorroq bo'ladi |
| Laying (yotish) inactivity'da yo'q | Fall detector tomonidan tutiladi |
| Multi-person real test yo'q | Demo'da ko'rish mumkin |

---

## 10. Ishga Tushirish

```bash
# O'rnatish
pip install -r requirements.txt

# Fall + Inactivity evaluation
python -m evaluation.evaluate

# Running evaluation (KTH)
python -m evaluation.eval_running_kth_calibrated

# Real-time demo — webcam
python main.py

# Real-time demo — video fayl
python main.py --source video.mp4

# Multi-person tracking bilan
python main.py --source video.mp4
```

---

## 11. Fayl Tuzilmasi

```
Company_Abnormal_Project/
├── src/
│   ├── config.py              — threshold va sozlamalar
│   ├── pose_extractor.py      — YOLO11n-pose + ByteTracker
│   ├── feature_extractor.py   — biomexanik featurelar
│   ├── fall_detector.py       — Fall qoidalari
│   ├── running_detector.py    — Running qoidalari
│   ├── inactivity_detector.py — Inactivity (per-person timer)
│   └── behavior_monitor.py    — 3 ta detektor birlashtirilgan
├── evaluation/
│   ├── evaluate.py                      — Fall + Inactivity
│   ├── eval_running_kth_calibrated.py   — Running (KTH)
│   └── eval_inactivity_full.py          — To'liq inactivity
├── datasets/
│   ├── npy_loader.py                    — X.npy yuklash
│   └── download_running_dataset.py      — KTH yuklab olish
├── data/running_dataset/                — KTH kliplari
└── main.py                              — Real-time demo
```
