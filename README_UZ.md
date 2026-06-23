# Ishchilarning Anormal Xatti-Harakatini Aniqlash

Ishchilarning anormal xatti-harakatini real vaqtda aniqlash tizimi. Pose estimation va ob'ektlarni kuzatish asosida ishlaydi. Model train qilish talab etilmaydi — faqat biomexanik qoidalar.

**[English](README.md) | [한국어](README_KO.md)**

---

## Aniqlanadigan Xatti-Harakatlar

| Xatti-Harakat | Ta'rif | Aniqlik |
|---|---|---|
| **Yiqilishni Aniqlash** | Ishchi to'satdan yiqilganda bildiradi | 92.4% |
| **Xavfli Yugurish** | Cheklangan zonada yugurish | 90.4% |
| **Uzoq Harakatsizlik** | 5+ daqiqa harakatsiz qolish | 95.8% |

> LOOCV (Leave-One-Out Cross-Validation) protokoli bilan baholangan.

---

## Tizim Qanday Ishlaydi

```
CCTV / Kamera
      ↓
YOLO11n-pose  →  Har bir odamdan 17 ta bo'g'im
      ↓
ByteTracker   →  Har ishchiga alohida ID
      ↓
┌──────────────┬─────────────────┬───────────────────┐
│ Yiqilish     │ Yugurish        │ Harakatsizlik     │
│ Detektori    │ Detektori       │ Detektori         │
└──────────────┴─────────────────┴───────────────────┘
      ↓
Ogohlantirish (FALL | RUNNING | INACTIVITY)
```

### Yiqilishni Aniqlash Mantiq
- **Tana burchagi** (vertikaldan) va **burchak o'zgarish tezligi** (°/sek) o'lchanadi
- Qoida: `tana_burchagi > 70° VA burchak_tezligi > 65°/sek`
- Yiqilish juda tez (74–140°/sek), ataylab yotish esa sekin (2–5°/sek)

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
| Yugurish | **90.42%** ± 1.9% | KTH Action (25 subject) | LOOCV |
| Harakatsizlik | **95.83%** ± 4.2% | UP-Fall (4 subject) | LOOCV |
| **O'rtacha** | **92.88%** | | |

---

## Datasetlar

### UP-Fall Detection Dataset
- Martinez-Velasco va b., *Data* 2019
- Aktivliklar: yiqilish (4 tur), yurish, turish, o'tirish, narsa olish
- Maqsad: Yiqilish va Harakatsizlik baholash

### KTH Action Dataset
- Schuldt va b., *ICPR* 2004
- 25 ta subject, 200 ta klip (100 yugurish + 100 yurish)
- Maqsad: Yugurish aniqlash baholash

---

## O'rnatish

```bash
pip install -r requirements.txt
```

**Talablar:** Python 3.10+, PyTorch, Ultralytics YOLO, OpenCV, SciPy

---

## Ishlatish

### Real-vaqt Demo
```bash
# Webcam
python main.py

# Video fayl
python main.py --source video.mp4

# RTSP oqim
python main.py --source rtsp://192.168.1.10/stream

# Kuzatishsiz (bitta kishi)
python main.py --no-tracking
```

### Baholash
```bash
# Yiqilish + Harakatsizlik (UP-Fall)
python -m evaluation.evaluate

# Yugurish (KTH)
python -m evaluation.eval_running_kth_calibrated

# To'liq harakatsizlik baholash
python -m evaluation.eval_inactivity_full
```

### KTH Datasetini Yuklash
```bash
python -m datasets.download_running_dataset
```

---

## Loyiha Tuzilmasi

```
├── src/
│   ├── config.py               # Threshold va sozlamalar
│   ├── pose_extractor.py       # YOLO11n-pose + ByteTracker
│   ├── feature_extractor.py    # Biomexanik featurelar
│   ├── fall_detector.py        # Yiqilish qoidalari
│   ├── running_detector.py     # Yugurish qoidalari
│   ├── inactivity_detector.py  # Har ishchi uchun taymer
│   └── behavior_monitor.py     # Barcha detektor boshqaruvi
├── evaluation/
│   ├── evaluate.py                      # Yiqilish + Harakatsizlik LOOCV
│   ├── eval_running_kth_calibrated.py   # Yugurish LOOCV (KTH)
│   └── eval_inactivity_full.py          # To'liq harakatsizlik baholash
├── datasets/
│   ├── npy_loader.py                    # X.npy yuklash
│   └── download_running_dataset.py      # KTH yuklab olish
├── main.py                              # Real-vaqt demo
├── requirements.txt
└── REPORT.md                            # Texnik hisobot
```

---

## Asosiy Texnologiyalar

- **YOLO11n-pose** — Real vaqtda 17 ta bo'g'im aniqlash
- **ByteTracker** — Ko'p kishilik doimiy ID kuzatish
- **Butterworth Filter** — Yiqilish kinematikasi uchun signal tekislash
- **Rule-based Logic** — Model train qilinmagan, to'liq tushuntiriladi

---

## Muhim Dizayn Qarorlari

- **Train talab yo'q** — Qoidalar biomexanikadan olingan, dataset statistikasi bo'yicha kalibrlangan
- **Ko'p kishi** — ByteTracker har ishchiga alohida ID va mustaqil taymer beradi
- **Cascade arxitektura** — Yiqilish → Yugurish → Harakatsizlik (ketma-ket tekshiruv)
- **Threshold kalibrasyon** — LOOCV ichida per-fold kalibrasyon cross-subject umumlashishni ta'minlaydi
