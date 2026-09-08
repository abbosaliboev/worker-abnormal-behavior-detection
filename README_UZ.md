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
│   ├── pose_extractor.py         # YOLO11s-pose + ByteTracker
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
YOLO11s-pose  →  Har bir odamdan 17 ta bo'g'im
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

**1-qadam — Burchakni o'lchash:** Har kadrda tana umurtqasining vertikaldan qanchalik qiyshayganini hisoblanadi (0° = tik, 90° = gorizontal).

**2-qadam — Tezlikni o'lchash:** Burchakning qanchalik tez o'zgarayotgani aniqlanadi (darajalar/soniya).

**Qaror:** Tana 70°dan ko'proq qiyshaygan VA bu 65°/sek dan tez sodir bo'lgan bo'lsa → **YIQILISH aniqlandi**

> Nima uchun tezlik muhim: haqiqiy yiqilish 0.3–0.5 sekundda sodir bo'ladi (juda tez). Ataylab yotish esa 3–5 soniya davom etadi (sekin). Tezlik ikkalasini farqlaydi.

---

### Xavfli Yugurish Mantiq

**1-qadam — Gorizontal harakatni kuzatish:** Har kadrda markaziy massaning kamera bo'yicha gorizontal tezligi o'lchanadi.

**2-qadam — Kalibrlash:** Chegara har bir muhit uchun training subjectlarida kalibrlash orqali topiladi.

**Qaror:** Gorizontal tezlik kalibrlangan chegaradan oshsa → **YUGURISH aniqlandi**

> Yugurish kamera kadrida yurgandan ~2x tezroq harakatlanadi.

---

### Harakatsizlik Mantiq

**1-qadam — Sokin kadrlarni hisoblash:** So'nggi kadrlarda gavda bo'g'imlari deyarli qimirlamagan kadrlar ulushi hisoblanadi.

**2-qadam — Poza barqarorligini tekshirish:** Tana burchagining barqarorligi ham tekshiriladi (narsa olayotgan odam egilib-to'g'rilanadi — u chiqarib tashlanadi).

**3-qadam — Taymer:** Ikkala shart bajarilsa, 5 daqiqalik sanash boshlanadi. Har qanday harakat taymerni nolga qaytaradi.

**Qaror:** Ishchi 5 daqiqa uzluksiz harakatsiz qolsa → **HARAKATSIZLIK ogohlantiriladi**

---

## Datasetlar

### UP-Fall Detection Dataset
- **Manba:** Martinez-Velasco va b., *Data* 2019 — [https://sites.google.com/up.edu.mx/har-up/](https://sites.google.com/up.edu.mx/har-up/)
- **Ishlatilgan subjectlar:** 4 ta (Subject 1–4, jami 17 tadan)
- **Kamera:** RGB, ~17 fps, ichki muhit
- **Jami oynalar:** 4,479 ta (har biri 30 kadr, stride 15)

| Aktivlik | Label | Oynalar | Maqsad |
|---|---|---|---|
| Act 1–5 (5 tur yiqilish) | Yiqilish | 629 | Fall baholash |
| Act 6 (yurish) | Aktiv | 854 | Inactivity negative |
| Act 7 (tik turish) | Harakatsiz | 844 | Inactivity positive |
| Act 8 (o'tirish) | Harakatsiz | 834 | Inactivity positive |
| Act 9 (narsa olish) | Aktiv | 120 | Inactivity negative |

### KTH Action Dataset
- **Manba:** Schuldt va b., *ICPR* 2004 — [https://www.csc.kth.se/cvap/actions/](https://www.csc.kth.se/cvap/actions/)
- **Ishlatilgan subjectlar:** 25 ta (hammasi)
- **Kamera:** Lateral ko'rinish, 25 fps, tashqi/ichki muhit
- **Jami kliplar:** 200 ta (100 yugurish + 100 yurish)
- **Klip uzunligi:** ~15 soniya (150 kadr ishlatildi)
- **Maqsad:** Running aniqlash baholash

### Train / Test Taqsimoti (LOOCV)

LOOCV'da qat'iy statik split yo'q — har bir subject navbat bilan test (ko'rmagan) qismga aylanadi, qolganlari threshold fit qilish uchun ishlatiladi. Quyidagi raqamlar fold bo'yicha o'rtacha. Yiqilish va Harakatsizlik **frame** ketma-ketliklari (pose keypoint window'lari) ustida, Yugurish esa to'liq **video klip**lar ustida baholanadi.

| Detektor | Jami frame | Train (o'rtacha/fold) | Test (o'rtacha/fold) | Fold soni |
|---|---|---|---|---|
| Yiqilish | 69,150 frame (yiqilish: 10,320 / normal: 58,830), 131 ta frame ketma-ketligi bo'yicha (59 yiqilish + 72 normal) | ~51,863 frame | ~17,288 frame | 4 — har bir UP-Fall subject uchun 1 ta |
| Harakatsizlik | 40,500 frame (harakatsiz: 25,530 / aktiv: 14,970), 48 ta frame ketma-ketligi bo'yicha (24 harakatsiz + 24 aktiv), 2,652 ta pose-estimation window'dan guruhlangan | ~30,375 frame | ~10,125 frame | 4 — har bir UP-Fall subject uchun 1 ta |

| Detektor | Jami video klip | Ishlatilgan frame | Train (o'rtacha/fold) | Test (o'rtacha/fold) | Fold soni |
|---|---|---|---|---|---|
| Yugurish | 200 ta video klip (100 yugurish + 100 yurish) | 30,000 frame (video klip boshiga 150 frame chegara; klip o'rtacha 521 frame) | ~100 video klip | ~100 video klip | 2 — toq va juft KTH subjectlar |

### Test Uchun Namuna Papkalar

Yuqoridagi pre-extracted LOOCV pool'dan alohida, `data/` ichida **to'liq, bitta fold'lik test to'plamlari** ham bor — bular xom (real) datadan tuzilgan, bitta to'liq LOOCV fold'ning haqiqiy ushlab qolingan (held-out) qismi, shunchaki namuna emas. Yiqilish va Harakatsizlik — frame dataси: ushlab qolingan Subject1'ning xom **frame** ketma-ketliklari (barcha 11 activity, faqat Camera1); Yugurish esa ushlab qolingan toq-raqamli subject guruhini to'liq **video klip** sifatida ishlatadi.

| Papka | Tarkibi | Xatti-harakat | Shu fold natijasini ko'rish |
|---|---|---|---|
| `data/test_upfall/` | Subject1'ning to'liq foldi — **33 ta frame ketma-ketligi bo'yicha 17,932 frame** (yiqilish activity 1–5: 2,832 frame; normal activity 6–11: 15,100 frame) | Yiqilish | `python -m fall_detection.evaluate` — Subject1 qatori: **90.9%** (TN=15 FP=3 FN=0 TP=15) |
| `data/test_inactivity/` | Subject1'ning to'liq foldi — **12 ta frame ketma-ketligi bo'yicha 10,431 frame** (harakatsiz Act7+8: 6,527 frame; aktiv Act6+9: 3,904 frame) | Harakatsizlik | `python -m inactivity_detection.evaluate` — Subject1 qatori: **100.0%** (TN=6 FP=0 FN=0 TP=6) |
| `data/test_running/` | Toq-raqamli subject guruhining to'liq foldi — **104 ta video klip** (13 ta KTH subject) | Yugurish | `python main.py --source data/test_running/person01_running_d1_uncomp.avi` (vizual tekshiruv), yoki `python -m running_detection.evaluate`ning Fold 1 qismi |

> Bu xom papkalar raqamlar ortida haqiqiy fayllar turganini ko'zga ko'rinadigan tarzda isbotlash uchun bor — aniqlikning o'zi esa yuqoridagi oddiy `evaluate` buyruqlari orqali (qo'shimcha flag'siz) chiqadi, va README'dagi raqamlarni qayta olish uchun aynan shularni ishga tushirish kerak.

> **Git'ga committed emas.** Bu papkalar jami ~10 GB (xom PNG frame'lar og'ir) — bu git repo uchun juda katta hajm, shuning uchun `data/` `.gitignore`da qoladi. Ular git orqali emas, **cloud drive** orqali ulashiladi — `git clone` qilgach avtomatik kelmaydi, havolani so'rab oling.

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

- **YOLO11s-pose** — Real vaqtda 17 ta bo'g'im aniqlash
- **ByteTracker** — Ko'p kishilik doimiy ID kuzatish
- **Butterworth Filter** — Yiqilish kinematikasi uchun signal tekislash
- **Rule-based Logic** — Model train qilinmagan, to'liq tushuntiriladi
