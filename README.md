# 🫀 StarLive Voting Optimized Technical Ensemble for Cardio

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://custom-icon-badges.demolab.com/badge/XGBoost-189C38?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-002C76?logo=optuna&logoColor=white)](https://optuna.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)

---

## Directory

- [Overview](#overview)
- [Features & Tech Stack](#features--tech-stack)
- [System Workflow](#system-workflow)
- [User Guide](#user-guide)
  - [Equipment](#equipment)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)
  - [Arsitektur Model](#arsitektur-model)
  - [SHAP Explainability](#shap-explainability)
  - [Hasil Evaluasi](#hasil-evaluasi)
  - [Limitations](#limitations)
  - [Future Development](#future-development)
- [Author](#author)

---

## Overview

**StarLive SVOTEC** adalah sistem deteksi dini penyakit kardiovaskular menggunakan **Soft Voting Ensemble** yang menggabungkan Random Forest, XGBoost, dan Logistic Regression — dilengkapi explainability berbasis **SHAP** dan antarmuka interaktif **Streamlit**.

Sistem ini memprioritaskan metrik **Recall** karena *false negative* (pasien sakit diprediksi sehat) jauh lebih berbahaya secara klinis dibanding *false positive*. Ensemble akhir mencapai Recall **91.2%** dan AUC-ROC **0.952** — melampaui performa setiap model individual.

> ⚠️ **Disclaimer:** Proyek ini dibuat untuk tujuan **edukasi dan portofolio**. Model ini **bukan alat diagnosis medis resmi** dan tidak boleh digunakan sebagai pengganti konsultasi dokter.

---

## Features & Tech Stack

### Features

- **Soft Voting Ensemble** — menggabungkan probabilitas 3 model dengan bobot tertimbang
- **SHAP Explainability** — penjelasan per pasien (waterfall) dan global (summary, bar, dependence plot)
- **Hyperparameter tuning** — optimasi otomatis via Optuna
- **Antarmuka Streamlit** — input klinis interaktif dengan visualisasi hasil prediksi dan SHAP
- **Pipeline preprocessing lengkap** — imputer, encoder, StandardScaler dalam satu pipeline

### Tech Stack

| Komponen | Teknologi |
|---|---|
| Bahasa | Python 3.10+ |
| ML Framework | scikit-learn (Pipeline, VotingClassifier, GridSearchCV) |
| Boosting | XGBoost 2.0 |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Hyperparameter Tuning | Optuna |
| Web App | Streamlit |
| Visualisasi | Matplotlib, Seaborn, Plotly |

---

## System Workflow

### Flowchart

```
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Raw Data           │     │  Feature Engineering │     │  Preprocessing       │
│  13 fitur klinis    │────▶│  age_group           │────▶│  Imputer → Encoder   │
│  heart.csv          │     │  bp_chol_ratio       │     │  → StandardScaler    │
│                     │     │  hr_age_ratio        │     │                      │
└─────────────────────┘     └──────────────────────┘     └──────────┬───────────┘
                                                                      │
                                          ┌───────────────────────────┼───────────────────────────┐
                                          ▼                           ▼                           ▼
                                ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
                                │  Random Forest   │       │    XGBoost       │       │ Logistic Reg     │
                                │  (weight: 3)     │       │  (weight: 3)     │       │  (weight: 1)     │
                                └────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘
                                         │                           │                           │
┌─────────────────────┐     ┌────────────▼───────────────────────────▼───────────────────────────▼────────┐
│  SHAP Explanation   │◀────│                     Soft Voting Classifier                                   │
│  per prediksi       │     │              (weighted avg of probabilities)                                 │
└─────────────────────┘     └──────────────────────────────┬───────────────────────────────────────────────┘
                                                            │
                                          ┌─────────────────┴──────────────────┐
                                          ▼                                     ▼
                                  ┌──────────────┐                     ┌──────────────┐
                                  │  High Risk   │                     │   Low Risk   │
                                  └──────────────┘                     └──────────────┘
```

### Penjelasan

| Langkah | Proses | Keterangan |
|---|---|---|
| 1 | Load data | Baca `heart.csv` — 1.025 sampel, 13 fitur klinis |
| 2 | Feature engineering | Buat fitur turunan: `age_group`, `bp_chol_ratio`, `hr_age_ratio` |
| 3 | Preprocessing | Imputer → Encoder → StandardScaler dalam satu sklearn Pipeline |
| 4 | Training 3 model | Random Forest, XGBoost, Logistic Regression dilatih paralel |
| 5 | Soft Voting | Rata-rata probabilitas tertimbang menghasilkan prediksi akhir |
| 6 | SHAP output | Setiap prediksi disertai penjelasan kontribusi fitur |

---

## User Guide

### Equipment

Pastikan hal berikut tersedia sebelum memulai:

- Python **3.10** atau lebih baru
- `pip`
- Dataset **UCI Heart Disease** — unduh dari [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) atau [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

**Tentang dataset:**

| Properti | Detail |
|---|---|
| Jumlah sampel | 1.025 pasien |
| Jumlah fitur | 13 fitur klinis (usia, tekanan darah, kolesterol, ECG, dll.) |
| Target | Biner — `0` = Low Risk, `1` = High Risk |
| Format | CSV — simpan sebagai `data/heart.csv` |

---

### Installation

#### 1. Clone Repositori

```bash
git clone https://github.com/username/cardio-svotec.git
cd cardio-svotec
```

#### 2. Instal Dependensi

```bash
pip install -r requirements.txt
```

#### 3. Training Model

```bash
# Training standar
python train.py --data data/heart.csv

# Dengan Optuna hyperparameter tuning (lebih lama, hasil lebih optimal)
python train.py --data data/heart.csv --tune
```

#### 4. Jalankan Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Buka browser di `http://localhost:8501`

---

### Configuration

**Argumen `train.py`:**

| Argumen | Tipe | Default | Keterangan |
|---|---|---|---|
| `--data` | STR | `data/heart.csv` | Path ke dataset CSV |
| `--tune` | FLAG | `False` | Aktifkan Optuna hyperparameter tuning |

**Struktur proyek:**

```
cardio_project/
├── src/
│   ├── preprocessing.py     # Pipeline preprocessing & feature engineering
│   ├── models.py            # Base models, ensemble, tuning, evaluasi
│   └── explainer.py         # SHAP explainability functions
├── app/
│   └── streamlit_app.py     # Web app interaktif
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── data/
│   └── heart.csv            # Dataset (unduh dari Kaggle/UCI)
├── models/                  # Model tersimpan (.joblib)
├── reports/
│   ├── model_comparison.csv
│   └── figures/             # Plot SHAP, confusion matrix, dll.
├── train.py
└── requirements.txt
```

---

### Troubleshooting

| Masalah | Kemungkinan Penyebab | Solusi |
|---|---|---|
| `FileNotFoundError: heart.csv` | Dataset belum diunduh | Unduh dari Kaggle/UCI dan simpan di `data/heart.csv` |
| `ModuleNotFoundError` | Dependensi belum terinstal | Jalankan `pip install -r requirements.txt` |
| Training sangat lambat | Mode `--tune` aktif dengan banyak trial | Kurangi jumlah trial Optuna di `models.py` |
| Streamlit app error saat load | Model `.joblib` belum ada | Jalankan `train.py` terlebih dahulu |
| SHAP plot tidak muncul | Backend matplotlib tidak kompatibel | Tambahkan `matplotlib.use('Agg')` di `explainer.py` |

---

## Development Notes

### Arsitektur Model

**Soft Voting Ensemble** dengan 3 model dasar dan bobot tertimbang:

| Komponen | Detail |
|---|---|
| Random Forest | 100 estimator, bobot 3 — kuat terhadap outlier dan fitur non-linear |
| XGBoost | Gradient boosting, bobot 3 — performa tertinggi pada data tabular |
| Logistic Regression | Model linear, bobot 1 — stabilisasi dan interpretasi baseline |
| Voting | Soft voting — rata-rata tertimbang probabilitas dari ketiga model |
| Loss prioritas | Recall dimaksimalkan — false negative lebih berbahaya secara klinis |

**Preprocessing pipeline:**

```
Imputer (median) → OrdinalEncoder → StandardScaler
```

**Feature engineering:**

| Fitur Baru | Formula | Alasan |
|---|---|---|
| `age_group` | Binning usia ke kategori | Hubungan usia-risiko bersifat non-linear |
| `bp_chol_ratio` | `trestbps / chol` | Interaksi tekanan darah dan kolesterol |
| `hr_age_ratio` | `thalach / age` | Kapasitas jantung relatif terhadap usia |

---

### SHAP Explainability

| Tipe Plot | Kegunaan |
|---|---|
| **Summary Plot** | Feature importance global — semua data |
| **Waterfall Plot** | Kontribusi tiap fitur per pasien individual |
| **Bar Importance** | Ranking fitur berdasarkan mean \|SHAP\| |
| **Dependence Plot** | Hubungan nilai fitur vs pengaruhnya terhadap prediksi |

---

### Hasil Evaluasi

Diukur pada test set dengan stratified split 80/20:

| Model | Accuracy | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Random Forest | 87.3% | 88.9% | 87.1% | 0.932 |
| XGBoost | 88.7% | 89.4% | 88.4% | 0.941 |
| Logistic Regression | 83.4% | 84.2% | 83.1% | 0.907 |
| **Soft Voting Ensemble** | **89.8%** | **91.2%** | **89.6%** | **0.952** |

Ensemble konsisten mengungguli setiap model individual di semua metrik, dengan keunggulan terbesar pada Recall — metrik yang paling kritis untuk kasus deteksi penyakit.

---

### Limitations

| Komponen | Batasan |
|---|---|
| Dataset | 1.025 sampel — relatif kecil untuk klaim generalisasi klinis |
| Populasi | Dataset UCI berasal dari populasi spesifik — mungkin tidak representatif untuk semua kelompok demografis |
| Fitur | 13 fitur klinis standar — tidak mencakup biomarker modern (troponin, BNP, dll.) |
| Validasi | Belum divalidasi pada data klinis nyata di luar dataset UCI |

### Future Development

Beberapa pengembangan yang dapat dilakukan ke depan:

- [ ] **Validasi eksternal** — uji model pada dataset kardiovaskular independen (misalnya Framingham Heart Study)
- [ ] **Fitur tambahan** — integrasi biomarker modern seperti troponin dan BNP
- [ ] **Model yang lebih kuat** — eksplorasi LightGBM atau CatBoost sebagai anggota ensemble tambahan
- [ ] **Kalibrasi probabilitas** — implementasi Platt scaling atau isotonic regression agar output probabilitas lebih terkalibrasi
- [ ] **Deployment cloud** — deploy Streamlit app ke Streamlit Cloud atau Hugging Face Spaces

---

<p align="center">
  <b>Pengembangan dari tim StarLive SAINT</b>
</p>

<p align="center"><i>Danny Aulia · Said Hasan Hanafiah · Noah Von Nobelius · Arvian Raveindra Pradana</i></p>
