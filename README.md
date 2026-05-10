# 🫀 StarLive Voting Optimized Technical Ensemble for Cardio

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)

Sistem deteksi dini penyakit kardiovaskular menggunakan **Soft Voting Ensemble** yang menggabungkan Random Forest, XGBoost, dan Logistic Regression — dilengkapi explainability berbasis **SHAP** dan antarmuka interaktif **Streamlit**.

---

## 🎯 Hasil Performa

| Model               | Accuracy | Recall | F1-Score | AUC-ROC |
|---------------------|----------|--------|----------|---------|
| Random Forest       | 87.3%    | 88.9%  | 87.1%    | 0.932   |
| XGBoost             | 88.7%    | 89.4%  | 88.4%    | 0.941   |
| Logistic Regression | 83.4%    | 84.2%  | 83.1%    | 0.907   |
| **Soft Voting Ensemble** | **89.8%** | **91.2%** | **89.6%** | **0.952** |

> Recall diprioritaskan karena false negative (pasien sakit diprediksi sehat) jauh lebih berbahaya.

---

## 📁 Struktur Proyek

```
cardio_project/
├── src/
│   ├── preprocessing.py     # Pipeline preprocessing & feature engineering
│   ├── models.py            # Base models, ensemble, tuning, evaluasi
│   └── explainer.py         # SHAP explainability functions
├── app/
│   └── streamlit_app.py     # Web app interaktif
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── data/
│   └── heart.csv            # Dataset (unduh dari Kaggle/UCI)
├── models/                  # Model tersimpan (.joblib)
├── reports/
│   ├── model_comparison.csv
│   └── figures/             # Plot SHAP, confusion matrix, dll
├── train.py                 # Script training utama
└── requirements.txt
```

---

## 🚀 Cara Menjalankan

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

Dataset: [UCI Heart Disease](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) dari Kaggle atau UCI ML Repository.

Simpan sebagai `data/heart.csv`.

### 3. Training model

```bash
# Training standar
python train.py --data data/heart.csv

# Dengan Optuna hyperparameter tuning (lebih lama, hasil lebih optimal)
python train.py --data data/heart.csv --tune
```

### 4. Jalankan Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Buka browser di `http://localhost:8501`

---

## 🧠 Arsitektur Model

```
Raw Data (13 fitur klinis)
        │
        ▼
Feature Engineering
(age_group, bp_chol_ratio, hr_age_ratio)
        │
        ▼
Preprocessing Pipeline
(Imputer → Encoder → StandardScaler)
        │
   ┌────┴────────────┬────────────┐
   ▼                 ▼            ▼
Random Forest     XGBoost    Logistic Reg
(weight: 3)      (weight: 3)  (weight: 1)
   └────┬────────────┴────────────┘
        │
        ▼
Soft Voting Classifier
(weighted avg of probabilities)
        │
   ┌────┴────┐
   ▼         ▼
High Risk  Low Risk  +  SHAP Explanation
```

---

## 🔬 SHAP Explainability

Proyek ini menggunakan SHAP untuk menjelaskan prediksi model:

- **Summary Plot** — feature importance global (semua data)
- **Waterfall Plot** — kontribusi tiap fitur per pasien
- **Bar Importance** — ranking fitur berdasarkan mean |SHAP|
- **Dependence Plot** — hubungan nilai fitur vs pengaruhnya

---

## 📊 Dataset

**UCI Heart Disease Dataset**
- 1,025 sampel pasien
- 13 fitur klinis (usia, tekanan darah, kolesterol, ECG, dll)
- Target biner: 0 = Low Risk, 1 = High Risk
- Sumber: [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

---

## ⚠️ Disclaimer

Proyek ini dibuat untuk tujuan **edukasi dan portofolio**. Model ini **bukan alat diagnosis medis resmi** dan tidak boleh digunakan sebagai pengganti konsultasi dokter.

---

## 🛠 Tech Stack

- Python 3.10+
- Scikit-learn (Pipeline, VotingClassifier, GridSearchCV)
- XGBoost
- imbalanced-learn (SMOTE)
- SHAP
- Optuna (hyperparameter tuning)
- Streamlit (web app)
- Matplotlib / Seaborn / Plotly
