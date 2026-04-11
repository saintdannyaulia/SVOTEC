"""
preprocessing.py
----------------
Pipeline preprocessing data untuk proyek Early Detection of Cardiovascular Disease.

Langkah-langkah:
1. Load & validasi dataset
2. Handle missing values
3. Encode fitur kategorikal
4. Feature engineering sederhana
5. Scale fitur numerik
6. Handle class imbalance dengan SMOTE
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Nama kolom sesuai UCI Heart Disease Dataset
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    "age", "sex", "chest_pain_type", "resting_bp",
    "cholesterol", "fasting_blood_sugar", "resting_ecg",
    "max_heart_rate", "exercise_angina", "st_depression",
    "st_slope", "num_major_vessels", "thalassemia"
]
TARGET_COL = "target"

NUMERIC_FEATURES = [
    "age", "resting_bp", "cholesterol",
    "max_heart_rate", "st_depression"
]
CATEGORICAL_FEATURES = [
    "sex", "chest_pain_type", "fasting_blood_sugar",
    "resting_ecg", "exercise_angina", "st_slope",
    "num_major_vessels", "thalassemia"
]

# Label ramah manusia untuk Streamlit app
FEATURE_LABELS = {
    "age":                 "Usia (tahun)",
    "sex":                 "Jenis Kelamin (1=Pria, 0=Wanita)",
    "chest_pain_type":     "Tipe Nyeri Dada (0-3)",
    "resting_bp":          "Tekanan Darah Istirahat (mmHg)",
    "cholesterol":         "Kolesterol Serum (mg/dl)",
    "fasting_blood_sugar": "Gula Darah Puasa > 120 mg/dl (1=Ya)",
    "resting_ecg":         "Hasil ECG Istirahat (0-2)",
    "max_heart_rate":      "Detak Jantung Maksimum",
    "exercise_angina":     "Angina saat Olahraga (1=Ya)",
    "st_depression":       "Depresi ST (0.0 - 6.2)",
    "st_slope":            "Kemiringan ST (0-2)",
    "num_major_vessels":   "Jumlah Pembuluh Utama (0-3)",
    "thalassemia":         "Thalassemia (1=Normal, 2=Fixed, 3=Reversible)"
}


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load dataset dari CSV.
    Mendukung format UCI asli (dengan header) maupun tanpa header.

    Parameters
    ----------
    filepath : str
        Path ke file CSV dataset.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath)

    # Jika kolom belum punya nama yang sesuai, rename otomatis
    if list(df.columns) == list(range(len(df.columns))):
        df.columns = FEATURE_NAMES + [TARGET_COL]

    # Pastikan kolom target ada
    assert TARGET_COL in df.columns, \
        f"Kolom '{TARGET_COL}' tidak ditemukan. Pastikan dataset sudah benar."

    # Konversi target: UCI memakai 0-4, kita binarize jadi 0 vs 1
    if df[TARGET_COL].nunique() > 2:
        df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    print(f"[load] Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"[load] Distribusi kelas:\n{df[TARGET_COL].value_counts(normalize=True).round(3)}\n")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan fitur turunan sederhana yang bisa meningkatkan performa model.

    Fitur baru:
    - age_group       : kategori usia (young/middle/senior)
    - bp_chol_ratio   : rasio tekanan darah / kolesterol
    - hr_age_ratio    : max_heart_rate / age (fitness proxy)
    """
    df = df.copy()

    # Kategorisasi usia
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 55, 120],
        labels=[0, 1, 2]    # 0=young, 1=middle, 2=senior
    ).astype(float)

    # Rasio tekanan darah / kolesterol
    df["bp_chol_ratio"] = df["resting_bp"] / (df["cholesterol"].replace(0, np.nan))
    df["bp_chol_ratio"].fillna(df["bp_chol_ratio"].median(), inplace=True)

    # Proxy kebugaran jantung
    df["hr_age_ratio"] = df["max_heart_rate"] / df["age"]

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Buat sklearn ColumnTransformer untuk preprocessing otomatis.

    Pipeline numerik  : Imputer (median) → StandardScaler
    Pipeline kategorikal: Imputer (most_frequent) → OrdinalEncoder
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    # Tambahkan fitur engineered ke numerik
    all_numeric = NUMERIC_FEATURES + ["bp_chol_ratio", "hr_age_ratio"]
    all_categorical = CATEGORICAL_FEATURES + ["age_group"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,  all_numeric),
            ("cat", categorical_pipeline, all_categorical),
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True
):
    """
    Split data train/test dan opsional terapkan SMOTE pada data training.

    Parameters
    ----------
    df           : DataFrame yang sudah melalui feature_engineering()
    test_size    : proporsi data test
    random_state : seed reproducibility
    apply_smote  : aktifkan SMOTE untuk mengatasi class imbalance

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays siap training
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"[SMOTE] Data training setelah resampling: {X_train.shape[0]} sampel")
        print(f"[SMOTE] Distribusi kelas baru: {pd.Series(y_train).value_counts().to_dict()}\n")

    print(f"[split] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")
    return X_train, X_test, y_train, y_test


def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """Dapatkan nama fitur output setelah preprocessing (berguna untuk SHAP)."""
    num_feats = NUMERIC_FEATURES + ["bp_chol_ratio", "hr_age_ratio"]
    cat_feats = CATEGORICAL_FEATURES + ["age_group"]
    return num_feats + cat_feats
