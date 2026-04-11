"""
models.py
---------
Definisi dan training base models + Soft Voting Ensemble
untuk proyek Early Detection of Cardiovascular Disease.

Model yang digunakan:
- RandomForestClassifier
- XGBoostClassifier
- LogisticRegression
- VotingClassifier (Soft Voting — kombinasi probabilitas)

Juga menyediakan:
- Hyperparameter tuning dengan Optuna
- Cross-validation evaluation
- Penyimpanan & loading model (joblib)
"""

import numpy as np
import pandas as pd
import joblib
import optuna
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────
# Default Hyperparameter (sudah di-tune)
# ─────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":      200,
    "max_depth":         10,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "max_features":      "sqrt",
    "class_weight":      "balanced",
    "random_state":      42,
    "n_jobs":            -1,
}

XGB_PARAMS = {
    "n_estimators":   300,
    "max_depth":      6,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":          0.1,
    "reg_alpha":      0.1,
    "reg_lambda":     1.0,
    "scale_pos_weight": 1,
    "use_label_encoder": False,
    "eval_metric":    "logloss",
    "random_state":   42,
    "n_jobs":         -1,
}

LR_PARAMS = {
    "C":           1.0,
    "solver":      "lbfgs",
    "max_iter":    1000,
    "class_weight": "balanced",
    "random_state": 42,
}

# Bobot tiap model dalam voting (berdasarkan AUC individual)
VOTING_WEIGHTS = [3, 3, 1]   # RF, XGB, LR


# ─────────────────────────────────────────────
# Builder Functions
# ─────────────────────────────────────────────

def build_random_forest(params: dict = None) -> RandomForestClassifier:
    """Buat instance RandomForestClassifier."""
    p = params or RF_PARAMS
    return RandomForestClassifier(**p)


def build_xgboost(params: dict = None) -> XGBClassifier:
    """Buat instance XGBClassifier."""
    p = dict(XGB_PARAMS)
    if params:
        p.update(params)
    # Hapus param yang tidak dikenal oleh versi XGB tertentu
    p.pop("use_label_encoder", None)
    return XGBClassifier(**p)


def build_logistic_regression(params: dict = None) -> LogisticRegression:
    """Buat instance LogisticRegression."""
    p = params or LR_PARAMS
    return LogisticRegression(**p)


def build_ensemble(
    rf_params: dict = None,
    xgb_params: dict = None,
    lr_params: dict = None,
    weights: list = None
) -> VotingClassifier:
    """
    Buat Soft Voting Classifier dari ketiga base model.

    Soft voting mengambil rata-rata tertimbang dari probabilitas
    tiap kelas, menghasilkan prediksi yang lebih halus dan akurat
    dibanding hard voting (majority vote).
    """
    rf  = build_random_forest(rf_params)
    xgb = build_xgboost(xgb_params)
    lr  = build_logistic_regression(lr_params)

    ensemble = VotingClassifier(
        estimators=[
            ("random_forest",        rf),
            ("xgboost",              xgb),
            ("logistic_regression",  lr),
        ],
        voting="soft",
        weights=weights or VOTING_WEIGHTS,
        n_jobs=-1,
    )
    return ensemble


def build_full_pipeline(preprocessor, model) -> Pipeline:
    """
    Bungkus preprocessor + model ke dalam satu sklearn Pipeline.
    Memudahkan inference dan mencegah data leakage.
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

def evaluate_model(
    pipeline: Pipeline,
    X_train, y_train,
    X_test,  y_test,
    model_name: str = "Model",
    cv_folds: int = 5
) -> dict:
    """
    Evaluasi lengkap: cross-validation + test set metrics.

    Returns dict berisi semua metrik penting.
    """
    print(f"\n{'='*55}")
    print(f"  Evaluasi: {model_name}")
    print(f"{'='*55}")

    # ── Cross-validation ──
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=["accuracy", "recall", "f1", "roc_auc"],
        return_train_score=False,
        n_jobs=-1
    )
    print(f"\n[CV {cv_folds}-fold] Hasil rata-rata ± std:")
    for metric in ["accuracy", "recall", "f1", "roc_auc"]:
        scores = cv_results[f"test_{metric}"]
        print(f"  {metric:12s}: {scores.mean():.4f} ± {scores.std():.4f}")

    # ── Fit & Test ──
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc      = accuracy_score(y_test, y_pred)
    recall   = recall_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_proba)

    print(f"\n[Test Set] Hasil akhir:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Recall    : {recall:.4f}   ← prioritas utama (FN berbahaya!)")
    print(f"  Precision : {prec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Low Risk','High Risk'])}")

    return {
        "model_name":  model_name,
        "accuracy":    acc,
        "recall":      recall,
        "precision":   prec,
        "f1":          f1,
        "auc_roc":     auc,
        "cv_auc_mean": cv_results["test_roc_auc"].mean(),
        "cv_auc_std":  cv_results["test_roc_auc"].std(),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Tampilkan tabel perbandingan semua model."""
    df = pd.DataFrame(results).set_index("model_name")
    df = df.sort_values("auc_roc", ascending=False)
    print("\n" + "="*55)
    print("  Perbandingan Model")
    print("="*55)
    print(df.round(4).to_string())
    return df


def plot_confusion_matrix(
    pipeline: Pipeline,
    X_test, y_test,
    model_name: str,
    save_path: str = None
):
    """Plot dan optionally simpan confusion matrix."""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low Risk", "High Risk"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# Hyperparameter Tuning dengan Optuna
# ─────────────────────────────────────────────

def tune_random_forest(X_train, y_train, n_trials: int = 50) -> dict:
    """Cari hyperparameter terbaik untuk Random Forest menggunakan Optuna."""
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight":      "balanced",
            "random_state":      42,
        }
        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n[Optuna RF] Best AUC: {study.best_value:.4f}")
    print(f"[Optuna RF] Best params: {study.best_params}")
    return study.best_params


def tune_xgboost(X_train, y_train, n_trials: int = 50) -> dict:
    """Cari hyperparameter terbaik untuk XGBoost menggunakan Optuna."""
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 1),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0, 2),
            "eval_metric":      "logloss",
            "random_state":     42,
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n[Optuna XGB] Best AUC: {study.best_value:.4f}")
    print(f"[Optuna XGB] Best params: {study.best_params}")
    return study.best_params


# ─────────────────────────────────────────────
# Simpan & Load Model
# ─────────────────────────────────────────────

def save_pipeline(pipeline: Pipeline, path: str):
    """Simpan pipeline ke disk menggunakan joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[save] Model disimpan ke: {path}")


def load_pipeline(path: str) -> Pipeline:
    """Load pipeline dari disk."""
    pipeline = joblib.load(path)
    print(f"[load] Model dimuat dari: {path}")
    return pipeline
