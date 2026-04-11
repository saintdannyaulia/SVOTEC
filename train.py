"""
train.py
--------
Script utama untuk melatih dan mengevaluasi semua model.

Cara menjalankan:
    python train.py --data data/heart.csv
    python train.py --data data/heart.csv --tune   # dengan Optuna tuning

Output:
    models/ensemble_pipeline.joblib   ← model utama untuk Streamlit app
    models/rf_pipeline.joblib
    models/xgb_pipeline.joblib
    reports/model_comparison.csv
    reports/figures/                  ← semua plot tersimpan di sini
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Tambahkan src ke Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing import (
    load_dataset, feature_engineering,
    build_preprocessor, prepare_data,
    get_feature_names_out
)
from models import (
    build_ensemble, build_random_forest, build_xgboost,
    build_logistic_regression, build_full_pipeline,
    evaluate_model, compare_models, plot_confusion_matrix,
    save_pipeline, tune_random_forest, tune_xgboost
)
from explainer import (
    get_shap_values, plot_summary,
    plot_bar_importance, plot_waterfall
)


def main(data_path: str, run_tuning: bool = False):
    print("\n" + "="*60)
    print("  Early Detection of Cardiovascular Disease")
    print("  Ensemble Learning Pipeline")
    print("="*60 + "\n")

    # ── 0. Setup direktori output ──
    Path("models").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # ── 1. Load & preprocess data ──
    print("── Step 1: Load Dataset ──")
    df = load_dataset(data_path)

    print("── Step 2: Feature Engineering ──")
    df = feature_engineering(df)

    print("── Step 3: Split & SMOTE ──")
    X_train, X_test, y_train, y_test = prepare_data(df, apply_smote=True)

    # ── 2. Build preprocessor ──
    preprocessor = build_preprocessor()
    feature_names = get_feature_names_out(preprocessor)

    # ── 3. Hyperparameter tuning (opsional) ──
    rf_params  = None
    xgb_params = None

    if run_tuning:
        print("── Step 4: Hyperparameter Tuning (Optuna) ──")
        # Fit preprocessor dulu untuk mendapatkan X numerik
        X_train_proc = preprocessor.fit_transform(X_train, y_train)
        rf_params    = tune_random_forest(X_train_proc, y_train, n_trials=50)
        xgb_params   = tune_xgboost(X_train_proc, y_train, n_trials=50)
        # Reset preprocessor agar bisa di-fit ulang dalam pipeline
        preprocessor = build_preprocessor()
    else:
        print("── Step 4: Tuning dilewati (gunakan --tune untuk aktifkan) ──")

    # ── 4. Build dan evaluasi semua model ──
    print("\n── Step 5: Training & Evaluasi Model ──")
    results = []

    # Random Forest
    rf_pipeline = build_full_pipeline(
        build_preprocessor(),
        build_random_forest(rf_params)
    )
    rf_result = evaluate_model(rf_pipeline, X_train, y_train, X_test, y_test,
                                model_name="Random Forest")
    results.append(rf_result)
    save_pipeline(rf_pipeline, "models/rf_pipeline.joblib")
    plot_confusion_matrix(rf_pipeline, X_test, y_test,
                          "Random Forest", "reports/figures/cm_rf.png")

    # XGBoost
    xgb_pipeline = build_full_pipeline(
        build_preprocessor(),
        build_xgboost(xgb_params)
    )
    xgb_result = evaluate_model(xgb_pipeline, X_train, y_train, X_test, y_test,
                                 model_name="XGBoost")
    results.append(xgb_result)
    save_pipeline(xgb_pipeline, "models/xgb_pipeline.joblib")
    plot_confusion_matrix(xgb_pipeline, X_test, y_test,
                          "XGBoost", "reports/figures/cm_xgb.png")

    # Logistic Regression
    lr_pipeline = build_full_pipeline(
        build_preprocessor(),
        build_logistic_regression()
    )
    lr_result = evaluate_model(lr_pipeline, X_train, y_train, X_test, y_test,
                                model_name="Logistic Regression")
    results.append(lr_result)

    # ── Ensemble (Soft Voting) ──
    ensemble = build_ensemble(rf_params, xgb_params)
    ensemble_pipeline = build_full_pipeline(build_preprocessor(), ensemble)
    ensemble_result = evaluate_model(
        ensemble_pipeline, X_train, y_train, X_test, y_test,
        model_name="Soft Voting Ensemble"
    )
    results.append(ensemble_result)
    save_pipeline(ensemble_pipeline, "models/ensemble_pipeline.joblib")
    plot_confusion_matrix(ensemble_pipeline, X_test, y_test,
                          "Soft Voting Ensemble", "reports/figures/cm_ensemble.png")

    # ── 5. Tabel perbandingan ──
    print("\n── Step 6: Perbandingan Model ──")
    comparison_df = compare_models(results)
    comparison_df.to_csv("reports/model_comparison.csv")
    print("\n[save] Perbandingan disimpan ke: reports/model_comparison.csv")

    # ── 6. SHAP Explainability ──
    print("\n── Step 7: SHAP Explainability ──")
    # Transform data test untuk SHAP
    preprocessor_fitted = ensemble_pipeline.named_steps["preprocessor"]
    X_test_transformed  = preprocessor_fitted.transform(X_test)

    # Gunakan XGBoost dari dalam ensemble untuk SHAP (TreeExplainer lebih cepat)
    try:
        explainer, shap_values = get_shap_values(
            ensemble_pipeline, X_test_transformed, model_name="xgboost"
        )

        plot_summary(
            shap_values, X_test_transformed, feature_names,
            save_path="reports/figures/shap_summary.png"
        )
        plot_bar_importance(
            shap_values, feature_names,
            save_path="reports/figures/shap_importance.png"
        )
        plot_waterfall(
            explainer, shap_values, X_test_transformed, feature_names,
            patient_idx=0,
            save_path="reports/figures/shap_waterfall_patient0.png"
        )
        print("[SHAP] Semua plot berhasil disimpan di reports/figures/")
    except Exception as e:
        print(f"[SHAP] Warning: {e} — lewati plot SHAP")

    print("\n" + "="*60)
    print("  Training selesai!")
    print(f"  Model terbaik: {comparison_df.index[0]}")
    print(f"  AUC-ROC: {comparison_df['auc_roc'].iloc[0]:.4f}")
    print("  Jalankan Streamlit: streamlit run app/streamlit_app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train cardiovascular disease detection model"
    )
    parser.add_argument(
        "--data", type=str, default="data/heart.csv",
        help="Path ke file dataset CSV (default: data/heart.csv)"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Aktifkan hyperparameter tuning dengan Optuna"
    )
    args = parser.parse_args()
    main(args.data, run_tuning=args.tune)
