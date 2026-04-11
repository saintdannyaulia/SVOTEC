"""
explainer.py
------------
Modul explainability berbasis SHAP untuk proyek cardiovascular disease detection.

Fungsi yang tersedia:
- get_shap_values()     : hitung SHAP values dari ensemble model
- plot_summary()        : SHAP beeswarm plot (feature importance global)
- plot_waterfall()      : SHAP waterfall plot (penjelasan per pasien)
- plot_dependence()     : SHAP dependence plot (hubungan fitur vs SHAP value)
- plot_bar_importance() : bar chart feature importance dari SHAP
- explain_patient()     : teks penjelasan otomatis per pasien
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")

# Non-interactive backend untuk kompatibilitas server
matplotlib.use("Agg")


def get_shap_values(pipeline, X_transformed: np.ndarray, model_name: str = "xgboost"):
    """
    Hitung SHAP values dari model di dalam pipeline.

    Parameters
    ----------
    pipeline      : sklearn Pipeline yang sudah di-fit
    X_transformed : data yang sudah melalui preprocessor (numpy array)
    model_name    : nama sub-model untuk diambil dari VotingClassifier
                    ("random_forest", "xgboost", "logistic_regression")

    Returns
    -------
    explainer   : SHAP explainer object
    shap_values : numpy array SHAP values (shape: n_samples x n_features)
    """
    # Ambil model dari VotingClassifier
    voting = pipeline.named_steps["classifier"]

    if hasattr(voting, "estimators_"):
        # Ambil estimator berdasarkan nama
        estimator_dict = dict(zip(
            [name for name, _ in voting.estimators],
            voting.estimators_
        ))
        model = estimator_dict.get(model_name, voting.estimators_[0])
    else:
        model = voting

    # Pilih explainer yang sesuai
    if "XGB" in type(model).__name__:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        # XGBoost TreeExplainer kadang return 2D langsung (untuk binary)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    elif "RandomForest" in type(model).__name__:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # kelas positif (High Risk)

    else:
        # Logistic Regression → Linear Explainer
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    return explainer, shap_values


def plot_summary(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list,
    save_path: str = None,
    max_display: int = 13
):
    """
    SHAP Beeswarm Summary Plot — menunjukkan feature importance global
    dan arah pengaruh tiap fitur terhadap prediksi.

    Merah = nilai fitur tinggi, Biru = nilai fitur rendah.
    Posisi horizontal = besar pengaruh (SHAP value).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None
    )
    plt.title("SHAP Summary Plot — Feature Importance Global", fontsize=13, pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Summary plot disimpan ke: {save_path}")
    return fig


def plot_bar_importance(
    shap_values: np.ndarray,
    feature_names: list,
    save_path: str = None,
    max_display: int = 13
):
    """
    Bar chart mean(|SHAP|) — versi yang lebih mudah dibaca
    untuk presentasi/README GitHub.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs
    }).sort_values("importance", ascending=True).tail(max_display)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color="#1D9E75",
        alpha=0.85
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Feature Importance (SHAP)", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)

    # Tambahkan label nilai di ujung bar
    for bar, val in zip(bars, importance_df["importance"]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=9, color="#444"
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Bar importance plot disimpan ke: {save_path}")
    return fig


def plot_waterfall(
    explainer,
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list,
    patient_idx: int = 0,
    save_path: str = None
):
    """
    SHAP Waterfall Plot untuk satu pasien.
    Menunjukkan kontribusi tiap fitur terhadap prediksi individu.
    Sangat berguna untuk menjelaskan keputusan model ke dokter/pasien.

    Parameters
    ----------
    patient_idx : index pasien di X_transformed (default=0)
    """
    # Buat Explanation object SHAP
    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    explanation = shap.Explanation(
        values=shap_values[patient_idx],
        base_values=base_value,
        data=X_transformed[patient_idx],
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.title(
        f"SHAP Waterfall — Penjelasan Prediksi Pasien #{patient_idx}",
        fontsize=12, pad=10
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Waterfall plot disimpan ke: {save_path}")
    return fig


def plot_dependence(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list,
    feature: str,
    interaction_feature: str = "auto",
    save_path: str = None
):
    """
    SHAP Dependence Plot — hubungan antara nilai satu fitur
    dengan SHAP value-nya, diwarnai oleh fitur interaksi.
    """
    feat_idx = feature_names.index(feature)
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X_transformed,
        feature_names=feature_names,
        interaction_index=interaction_feature,
        ax=ax,
        show=False
    )
    ax.set_title(f"SHAP Dependence — {feature}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def explain_patient(
    shap_values_patient: np.ndarray,
    feature_names: list,
    feature_values: np.ndarray,
    prediction_proba: float,
    top_n: int = 5
) -> str:
    """
    Generate penjelasan teks otomatis untuk satu pasien.
    Berguna untuk ditampilkan di Streamlit app.

    Returns
    -------
    str : penjelasan dalam Bahasa Indonesia
    """
    risk_label = "TINGGI" if prediction_proba >= 0.5 else "RENDAH"
    risk_pct   = prediction_proba * 100

    # Urutkan fitur berdasarkan |SHAP value|
    contributions = sorted(
        zip(feature_names, shap_values_patient, feature_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    lines = [
        f"Prediksi risiko kardiovaskular: **{risk_label}** ({risk_pct:.1f}%)\n",
        f"**{top_n} faktor utama yang mempengaruhi prediksi ini:**\n"
    ]

    for i, (feat, shap_val, feat_val) in enumerate(contributions, 1):
        direction = "meningkatkan" if shap_val > 0 else "menurunkan"
        impact    = "signifikan" if abs(shap_val) > 0.1 else "moderat"
        lines.append(
            f"{i}. **{feat}** (nilai: {feat_val:.2f}) — "
            f"*{direction}* risiko secara *{impact}* "
            f"(SHAP: {shap_val:+.3f})"
        )

    lines.append(
        "\n> ⚠️ Prediksi ini hanya alat bantu. "
        "Selalu konsultasikan dengan dokter untuk diagnosis resmi."
    )
    return "\n".join(lines)
