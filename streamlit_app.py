"""
streamlit_app.py
----------------
Aplikasi web interaktif untuk prediksi risiko penyakit kardiovaskular.

Cara menjalankan:
    streamlit run app/streamlit_app.py

Fitur:
- Input data pasien via sidebar (13 fitur klinis)
- Prediksi risiko + probabilitas dari Ensemble Model
- SHAP Waterfall Chart: penjelasan per pasien
- SHAP Summary Plot: feature importance global
- Perbandingan metrik semua model
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tambahkan src ke path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import (
    feature_engineering, build_preprocessor,
    get_feature_names_out, FEATURE_LABELS
)
from models import load_pipeline
from explainer import (
    get_shap_values, plot_waterfall,
    plot_bar_importance, explain_patient
)


# ─────────────────────────────────────────────
# Config Halaman
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS minimalis
st.markdown("""
<style>
  .risk-high { background:#fff1f0; border-left:4px solid #e03e3e;
               padding:1rem 1.2rem; border-radius:8px; }
  .risk-low  { background:#f0fff4; border-left:4px solid #2e7d32;
               padding:1rem 1.2rem; border-radius:8px; }
  .metric-card { background:#f8f9fa; border-radius:8px;
                 padding:0.8rem 1rem; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    model_dir = ROOT / "models"
    for name, path in [
        ("ensemble",   "ensemble_pipeline.joblib"),
        ("rf",         "rf_pipeline.joblib"),
        ("xgboost",    "xgb_pipeline.joblib"),
    ]:
        full_path = model_dir / path
        if full_path.exists():
            models[name] = load_pipeline(str(full_path))
    return models


@st.cache_data
def load_comparison():
    path = ROOT / "reports" / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


# ─────────────────────────────────────────────
# Sidebar: Input Pasien
# ─────────────────────────────────────────────
def render_sidebar():
    st.sidebar.title("🫀 CardioGuard AI")
    st.sidebar.markdown("Masukkan data klinis pasien di bawah ini.")
    st.sidebar.divider()

    with st.sidebar.expander("📋 Data Demografis", expanded=True):
        age = st.slider("Usia (tahun)", 20, 80, 50)
        sex = st.radio("Jenis Kelamin", [1, 0],
                       format_func=lambda x: "Pria" if x == 1 else "Wanita")

    with st.sidebar.expander("🩺 Data Klinis", expanded=True):
        chest_pain = st.select_slider(
            "Tipe Nyeri Dada",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 — Typical Angina",
                1: "1 — Atypical Angina",
                2: "2 — Non-anginal Pain",
                3: "3 — Asymptomatic"
            }[x],
            value=0
        )
        resting_bp  = st.slider("Tekanan Darah Istirahat (mmHg)", 80, 200, 120)
        cholesterol = st.slider("Kolesterol Serum (mg/dl)", 100, 600, 240)
        fbs         = st.radio("Gula Darah Puasa > 120 mg/dl",
                               [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        vessels     = st.select_slider("Jumlah Pembuluh Utama (0-3)",
                                       options=[0, 1, 2, 3])
        thal        = st.select_slider(
            "Thalassemia",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "1 — Normal",
                2: "2 — Fixed Defect",
                3: "3 — Reversible Defect"
            }[x]
        )

    with st.sidebar.expander("📈 Data ECG & Olahraga", expanded=True):
        resting_ecg = st.select_slider(
            "Hasil ECG Istirahat",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 — Normal",
                1: "1 — ST-T Abnormality",
                2: "2 — Left Ventricular Hypertrophy"
            }[x]
        )
        max_hr     = st.slider("Detak Jantung Maks.", 60, 220, 150)
        ex_angina  = st.radio("Angina saat Olahraga",
                              [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        st_depress = st.slider("Depresi ST", 0.0, 6.2, 1.0, step=0.1)
        st_slope   = st.select_slider(
            "Kemiringan ST",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 — Upsloping",
                1: "1 — Flat",
                2: "2 — Downsloping"
            }[x]
        )

    # Buat DataFrame dari input
    patient_data = pd.DataFrame([{
        "age":                 age,
        "sex":                 sex,
        "chest_pain_type":     chest_pain,
        "resting_bp":          resting_bp,
        "cholesterol":         cholesterol,
        "fasting_blood_sugar": fbs,
        "resting_ecg":         resting_ecg,
        "max_heart_rate":      max_hr,
        "exercise_angina":     ex_angina,
        "st_depression":       st_depress,
        "st_slope":            st_slope,
        "num_major_vessels":   vessels,
        "thalassemia":         thal,
    }])
    return patient_data


# ─────────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────────
def main():
    st.title("🫀 CardioGuard AI")
    st.markdown(
        "Sistem deteksi dini penyakit kardiovaskular menggunakan "
        "**Soft Voting Ensemble** (Random Forest + XGBoost + Logistic Regression) "
        "dilengkapi explainability berbasis **SHAP**."
    )

    # Load model
    models = load_models()
    if not models:
        st.error(
            "Model belum ditemukan. Jalankan `python train.py --data data/heart.csv` "
            "terlebih dahulu untuk melatih model."
        )
        st.stop()

    # Input pasien
    patient_df = render_sidebar()

    # Tombol prediksi
    predict_btn = st.sidebar.button("🔍 Prediksi Sekarang", type="primary",
                                    use_container_width=True)

    # ── Tab utama ──
    tab_pred, tab_shap, tab_compare = st.tabs([
        "🎯 Hasil Prediksi",
        "🔬 Penjelasan SHAP",
        "📊 Perbandingan Model"
    ])

    # ─────────────────────────────────────────
    # Tab 1: Hasil Prediksi
    # ─────────────────────────────────────────
    with tab_pred:
        if not predict_btn:
            st.info("Isi data pasien di sidebar, lalu klik **Prediksi Sekarang**.")
            st.markdown("### Ringkasan Data yang Dimasukkan")
            st.dataframe(
                patient_df.rename(columns=FEATURE_LABELS).T.rename(columns={0: "Nilai"}),
                use_container_width=True
            )
            return

        # Feature engineering
        patient_fe = feature_engineering(
            pd.concat([patient_df, pd.Series([0], name="target")
                       .to_frame().T.reset_index(drop=True)], axis=1)
        ).drop(columns=["target"])

        model = models.get("ensemble", list(models.values())[0])

        try:
            proba       = model.predict_proba(patient_fe)[0, 1]
            prediction  = int(proba >= 0.5)
        except Exception as e:
            st.error(f"Error saat prediksi: {e}")
            return

        # ── Tampilkan hasil ──
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            risk_label = "TINGGI 🔴" if prediction == 1 else "RENDAH 🟢"
            st.metric("Prediksi Risiko", risk_label)
        with col2:
            st.metric("Probabilitas Risiko Tinggi", f"{proba*100:.1f}%")
        with col3:
            confidence = abs(proba - 0.5) * 2
            st.metric("Tingkat Keyakinan Model", f"{confidence*100:.1f}%")

        # Gauge / progress bar
        st.markdown("#### Skala Risiko")
        st.progress(float(proba), text=f"Risiko: {proba*100:.1f}%")

        # Alert box
        if prediction == 1:
            st.markdown(
                '<div class="risk-high">'
                '<b>⚠️ Risiko Kardiovaskular TINGGI</b><br>'
                'Model mendeteksi indikasi risiko penyakit jantung. '
                'Segera konsultasikan dengan dokter spesialis jantung.'
                '</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="risk-low">'
                '<b>✅ Risiko Kardiovaskular RENDAH</b><br>'
                'Tidak terdeteksi indikasi signifikan. Tetap jaga gaya hidup sehat '
                'dan lakukan pemeriksaan rutin.'
                '</div>', unsafe_allow_html=True
            )

        st.caption(
            "⚠️ Hasil ini hanya alat bantu skrining berbasis machine learning. "
            "Bukan pengganti diagnosis medis dari dokter."
        )

    # ─────────────────────────────────────────
    # Tab 2: SHAP Explainability
    # ─────────────────────────────────────────
    with tab_shap:
        if not predict_btn:
            st.info("Lakukan prediksi terlebih dahulu untuk melihat penjelasan SHAP.")
            return

        st.markdown("### Mengapa Model Membuat Prediksi Ini?")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) menunjukkan kontribusi "
            "tiap fitur terhadap prediksi untuk pasien ini."
        )

        model = models.get("ensemble", list(models.values())[0])
        preprocessor = model.named_steps["preprocessor"]

        try:
            # Feature engineering untuk SHAP
            patient_fe = feature_engineering(
                pd.concat([patient_df,
                           pd.Series([0], name="target").to_frame().T
                           .reset_index(drop=True)], axis=1)
            ).drop(columns=["target"])

            X_transformed  = preprocessor.transform(patient_fe)
            feature_names  = get_feature_names_out(preprocessor)

            # Hitung SHAP
            with st.spinner("Menghitung SHAP values..."):
                explainer, shap_values = get_shap_values(
                    model, X_transformed, model_name="xgboost"
                )

            # Waterfall chart
            st.markdown("#### SHAP Waterfall Chart — Kontribusi Tiap Fitur")
            st.caption("Merah = meningkatkan risiko | Biru = menurunkan risiko")
            fig_wf = plot_waterfall(
                explainer, shap_values, X_transformed,
                feature_names, patient_idx=0
            )
            st.pyplot(fig_wf, use_container_width=True)
            plt.close()

            # Bar importance untuk pasien ini
            st.markdown("#### Urutan Fitur Berdasarkan Pengaruh")
            fig_bar = plot_bar_importance(shap_values, feature_names)
            st.pyplot(fig_bar, use_container_width=True)
            plt.close()

            # Penjelasan teks
            st.markdown("#### Ringkasan Penjelasan")
            proba      = model.predict_proba(patient_fe)[0, 1]
            explanation = explain_patient(
                shap_values[0], feature_names,
                X_transformed[0], proba
            )
            st.markdown(explanation)

        except Exception as e:
            st.warning(f"SHAP tidak tersedia: {e}")
            st.info(
                "Pastikan model XGBoost tersedia di dalam ensemble. "
                "Coba jalankan ulang training."
            )

    # ─────────────────────────────────────────
    # Tab 3: Perbandingan Model
    # ─────────────────────────────────────────
    with tab_compare:
        st.markdown("### Perbandingan Performa Semua Model")
        comp_df = load_comparison()

        if comp_df is not None:
            # Format tabel dengan highlight
            styled = (
                comp_df[["accuracy", "recall", "precision", "f1", "auc_roc"]]
                .style
                .format("{:.4f}")
                .highlight_max(axis=0, color="#d4edda")
                .set_caption("Metrik evaluasi pada test set (20% data)")
            )
            st.dataframe(styled, use_container_width=True)

            # Chart perbandingan
            st.markdown("#### Visualisasi Perbandingan")
            metrics = ["accuracy", "recall", "f1", "auc_roc"]
            fig, ax = plt.subplots(figsize=(9, 4))
            x = np.arange(len(comp_df))
            width = 0.2
            colors = ["#1D9E75", "#534AB7", "#D85A30", "#BA7517"]
            for i, metric in enumerate(metrics):
                ax.bar(x + i * width, comp_df[metric], width,
                       label=metric.upper().replace("_", "-"), color=colors[i], alpha=0.85)
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(comp_df.index, rotation=15, ha="right")
            ax.set_ylim(0.7, 1.0)
            ax.set_ylabel("Score")
            ax.set_title("Perbandingan Metrik Antar Model")
            ax.legend(loc="lower right", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Insight
            best = comp_df["auc_roc"].idxmax()
            st.success(
                f"**Model terbaik: {best}** "
                f"dengan AUC-ROC = {comp_df.loc[best, 'auc_roc']:.4f}"
            )
        else:
            st.warning(
                "File `reports/model_comparison.csv` belum ditemukan. "
                "Jalankan `python train.py` untuk generate data perbandingan."
            )

        st.markdown("---")
        st.markdown("""
        **Catatan Metrik:**
        - **Recall** diprioritaskan — false negative (pasien sakit diprediksi sehat) lebih berbahaya
        - **AUC-ROC** mengukur kemampuan model membedakan risiko tinggi vs rendah
        - **F1-Score** keseimbangan antara precision dan recall
        - Semua model dievaluasi dengan stratified 5-fold cross-validation
        """)


if __name__ == "__main__":
    main()
