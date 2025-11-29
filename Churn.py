# App Churn.py (fixed)
import os
import joblib
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

# =========================================================
# BASE DIR / PAGE CONFIG & GLOBAL STYLE
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📉",
    layout="wide",
)

# Custom CSS styling for bright, modern UI
CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #778fa1, #4c8e9c);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ede7f6, #e0f7fa);
}
.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: #0c4c85; #Biru tua
    margin-bottom: 0.25rem;
}
.sub-title {
    font-size: 1.1rem;
    color: #0c4c85;
    margin-bottom: 1.5rem;
}
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0c4c85;
    margin-top: 1.2rem;
    margin-bottom: 0.2rem;
}
.metric-card {
    background-color: rgba(255, 255, 255, 0.90);
    padding: 1rem 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    margin-bottom: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
MODEL_FILENAME = "Ecommerce_Customer_Churn_LGBM.pkl"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# Columns list based on df_data_cleaning.csv
NUMERIC_FEATURES_DEFAULT = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

CATEGORICAL_FEATURES_DEFAULT = [
    "PreferredLoginDevice",
    "CityTier",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
    "Complain",
]


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def load_local_sample_data(
    paths: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Try to load a local sample dataset for EDA if it exists.
    Checks multiple likely locations relative to BASE_DIR.
    """
    if paths is None:
        paths = [
            os.path.join(BASE_DIR, "df_data_cleaning.csv"),
            os.path.join(BASE_DIR, "df_data cleaning.csv"),
            os.path.join(BASE_DIR, "data", "df_data_cleaning.csv"),
            "df_data_cleaning.csv",
            "df_data cleaning.csv",
            os.path.join("data", "df_data_cleaning.csv"),
        ]

    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return None


def read_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Read uploaded CSV or Excel file into a pandas DataFrame.
    Works with Streamlit's UploadedFile.
    """
    if uploaded_file is None:
        return None

    _, ext = os.path.splitext(uploaded_file.name.lower())

    try:
        if ext == ".csv":
            return pd.read_csv(uploaded_file)
        if ext in (".xls", ".xlsx"):
            return pd.read_excel(uploaded_file)
    except Exception:
        return None

    return None


@st.cache_resource(show_spinner=False)
def load_model_and_scaler(
    model_path: str,
) -> Tuple[Optional[object], Optional[object]]:
    """
    Load machine learning model and optional scaler from disk using joblib.
    Returns (model, scaler) or (None, None) if problems occur.
    """
    model = None
    scaler = None

    # Load model (joblib)
    try:
        if not os.path.exists(model_path):
            st.error(
                f"Model file '{os.path.basename(model_path)}' tidak ditemukan di:\n{model_path}\n\n"
                "Pastikan file model berada di direktori yang sama dengan App Churn.py."
            )
            return None, None

        model = joblib.load(model_path)
    except Exception as exc:
        st.error(f"Gagal memuat model: {exc}")
        return None, None

    # Load scaler jika ada (joblib)
    try:
        scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = None
    except Exception:
        scaler = None

    return model, scaler


def preprocess_input(
    input_df: pd.DataFrame,
    scaler: Optional[object] = None,
) -> pd.DataFrame:
    """
    Preprocess input data before prediction.
    Drops 'Churn' if present and applies provided scaler only to numeric columns.
    Note: If your model is a full pipeline (preprocessing included), don't pass a scaler.
    """
    df_processed = input_df.copy()

    # Drop target column if exists
    if "Churn" in df_processed.columns:
        df_processed = df_processed.drop(columns=["Churn"])

    # Optional scaling: only numeric columns
    if scaler is not None:
        try:
            numeric_cols = df_processed.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])
        except Exception:
            # If scaling fails, continue without raising error
            pass

    return df_processed


def predict_churn(
    model: object,
    input_df: pd.DataFrame,
    scaler: Optional[object] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate churn predictions and probabilities.
    Strategy:
      - Try calling model.predict on raw input (works if model is full pipeline).
      - If that fails due to shape/column mismatch, apply preprocess_input and retry.
    """
    # attempt 1: predict directly (for full-pipeline models)
    try:
        predictions = model.predict(input_df)
    except Exception:
        # fallback: preprocess (use scaler if provided) then predict
        try:
            features = preprocess_input(input_df, scaler)
            predictions = model.predict(features)
        except Exception:
            raise

    # attempt to get probabilities
    try:
        # If predict_proba exists and works on same input
        try:
            proba = model.predict_proba(input_df)
        except Exception:
            features = preprocess_input(input_df, scaler)
            proba = model.predict_proba(features)
        churn_probas = proba[:, 1]
    except Exception:
        churn_probas = np.full(shape=(len(predictions),), fill_value=0.5)

    return np.array(predictions), np.array(churn_probas)


def map_churn_label(label) -> str:
    """
    Map raw model output to human-readable label.
    Accepts numeric or stringy labels.
    """
    try:
        if str(int(label)) == "1":
            return "Churn"
    except Exception:
        pass
    if str(label).lower() in ("1", "true", "yes", "churn"):
        return "Churn"
    return "Not Churn"


# =========================================================
# EDA PAGE
# =========================================================
def render_eda_page() -> None:
    """
    Render Exploratory Data Analysis (EDA) page.
    """
    st.markdown(
        "<div class='main-title'>📊 Exploratory Data Analysis</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='sub-title'>Eksplorasi dataset customer churn "
        "untuk memahami pola dan karakteristik pelanggan.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 🔼 Upload Dataset")
    col_upload1, col_upload2 = st.columns([2, 1])
    with col_upload1:
        uploaded_file = st.file_uploader(
            "Upload dataset untuk EDA (format: CSV atau Excel)",
            type=["csv", "xls", "xlsx"],
            key="eda_uploader",
        )
    with col_upload2:
        st.info(
            "Jika tidak ada file di-upload, aplikasi akan mencoba "
            "menggunakan `df_data_cleaning.csv` sebagai sample data "
            "(jika tersedia di server)."
        )

    if uploaded_file is not None:
        df = read_uploaded_file(uploaded_file)
        if df is None:
            st.error("Gagal membaca file. Pastikan format file sesuai.")
            return
    else:
        df = load_local_sample_data()
        if df is None:
            st.info("Silakan upload dataset terlebih dahulu untuk memulai EDA.")
            return

    st.success("Dataset berhasil dimuat ✅")

    # Detect numeric & categorical columns dynamically
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Override default lists based on actual columns
    numeric_features = [
        col for col in NUMERIC_FEATURES_DEFAULT if col in df.columns and col in numeric_cols
    ]
    categorical_features = [
        col for col in CATEGORICAL_FEATURES_DEFAULT if col in df.columns and col in categorical_cols
    ]

    # Tabs for EDA sections
    tab_overview, tab_dist, tab_corr, tab_churn, tab_interactive = st.tabs(
        [
            "📌 Overview",
            "📈 Feature Distribution",
            "🌡️ Correlation Heatmap",
            "🧩 Churn Analysis",
            "⚙️ Interactive Plotly",
        ]
    )

    # ---------------------- OVERVIEW TAB ----------------------
    with tab_overview:
        st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
        st.write("Ukuran data:", df.shape)

        st.markdown("#### 🗂️ Dataframe")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("#### 📊 Summary Statistics")
        with st.expander("Lihat ringkasan statistik (describe)", expanded=False):
            st.write(df.describe(include="all"))

        st.markdown("#### 🔍 Missing Values")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.write("Tidak ada missing values pada dataset.")
        else:
            st.write(missing[missing > 0])

    # ---------------------- DISTRIBUTION TAB ----------------------
    with tab_dist:
        st.markdown("<div class='section-header'>Feature Distribution</div>", unsafe_allow_html=True)

        if not numeric_cols:
            st.warning("Tidak ada kolom numerik yang tersedia untuk distribusi.")
        else:
            target_col = "Churn" if "Churn" in df.columns else None

            selected_num = st.selectbox(
                "Pilih feature numerik",
                numeric_features if numeric_features else numeric_cols,
            )

            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                st.markdown("##### Histogram")
                fig_hist = px.histogram(
                    df,
                    x=selected_num,
                    color=target_col,
                    marginal="box",
                    nbins=30,
                    title=f"Distribusi {selected_num}",
                    template="plotly",
                )
                fig_hist.update_layout(bargap=0.1, plot_bgcolor="#f3e5f5", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hist, use_container_width=True)

            with plot_col2:
                st.markdown("##### Boxplot")
                fig_box = px.box(df, y=selected_num, color=target_col, title=f"Boxplot {selected_num}", template="plotly")
                fig_box.update_layout(plot_bgcolor="#e3f2fd", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_box, use_container_width=True)

    # ---------------------- CORRELATION TAB ----------------------
    with tab_corr:
        st.markdown("<div class='section-header'>Correlation Heatmap</div>", unsafe_allow_html=True)
        if len(numeric_cols) < 2:
            st.warning("Minimal dua kolom numerik diperlukan untuk korelasi.")
        else:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdPu", title="Correlation Heatmap (Numerical Features)")
            fig_corr.update_layout(plot_bgcolor="#fce4ec", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)

    # ---------------------- CHURN ANALYSIS TAB ----------------------
    with tab_churn:
        st.markdown("<div class='section-header'>Churn Analysis</div>", unsafe_allow_html=True)

        if "Churn" not in df.columns:
            st.warning("Kolom 'Churn' tidak ditemukan di dataset.")
        else:
            st.markdown("#### 🥧 Churn Distribution")
            labels_map = {0: "Not Churn", 1: "Churn"}
            churn_series = df["Churn"].map(labels_map).fillna(df["Churn"])
            fig_pie = px.pie(values=churn_series.value_counts().values, names=churn_series.value_counts().index, hole=0.4, title="Distribusi Churn vs Not Churn")
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=True, plot_bgcolor="#e0f7fa", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("#### 📊 Churn vs Feature")
            candidate_features = [col for col in df.columns if col != "Churn"]
            selected_feature = st.selectbox("Pilih feature untuk dibandingkan dengan churn", candidate_features)

            if selected_feature:
                if (selected_feature in categorical_cols) or (df[selected_feature].nunique() <= 10):
                    fig_bar = px.histogram(df, x=selected_feature, color="Churn", barmode="group", title=f"Churn vs {selected_feature}")
                    fig_bar.update_layout(xaxis_title=selected_feature, plot_bgcolor="#f3e5f5", paper_bgcolor="rgba(0,0,0,0)")
                else:
                    fig_bar = px.box(df, x="Churn", y=selected_feature, title=f"{selected_feature} by Churn")
                    fig_bar.update_layout(plot_bgcolor="#f3e5f5", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------------- INTERACTIVE TAB ----------------------
    with tab_interactive:
        st.markdown("<div class='section-header'>Interactive Plotly</div>", unsafe_allow_html=True)

        if len(numeric_cols) < 2:
            st.warning("Minimal dua kolom numerik diperlukan untuk plot interaktif.")
        else:
            col_x, col_y = st.columns(2)
            with col_x:
                x_feature = st.selectbox("Pilih feature untuk sumbu X", numeric_cols, key="x_feature_interactive")
            with col_y:
                y_feature = st.selectbox("Pilih feature untuk sumbu Y", numeric_cols, index=1, key="y_feature_interactive")

            color_feature = "Churn" if "Churn" in df.columns else None

            fig_scatter = px.scatter(df, x=x_feature, y=y_feature, color=color_feature, title=f"{x_feature} vs {y_feature}", size_max=10)
            fig_scatter.update_layout(plot_bgcolor="#e3f2fd", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)


# =========================================================
# PREDICTION PAGE
# =========================================================
def render_prediction_page() -> None:
    """
    Render Customer Churn Prediction page.
    """
    st.markdown("<div class='main-title'>🤖 Customer Churn Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Upload data customer dan dapatkan prediksi kemungkinan churn secara otomatis.</div>", unsafe_allow_html=True)

    st.markdown("### 🔼 Upload Data Customer")
    uploaded_file = st.file_uploader("Upload data customer (single row atau multiple rows) (format: CSV atau Excel)", type=["csv", "xls", "xlsx"], key="pred_uploader")

    if uploaded_file is None:
        st.info("Silakan upload file data customer untuk melakukan prediksi churn.")
        return

    # Load input data
    input_df = read_uploaded_file(uploaded_file)
    if input_df is None or input_df.empty:
        st.error("Gagal membaca file atau file kosong.")
        return

    st.success("Data customer berhasil dimuat ✅")
    st.markdown("#### 🗂️ Preview Data")
    st.dataframe(input_df.head(20), use_container_width=True)

    # Load model and optional scaler
    model, scaler = load_model_and_scaler(MODEL_PATH)
    if model is None:
        return

    # Run prediction
    with st.spinner("Melakukan prediksi churn..."):
        try:
            predictions, churn_probas = predict_churn(model, input_df, scaler)
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
            return

    label_texts = [map_churn_label(lbl) for lbl in predictions]

    # If single row -> elegant card + gauge chart
    if len(input_df) == 1:
        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi (Single Customer)")

        prob_churn = float(churn_probas[0])
        label_text = label_texts[0]

        left_col, right_col = st.columns([1.2, 1])

        with left_col:
            html_card = f"""
            <div class="metric-card">
                <h3 style="margin-bottom:0.5rem;">Prediction Result</h3>
                <p style="font-size:1.1rem; margin-bottom:0.25rem;">
                    Status Customer:
                    <b style="color:{'#e53935' if label_text=='Churn' else '#43a047'};">
                        {label_text}
                    </b>
                </p>
                <p style="font-size:0.95rem; margin-bottom:0.2rem;">
                    Probability of Churn:
                    <b>{prob_churn:.2%}</b>
                </p>
                <p style="font-size:0.85rem; color:#546e7a;">
                    *Nilai probabilitas antara 0 (sangat kecil kemungkinan churn)
                    hingga 1 (sangat tinggi kemungkinan churn).
                </p>
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)

            st.markdown("##### Progress Probability")
            st.progress(min(max(prob_churn, 0.0), 1.0))

        with right_col:
            st.markdown("##### Gauge Chart")
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob_churn,
                    number={"valueformat": ".2f"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"thickness": 0.3},
                        "steps": [
                            {"range": [0, 0.33], "color": "#e0f7fa"},
                            {"range": [0.33, 0.66], "color": "#fff3e0"},
                            {"range": [0.66, 1.0], "color": "#ffebee"},
                        ],
                        "threshold": {
                            "line": {"color": "#d32f2f", "width": 4},
                            "thickness": 0.8,
                            "value": prob_churn,
                        },
                    },
                    title={"text": "Churn Probability"},
                )
            )
            fig_gauge.update_layout(margin=dict(t=50, b=0, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Multiple rows -> table with predictions
    else:
        st.markdown("---")
        st.markdown("### 📋 Hasil Prediksi (Multiple Customers)")

        results_df = input_df.copy()
        results_df["Churn_Prediction"] = label_texts
        results_df["Churn_Probability"] = churn_probas

        st.dataframe(results_df, use_container_width=True)

        st.markdown("#### 📊 Distribusi Prediksi Churn")
        fig_pred_pie = px.pie(results_df, names="Churn_Prediction", hole=0.4, title="Distribusi Prediksi Churn vs Not Churn")
        fig_pred_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pred_pie, use_container_width=True)


# =========================================================
# SIDEBAR & MAIN ROUTER
# =========================================================
def main() -> None:
    """
    Main function to control page routing and layout.
    """
    # Sidebar header
    st.sidebar.markdown("<h2 style='color:#1e88e5; margin-bottom:0.5rem;'>🧠 Churn App</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='font-size:0.9rem; color:#546e7a;'>Aplikasi untuk EDA dan prediksi Customer Churn.</p>", unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.radio("Pilih Halaman", ("Exploratory Data Analysis (EDA)", "Prediksi Churn"), index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<p style='font-size:0.8rem; color:#78909c;'>"
        f"Model: <b>{MODEL_FILENAME}</b><br>"
        "Pastikan file model berada di direktori yang sama dengan <b>App Churn.py</b>."
        "</p>",
        unsafe_allow_html=True,
    )

    # Route to selected page
    if page == "Exploratory Data Analysis (EDA)":
        render_eda_page()
    else:
        render_prediction_page()


if __name__ == "__main__":
    main()
