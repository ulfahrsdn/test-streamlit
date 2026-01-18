# Churn.py 
import os
import joblib
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# BASE DIR / PAGE CONFIG & GLOBAL STYLE
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📉",
    layout="wide",
)

# Custom CSS styling
CUSTOM_CSS = """
<style>
.stApp {
    background-color: #6f9680; /* solid cream muda */
    color: #00796b; /* default teks */
}
[data-testid="stSidebar"] {
    background-color: #fdf6e3; /* solid cream muda */
    color: #00796b;
}
.stSidebar div[role="radiogroup"] label {
    font-size: 1.1rem; /* radio button font size */
    font-weight: 600;   /* bold */
    color: #00796b;     /* kontras */
}
.stSidebar p, .stSidebar span {
    font-size: 0.95rem; /* info file, ukuran lebih besar */
    color: #00796b;     /* kontras */
}
.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: #00796b;
    margin-bottom: 0.25rem;
}
.sub-title {
    font-size: 1.1rem;
    color: #00796b;
    margin-bottom: 1.5rem;
}
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #00796b;
    margin-top: 1.2rem;
    margin-bottom: 0.2rem;
}
.metric-card {
    background-color: #fdf6e3;
    padding: 1rem 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    color: #00796b;
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
def load_local_sample_data(paths: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
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
def load_model_and_scaler(model_path: str) -> Tuple[Optional[object], Optional[object]]:
    model = None
    scaler = None
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file '{os.path.basename(model_path)}' tidak ditemukan di: {model_path}")
            return None, None
        model = joblib.load(model_path)
    except Exception as exc:
        st.error(f"Gagal memuat model: {exc}")
        return None, None
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    return model, scaler


def preprocess_input(input_df: pd.DataFrame, scaler: Optional[object] = None) -> pd.DataFrame:
    df_processed = input_df.copy()
    if "Churn" in df_processed.columns:
        df_processed = df_processed.drop(columns=["Churn"])
    if scaler is not None:
        numeric_cols = df_processed.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])
    return df_processed


def predict_churn(model: object, input_df: pd.DataFrame, scaler: Optional[object] = None) -> Tuple[np.ndarray, np.ndarray]:
    try:
        predictions = model.predict(input_df)
    except Exception:
        features = preprocess_input(input_df, scaler)
        predictions = model.predict(features)
    try:
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
    if str(label) in ("1", "Churn", "True", "Yes"):
        return "Churn"
    return "Not Churn"


# =========================================================
# EDA PAGE
# =========================================================
def render_eda_page() -> None:
    st.markdown("<div class='main-title'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Eksplorasi dataset customer churn untuk memahami pola pelanggan.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload dataset untuk EDA (CSV/Excel)", type=["csv", "xls", "xlsx"])
    df = read_uploaded_file(uploaded_file) if uploaded_file else load_local_sample_data()
    if df is None:
        st.info("Silakan upload dataset terlebih dahulu untuk memulai EDA.")
        return

    st.success("Dataset berhasil dimuat ✅")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Overview
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
    st.write("Ukuran data:", df.shape)
    st.dataframe(df.head(50), use_container_width=True)
    st.write(df.describe(include="all"))

    # Churn distribution
    if "Churn" in df.columns:
        st.markdown("<div class='section-header'>Churn Distribution</div>", unsafe_allow_html=True)
        labels_map = {0: "Not Churn", 1: "Churn"}
        churn_series = df["Churn"].map(labels_map).fillna(df["Churn"])
        fig_pie = px.pie(values=churn_series.value_counts().values, names=churn_series.value_counts().index, hole=0.4, title="Distribusi Churn vs Not Churn")
        fig_pie.update_layout(plot_bgcolor="#fdf6e3", paper_bgcolor="#fdf6e3")
        st.plotly_chart(fig_pie, use_container_width=True)


# =========================================================
# PREDICTION PAGE
# =========================================================
def render_prediction_page() -> None:
    st.markdown("<div class='main-title'>🤖 Customer Churn Prediction</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload data customer (CSV/Excel)", type=["csv", "xls", "xlsx"])
    if uploaded_file is None:
        st.info("Silakan upload file data customer untuk melakukan prediksi churn.")
        return

    input_df = read_uploaded_file(uploaded_file)
    if input_df is None or input_df.empty:
        st.error("Gagal membaca file atau file kosong.")
        return

    st.success("Data customer berhasil dimuat ✅")
    st.dataframe(input_df.head(20), use_container_width=True)

    model, scaler = load_model_and_scaler(MODEL_PATH)
    if model is None:
        return

    predictions, churn_probas = predict_churn(model, input_df, scaler)
    label_texts = [map_churn_label(lbl) for lbl in predictions]

    st.markdown("---")
    if len(input_df) == 1:
        prob_churn = float(churn_probas[0])
        label_text = label_texts[0]
        st.markdown(f"<div class='metric-card'>Status Customer: <b>{label_text}</b><br>Probability: {prob_churn:.2%}</div>", unsafe_allow_html=True)
    else:
        results_df = input_df.copy()
        results_df["ChurnPrediction"] = label_texts
        results_df["ChurnProbability"] = churn_probas
        st.dataframe(results_df, use_container_width=True)


# =========================================================
# MAIN APP
# =========================================================
st.sidebar.markdown("<div class='main-title'>Menu</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Pilih Halaman", ["EDA", "Prediction"])
if page == "EDA":
    render_eda_page()
else:
    render_prediction_page()
