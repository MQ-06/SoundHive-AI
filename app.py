"""SoundHive-AI — Interactive Beehive Health Monitoring Demo"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.bootstrap import ensure_artifacts
from src.csv_loader import load_user_csv
from src.inference import health_summary, run_inference_pipeline

st.set_page_config(page_title="SoundHive AI", page_icon="🐝", layout="wide")

with st.spinner("Initializing ML model (first load ~15 seconds)..."):
    ensure_artifacts()

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEMO_PATH = PROJECT_ROOT / "data" / "demo" / "sample_temperature.csv"
TEST_PATH = PROJECT_ROOT / "data" / "demo" / "test_large_variation.csv"

HEALTH_COLORS = {
    "Low Temperature": "#3b82f6",
    "Normal Temperature": "#22c55e",
    "High Temperature": "#f97316",
}

with st.sidebar:
    st.title("🐝 SoundHive AI")
    st.markdown("[GitHub](https://github.com/MQ-06/SoundHive-AI)")
    st.markdown("[Dataset](https://www.kaggle.com/datasets/se18m502/bee-hive)")

st.markdown("## 🐝 SoundHive AI")
st.markdown("End-to-end ML system for beehive temperature health monitoring.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sensor Readings", "400K+")
c2.metric("ML Models", "5+")
c3.metric("Features", "9")
c4.metric("Classes", "3")
st.divider()

st.header("🔬 Live Hive Health Prediction")
upload_col, btn_col = st.columns([3, 1])
with upload_col:
    uploaded = st.file_uploader("Upload CSV (`timestamp`, `temperature`)", type=["csv"])
with btn_col:
    st.write("")
    use_sample = st.button("🧪 Sample Data", use_container_width=True)
    use_test = st.button("📈 Large Variation", use_container_width=True)

input_df = None
if use_sample and DEMO_PATH.exists():
    input_df = pd.read_csv(DEMO_PATH)
    st.success(f"Loaded {len(input_df):,} sample rows")
if use_test and TEST_PATH.exists():
    input_df = load_user_csv(open(TEST_PATH, "rb"))
    st.success(f"Loaded {len(input_df):,} rows | {input_df['temperature'].min():.1f}–{input_df['temperature'].max():.1f}°C")
if uploaded:
    try:
        input_df = load_user_csv(uploaded)
        st.info(f"Uploaded {len(input_df):,} rows from {uploaded.name}")
    except ValueError as e:
        st.error(str(e))

if input_df is not None:
    try:
        with st.spinner("Running ML pipeline..."):
            _, _, pred_df = run_inference_pipeline(input_df)
            summary = health_summary(pred_df)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dominant State", summary["dominant_health"])
        m2.metric("Avg Confidence", f"{summary['avg_confidence']:.1%}")
        m3.metric("Confidence Range", f"{summary['min_confidence']:.0%}–{summary['max_confidence']:.0%}")
        m4.metric("Uncertain (<70%)", summary["low_confidence_windows"])

        chart_col, table_col = st.columns([2, 1])
        with chart_col:
            dist = pd.Series(summary["distribution"])
            fig, ax = plt.subplots(figsize=(6, 3.5))
            dist.plot(kind="bar", ax=ax, color=[HEALTH_COLORS.get(k, "#888") for k in dist.index])
            ax.set_title("Health Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with table_col:
            cols = ["temp_mean", "predicted_class", "confidence"]
            st.dataframe(pred_df[cols].tail(10).round(3), hide_index=True)

        st.line_chart(pred_df[["temp_mean"]])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.header("📊 Model Performance")
results_path = RESULTS_DIR / "tables" / "classical_ml_results.csv"
if results_path.exists():
    st.dataframe(pd.read_csv(results_path), hide_index=True)
cm = RESULTS_DIR / "figures" / "rf_confusion_matrix.png"
fi = RESULTS_DIR / "figures" / "rf_feature_importances.png"
v1, v2 = st.columns(2)
if cm.exists():
    v1.image(str(cm), use_container_width=True)
if fi.exists():
    v2.image(str(fi), use_container_width=True)
