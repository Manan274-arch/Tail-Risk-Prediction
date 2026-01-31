import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# Base directory (CRITICAL for deployment)
# ======================================================
BASE_DIR = Path(__file__).resolve().parent

# ======================================================
# Global plot compactness (dashboard friendly)
# ======================================================
plt.rcParams.update({
    "figure.dpi": 100,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# ======================================================
# Page config
# ======================================================
st.set_page_config(page_title="Tail Risk Dashboard", layout="wide")

# ======================================================
# Header
# ======================================================
st.markdown(
    """
    <h2 style='text-align:center;'>📉 Tail Risk Early Warning Dashboard</h2>
    <p style='text-align:center; font-size:14px; margin-top:-8px;'>
        Probabilistic early warning for large downside events (tail risk)
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ======================================================
# Load assets (PATH SAFE)
# ======================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(BASE_DIR / "tail_risk_model.keras")

@st.cache_resource
def load_scaler():
    return joblib.load(BASE_DIR / "scaler.pkl")

@st.cache_resource
def load_features():
    return joblib.load(BASE_DIR / "features.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "processed_data.csv")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date").sort_index()
    else:
        df.index = pd.date_range(start="1990-01-01", periods=len(df), freq="B")
        df.index.name = "Date"

    return df

# ======================================================
# Load everything
# ======================================================
model = load_model()
scaler = load_scaler()
features = load_features()
df = load_data()

# ======================================================
# Validate required columns
# ======================================================
missing_features = [c for c in features if c not in df.columns]
if missing_features:
    st.error(
        "Your CSV is missing feature columns required by features.pkl:\n\n"
        + ", ".join(missing_features[:30])
        + (" ..." if len(missing_features) > 30 else "")
    )
    st.stop()

if "forward_return" not in df.columns:
    st.error(
        "`processed_data.csv` must contain a `forward_return` column "
        "for crash labeling and drawdown plots."
    )
    st.stop()

# ======================================================
# Sidebar controls
# ======================================================
st.sidebar.header("⚙️ Controls")

selected_date = st.sidebar.date_input(
    "Evaluation date",
    value=df.index[-1],
    min_value=df.index.min(),
    max_value=df.index.max()
)

prob_threshold = st.sidebar.slider(
    "Probability threshold",
    0.05, 0.50, 0.25, 0.05
)

crash_dd = st.sidebar.slider(
    "Crash definition (drawdown)",
    0.02, 0.20, 0.05, 0.01,
    help="Crash day is defined as forward_return <= -drawdown"
)

# ======================================================
# Inference
# ======================================================
X_all = scaler.transform(df[features])
df["prob"] = model.predict(X_all, verbose=0).ravel()
df["is_crash"] = (df["forward_return"] <= -crash_dd).astype(int)

# Selected date handling
sel_ts = pd.to_datetime(selected_date)
if sel_ts not in df.index:
    sel_ts = df.index[df.index.get_indexer([sel_ts], method="pad")][0]

prob_today = float(df.loc[sel_ts, "prob"])

crash_probs = df.loc[df["is_crash"] == 1, "prob"]
noncrash_probs = df.loc[df["is_crash"] == 0, "prob"]

avg_crash = float(crash_probs.mean()) if len(crash_probs) else float("nan")
avg_noncrash = float(noncrash_probs.mean()) if len(noncrash_probs) else float("nan")

# ======================================================
# Tabs
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📍 Today’s Risk", "📊 Model Behaviour", "📉 Economic Impact", "ℹ️ Methodology"]
)

# ===================== TAB 1 ==========================
with tab1:
    colA, colB, colC = st.columns([1, 2, 1])

    elevated = prob_today >= prob_threshold
    risk_label = "ELEVATED RISK" if elevated else "LOW RISK"
    color = "red" if elevated else "green"

    colB.markdown(
        f"""
        <h3 style='text-align:center;'>Tail Risk on {sel_ts.date()}</h3>
        <h1 style='text-align:center; color:{color}; margin-top:-8px;'>
            {prob_today:.2%}
        </h1>
        <p style='text-align:center; color:{color}; font-weight:bold; margin-top:-10px;'>
            {risk_label}
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        **Interpretation**  
        This is the model-estimated probability of a **large downside event**.

        - Crash definition: **forward_return ≤ −{crash_dd:.0%}**
        - Signal threshold: **{prob_threshold:.0%}**
        """
    )

# ===================== TAB 2 ==========================
with tab2:
    left, right = st.columns(2)

    fig1, ax1 = plt.subplots(figsize=(4.0, 3.0))
    ax1.hist(crash_probs, bins=25, alpha=0.6, density=True, label="Crash days", color="red")
    ax1.hist(noncrash_probs, bins=25, alpha=0.6, density=True, label="Non-crash days", color="green")
    ax1.axvline(prob_threshold, color="black", linestyle="--", linewidth=1)
    ax1.set_title("Crash vs Non-Crash Distribution")
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
    ax2.bar(["Crash days", "Non-crash days"], [avg_crash, avg_noncrash], color=["red", "green"])
    ax2.set_ylim(0, 1)
    ax2.set_title("Average Model Risk by Outcome")
    st.pyplot(fig2)

# ===================== TAB 3 ==========================
with tab3:
    bins = [0.0, 0.2, 0.4, 0.6, 1.0]
    labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6+"]

    df["risk_bucket"] = pd.cut(df["prob"], bins=bins, labels=labels)

    def max_drawdown(r):
        r = r.dropna()
        if len(r) == 0:
            return np.nan
        c = (1 + r).cumprod()
        p = c.cummax()
        return float(((c - p) / p).min())

    bucket_dd = df.groupby("risk_bucket")["forward_return"].apply(max_drawdown)

    fig3, ax3 = plt.subplots(figsize=(5.0, 3.0))
    bucket_dd.plot(kind="bar", ax=ax3, color="firebrick")
    ax3.set_title("Maximum Drawdown by Predicted Risk Bucket")
    st.pyplot(fig3)

# ===================== TAB 4 ==========================
with tab4:
    st.markdown(
        f"""
        ### What this system estimates
        - A **probability** of a large downside event, not a price forecast.

        ### Crash definition
        - **forward_return ≤ −{crash_dd:.0%}**

        ### Why diagnostics matter
        - Distribution separation
        - Economic validity via drawdowns
        """
    )
