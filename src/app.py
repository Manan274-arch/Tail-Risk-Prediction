import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# Base directory (deployment-safe)
# ======================================================
BASE_DIR = Path(__file__).resolve().parent

# ======================================================
# Global plot compactness (dashboard friendly)
# ======================================================
plt.rcParams.update({
    "figure.dpi": 90,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
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
# Load assets
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

# ======================================================
# Load data + feature engineering
# ======================================================
@st.cache_data
def load_data():
    df_train = pd.read_csv(BASE_DIR / "tail_risk_train_data.csv")
    df_live  = pd.read_csv(BASE_DIR / "tail_risk_live_data.csv")

    for df in [df_train, df_live]:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)

        df["log_return_1d"] = np.log(df["Close"]).diff()
        df["vol_10"] = df["log_return_1d"].rolling(10).std()
        df["vol_20"] = df["log_return_1d"].rolling(20).std()
        df["vol_60"] = df["log_return_1d"].rolling(60).std()
        df["drawdown_20"] = df["Close"] / df["Close"].rolling(20).max() - 1.0
        df["mom_10"] = df["Close"].pct_change(10)
        df["mom_50"] = df["Close"].pct_change(50)
        df["volume_ratio_20"] = df["Volume"] / df["Volume"].rolling(20).mean()

    return df_train.dropna().sort_index(), df_live.dropna().sort_index()

# ======================================================
# Load everything
# ======================================================
model = load_model()
scaler = load_scaler()
features = load_features()
df_train, df_live = load_data()

# ======================================================
# Validate inputs
# ======================================================
missing_features = [c for c in features if c not in df_live.columns]
if missing_features:
    st.error(f"Missing required features: {missing_features}")
    st.stop()

# ======================================================
# Sidebar
# ======================================================
st.sidebar.header("🧭 Risk Settings")

selected_date = st.sidebar.date_input(
    "Evaluation date",
    value=df_live.index[-1],
    min_value=df_live.index.min(),
    max_value=df_live.index.max()
)

prob_threshold = st.sidebar.slider(
    "When should I be alerted?",
    0.05, 0.50, 0.20, 0.05,
    help="Alert when estimated crash probability exceeds this value"
)

st.sidebar.caption(
    "Crash definition is fixed at training time. "
    "This threshold only controls alert sensitivity."
)

# ======================================================
# Inference
# ======================================================
X_train = scaler.transform(df_train[features])
X_live  = scaler.transform(df_live[features])

df_train["prob"] = model.predict(X_train, verbose=0).ravel()
df_live["prob"]  = model.predict(X_live, verbose=0).ravel()

# Fixed training-time crash labels
df_train["is_crash"] = df_train["tail_event"].astype(int)

# ======================================================
# Selected date handling
# ======================================================
sel_ts = pd.to_datetime(selected_date)
if sel_ts not in df_live.index:
    sel_ts = df_live.index[df_live.index.get_indexer([sel_ts], method="pad")][0]

prob_today = float(df_live.loc[sel_ts, "prob"])
percentile = (df_train["prob"] < prob_today).mean() * 100

# ======================================================
# Tabs
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📍 Today’s Risk", "📊 Model Behaviour", "🎯 Model Relevancy", "ℹ️ Methodology"]
)

# ======================================================
# TAB 1 — Today’s Risk
# ======================================================
with tab1:

    if prob_today >= prob_threshold:
        color, label = "#C62828", "HEIGHTENED RISK"
    elif prob_today >= 0.75 * prob_threshold:
        color, label = "#F9A825", "CAUTION"
    else:
        color, label = "#2E7D32", "LOW RISK"

    st.html(f"""
    <div style="text-align:center;">
        <h2>Tail Risk on {sel_ts.date()}</h2>
        <div style="font-size:56px; color:{color};">{prob_today:.2%}</div>
        <div style="font-weight:600; color:{color};">{label}</div>
        <div style="color:#AAA; margin-top:10px;">
            Higher than {percentile:.0f}% of historical observations
        </div>
    </div>
    """)

    st.markdown(
        f"""
        **Interpretation**
        - Model outputs a **probability**, not a forecast
        - Crash definition is **fixed at training time**
        - Alert threshold: **{prob_threshold:.0%}**
        """
    )

# ======================================================
# TAB 2 — Model Behaviour
# ======================================================
with tab2:

    crash_probs = df_train[df_train["is_crash"] == 1]["prob"]
    noncrash_probs = df_train[df_train["is_crash"] == 0]["prob"]

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots()
        ax.hist(crash_probs, bins=20, alpha=0.6, label="Crash", color="red", density=True)
        ax.hist(noncrash_probs, bins=20, alpha=0.6, label="Non-crash", color="green", density=True)
        ax.axvline(prob_threshold, linestyle="--", color="black")
        ax.legend()
        st.pyplot(fig)

    with right:
        fig, ax = plt.subplots()
        ax.bar(["Crash", "Non-crash"], [crash_probs.mean(), noncrash_probs.mean()],
               color=["red", "green"])
        st.pyplot(fig)

# ======================================================
# TAB 3 — Model Relevancy
# ======================================================
with tab3:

    df_train["risk_bucket"] = pd.cut(
        df_train["prob"],
        bins=[0, 0.2, 0.4, 0.6, 1.0],
        labels=["0–0.2", "0.2–0.4", "0.4–0.6", "0.6+"]
    )

    bucket_rate = df_train.groupby("risk_bucket")["is_crash"].mean()

    fig, ax = plt.subplots()
    bucket_rate.plot(kind="bar", ax=ax, color="firebrick")
    st.pyplot(fig)

# ======================================================
# TAB 4 — Methodology
# ======================================================
with tab4:
    st.markdown(
        """
        ### What this system estimates
        - The model outputs a **probability**, not a price forecast.
        - This probability represents the likelihood of a **large downside (tail) event**
          occurring within the forecast horizon used during training.

        ### How “crash” is defined
        - A crash corresponds to a **fixed training-time definition** of a large downside move
          over a **10-day forward window**.
        - This definition is **not changed dynamically** to preserve statistical validity
          and ensure that diagnostics remain meaningful.

        ### What the user controls
        - Users can adjust the **probability threshold**, which controls:
            - When alerts are triggered
            - How conservative or aggressive the warning system is
        - User controls do **not** alter the model, its predictions, or historical labels.

        ### Why these diagnostics matter
        - **Crash vs non-crash distributions** check whether the model assigns
          systematically higher probabilities during crash periods.
        - **Average predicted risk by outcome** provides a basic sanity check.
        - **Actual crash frequency by risk bucket** tests whether higher predicted
          risk corresponds to higher realized tail event frequency.
        - Together, these focus on **probabilistic relevance**, not just statistical fit.

        ### What this is NOT
        - ❌ Not a crash date prediction
        - ❌ Not a price forecast
        - ❌ Not a trading signal by itself

        ### Data and reproducibility
        - Market data are sourced from **Yahoo Finance (SPY)**.
        - The dashboard updates daily and uses cached data to ensure stability
          and reproducibility across sessions.
        """
    )

