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
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.set_index("Date", inplace=True)
        else:
            df.index = pd.date_range(start="1990-01-01", periods=len(df), freq="B")
            df.index.name = "Date"

    # --------------------------------------------------
    # FEATURE ENGINEERING (only if missing)
    # --------------------------------------------------
    for df in [df_train, df_live]:

        if "log_return_1d" not in df.columns:
            df["log_return_1d"] = np.log(df["Close"]).diff()

        if "vol_10" not in df.columns:
            df["vol_10"] = df["log_return_1d"].rolling(10).std()

        if "vol_20" not in df.columns:
            df["vol_20"] = df["log_return_1d"].rolling(20).std()

        if "vol_60" not in df.columns:
            df["vol_60"] = df["log_return_1d"].rolling(60).std()

        if "drawdown_20" not in df.columns:
            df["drawdown_20"] = df["Close"] / df["Close"].rolling(20).max() - 1.0

        if "mom_10" not in df.columns:
            df["mom_10"] = df["Close"].pct_change(10)

        if "mom_50" not in df.columns:
            df["mom_50"] = df["Close"].pct_change(50)

        if "volume_ratio_20" not in df.columns and "Volume" in df.columns:
            df["volume_ratio_20"] = df["Volume"] / df["Volume"].rolling(20).mean()

    df_train = df_train.dropna().sort_index()
    df_live  = df_live.dropna().sort_index()

    return df_train, df_live

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
    st.error(
        "Your CSV is missing feature columns required by `features.pkl`:\n\n"
        + ", ".join(missing_features[:25])
    )
    st.stop()

# ======================================================
# Sidebar controls
# ======================================================
st.sidebar.header("⚙️ Controls")

selected_date = st.sidebar.date_input(
    "Evaluation date",
    value=df_live.index[-1],
    min_value=df_live.index.min(),
    max_value=df_live.index.max()
)

prob_threshold = st.sidebar.slider(
    "Probability threshold",
    0.05, 0.50, 0.20, 0.05
)

crash_dd = st.sidebar.slider(
    "Crash definition (drawdown)",
    0.02, 0.20, 0.05, 0.01,
    help="Crash day is defined as forward-return ≤ −drawdown"
)

# ======================================================
# Inference
# ======================================================
X_train_all = scaler.transform(df_train[features])
X_live_all  = scaler.transform(df_live[features])

pred_train = model.predict(X_train_all, verbose=0)
pred_live  = model.predict(X_live_all, verbose=0)

df_train["prob"] = pred_train[:, -1] if pred_train.ndim == 2 else pred_train.ravel()
df_live["prob"]  = pred_live[:, -1]  if pred_live.ndim == 2  else pred_live.ravel()

df_train["is_crash"] = df_train["tail_event"].astype(int)

# ======================================================
# Selected date
# ======================================================
sel_ts = pd.to_datetime(selected_date)
if sel_ts not in df_live.index:
    sel_ts = df_live.index[df_live.index.get_indexer([sel_ts], method="pad")][0]

prob_today = float(df_live.loc[sel_ts, "prob"])

crash_probs = df_train.loc[df_train["is_crash"] == 1, "prob"]
noncrash_probs = df_train.loc[df_train["is_crash"] == 0, "prob"]

avg_crash = crash_probs.mean()
avg_noncrash = noncrash_probs.mean()

# ======================================================
# Tabs
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📍 Today’s Risk", "📊 Model Behaviour", "📉 Economic Impact", "ℹ️ Methodology"]
)

# ======================================================
# TAB 1 — Today’s Risk
# ======================================================
with tab1:
    colA, colB, colC = st.columns([1, 2, 1])

    elevated = prob_today >= prob_threshold
    color = "red" if elevated else "green"
    label = "ELEVATED RISK" if elevated else "LOW RISK"

    colB.markdown(
        f"""
        <h3 style='text-align:center;'>Tail Risk on {sel_ts.date()}</h3>
        <h1 style='text-align:center; color:{color}; margin-top:-8px;'>
            {prob_today:.2%}
        </h1>
        <p style='text-align:center; color:{color}; font-weight:bold; margin-top:-10px;'>
            {label}
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        **Interpretation**  
        This number represents the model-estimated probability of a **large downside event**
        occurring within the forecast horizon used during training.

        - Crash definition for diagnostics: **forward-return over the next 10 days ≤ −{crash_dd:.0%}**
        - Probability threshold (signal line): **{prob_threshold:.0%}**
        """
    )

    with st.expander("What should I do with this probability?"):
        st.markdown(
            """
            - Treat this as an **early warning indicator**, not a deterministic forecast.
            - Elevated values suggest **heightened vulnerability**, not certainty.
            - Typical actions include reducing leverage, tightening stops, or adding hedges.
            """
        )

# ======================================================
# TAB 2 — Model Behaviour
# ======================================================
with tab2:
    st.markdown(
        """
        These diagnostics test whether the model assigns **systematically higher probabilities**
        on crash days relative to non-crash days.
        """
    )

    left, right = st.columns(2)

    with left:
        fig1, ax1 = plt.subplots(figsize=(3.6, 2.6))
        ax1.hist(crash_probs, bins=20, density=True, alpha=0.6, label="Crash days", color="red")
        ax1.hist(noncrash_probs, bins=20, density=True, alpha=0.6, label="Non-crash days", color="green")
        ax1.axvline(prob_threshold, linestyle="--", color="black", linewidth=1)
        ax1.set_title("Crash vs Non-Crash Distribution")
        ax1.set_xlabel("Predicted Probability")
        ax1.legend()
        ax1.grid(alpha=0.2)
        st.pyplot(fig1)

    with right:
        fig2, ax2 = plt.subplots(figsize=(3.6, 2.6))
        ax2.bar(["Crash", "Non-crash"], [avg_crash, avg_noncrash], color=["red", "green"])
        ax2.set_ylim(0, 1)
        ax2.set_title("Average Predicted Risk")
        ax2.grid(axis="y", alpha=0.2)
        st.pyplot(fig2)

    st.markdown(
        f"""
        **Quick read:**  
        - Mean probability on crash days: **{avg_crash:.3f}**  
        - Mean probability on non-crash days: **{avg_noncrash:.3f}**
        """
    )

# ======================================================
# TAB 3 — Economic Impact
# ======================================================
with tab3:
    st.markdown(
        """
        This plot examines whether **higher predicted risk** corresponds to a **higher realized
        frequency of tail events**, which is the core probabilistic validity check.
        """
    )

    bins = [0.0, 0.2, 0.4, 0.6, 1.0]
    labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6+"]

    df_train["risk_bucket"] = pd.cut(
        df_train["prob"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    bucket_crash_rate = (
        df_train
        .groupby("risk_bucket")["is_crash"]
        .mean()
        .reindex(labels)
    )

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bucket_crash_rate.plot(kind="bar", ax=ax, color="firebrick")

    ax.set_title("Actual Crash Rate by Predicted Risk Level")
    ax.set_xlabel("Predicted Risk Bucket")
    ax.set_ylabel("Actual Crash Frequency")
    ax.set_ylim(0, bucket_crash_rate.max() * 1.3)
    ax.grid(axis="y", alpha=0.2)

    st.pyplot(fig, use_container_width=False)

    with st.expander("Show bucket values (table)"):
        st.dataframe(bucket_crash_rate.rename("crash_rate"))

# ======================================================
# TAB 4 — Methodology
# ======================================================
with tab4:
    st.markdown(
        f"""
        ### What this system estimates
        - The model outputs a **probability**, not a price forecast.
        - This probability represents the likelihood of a **tail event**
          within the horizon used during training.

        ### How “crash” is defined here
        - A crash corresponds to:
          **forward-return over the next 10 days ≤ −{crash_dd:.0%}** over the forward window.

        ### Why these diagnostics matter
        - **Distribution separation** checks discriminative ability.
        - **Average risk by outcome** provides a basic sanity check.
        - **Drawdown by risk bucket** tests economic relevance, not just statistical fit.
        - **Live market data** are retrieved dynamically from Yahoo Finance (SPY) and cached for 24 hours to ensure stability and reproducibility.
        """
    )

