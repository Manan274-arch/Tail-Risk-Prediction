import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

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
# Load assets
# ======================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tail_risk_model.keras")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_features():
    return joblib.load("features.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("processed_data.csv")
    # If you later export a real Date column, we will use it automatically
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date").sort_index()
    else:
        # Placeholder business-day index (only because CSV has no dates)
        df.index = pd.date_range(start="1990-01-01", periods=len(df), freq="B")
        df.index.name = "Date"
    return df

# Load
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
    st.error("`processed_data.csv` must contain a `forward_return` column for crash labeling and drawdown plots.")
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
# Inference (compute all probabilities once)
# ======================================================
X_all = scaler.transform(df[features])
df["prob"] = model.predict(X_all, verbose=0).ravel()

# Derive crash label from forward returns (no crash column needed)
df["is_crash"] = (df["forward_return"] <= -crash_dd).astype(int)

# Today's risk
sel_ts = pd.to_datetime(selected_date)
if sel_ts not in df.index:
    # If user picks a non-index date (rare), fallback to nearest previous date
    nearest = df.index[df.index.get_indexer([sel_ts], method="pad")][0]
    sel_ts = nearest

prob_today = float(df.loc[sel_ts, "prob"])

# Precompute for plots
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

# ======================================================
# TAB 1 — Today’s Risk
# ======================================================
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
        This is the model-estimated probability of a **large downside event** (tail event)
        occurring within the forecast horizon used during training.

        - Crash definition used for diagnostics: **forward_return ≤ −{crash_dd:.0%}**
        - Probability threshold (signal line): **{prob_threshold:.0%}**
        """
    )

    with st.expander("What should I do with this probability?"):
        st.markdown(
            """
            - Treat it as an **early warning indicator**, not a deterministic crash predictor.
            - Use it to **tighten risk** (reduce leverage, tighten stops, hedge) when elevated.
            - Track how the probability behaves across regimes (quiet vs stressed periods).
            """
        )

# ======================================================
# TAB 2 — Model Behaviour (your 2 Kaggle plots)
# ======================================================
with tab2:
    st.markdown(
        """
        These plots check whether the model assigns **higher probabilities on crash days**
        than on non-crash days. This is the core “does it separate outcomes?” diagnostic.
        """
    )

    if len(crash_probs) == 0 or len(noncrash_probs) == 0:
        st.warning(
            "One of the groups (crash/non-crash) is empty under the current crash definition. "
            "Try adjusting the crash drawdown slider in the sidebar."
        )
    else:
        left, right = st.columns(2)

        # ---- Plot 1: Crash vs Non-crash distribution
        with left:
            fig1, ax1 = plt.subplots(figsize=(4.0, 3.0))

            ax1.hist(crash_probs, bins=25, alpha=0.6, density=True,
                     label="Crash days", color="red")
            ax1.hist(noncrash_probs, bins=25, alpha=0.6, density=True,
                     label="Non-crash days", color="green")

            ax1.axvline(prob_threshold, color="black", linestyle="--", linewidth=1, label="Threshold")
            ax1.set_title("Crash vs Non-Crash Distribution")
            ax1.set_xlabel("Predicted Crash Probability")
            ax1.set_ylabel("Density")
            ax1.legend()
            ax1.grid(alpha=0.2)
            st.pyplot(fig1, use_container_width=False)

        # ---- Plot 2: Avg risk by outcome
        with right:
            fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
            ax2.bar(["Crash days", "Non-crash days"], [avg_crash, avg_noncrash], color=["red", "green"])
            ax2.set_ylim(0, 1)
            ax2.set_title("Average Model Risk by Outcome")
            ax2.set_ylabel("Average Predicted Probability")
            ax2.grid(axis="y", alpha=0.2)
            st.pyplot(fig2, use_container_width=False)

        st.markdown(
            f"""
            **Quick read:**  
            - Mean probability on crash days: **{avg_crash:.3f}**  
            - Mean probability on non-crash days: **{avg_noncrash:.3f}**  
            """
        )

# ======================================================
# TAB 3 — Economic Impact (your risk bucket drawdown plot)
# ======================================================
with tab3:
    st.markdown(
        """
        This plot checks whether **higher predicted risk** corresponds to **worse realized downside**.
        That is, it tests the **economic relevance** of your probability output.
        """
    )

    bins = [0.0, 0.2, 0.4, 0.6, 1.0]
    labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6+"]

    df["risk_bucket"] = pd.cut(df["prob"], bins=bins, labels=labels)

    def max_drawdown(returns: pd.Series) -> float:
        r = returns.dropna()
        if len(r) == 0:
            return np.nan
        cumulative = (1 + r).cumprod()
        peak = cumulative.cummax()
        dd = (cumulative - peak) / peak
        return float(dd.min())

    bucket_drawdowns = (
        df.groupby("risk_bucket")["forward_return"]
        .apply(max_drawdown)
        .reindex(labels)
    )

    fig3, ax3 = plt.subplots(figsize=(5.0, 3.0))
    bucket_drawdowns.plot(kind="bar", ax=ax3, color="firebrick")
    ax3.set_title("Maximum Drawdown by Predicted Risk Bucket")
    ax3.set_xlabel("Predicted Risk Bucket")
    ax3.set_ylabel("Maximum Drawdown")
    ax3.axhline(0, color="black", linewidth=1)
    ax3.grid(axis="y", alpha=0.2)
    st.pyplot(fig3, use_container_width=False)

    with st.expander("Show bucket values (table)"):
        st.dataframe(bucket_drawdowns.rename("max_drawdown"))

# ======================================================
# TAB 4 — Methodology (full explanation text)
# ======================================================
with tab4:
    st.markdown(
        f"""
        ### What this system estimates
        - The model outputs a **probability** intended to represent the likelihood of a **tail event**
          (large downside move) occurring within the forecast horizon used during training.
        - It is not a point forecast of price, and it is not a guarantee of a crash.

        ### How we define “crash” in this dashboard
        - Crash in this model is defined over a period of 10 days exhibiting:
          **forward_return ≤ −{crash_dd:.0%}**

        ### Why these diagnostic plots matter
        - **Crash vs Non-crash pr`obability distributions**: checks whether the model assigns higher probabilities
          on crash days than on non-crash days (discriminative ability).
        - **Average risk by outcome**: a simple sanity check that the model’s average predicted risk is directionally correct.
        - **Max drawdown by predicted risk bucket**: tests whether high predicted risk corresponds to economically meaningful
          downside severity (economic validity)."""
    )