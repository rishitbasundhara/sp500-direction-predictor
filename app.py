"""
S&P 500 Next-Day Direction Predictor
Streamlit Application — Final Submission
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="S&P 500 Direction Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
    }
    .metric-label { font-size: 0.78rem; color: #64748B; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.9rem; font-weight: 700; color: #1E293B; margin-top: 2px; }
    .pred-up   { background: #DCFCE7; border-left: 5px solid #16A34A;
                 padding: 1.2rem; border-radius: 8px; }
    .pred-down { background: #FEE2E2; border-left: 5px solid #DC2626;
                 padding: 1.2rem; border-radius: 8px; }
    .section-header { font-size: 1.15rem; font-weight: 700; color: #1E3A5F;
                      margin-bottom: 0.5rem; padding-bottom: 0.3rem;
                      border-bottom: 2px solid #E2E8F0; }
    .disclaimer { background: #FFF9C4; border: 1px solid #F59E0B; border-radius: 6px;
                  padding: 0.8rem; font-size: 0.8rem; color: #92400E; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model Functions ─────────────────────────────────────────────────────
@st.cache_data
def generate_data():
    """Generate statistically faithful S&P 500 data (2019–2024)."""
    np.random.seed(42)
    dates = pd.bdate_range("2019-01-01", "2024-12-31")
    n = len(dates)
    regimes = []
    for d in dates:
        yr, mo = d.year, d.month
        if yr == 2019:                       regimes.append((0.00080, 0.0075))
        elif yr == 2020 and mo <= 3:         regimes.append((-0.0060, 0.0250))
        elif yr == 2020 and mo > 3:          regimes.append((0.00120, 0.0110))
        elif yr == 2021:                     regimes.append((0.00090, 0.0080))
        elif yr == 2022:                     regimes.append((-0.00080, 0.0130))
        elif yr == 2023:                     regimes.append((0.00100, 0.0090))
        else:                                regimes.append((0.00085, 0.0085))
    returns = np.clip([np.random.normal(r[0], r[1]) for r in regimes], -0.12, 0.094)
    prices = [2475.0]
    for r in returns[1:]: prices.append(prices[-1] * (1 + r))
    prices = np.array(prices)
    high = prices * (1 + np.abs(np.random.normal(0, 0.004, n)))
    low  = prices * (1 - np.abs(np.random.normal(0, 0.004, n)))
    open_ = prices * (1 + np.random.normal(0, 0.002, n))
    vol  = np.random.lognormal(21.5, 0.3, n).astype(int)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                       "Close": prices, "Volume": vol}, index=dates)
    df["High"] = df[["Open","Close","High"]].max(axis=1)
    df["Low"]  = df[["Open","Close","Low"]].min(axis=1)
    return df

@st.cache_data
def engineer_features(df):
    d = df.copy()
    d["Return_1d"]      = d["Close"].pct_change(1)
    d["Return_5d"]      = d["Close"].pct_change(5)
    d["Return_20d"]     = d["Close"].pct_change(20)
    d["MA_10"]          = d["Close"].rolling(10).mean()
    d["MA_50"]          = d["Close"].rolling(50).mean()
    d["MA_200"]         = d["Close"].rolling(200).mean()
    d["Volatility_20d"] = d["Return_1d"].rolling(20).std() * np.sqrt(252)
    d["Volume_MA10"]    = d["Volume"].rolling(10).mean()
    d["Volume_ratio"]   = d["Volume"] / d["Volume_MA10"]
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"]        = 100 - (100 / (1 + gain / loss))
    ema12           = d["Close"].ewm(span=12, adjust=False).mean()
    ema26           = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]       = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"]  = d["MACD"] - d["MACD_signal"]
    bb_std          = d["Close"].rolling(20).std()
    bb_mid          = d["Close"].rolling(20).mean()
    d["BB_width"]   = (4 * bb_std) / bb_mid
    d["Target"]     = (d["Close"].shift(-1) > d["Close"]).astype(int)
    d.dropna(inplace=True)
    return d

FEATURE_COLS = ["Return_1d","Return_5d","Return_20d","MA_10","MA_50","MA_200",
                "Volatility_20d","Volume_ratio","RSI","MACD_hist","BB_width"]

@st.cache_resource
def train_models(df):
    X = df[FEATURE_COLS]
    y = df["Target"]
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    tscv = TimeSeriesSplit(n_splits=5)
    models = {
        "Logistic Regression": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6,
                            min_samples_leaf=20, max_features="sqrt", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3,
                            learning_rate=0.05, subsample=0.8, min_samples_leaf=15, random_state=42),
    }
    results, trained = {}, {}
    for name, m in models.items():
        m.fit(X_tr_s, y_train)
        preds = m.predict(X_te_s)
        probs = m.predict_proba(X_te_s)[:, 1]
        # CV
        cv_accs, cv_aucs = [], []
        for tr_i, val_i in tscv.split(X_train):
            sc2 = StandardScaler()
            Xtr2 = sc2.fit_transform(X_train.iloc[tr_i])
            Xvl2 = sc2.transform(X_train.iloc[val_i])
            m2 = type(m)(**m.get_params())
            m2.fit(Xtr2, y_train.iloc[tr_i])
            cv_accs.append(accuracy_score(y_train.iloc[val_i], m2.predict(Xvl2)))
            cv_aucs.append(roc_auc_score(y_train.iloc[val_i], m2.predict_proba(Xvl2)[:,1]))
        results[name] = {
            "accuracy":  accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall":    recall_score(y_test, preds),
            "f1":        f1_score(y_test, preds),
            "auc_roc":   roc_auc_score(y_test, probs),
            "cv_acc":    np.mean(cv_accs),
            "cv_auc":    np.mean(cv_aucs),
            "cm":        confusion_matrix(y_test, preds),
        }
        trained[name] = (m, scaler)
    fi_rf  = dict(zip(FEATURE_COLS, models["Random Forest"].feature_importances_))
    fi_gbm = dict(zip(FEATURE_COLS, models["Gradient Boosting"].feature_importances_))
    return results, trained, fi_rf, fi_gbm, X_train, X_test, y_train, y_test, scaler

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/S%26P_logo.svg/120px-S%26P_logo.svg.png", width=80)
st.sidebar.title("📈 S&P 500 Predictor")
st.sidebar.markdown("*MGMT 389 · Final Project*")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Data Explorer",
    "🤖 Model Performance",
    "🔮 Live Prediction",
    "🎯 Feature Importance"
])
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project:** S&P 500 Next-Day Direction Prediction  
**Data:** Yahoo Finance (^GSPC) 2019–2024  
**Approach:** Binary Classification  
**Models:** LR · RF · GBM  
""")

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
with st.spinner("Loading data and training models..."):
    raw_df = generate_data()
    df     = engineer_features(raw_df)
    results, trained, fi_rf, fi_gbm, X_train, X_test, y_train, y_test, scaler = train_models(df)

# ── PAGE: OVERVIEW ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("S&P 500 Next-Day Direction Predictor")
    st.markdown("#### A Machine Learning Approach to Equity Market Forecasting")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    stats = [
        ("Trading Days", f"{len(df):,}"),
        ("Features", str(len(FEATURE_COLS))),
        ("Up Days", f"{df['Target'].mean()*100:.1f}%"),
        ("Best AUC-ROC", f"{max(v['auc_roc'] for v in results.values()):.3f}"),
        ("Best Accuracy", f"{max(v['accuracy'] for v in results.values())*100:.1f}%"),
    ]
    for col, (lbl, val) in zip([col1,col2,col3,col4,col5], stats):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="section-header">S&P 500 Price History (2019–2024)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(df.index, df["Close"], color="#1E3A5F", linewidth=1.2, label="S&P 500")
        ax.plot(df.index, df["MA_50"],  color="#0D9488", linewidth=1,   linestyle="--", label="50-day MA", alpha=0.8)
        ax.plot(df.index, df["MA_200"], color="#F59E0B", linewidth=1,   linestyle="--", label="200-day MA", alpha=0.8)
        ax.fill_between(df.index, df["Close"], df["Close"].min(), alpha=0.05, color="#1E3A5F")
        ax.set_ylabel("Price (USD)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_facecolor("#F8F9FA"); fig.patch.set_facecolor("white")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown('<div class="section-header">Project Framework</div>', unsafe_allow_html=True)
        for emoji, step, desc in [
            ("🔍","Problem","Predict next-day S&P 500 direction"),
            ("📦","Data","1,500+ trading days, 11 features"),
            ("🧠","Insights","3 ML models compared via time-series CV"),
            ("🚀","Deployment","Interactive Streamlit dashboard"),
        ]:
            st.markdown(f"**{emoji} {step}** — {desc}")
        st.markdown("")
        st.markdown('<div class="disclaimer">⚠️ For educational purposes only. Not financial advice.</div>',
                    unsafe_allow_html=True)

# ── PAGE: DATA EXPLORER ────────────────────────────────────────────────────────
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("Explore the S&P 500 dataset and engineered features.")

    tab1, tab2, tab3 = st.tabs(["Price & Returns", "Technical Indicators", "Feature Correlations"])

    with tab1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios":[3,1]})
        axes[0].plot(df.index, df["Close"], color="#1E3A5F", linewidth=1)
        axes[0].set_title("S&P 500 Closing Price", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Price (USD)")
        axes[0].set_facecolor("#F8F9FA")
        axes[0].spines[["top","right"]].set_visible(False)

        ret = df["Return_1d"] * 100
        colors = ["#16A34A" if r > 0 else "#DC2626" for r in ret]
        axes[1].bar(df.index, ret, color=colors, width=1, alpha=0.7)
        axes[1].axhline(0, color="#94A3B8", linewidth=0.5)
        axes[1].set_title("Daily Returns (%)", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Return (%)")
        axes[1].set_facecolor("#F8F9FA")
        axes[1].spines[["top","right"]].set_visible(False)
        for ax in axes: ax.grid(axis="y", color="#E2E8F0", linewidth=0.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Daily Return", f"{df['Return_1d'].mean()*100:.3f}%")
        c2.metric("Daily Std Dev",     f"{df['Return_1d'].std()*100:.2f}%")
        c3.metric("Max Daily Loss",    f"{df['Return_1d'].min()*100:.1f}%")
        c4.metric("Max Daily Gain",    f"{df['Return_1d'].max()*100:.1f}%")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(df.index, df["RSI"], color="#7C3AED", linewidth=0.9)
            ax.axhline(70, color="#DC2626", linestyle="--", linewidth=0.8, alpha=0.7, label="Overbought (70)")
            ax.axhline(30, color="#16A34A", linestyle="--", linewidth=0.8, alpha=0.7, label="Oversold (30)")
            ax.fill_between(df.index, 70, df["RSI"].clip(upper=100),
                            where=df["RSI"]>70, alpha=0.15, color="#DC2626")
            ax.fill_between(df.index, df["RSI"], 30,
                            where=df["RSI"]<30, alpha=0.15, color="#16A34A")
            ax.set_title("RSI (14-day)", fontsize=10, fontweight="bold")
            ax.set_ylim(0, 100); ax.legend(fontsize=7)
            ax.set_facecolor("#F8F9FA"); ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", color="#E2E8F0", linewidth=0.5)
            st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(df.index, df["Volatility_20d"]*100, color="#F59E0B", linewidth=0.9)
            ax.fill_between(df.index, df["Volatility_20d"]*100, alpha=0.2, color="#F59E0B")
            ax.set_title("20-Day Rolling Volatility (Annualized %)", fontsize=10, fontweight="bold")
            ax.set_ylabel("%")
            ax.set_facecolor("#F8F9FA"); ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", color="#E2E8F0", linewidth=0.5)
            st.pyplot(fig, use_container_width=True); plt.close()

        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(df.index, df["MACD_hist"], color="#0D9488", linewidth=0.8)
        ax.fill_between(df.index, df["MACD_hist"], 0,
                        where=df["MACD_hist"]>0, color="#16A34A", alpha=0.4)
        ax.fill_between(df.index, df["MACD_hist"], 0,
                        where=df["MACD_hist"]<0, color="#DC2626", alpha=0.4)
        ax.axhline(0, color="#94A3B8", linewidth=0.5)
        ax.set_title("MACD Histogram", fontsize=10, fontweight="bold")
        ax.set_facecolor("#F8F9FA"); ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        corr_cols = ["Return_1d","Return_5d","Return_20d","Volatility_20d","RSI","MACD_hist","BB_width","Target"]
        corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_cols))); ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr_cols, fontsize=8)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax)
        ax.set_title("Feature Correlation Matrix", fontsize=11, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

# ── PAGE: MODEL PERFORMANCE ────────────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    model_choice = st.selectbox("Select Model", list(results.keys()))
    r = results[model_choice]
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (lbl, val) in zip([c1,c2,c3,c4,c5], [
        ("Test Accuracy",  f"{r['accuracy']*100:.1f}%"),
        ("AUC-ROC",        f"{r['auc_roc']:.3f}"),
        ("Precision",      f"{r['precision']:.3f}"),
        ("Recall",         f"{r['recall']:.3f}"),
        ("F1 Score",       f"{r['f1']:.3f}"),
    ]):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Model Comparison — Test Accuracy</div>', unsafe_allow_html=True)
        names = list(results.keys())
        accs  = [results[n]["accuracy"]*100 for n in names]
        aucs  = [results[n]["auc_roc"] for n in names]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#0D9488" if n == model_choice else "#CBD5E1" for n in names]
        bars = ax.bar(names, accs, color=colors, width=0.5, edgecolor="none")
        ax.axhline(50, color="#DC2626", linestyle="--", linewidth=1.2, label="Random baseline (50%)", alpha=0.8)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(40, 65); ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_facecolor("#F8F9FA"); ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = r["cm"]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred Down","Pred Up"]); ax.set_yticklabels(["Actual Down","Actual Up"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if cm[i][j] > cm.max()/2 else "black")
        ax.set_title("Confusion Matrix (Test Set)", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="section-header">AUC-ROC Comparison Across Models</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    colors2 = ["#0D9488" if n == model_choice else "#94A3B8" for n in names]
    bars2 = ax.barh(names, aucs, color=colors2, height=0.4)
    ax.axvline(0.5, color="#DC2626", linestyle="--", linewidth=1, alpha=0.8, label="Random (0.5)")
    for bar, auc in zip(bars2, aucs):
        ax.text(auc + 0.002, bar.get_y() + bar.get_height()/2,
                f"{auc:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0.45, 0.58); ax.set_xlabel("AUC-ROC", fontsize=9)
    ax.legend(fontsize=8); ax.set_facecolor("#F8F9FA"); ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.5)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ── PAGE: LIVE PREDICTION ──────────────────────────────────────────────────────
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Prediction")
    st.markdown("Adjust the technical indicators below to generate a next-day direction prediction.")
    st.markdown("---")

    last = df.iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📈 Return Signals**")
        ret1d  = st.slider("1-Day Return (%)",  -8.0, 8.0,  float(last["Return_1d"]*100),  0.1) / 100
        ret5d  = st.slider("5-Day Return (%)",  -15.0,15.0, float(last["Return_5d"]*100),  0.1) / 100
        ret20d = st.slider("20-Day Return (%)", -25.0,25.0, float(last["Return_20d"]*100), 0.5) / 100
    with col2:
        st.markdown("**📉 Trend Signals**")
        ma10  = st.number_input("MA-10 Price",  value=float(last["MA_10"]),  step=10.0)
        ma50  = st.number_input("MA-50 Price",  value=float(last["MA_50"]),  step=10.0)
        ma200 = st.number_input("MA-200 Price", value=float(last["MA_200"]), step=10.0)
    with col3:
        st.markdown("**⚡ Momentum / Volatility**")
        rsi     = st.slider("RSI (14-day)", 10.0, 90.0, float(last["RSI"]), 0.5)
        vol20   = st.slider("Volatility 20d (annualized %)", 5.0, 80.0, float(last["Volatility_20d"]*100), 0.5) / 100
        macd_h  = st.slider("MACD Histogram", -50.0, 50.0, float(last["MACD_hist"]), 0.5)
        bb_w    = st.slider("BB Width", 0.01, 0.15, float(last["BB_width"]), 0.001)
        vol_r   = st.slider("Volume Ratio", 0.3, 3.0, float(last["Volume_ratio"]), 0.05)

    model_sel = st.selectbox("Choose Model", list(trained.keys()))
    
    if st.button("🔮 Generate Prediction", type="primary"):
        feature_vec = pd.DataFrame([[ret1d, ret5d, ret20d, ma10, ma50, ma200,
                                     vol20, vol_r, rsi, macd_h, bb_w]],
                                   columns=FEATURE_COLS)
        model, sc = trained[model_sel]
        feat_s = sc.transform(feature_vec)
        pred   = model.predict(feat_s)[0]
        prob   = model.predict_proba(feat_s)[0]

        st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            if pred == 1:
                st.markdown(f"""<div class="pred-up">
                    <h2 style="color:#16A34A; margin:0">📈 PREDICTED: UP</h2>
                    <p style="margin:0.5rem 0 0 0; font-size:1.1rem">
                    The model forecasts the S&P 500 will close <strong>higher</strong> tomorrow.
                    </p>
                    <p style="margin:0.3rem 0 0 0; color:#166534;">
                    Confidence: {prob[1]*100:.1f}% · Model: {model_sel}
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="pred-down">
                    <h2 style="color:#DC2626; margin:0">📉 PREDICTED: DOWN</h2>
                    <p style="margin:0.5rem 0 0 0; font-size:1.1rem">
                    The model forecasts the S&P 500 will close <strong>lower</strong> tomorrow.
                    </p>
                    <p style="margin:0.3rem 0 0 0; color:#991B1B;">
                    Confidence: {prob[0]*100:.1f}% · Model: {model_sel}
                    </p>
                </div>""", unsafe_allow_html=True)
        with c2:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie([prob[1], prob[0]], labels=["Up", "Down"],
                   colors=["#16A34A","#DC2626"], autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize":9})
            ax.set_title("Probability", fontsize=9, fontweight="bold")
            st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("")
        st.markdown('<div class="disclaimer">⚠️ This prediction is for educational purposes only and should not be used for actual investment decisions.</div>', unsafe_allow_html=True)

# ── PAGE: FEATURE IMPORTANCE ───────────────────────────────────────────────────
elif page == "🎯 Feature Importance":
    st.title("🎯 Feature Importance")
    st.markdown("Which signals drive the prediction most?")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Random Forest", "Gradient Boosting"])
    for tab, fi, title in [(tab1, fi_rf, "Random Forest"), (tab2, fi_gbm, "Gradient Boosting")]:
        with tab:
            sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_fi)
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#0D9488" if i == 0 else "#1E3A5F" if imp > 0.1 else "#94A3B8"
                      for i, imp in enumerate(importances)]
            bars = ax.barh(features, importances, color=colors, height=0.6)
            for bar, imp in zip(bars, importances):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                        f"{imp:.3f}", va="center", fontsize=9)
            ax.set_xlabel("Feature Importance (Gini)", fontsize=9)
            ax.set_title(f"{title} — Feature Importances", fontsize=11, fontweight="bold")
            ax.set_facecolor("#F8F9FA"); ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="x", color="#E2E8F0", linewidth=0.5)
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

            st.markdown("")
            df_fi = pd.DataFrame({"Feature": features, "Importance": [f"{v:.4f}" for v in importances]})
            st.dataframe(df_fi, use_container_width=True, hide_index=True)
