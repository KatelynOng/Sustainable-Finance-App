import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# 1. Page Configuration & Professional Styling
# =========================================================
st.set_page_config(page_title="ESG Portfolio Optimiser", layout="wide")

# Custom CSS for modern UI and "FinTech" look
st.markdown("""
    <style>
    .stApp { background-color: #F4F6F3; }
    [data-testid="stMetricContainer"] {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

def add_styled_header(subtitle="Mean-variance optimisation with ESG screening"):
    st.markdown(f"""
        <div style="padding: 1rem 0 1.5rem;">
            <h1 style="font-size: 1.6rem; font-weight: 600; color: #1B1B1B; margin-bottom: 0.3rem;">
                ESG Portfolio Optimiser
            </h1>
            <p style="color: #6B7280; font-size: 0.95rem;">
                {subtitle}
            </p>
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# 2. Defaults & Session State
# =========================================================
DEFAULTS = {
    "page": "inputs",
    "mu1_pct": 5.00, "mu2_pct": 12.00,
    "sigma1_pct": 9.00, "sigma2_pct": 20.00,
    "rf_pct": 2.00, "rho": -0.20,
    "esg1": 35.0, "esg2": 80.0,
    "lambda_esg": 0.30, "gamma": 3.0,
    "num_points": 1001, "frontier_points": 80,
    "trading_days": 252, "onboarding_complete": False,
    "onboarding_step": "investor_type", "beginner_mode": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go_to(page_name: str):
    st.session_state.page = page_name
    st.rerun()

# =========================================================
# 3. Calculation Logic (Teaching Model)
# =========================================================
def var_covar(sigmas, rho):
    return np.array([[sigmas[0]**2, rho*sigmas[0]*sigmas[1]], [rho*sigmas[0]*sigmas[1], sigmas[1]**2]])

def build_portfolio_grid(mu, sigma, rho, rf, esg_scores, gamma, lambda_esg, num_points):
    cov = var_covar(sigma, rho)
    weights = np.linspace(0, 1, num_points)
    rows = []
    for w1 in weights:
        w = np.array([w1, 1 - w1])
        exp_return = float(np.dot(mu, w))
        variance = float(np.dot(w, np.dot(cov, w)))
        std_dev = np.sqrt(max(variance, 0))
        esg_score = float(np.dot(esg_scores, w))
        sharpe = (exp_return - rf) / std_dev if std_dev > 0 else 0
        rows.append({"Weight Asset 1": w1, "Weight Asset 2": 1-w1, "Expected Return": exp_return, "Std Dev": std_dev, "ESG Score": esg_score, "Sharpe Ratio": sharpe})
    return pd.DataFrame(rows)

# =========================================================
# 4. Page Logic
# =========================================================

# Onboarding Skip Logic (Simplified for brevity)
if not st.session_state.onboarding_complete:
    add_styled_header("Investor Onboarding")
    if st.button("Start App with Defaults"):
        st.session_state.onboarding_complete = True
        st.rerun()
    st.stop()

# --- INPUTS PAGE ---
if st.session_state.page == "inputs":
    add_styled_header("Configure Your Portfolio Parameters")
    
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Asset 1 (Low ESG)")
            mu1_pct = st.number_input("Return (%)", 0.0, 100.0, float(st.session_state.mu1_pct))
            esg1 = st.number_input("ESG Score (0-100)", 0.0, 100.0, float(st.session_state.esg1))
        with col2:
            st.subheader("Asset 2 (High ESG)")
            mu2_pct = st.number_input("Return (%)", 0.0, 100.0, float(st.session_state.mu2_pct))
            esg2 = st.number_input("ESG Score (0-100)", 0.0, 100.0, float(st.session_state.esg2))
        
        st.divider()
        lambda_esg = st.slider("ESG Preference Intensity (λ)", 0.0, 1.0, float(st.session_state.lambda_esg))
        
        if st.form_submit_button("Calculate Results"):
            st.session_state.mu1_pct, st.session_state.mu2_pct = mu1_pct, mu2_pct
            st.session_state.esg1, st.session_state.esg2 = esg1, esg2
            st.session_state.lambda_esg = lambda_esg
            go_to("results")

# --- RESULTS PAGE ---
elif st.session_state.page == "results":
    add_styled_header("Analysis & Optimisation Results")
    
    # 1. Data Prep
    mu = np.array([st.session_state.mu1_pct, st.session_state.mu2_pct]) / 100.0
    sigma = np.array([st.session_state.sigma1_pct, st.session_state.sigma2_pct]) / 100.0
    rf = st.session_state.rf_pct / 100.0
    df_all = build_portfolio_grid(mu, sigma, st.session_state.rho, rf, np.array([st.session_state.esg1, st.session_state.esg2])/100, st.session_state.gamma, st.session_state.lambda_esg, 500)
    
    tan_pt = df_all.loc[df_all["Sharpe Ratio"].idxmax()]

    # 2. KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Optimal Return", f"{tan_pt['Expected Return']:.2%}")
    m2.metric("Optimal Risk", f"{tan_pt['Std Dev']:.2%}")
    m3.metric("Sharpe Ratio", f"{tan_pt['Sharpe Ratio']:.3f}")
    m4.metric("Portfolio ESG", f"{tan_pt['ESG Score']*100:.1f}/100")

    # 3. Interactive Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_all["Std Dev"]*100, y=df_all["Expected Return"]*100,
        mode='lines', name='Frontier', line=dict(color='#1D9E75', width=3),
        hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[tan_pt["Std Dev"]*100], y=[tan_pt["Expected Return"]*100],
        mode='markers', marker=dict(size=12, color='Gold', symbol='star'),
        name='Tangency Portfolio'
    ))
    fig.update_layout(
        title="Interactive Efficient Frontier",
        xaxis_title="Volatility (Risk %)", yaxis_title="Expected Return (%)",
        template="plotly_white", height=500,
        hovermode="closest"
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("← Edit Inputs"):
        go_to("inputs")
