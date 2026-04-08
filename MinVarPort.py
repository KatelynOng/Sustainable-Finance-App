import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# 1. Page Configuration & Professional Styling
# =========================================================
st.set_page_config(page_title="ESG Portfolio Optimiser", layout="wide")

# Custom CSS for modern UI
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
    .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

def add_styled_header(subtitle="Mean-variance optimisation with ESG screening"):
    st.markdown(f"""
        <div style="padding: 0.5rem 0 1.5rem;">
            <h1 style="font-size: 2rem; font-weight: 700; color: #1B1B1B; margin-bottom: 0.3rem;">
                ESG Portfolio Optimiser
            </h1>
            <p style="color: #6B7280; font-size: 1.05rem;">
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

def reset_onboarding():
    for k, v in DEFAULTS.items(): st.session_state[k] = v
    st.rerun()

# =========================================================
# 3. Calculation Logic
# =========================================================
def var_covar(sigmas, rho):
    return np.array([[sigmas[0]**2, rho*sigmas[0]*sigmas[1]], [rho*sigmas[0]*sigmas[1], sigmas[1]**2]])

def build_portfolio_grid(mu, sigma, rho, rf, esg_scores, gamma, lambda_esg, num_points):
    cov = var_covar(sigma, rho)
    weights = np.linspace(0, 1, num_points)
    rows = []
    for w1 in weights:
        w = np.array([w1, 1 - w1])
        exp_ret = float(np.dot(mu, w))
        var = float(np.dot(w, np.dot(cov, w)))
        std = np.sqrt(max(var, 0))
        esg = float(np.dot(esg_scores, w))
        sharpe = (exp_ret - rf) / std if std > 0 else 0
        utility = exp_ret - 0.5 * gamma * var + lambda_esg * esg
        rows.append({"Weight Asset 1": w1, "Weight Asset 2": 1-w1, "Expected Return": exp_ret, "Std Dev": std, "ESG Score": esg, "Sharpe Ratio": sharpe, "Utility": utility})
    return pd.DataFrame(rows)

@st.cache_data
def load_fast_workbook():
    file_path = Path("esg_crsp_2025_matched_workbook.xlsx")
    if not file_path.exists():
        st.error("Excel file not found!")
        return None, None, None
    summary = pd.read_excel(file_path, sheet_name="Summary")
    cov = pd.read_excel(file_path, sheet_name="Covariance Matrix", index_col=0)
    # Basic cleaning
    summary = summary.rename(columns={"ESG Company Name (2025)": "Company", "ESGCombinedScore 2025": "ESG Score", "Latest Available Annual Return (2024)": "Expected Return"})
    return summary, cov, None

# =========================================================
# 4. App Pages
# =========================================================

# --- Onboarding ---
if not st.session_state.onboarding_complete:
    add_styled_header("Investor Onboarding")
    st.info("Please select your investor profile to continue.")
    investor_type = st.radio("Are you an experienced investor?", ["Experienced Investor", "New to Investing"])
    if st.button("Start Analysis"):
        st.session_state.investor_type = investor_type
        st.session_state.onboarding_complete = True
        st.rerun()
    st.stop()

# --- Inputs Page ---
if st.session_state.page == "inputs":
    add_styled_header("Portfolio Configuration")
    with st.form("input_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Asset 1 (Low ESG)")
            mu1 = st.number_input("Return (%)", 0.0, 100.0, float(st.session_state.mu1_pct))
            sig1 = st.number_input("Risk (%)", 0.1, 100.0, float(st.session_state.sigma1_pct))
            esg1 = st.number_input("ESG Score", 0.0, 100.0, float(st.session_state.esg1))
        with c2:
            st.subheader("Asset 2 (High ESG)")
            mu2 = st.number_input("Return (%)", 0.0, 100.0, float(st.session_state.mu2_pct))
            sig2 = st.number_input("Risk (%)", 0.1, 100.0, float(st.session_state.sigma2_pct))
            esg2 = st.number_input("ESG Score", 0.0, 100.0, float(st.session_state.esg2))
        
        st.divider()
        rho = st.slider("Correlation", -1.0, 1.0, float(st.session_state.rho))
        lambda_esg = st.slider("ESG Preference (λ)", 0.0, 1.0, float(st.session_state.lambda_esg))
        
        if st.form_submit_button("Generate Results"):
            st.session_state.mu1_pct, st.session_state.mu2_pct = mu1, mu2
            st.session_state.sigma1_pct, st.session_state.sigma2_pct = sig1, sig2
            st.session_state.esg1, st.session_state.esg2 = esg1, esg2
            st.session_state.rho, st.session_state.lambda_esg = rho, lambda_es
