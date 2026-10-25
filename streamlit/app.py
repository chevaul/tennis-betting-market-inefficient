import streamlit as st
import json
from pathlib import Path

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Tennis Betting Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rename "app" to "Home" in sidebar
st.markdown("""
<style>
    [data-testid="stSidebarNav"] li:first-child a {
        visibility: hidden;
        position: relative;
    }
    [data-testid="stSidebarNav"] li:first-child a::before {
        content: "🏠 Home";
        visibility: visible;
        position: absolute;
        left: 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL STATS
# ============================================================================

@st.cache_data
def load_stats():
    """Load model statistics with robust path handling"""
    try:
        # Get directory where this script is located
        script_dir = Path(__file__).resolve().parent
        # Go up one level to repo root, then into models
        stats_path = script_dir.parent / 'models' / 'model_stats.json'
        
        with open(stats_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Using fallback stats (file not found: {e})")
        # Fallback data - hardcoded from your results
        return {
            'accuracy': 0.7361,
            'auc': 0.815,
            'log_loss': 0.5238,
            'roi_conservative': 42.17,
            'roi_aggressive': 79.75,
            'annual_roi_conservative': 3.6,
            'annual_roi_aggressive': 6.0,
            'sharpe_conservative': 4.21,
            'sharpe_aggressive': 3.44,
            'baseline_roi': -3.68,
            'baseline_sharpe': -1.23,
            'test_matches': 4972,
            'value_bets_conservative': 1702,
            'value_bets_aggressive': 708
        }

stats = load_stats()

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("🎾 Tennis Match Prediction & Betting Analytics")
st.markdown("### ML-Powered Tennis Betting Strategy Tool")
st.markdown("*Developed by LSE Data Science Society*")

st.markdown("---")

# ============================================================================
# KEY METRICS DISPLAY
# ============================================================================

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy", 
        f"{stats['accuracy']*100:.2f}%",
        delta="+9.6 pts vs Elo",
        help="Correctly predicts 73.6% of matches"
    )

with col2:
    st.metric(
        "AUC", 
        f"{stats['auc']:.3f}",
        delta="+0.115 vs Elo",
        help="Excellent discrimination between winners and losers"
    )

with col3:
    st.metric(
        "Log Loss",
        f"{stats['log_loss']:.3f}",
        delta="-9.9% vs Market",
        delta_color="inverse",
        help="Better probability calibration than bookmakers"
    )

with col4:
    st.metric(
        "Annual ROI",
        f"{stats['annual_roi_conservative']:.1f}%",
        delta=f"+{stats['annual_roi_conservative'] - stats['baseline_roi']:.1f}% vs Baseline",
        help="Conservative betting strategy returns"
    )

st.markdown("---")

# ============================================================================
# PROJECT OVERVIEW
# ============================================================================

st.subheader("🔍 Project Overview")

st.markdown(f"""
Our machine learning system analyzes ATP tennis matches to identify betting market inefficiencies.

**What We Built:**
- Trained on **24,875 matches** (2015-2024)
- Tested on **{stats['test_matches']:,} unseen matches**
- CatBoost model with **44 features** (Elo, rankings, form, head-to-head)
- Validated with temporal testing (71.7% accuracy on 2024 data)

**Key Findings:**
- Model beats market by **9.9% in probability calibration**
- Identified **favorite-longshot bias**: markets systematically overprice favorites
- Conservative strategy: **3.6% annual ROI** with Sharpe ratio **{stats['sharpe_conservative']:.1f}**
- Aggressive strategy: **6.0% annual ROI** with Sharpe ratio **{stats['sharpe_aggressive']:.1f}**
- Market baseline **loses 3.7% annually** betting on favorites
""")

# ============================================================================
# NAVIGATION GUIDE
# ============================================================================

st.markdown("---")

st.subheader("📱 Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🎯 Match Predictor**
    - Predict any ATP match
    - Get win probabilities and edge vs market
    - Kelly criterion bet sizing
    
    **📈 Wimbledon 2024 Case Study**
    - Real tournament P&L analysis
    - Strategy performance metrics
    - Risk-adjusted returns
    """)

with col2:
    st.markdown("""
    **💰 Kelly Calculator**
    - Interactive bet sizing tool
    - Risk management calculator
    - Optimal stake recommendations
    
    **📊 Portfolio Dashboard**
    - Risk metrics visualization
    - Surface diversification
    - Drawdown analysis
    """)

st.info("👈 **Select a page from the sidebar to get started!**")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>LSE Data Science Society - AT Project 2026</strong></p>
    <p>Ecem Yildirim (Lead) | Haoyuan Xu | Minh Hoang | Jennifer Cheung</p>
    <p>For March 2026 Data Science Conference</p>
</div>
""", unsafe_allow_html=True)
