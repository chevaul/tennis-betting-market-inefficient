"""
PAGE 3: Kelly Criterion Calculator
Interactive tool for optimal bet sizing in tennis betting
Tab 1: Manual calculator with interactive charts
Tab 2: Upload model predictions CSV → auto-identify betting opportunities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# PAGE SETUP
# ============================================================================

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

st.set_page_config(page_title="Kelly Calculator", page_icon="🎰", layout="wide")

st.title("🎰 Kelly Criterion Calculator")
st.markdown("### Optimal Bet Sizing for Tennis Betting")

# ============================================================================
# KELLY CRITERION FUNCTIONS
# ============================================================================

def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """
    Calculate the Kelly fraction for a given bet.
    f* = (bp - q) / b
    """
    b = decimal_odds - 1
    q = 1 - prob
    f_star = (b * prob - q) / b
    return max(0.0, f_star)


def calculate_edge(prob: float, decimal_odds: float) -> float:
    """Calculate edge: model probability minus implied probability."""
    return prob - (1 / decimal_odds)


def expected_growth_rate(prob: float, fraction: float, decimal_odds: float) -> float:
    """Calculate expected log growth rate: G = p*ln(1+fb) + q*ln(1-f)"""
    if fraction <= 0 or fraction >= 1:
        return 0.0
    b = decimal_odds - 1
    q = 1 - prob
    return prob * np.log(1 + fraction * b) + q * np.log(1 - fraction)


# ============================================================================
# SIDEBAR (shared across tabs)
# ============================================================================

st.sidebar.header("Bet Parameters")

st.sidebar.markdown("**Your Estimate**")
prob_col1, prob_col2 = st.sidebar.columns([2, 1])
with prob_col1:
    prob_slider = st.slider(
        "Model Win Probability (%)", 
        min_value=1.0, max_value=99.0, value=60.0, step=0.1,
        help="Your model's estimated probability of winning"
    )
with prob_col2:
    prob_input = st.number_input(
        "Exact %", 
        min_value=1.0, max_value=99.0, value=prob_slider, step=0.1,
        help="Type exact probability"
    )
model_prob = prob_input / 100

st.sidebar.markdown("**Market Odds**")
decimal_odds = st.sidebar.number_input(
    "Decimal Odds", 
    min_value=1.01, max_value=20.0, value=2.00, step=0.05,
    help="Bookmaker decimal odds (e.g., 2.00 = evens)"
)

st.sidebar.markdown("**Bankroll**")
bankroll = st.sidebar.number_input(
    "Current Bankroll (£)", 
    min_value=10, max_value=1_000_000, value=1000, step=100
)

st.sidebar.markdown("---")

st.sidebar.markdown("**Kelly Fractions**")
show_half = st.sidebar.checkbox("½ Kelly", value=True)
show_quarter = st.sidebar.checkbox("¼ Kelly", value=True)

# Core calculations (used by Tab 1)
implied_prob = 1 / decimal_odds
edge = calculate_edge(model_prob, decimal_odds)
full_kelly = kelly_fraction(model_prob, decimal_odds)
half_kelly = full_kelly / 2
quarter_kelly = full_kelly / 4
full_stake = full_kelly * bankroll
half_stake = half_kelly * bankroll
quarter_stake = quarter_kelly * bankroll

# ============================================================================
# TABS
# ============================================================================

tab1, tab2 = st.tabs(["📊 Manual Calculator", "📂 Betting Opportunities"])

# ============================================================================
# TAB 1: MANUAL CALCULATOR
# ============================================================================

with tab1:

    # --- Key Metrics ---
    st.subheader("📊 Bet Analysis")

    met_col1, met_col2, met_col3, met_col4 = st.columns(4)

    with met_col1:
        edge_color = "normal" if edge > 0 else "inverse"
        st.metric(
            "Edge", 
            f"{edge*100:.1f}%",
            delta="Positive ✓" if edge > 0 else "No Edge ✗",
            delta_color=edge_color
        )
    with met_col2:
        st.metric("Implied Probability", f"{implied_prob*100:.1f}%")
    with met_col3:
        st.metric("Full Kelly Fraction", f"{full_kelly*100:.1f}%")
    with met_col4:
        st.metric("Full Kelly Stake", f"£{full_stake:,.2f}")

    if show_half or show_quarter:
        frac_col1, frac_col2 = st.columns(2)
        if show_half:
            with frac_col1:
                st.metric("½ Kelly Stake", f"£{half_stake:,.2f}", 
                          delta=f"{half_kelly*100:.1f}% of bankroll")
        if show_quarter:
            with frac_col2:
                st.metric("¼ Kelly Stake", f"£{quarter_stake:,.2f}", 
                          delta=f"{quarter_kelly*100:.1f}% of bankroll")

    if edge <= 0:
        st.warning("⚠️ No positive edge detected. Kelly recommends **no bet** at these parameters.")

    # --- Kelly Fractions Helper ---
    with st.expander("❓ What do Full, ½, and ¼ Kelly mean?", expanded=False):
        help_col1, help_col2, help_col3 = st.columns(3)
        
        with help_col1:
            st.markdown("**🔵 Full Kelly**")
            st.markdown(
                "Bet the *mathematically optimal* amount to maximise long-run bankroll growth. "
                "Sounds perfect — but in practice it's **aggressive**. You'll experience large "
                "swings: drawdowns of 40–50% are normal, even when your model is correct. "
                "Best suited for situations where you have **very high confidence** in your "
                "probability estimates."
            )
        with help_col2:
            st.markdown("**🟣 ½ Kelly (Recommended)**")
            st.markdown(
                "Bet **half** of what full Kelly suggests. You give up roughly 25% of the "
                "growth rate, but your drawdowns shrink dramatically. This is the **most common "
                "choice among professional bettors and traders** because it protects against "
                "the reality that your model's probabilities are estimates, not certainties. "
                "**Start here if you're unsure.**"
            )
        with help_col3:
            st.markdown("**🟠 ¼ Kelly**")
            st.markdown(
                "Bet **a quarter** of the full Kelly amount. Very conservative — almost all "
                "paths end in profit, but growth is slow. Use this when you're **less confident "
                "in your model**, when you're early in testing, or when you simply can't afford "
                "to lose much of your bankroll. A good choice for the first few weeks of live betting."
            )
        st.markdown("---")
        st.markdown(
            "**Rule of thumb:** If you're not sure, pick **½ Kelly**. "
            "It's the standard in both quantitative finance and sports betting because "
            "it balances growth against the inevitable uncertainty in any model's predictions."
        )

    # --- Main Chart: Optimal Bet Size by Edge ---
    st.markdown("---")
    st.subheader("📈 Optimal Bet Size by Edge")
    st.markdown("How the recommended stake changes as your estimated edge varies")

    chart_odds = st.slider(
        "Adjust Decimal Odds for Chart", 
        min_value=1.20, max_value=5.00, value=float(decimal_odds), step=0.05,
        help="Change the odds to see how the Kelly curve shifts"
    )

    chart_implied = 1 / chart_odds
    probs_range = np.linspace(max(chart_implied + 0.001, 0.05), 0.95, 200)
    edges_range = probs_range - chart_implied

    full_kelly_curve = [kelly_fraction(p, chart_odds) * 100 for p in probs_range]
    half_kelly_curve = [kelly_fraction(p, chart_odds) * 50 for p in probs_range]
    quarter_kelly_curve = [kelly_fraction(p, chart_odds) * 25 for p in probs_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edges_range * 100, y=full_kelly_curve,
        name='Full Kelly', line=dict(color='#2E86AB', width=3),
        hovertemplate='Edge: %{x:.1f}%<br>Stake: %{y:.1f}% of bankroll<extra>Full Kelly</extra>'
    ))
    if show_half:
        fig.add_trace(go.Scatter(
            x=edges_range * 100, y=half_kelly_curve,
            name='½ Kelly', line=dict(color='#A23B72', width=2, dash='dash'),
            hovertemplate='Edge: %{x:.1f}%<br>Stake: %{y:.1f}% of bankroll<extra>½ Kelly</extra>'
        ))
    if show_quarter:
        fig.add_trace(go.Scatter(
            x=edges_range * 100, y=quarter_kelly_curve,
            name='¼ Kelly', line=dict(color='#F18F01', width=2, dash='dot'),
            hovertemplate='Edge: %{x:.1f}%<br>Stake: %{y:.1f}% of bankroll<extra>¼ Kelly</extra>'
        ))
    if edge > 0:
        fig.add_trace(go.Scatter(
            x=[edge * 100], y=[full_kelly * 100],
            mode='markers', name='Your Bet',
            marker=dict(size=14, color='#E63946', symbol='star', 
                        line=dict(width=2, color='white')),
            hovertemplate='Your Edge: %{x:.1f}%<br>Kelly Stake: %{y:.1f}%<extra>Your Bet</extra>'
        ))

    fig.update_layout(
        title=f"Kelly Criterion at {chart_odds:.2f} Decimal Odds (Implied: {chart_implied*100:.1f}%)",
        xaxis_title="Edge (%)", yaxis_title="Recommended Stake (% of Bankroll)",
        height=500, hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='grey'),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='grey', 
                   range=[0, max(full_kelly_curve) * 1.1 if full_kelly_curve else 50])
    )
    st.plotly_chart(fig, width="stretch")

    # --- Growth Rate Chart ---
    st.markdown("---")
    st.subheader("📊 Expected Growth Rate by Stake Size")
    st.markdown("Why Kelly is optimal: it maximises long-run bankroll growth")

    if edge > 0:
        fractions = np.linspace(0.001, min(full_kelly * 2.5, 0.95), 200)
        growth_rates = [expected_growth_rate(model_prob, f, decimal_odds) * 100 for f in fractions]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=fractions * 100, y=growth_rates,
            name='Growth Rate', line=dict(color='#2E86AB', width=3),
            fill='tozeroy', fillcolor='rgba(46, 134, 171, 0.1)',
            hovertemplate='Stake: %{x:.1f}%<br>Growth: %{y:.3f}% per bet<extra></extra>'
        ))
        
        markers_x = [full_kelly * 100]
        markers_y = [expected_growth_rate(model_prob, full_kelly, decimal_odds) * 100]
        markers_names = ['Full Kelly']
        markers_colors = ['#2E86AB']
        
        if show_half:
            markers_x.append(half_kelly * 100)
            markers_y.append(expected_growth_rate(model_prob, half_kelly, decimal_odds) * 100)
            markers_names.append('½ Kelly')
            markers_colors.append('#A23B72')
        if show_quarter:
            markers_x.append(quarter_kelly * 100)
            markers_y.append(expected_growth_rate(model_prob, quarter_kelly, decimal_odds) * 100)
            markers_names.append('¼ Kelly')
            markers_colors.append('#F18F01')
        
        for mx, my, mn, mc in zip(markers_x, markers_y, markers_names, markers_colors):
            fig2.add_trace(go.Scatter(
                x=[mx], y=[my], mode='markers+text', name=mn,
                marker=dict(size=12, color=mc, line=dict(width=2, color='white')),
                text=[mn], textposition='top center',
                hovertemplate=f'{mn}<br>Stake: {mx:.1f}%<br>Growth: {my:.3f}%<extra></extra>'
            ))
        
        fig2.update_layout(
            title="Expected Log Growth Rate vs. Stake Size",
            xaxis_title="Stake (% of Bankroll)",
            yaxis_title="Expected Growth Rate (% per bet)",
            height=450, showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)")
        )
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("📉 No positive edge — growth rate chart available when edge > 0%.")

    # --- Expanders ---
    st.markdown("---")

    with st.expander("🔍 Kelly Criterion — Key Insights"):
        st.markdown(f"""
        **Current Bet Analysis:**
        
        - Model probability: **{model_prob*100:.1f}%** vs. implied probability: **{implied_prob*100:.1f}%**
        - Edge: **{edge*100:.1f}%** → Kelly recommends staking **{full_kelly*100:.1f}%** of bankroll
        - At £{bankroll:,} bankroll: Full Kelly = **£{full_stake:,.2f}**, 
          ½ Kelly = **£{half_stake:,.2f}**, ¼ Kelly = **£{quarter_stake:,.2f}**
        
        **Why Fractional Kelly?**
        
        - Full Kelly maximises long-run growth but has **high variance** (~40% drawdowns are common)
        - ½ Kelly achieves **~75% of the growth rate** with significantly lower volatility
        - ¼ Kelly is very conservative — suitable when model uncertainty is high
        - In practice, most professional bettors use ½ to ¼ Kelly
        
        **Important Assumptions:**
        
        - Kelly assumes your probability estimate is **accurate** — model calibration is critical
        - Assumes bets are **independent** (no correlation between matches)
        - Assumes you can place bets at the **stated odds** (no slippage or limits)
        - Over-betting (staking more than Kelly) **reduces** long-run growth
        
        **For Your Tennis Model:**
        
        - Use model-predicted probabilities as your win probability input
        - Compare against bookmaker odds to find the edge
        - Apply ½ Kelly as a sensible default for the betting simulation
        """)

    with st.expander("📚 The Mathematics"):
        st.markdown(r"""
        **Kelly Formula:**
        
        $$f^* = \frac{bp - q}{b}$$
        
        Where:
        - $f^*$ = optimal fraction of bankroll to wager
        - $b$ = net odds (decimal odds − 1)
        - $p$ = probability of winning
        - $q$ = probability of losing $(1 - p)$
        
        **Expected Growth Rate:**
        
        $$G(f) = p \cdot \ln(1 + fb) + q \cdot \ln(1 - f)$$
        
        The Kelly fraction $f^*$ maximises $G(f)$. Any stake above $f^*$ 
        actually **decreases** long-run growth — this is why over-betting is dangerous.
        
        **Edge Definition:**
        
        $$\text{Edge} = p_{\text{model}} - p_{\text{implied}} = p_{\text{model}} - \frac{1}{\text{odds}}$$
        """)


# ============================================================================
# TAB 2: BETTING OPPORTUNITIES (CSV Upload)
# ============================================================================

with tab2:

    st.subheader("📂 Betting Opportunities")
    st.markdown(
        "Model predictions are loaded automatically from your project data. "
        "You can also upload a different CSV to override."
    )
    
    # --- Default data path ---
    from pathlib import Path
    
    DEFAULT_CSV = Path(__file__).parent.parent.parent / 'data' / 'streamlit_data' / 'betting_df_for_streamlit.csv'
    
    # --- Optional upload override ---
    uploaded_file = st.file_uploader(
        "Upload a different predictions CSV (optional — overrides default data)",
        type=['csv'],
        help="Leave empty to use the default dataset from data/streamlit_data/"
    )

    # --- Expected format helper ---
    with st.expander("📋 Expected CSV Format"):
        st.markdown("""
        Your CSV should contain at minimum these columns:
        
        | Column | Description | Example |
        |--------|-------------|---------|
        | `Player` | Player name | Djokovic N. |
        | `Opponent` | Opponent name | Alcaraz C. |
        | `model_prob` | Model's predicted win probability (0–1) | 0.65 |
        | `decimal_odds` | Best available decimal odds | 1.80 |
        
        **Optional columns** (auto-detected if present):
        
        | Column | Description |
        |--------|-------------|
        | `Tournament` | Tournament name |
        | `Date` | Match date |
        | `B365W`, `PSW`, `MaxW`, `AvgW` | Bookmaker odds |
        | `Won` or `actual_outcome` | Actual result (for backtesting) |
        | `market_prob` | Implied probability from odds |
        | `edge` | Pre-calculated edge |
        """)

    # --- Load data: upload takes priority, otherwise default ---
    df = None
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.caption("📤 Using uploaded file")
    elif DEFAULT_CSV.exists():
        df = pd.read_csv(DEFAULT_CSV)
        st.caption(f"📁 Auto-loaded from `data/streamlit_data/betting_df_for_streamlit.csv`")
    else:
        st.warning(
            "⚠️ Default data file not found. Please upload a CSV or check that "
            "`data/streamlit_data/betting_df_for_streamlit.csv` exists in your project."
        )

    if df is not None:
        
        # Validate required columns
        required = ['model_prob', 'decimal_odds']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        # Calculate Kelly columns
        df['market_prob'] = 1 / df['decimal_odds']
        df['edge'] = df['model_prob'] - df['market_prob']
        df['full_kelly'] = df.apply(
            lambda r: kelly_fraction(r['model_prob'], r['decimal_odds']), axis=1
        )
        df['half_kelly'] = df['full_kelly'] / 2
        df['quarter_kelly'] = df['full_kelly'] / 4
        df['full_stake'] = df['full_kelly'] * bankroll
        df['half_stake'] = df['half_kelly'] * bankroll
        df['quarter_stake'] = df['quarter_kelly'] * bankroll

        # Detect if we have actual outcomes (backtest mode)
        has_outcomes = 'Won' in df.columns or 'actual_outcome' in df.columns
        outcome_col = 'Won' if 'Won' in df.columns else ('actual_outcome' if 'actual_outcome' in df.columns else None)

        st.success(f"✓ Loaded **{len(df):,}** matches")

        # ----------------------------------------------------------------
        # FILTERS
        # ----------------------------------------------------------------
        st.markdown("---")
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            min_edge_pct = st.slider(
                "Minimum Edge (%)", 0.0, 30.0, 0.0, 0.5,
                help="Only show matches where model edge exceeds this threshold",
                key="opp_min_edge"
            )
        with filter_col2:
            kelly_strategy = st.selectbox(
                "Kelly Strategy", 
                ["Full Kelly", "½ Kelly", "¼ Kelly"],
                index=1,
                help="Which Kelly fraction to use for stake calculations"
            )
        with filter_col3:
            if 'Tournament' in df.columns:
                tournaments = ['All'] + sorted(df['Tournament'].dropna().unique().tolist())
                selected_tournament = st.selectbox("Tournament", tournaments)
            else:
                selected_tournament = 'All'

        # Apply filters
        kelly_col = {'Full Kelly': 'full_kelly', '½ Kelly': 'half_kelly', '¼ Kelly': 'quarter_kelly'}[kelly_strategy]
        stake_col = {'Full Kelly': 'full_stake', '½ Kelly': 'half_stake', '¼ Kelly': 'quarter_stake'}[kelly_strategy]
        
        df_value = df[df['edge'] > (min_edge_pct / 100)].copy()
        if selected_tournament != 'All' and 'Tournament' in df.columns:
            df_value = df_value[df_value['Tournament'] == selected_tournament]
        df_value = df_value.sort_values('edge', ascending=False)

        # ----------------------------------------------------------------
        # SUMMARY METRICS
        # ----------------------------------------------------------------
        st.markdown("---")
        st.subheader("🎯 Betting Opportunities")

        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

        with sum_col1:
            st.metric("Value Bets Found", f"{len(df_value)}")
        with sum_col2:
            st.metric("Avg Edge", f"{df_value['edge'].mean()*100:.1f}%" if len(df_value) > 0 else "N/A")
        with sum_col3:
            total_exposure = df_value[stake_col].sum()
            st.metric(f"Total {kelly_strategy} Exposure", f"£{total_exposure:,.0f}")
        with sum_col4:
            if has_outcomes and len(df_value) > 0:
                win_rate = df_value[outcome_col].mean() * 100
                st.metric("Win Rate (actual)", f"{win_rate:.1f}%")
            else:
                st.metric("Max Edge", f"{df_value['edge'].max()*100:.1f}%" if len(df_value) > 0 else "N/A")

        # ----------------------------------------------------------------
        # OPPORTUNITIES TABLE
        # ----------------------------------------------------------------
        if len(df_value) > 0:
            display = df_value.copy()
            
            display['Model Prob'] = (display['model_prob'] * 100).round(1).astype(str) + '%'
            display['Market Prob'] = (display['market_prob'] * 100).round(1).astype(str) + '%'
            display['Edge'] = (display['edge'] * 100).round(1).astype(str) + '%'
            display['Kelly %'] = (display[kelly_col] * 100).round(2).astype(str) + '%'
            display['Stake'] = '£' + display[stake_col].round(0).astype(int).astype(str)
            display['Odds'] = display['decimal_odds'].round(2)

            if has_outcomes:
                display['Result'] = display[outcome_col].map({1: '✅ Won', 0: '❌ Lost'})
            
            show_cols = []
            if 'Date' in display.columns:
                show_cols.append('Date')
            if 'Tournament' in display.columns:
                show_cols.append('Tournament')
            show_cols += ['Player', 'Opponent', 'Model Prob', 'Market Prob', 'Edge', 'Odds', 'Kelly %', 'Stake']
            if has_outcomes:
                show_cols.append('Result')
            
            st.dataframe(display[show_cols], width="stretch", hide_index=True, height=500)

            # ----------------------------------------------------------------
            # CHARTS
            # ----------------------------------------------------------------
            st.markdown("---")
            st.subheader("📈 Edge Distribution of Value Bets")

            fig_edge = go.Figure()
            fig_edge.add_trace(go.Histogram(
                x=df_value['edge'] * 100, nbinsx=40,
                marker_color='#2E86AB', opacity=0.8,
                hovertemplate='Edge: %{x:.1f}%<br>Count: %{y}<extra></extra>'
            ))
            fig_edge.update_layout(
                title=f"Distribution of Edges ({len(df_value)} value bets)",
                xaxis_title="Edge (%)", yaxis_title="Number of Matches", height=400,
            )
            st.plotly_chart(fig_edge, width="stretch")

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                st.subheader("💰 Stake Distribution")
                fig_stake = go.Figure()
                fig_stake.add_trace(go.Histogram(
                    x=df_value[stake_col], nbinsx=30,
                    marker_color='#A23B72', opacity=0.8,
                    hovertemplate='Stake: £%{x:,.0f}<br>Count: %{y}<extra></extra>'
                ))
                fig_stake.update_layout(
                    title=f"{kelly_strategy} Stake Distribution",
                    xaxis_title="Stake (£)", yaxis_title="Number of Bets", height=400,
                )
                st.plotly_chart(fig_stake, width="stretch")

            with viz_col2:
                st.subheader("🏆 Top Players by Value Bets")
                if 'Player' in df_value.columns:
                    top_players = df_value['Player'].value_counts().head(10).reset_index()
                    top_players.columns = ['Player', 'Value Bets']
                    fig_players = px.bar(
                        top_players, x='Value Bets', y='Player',
                        orientation='h', color='Value Bets',
                        color_continuous_scale='Greens',
                        title="Top 10 Players (Most Value Bets)"
                    )
                    fig_players.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_players, width="stretch")

            # ----------------------------------------------------------------
            # BACKTEST RESULTS (if outcomes available)
            # ----------------------------------------------------------------
            if has_outcomes:
                st.markdown("---")
                st.subheader("📊 Backtest Performance")

                df_value['profit'] = np.where(
                    df_value[outcome_col] == 1,
                    df_value[stake_col] * (df_value['decimal_odds'] - 1),
                    -df_value[stake_col]
                )
                
                total_staked = df_value[stake_col].sum()
                total_profit = df_value['profit'].sum()
                roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
                n_wins = int(df_value[outcome_col].sum())
                n_bets = len(df_value)

                bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)

                with bt_col1:
                    st.metric("Total Staked", f"£{total_staked:,.0f}")
                with bt_col2:
                    profit_delta = "Profit ✓" if total_profit > 0 else "Loss ✗"
                    st.metric("Total Profit/Loss", f"£{total_profit:,.0f}", delta=profit_delta)
                with bt_col3:
                    st.metric("ROI", f"{roi:.1f}%")
                with bt_col4:
                    st.metric("Record", f"{n_wins}W – {n_bets - n_wins}L")

                # Cumulative P&L chart
                df_pnl = df_value.sort_values('Date').copy() if 'Date' in df_value.columns else df_value.copy()
                df_pnl['cumulative_profit'] = df_pnl['profit'].cumsum()
                
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(
                    x=list(range(1, len(df_pnl) + 1)),
                    y=df_pnl['cumulative_profit'],
                    mode='lines', line=dict(color='#2E86AB', width=2),
                    fill='tozeroy', fillcolor='rgba(46, 134, 171, 0.1)',
                    hovertemplate='Bet #%{x}<br>Cumulative P&L: £%{y:,.0f}<extra></extra>'
                ))
                fig_pnl.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
                fig_pnl.update_layout(
                    title=f"Cumulative P&L — {kelly_strategy} Strategy",
                    xaxis_title="Bet Number", yaxis_title="Cumulative Profit (£)", height=400,
                )
                st.plotly_chart(fig_pnl, width="stretch")

        else:
            st.info(f"No matches found with edge > {min_edge_pct:.1f}%. Try lowering the threshold.")

    else:
        st.info("👆 Upload a CSV file or ensure `data/streamlit_data/betting_df_for_streamlit.csv` exists in your project.")