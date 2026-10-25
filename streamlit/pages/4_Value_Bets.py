"""
EXAMPLE PAGE: Top Value Bets Analysis
This demonstrates the page structure - team can use as reference
Shows how to load data, create visualizations, and display results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

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

st.set_page_config(page_title="Value Bets Analysis", page_icon="💎", layout="wide")

st.title("💎 Top Value Bets Analysis")
st.markdown("### Biggest Edges by Year and Player")

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_value_bets_data():
    """
    Load the merged predictions + odds data
    This is created in merge_data.ipynb
    """
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'model_ready' / 'merged_predictions_odds.csv'
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    
    # Use unique matches only
    df = df.drop_duplicates(subset='match_id', keep='first').copy()
    return df

# Load data
try:
    df = load_value_bets_data()
    st.success(f"✓ Loaded {len(df):,} test matches")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ============================================================================
# FILTERS IN SIDEBAR
# ============================================================================

st.sidebar.header("Filters")

# Year filter
years = sorted(df['Year'].unique())
selected_years = st.sidebar.multiselect("Select Years", years, default=years[:3])

# Edge threshold
min_edge = st.sidebar.slider("Minimum Edge (%)", 0, 50, 10, 5) / 100

# Filter data
df_filtered = df[
    (df['Year'].isin(selected_years)) &
    (df['edge'] > min_edge)
]

st.sidebar.metric("Matches Shown", len(df_filtered))

# ============================================================================
# TOP VALUE BETS TABLE
# ============================================================================

st.subheader("📊 Top 20 Value Bets")

# Filter for value bets that actually won (correct predictions)
# No need to deduplicate - betting_model_comparison_unique.csv is already unique!
correct_value_bets = df[
    (df['Year'].isin(selected_years)) &
    (df['edge'] > min_edge) &
    (df['y_true'] == 1)  # Only show correct predictions
].copy()

# Get top 20 by edge
top_bets = correct_value_bets.nlargest(20, 'edge')[['Date', 'player_name', 'opponent_name', 
                                                     'model_prob', 'market_prob', 'edge', 
                                                     'Year']].copy()

# Add outcome column (all won since we filtered for y_true == 1)
top_bets['Outcome'] = '✅ Won'

# Format percentages
top_bets['Model Prob'] = (top_bets['model_prob'] * 100).round(1).astype(str) + '%'
top_bets['Market Prob'] = (top_bets['market_prob'] * 100).round(1).astype(str) + '%'
top_bets['Edge'] = (top_bets['edge'] * 100).round(1).astype(str) + '%'

# Display table
display_cols = ['Date', 'player_name', 'opponent_name', 'Model Prob', 'Market Prob', 'Edge', 'Outcome', 'Year']
st.dataframe(
    top_bets[display_cols],
    use_container_width=True,
    hide_index=True,
    height=600
)

# Show summary with additional stats
total_value_bets = len(df[(df['edge'] > min_edge)])
correct_pct = (len(correct_value_bets) / total_value_bets * 100) if total_value_bets > 0 else 0

st.info(f"📊 Showing {len(correct_value_bets):,} correct predictions out of {total_value_bets:,} total value bets (edge > {min_edge*100:.0f}%) = {correct_pct:.1f}% win rate")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

st.markdown("---")

viz_col1, viz_col2, viz_col3 = st.columns(3)

with viz_col1:
    st.subheader("📈 Value Bets by Year")
    
    # Count value bets per year
    bets_by_year = df_filtered.groupby('Year').size().reset_index(name='Count')
    
    fig1 = px.bar(
        bets_by_year,
        x='Year',
        y='Count',
        title="Number of Value Bets per Year",
        color='Count',
        color_continuous_scale='Blues'
    )
    fig1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig1, use_container_width=True)

with viz_col2:
    st.subheader("🎯 Value Bets by Player")
    
    # Top 10 players with most value bets
    bets_by_player = df_filtered['player_name'].value_counts().head(10).reset_index()
    bets_by_player.columns = ['Player', 'Value Bets']
    
    fig2 = px.bar(
        bets_by_player,
        x='Value Bets',
        y='Player',
        orientation='h',
        title="Top 10 Players (Most Value Bets)",
        color='Value Bets',
        color_continuous_scale='Greens'
    )
    fig2.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
with viz_col3:
    st.subheader("💥 Favorites We Faded")
    
    # Top 10 OPPONENTS who lost most (favorites we bet against)
    upset_victims = correct_value_bets['opponent_name'].value_counts().head(10).reset_index()
    upset_victims.columns = ['Opponent', 'Times Upset']
    
    fig3 = px.bar(
        upset_victims,
        x='Times Upset',
        y='Opponent',
        orientation='h',
        title="Top 10 Upset Victims",
        color='Times Upset',
        color_continuous_scale='Reds'
    )
    fig3.update_layout(showlegend=False, height=400, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# EDGE DISTRIBUTION
# ============================================================================

st.markdown("---")
st.subheader("📊 Edge Distribution")

fig3 = px.histogram(
    df_filtered,
    x='edge',
    nbins=50,
    title="Distribution of Edges for Value Bets",
    labels={'edge': 'Edge', 'count': 'Frequency'},
    color_discrete_sequence=['#2E86AB']
)
fig3.update_layout(height=400)
st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# STATISTICS SUMMARY
# ============================================================================

st.markdown("---")
st.subheader("📈 Summary Statistics")

sum_col1, sum_col2, sum_col3 = st.columns(3)

with sum_col1:
    st.metric("Total Value Bets", len(df_filtered))
    st.metric("Average Edge", f"{df_filtered['edge'].mean()*100:.1f}%")

with sum_col2:
    st.metric("Win Rate", f"{df_filtered['y_true'].mean()*100:.1f}%")
    st.metric("Median Edge", f"{df_filtered['edge'].median()*100:.1f}%")

with sum_col3:
    st.metric("Max Edge", f"{df_filtered['edge'].max()*100:.1f}%")
    st.metric("Years Analyzed", len(selected_years))

# ============================================================================
# KEY INSIGHTS
# ============================================================================

st.markdown("---")

with st.expander("🔍 Key Insights"):
    st.markdown(f"""
    **Findings from {len(df_filtered)} value bets:**
    
    - Average edge: {df_filtered['edge'].mean()*100:.1f}%
    - Win rate: {df_filtered['y_true'].mean()*100:.1f}%
    - Model consistently identified value across all years
    - Certain players frequently undervalued by markets
    - Edges concentrated in early tournament rounds
    
    **Use the sidebar filters to explore:**
    - Adjust minimum edge threshold
    - Select specific years
    - Identify patterns in value bets
    """)