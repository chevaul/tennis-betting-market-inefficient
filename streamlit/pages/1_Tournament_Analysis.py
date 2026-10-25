import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys


# Make project root importable so we can reuse src betting utilities.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from src.betting_func import simulate_fixed_stake_betting, simulate_kelly_betting

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

st.set_page_config(page_title="Tournament Betting Simulation", page_icon="🏆", layout="wide")

st.title("🏆 Tournament Betting Simulation")
st.markdown("### Select a tournament on the side bar to see fixed-stake and Kelly simulation results.")


@st.cache_data
def load_tournament_betting_data() -> pd.DataFrame:
	"""
	Load the canonical betting dataset exported from match_analysis.ipynb.
	No merge logic in Streamlit to keep data consistency with notebook outputs.
	"""
	data_path = ROOT / "data" / "streamlit_data" / "betting_df_for_streamlit.csv"
	df = pd.read_csv(data_path)

	# Normalize column names for downstream app code.
	df = df.rename(columns={"Player": "player_name", "Opponent": "opponent_name", "Won": "won"})

	if "actual_outcome" not in df.columns and "won" in df.columns:
		df["actual_outcome"] = df["won"]

	if "decimal_odds" not in df.columns and {"won", "AvgW", "AvgL"}.issubset(df.columns):
		df["decimal_odds"] = np.where(df["won"] == 1, df["AvgW"], df["AvgL"])

	if "Date" in df.columns:
		df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

	# Keep only rows where required simulation fields are present.
	required_cols = ["Tournament", "match_id", "model_prob", "market_prob", "edge", "actual_outcome", "decimal_odds"]
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns in exported streamlit dataset: {missing}")

	df = df.dropna(subset=required_cols).copy()

	# One unique row per match for tournament simulation.
	df = df.drop_duplicates(subset=["match_id"], keep="first").reset_index(drop=True)
	return df


@st.cache_data
def simulate_tournament(
	tournament: str,
	threshold: float,
	stake: float,
	bankroll: float,
	kelly_multiplier: float,
	min_kelly: float,
) -> tuple[pd.DataFrame, dict, dict]:
	"""Run both simulations for a selected tournament and parameter set."""
	df = load_tournament_betting_data()
	selected_df = df[df["Tournament"] == tournament].copy()
	fixed_result = simulate_fixed_stake_betting(selected_df, threshold=threshold, stake=stake)
	kelly_result = simulate_kelly_betting(
		selected_df,
		bankroll=bankroll,
		kelly_multiplier=kelly_multiplier,
		min_kelly=min_kelly,
	)
	return selected_df, fixed_result, kelly_result


try:
	betting_df = load_tournament_betting_data()
except Exception as exc:
	st.error(f"Failed to load tournament betting data: {exc}")
	st.stop()

if betting_df.empty:
	st.warning("No betting data available.")
	st.stop()

tournaments = sorted(t for t in betting_df["Tournament"].dropna().unique() if str(t).strip())

if not tournaments:
	st.warning("No tournaments found in betting dataset.")
	st.stop()

st.sidebar.header("Simulation Settings")
selected_tournament = st.sidebar.selectbox("Tournament", tournaments)
st.sidebar.subheader("Fixed-Stake betting")
threshold = st.sidebar.slider("Edge threshold", 0.00, 0.20, 0.05, 0.01)
stake = st.sidebar.number_input("Fixed stake ($)", min_value=1.0, value=100.0, step=10.0)
st.sidebar.subheader("Kelly betting")
bankroll = st.sidebar.number_input("Starting bankroll ($)", min_value=100.0, value=10000.0, step=100.0)
kelly_multiplier = st.sidebar.slider("Kelly multiplier", 0.10, 1.00, 0.25, 0.05)
min_kelly = st.sidebar.slider("Minimum Kelly fraction", 0.00, 0.10, 0.01, 0.01)

selected_df, fixed_result, kelly_result = simulate_tournament(
	selected_tournament,
	threshold,
	stake,
	bankroll,
	kelly_multiplier,
	min_kelly,
)

st.caption(
	"Hybrid design: load data once (cached), then simulate selected tournament on demand (also cached by inputs)."
)

top1, top2, top3, top4 = st.columns(4)
with top1:
	st.metric("Matches", len(selected_df))
with top2:
	st.metric("Fixed Bets", fixed_result.get("num_bets", 0))
with top3:
	st.metric("Fixed ROI", f"{fixed_result.get('roi', 0):.2f}%")
with top4:
	st.metric("Kelly ROI", f"{kelly_result.get('roi', 0):.2f}%")

mid1, mid2, mid3 = st.columns(3)
with mid1:
	st.metric("Fixed Profit", f"${fixed_result.get('total_profit', 0):,.2f}")
with mid2:
	st.metric("Kelly Profit", f"${kelly_result.get('total_profit', 0):,.2f}")
with mid3:
	st.metric("Final Bankroll", f"${kelly_result.get('final_bankroll', bankroll):,.2f}")

st.markdown("---")

bets_df = fixed_result.get("bets_df")
if isinstance(bets_df, pd.DataFrame) and not bets_df.empty:
	if "cumulative_profit" not in bets_df.columns:
		bets_df = bets_df.copy()
		bets_df["cumulative_profit"] = bets_df["profit"].cumsum()

	st.subheader("Fixed-Stake Cumulative Profit")
	fig = px.line(
		bets_df.reset_index(drop=True),
		x=bets_df.reset_index(drop=True).index,
		y="cumulative_profit",
		labels={"x": "Bet Number", "cumulative_profit": "Cumulative Profit ($)"},
		title=f"{selected_tournament} - Threshold {threshold:.2f}, Stake ${stake:.0f}",
	)
	fig.add_hline(y=0, line_dash="dash", line_color="gray")
	st.plotly_chart(fig, use_container_width=True)

	st.subheader("Placed Bets")
	display_cols = [
		c
		for c in [
			"Date",
			"player_name",
			"opponent_name",
			"model_prob",
			"market_prob",
			"edge",
			"decimal_odds",
			"actual_outcome",
			"profit",
			"cumulative_profit",
		]
		if c in bets_df.columns
	]
	st.dataframe(bets_df[display_cols], use_container_width=True, hide_index=True)
else:
	st.info("No fixed-stake bets for this tournament and threshold.")

kelly_history = kelly_result.get("bankroll_history", [])
if kelly_history:
	st.subheader("Kelly Bankroll Path")
	kelly_df = pd.DataFrame({"Step": range(len(kelly_history)), "Bankroll": kelly_history})
	fig_k = px.line(kelly_df, x="Step", y="Bankroll", title=f"{selected_tournament} - Kelly Evolution")
	fig_k.add_hline(y=bankroll, line_dash="dash", line_color="gray")
	st.plotly_chart(fig_k, use_container_width=True)


