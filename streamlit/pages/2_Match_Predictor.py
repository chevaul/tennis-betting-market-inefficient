"""
2_Match_Predictor.py  —  Streamlit page
========================================
Loads the trained CatBoost model and lets users enter two players' stats
to get a win-probability prediction, an Elo baseline comparison, and
(optionally) a betting value signal against bookmaker odds.

Requires:
    models/catboost_model.pkl        — trained model (joblib)
    models/feature_names.csv         — ordered list of feature columns
    models/model_stats.json          — accuracy / AUC / log-loss metrics

Run from repo root:
    streamlit run streamlit/pages/2_Match_Predictor.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
MODEL_PATH  = _REPO_ROOT / "models" / "catboost_model.pkl"
FEAT_PATH   = _REPO_ROOT / "models" / "feature_names.csv"
STATS_PATH  = _REPO_ROOT / "models" / "model_stats.json"

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

# ---------------------------------------------------------------------------
# Load model artifacts (cached so they only load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_feature_names():
    if not FEAT_PATH.exists():
        return []
    return pd.read_csv(FEAT_PATH)["feature"].tolist()

@st.cache_data
def load_model_stats():
    if not STATS_PATH.exists():
        return {}
    with open(STATS_PATH) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Feature engineering — must mirror the training pipeline exactly
# ---------------------------------------------------------------------------
def build_feature_row(p: dict, o: dict, match: dict, feature_names: list) -> pd.DataFrame:
    """
    Takes dicts of player (p), opponent (o), and match context,
    builds the same engineered features used during training,
    and returns a single-row DataFrame aligned to feature_names.
    """
    row = {}

    # --- Raw numeric features ---
    row["Best of"]          = match["best_of"]
    row["player_rank"]      = p["rank"]
    row["player_points"]    = p["points"]
    row["player_elo"]       = p["elo"]
    row["player_h2h"]       = p["h2h"]
    row["player_height"]    = p["height"]
    row["player_age"]       = p["age"]
    row["player_rating"]    = p["rating"]
    row["player_form"]      = p["form"]
    row["opponent_rank"]    = o["rank"]
    row["opponent_points"]  = o["points"]
    row["opponent_elo"]     = o["elo"]
    row["opponent_h2h"]     = o["h2h"]
    row["opponent_height"]  = o["height"]
    row["opponent_age"]     = o["age"]
    row["opponent_rating"]  = o["rating"]
    row["opponent_form"]    = o["form"]

    # --- Elo win probability ---
    elo_diff = p["elo"] - o["elo"]
    row["player_win_prob_elo"] = 1 / (1 + 10 ** (-elo_diff / 400))

    # --- Engineered difference / interaction features ---
    row["rank_diff"]    = p["rank"]    - o["rank"]
    row["elo_diff"]     = p["elo"]     - o["elo"]
    row["rating_diff"]  = p["rating"]  - o["rating"]
    row["height_diff"]  = p["height"]  - o["height"]
    row["points_diff"]  = p["points"]  - o["points"]
    row["age_diff"]     = p["age"]     - o["age"]
    row["form_diff"]    = p["form"]    - o["form"]

    # Sigmoid rank probability (scale=200, as in training)
    row["proba_rank_200"] = 1 / (1 + 10 ** ((p["rank"] - o["rank"]) / 200))

    # Sigmoid form probability (scale=50, as in training)
    row["proba_form_50"] = 1 / (1 + 10 ** ((p["form"] - o["form"]) / 50))

    # H2H win rate
    total_h2h = p["h2h"] + o["h2h"]
    row["player_h2h_rate"] = p["h2h"] / total_h2h if total_h2h > 0 else 0.5

    # --- One-hot encoded categoricals ---
    # Surface
    for s in ["Clay", "Grass", "Hard"]:
        row[f"Surface_{s}"] = int(match["surface"] == s)

    # Court
    row["Court_Outdoor"] = int(match["court"] == "Outdoor")

    # Round
    round_cats = [
        "2nd Round", "3rd Round", "4th Round",
        "Quarterfinals", "Round Robin",
        "Semifinals", "The Final",
    ]
    for r in round_cats:
        row[f"Round_{r}"] = int(match["round"] == r)

    # Series
    series_cats = ["ATP500", "Grand Slam", "Masters 1000", "Masters Cup"]
    for s in series_cats:
        row[f"Series_{s}"] = int(match["series"] == s)

    # Handedness
    row["player_hand_R"]      = int(p["hand"] == "R")
    row["player_hand_Unknown"]= int(p["hand"] == "Unknown")
    row["opponent_hand_R"]    = int(o["hand"] == "R")
    row["opponent_hand_Unknown"] = int(o["hand"] == "Unknown")

    # Build DataFrame and align to training feature order
    df = pd.DataFrame([row])

    # Add any missing columns as 0, drop any extras
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    return df


# ---------------------------------------------------------------------------
# Betting value calculation
# ---------------------------------------------------------------------------
def implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / odds if odds > 1 else 0.0

def expected_value(model_prob: float, odds: float) -> float:
    """
    EV per £1 staked.
    Positive = value bet (model thinks outcome more likely than bookmaker does).
    """
    return model_prob * (odds - 1) - (1 - model_prob)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Match Predictor",
        page_icon="🎾",
        layout="wide",
    )

    st.title("🎾 Tennis Match Predictor")
    st.caption("Enter pre-match stats for both players to get a win probability from the CatBoost model.")

    # Load artifacts
    model         = load_model()
    feature_names = load_feature_names()
    stats         = load_model_stats()

    if model is None:
        st.error(
            "Model not found. Run the training notebook first to generate "
            "`models/catboost_model.pkl`."
        )
        return

    # Model performance banner
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Accuracy", f"{stats.get('accuracy', 0):.1%}")
        c2.metric("AUC", f"{stats.get('auc', 0):.3f}")
        c3.metric("Log Loss", f"{stats.get('log_loss', 0):.3f}")
    st.divider()

    # -----------------------------------------------------------------------
    # Match context
    # -----------------------------------------------------------------------
    st.subheader("Match Context")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    with col2:
        court = st.selectbox("Court", ["Outdoor", "Indoor"])
    with col3:
        round_ = st.selectbox("Round", [
            "1st Round", "2nd Round", "3rd Round", "4th Round",
            "Quarterfinals", "Semifinals", "The Final", "Round Robin",
        ])
    with col4:
        series = st.selectbox("Series", [
            "ATP250", "ATP500", "Masters 1000", "Grand Slam", "Masters Cup",
        ])

    best_of = st.radio("Best of", [3, 5], horizontal=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Player inputs
    # -----------------------------------------------------------------------
    left, right = st.columns(2)

    with left:
        st.subheader("🟦 Player")
        p_name   = st.text_input("Name", value="Player A", key="p_name")
        p_rank   = st.number_input("ATP Ranking", min_value=1, max_value=2000, value=20,  key="p_rank")
        p_points = st.number_input("Ranking Points", min_value=0, max_value=20000, value=2000, key="p_pts")
        p_elo    = st.number_input("Elo Rating", min_value=1200, max_value=2800, value=2100, key="p_elo")
        p_form   = st.slider("Recent Form (0–10)", 0.0, 10.0, 6.0, 0.1, key="p_form",
                             help="Higher = better recent results")
        p_h2h    = st.number_input("H2H Wins vs Opponent", min_value=0, max_value=50, value=3, key="p_h2h")

        with st.expander("More stats (optional)"):
            p_height = st.number_input("Height (cm)", 150, 220, 185, key="p_height")
            p_age    = st.number_input("Age", 16, 45, 26, key="p_age")
            p_rating = st.number_input("Serve Rating", -5.0, 5.0, 0.0, 0.1, key="p_rating")
            p_hand   = st.selectbox("Dominant Hand", ["R", "L", "Unknown"], key="p_hand")

    with right:
        st.subheader("🟥 Opponent")
        o_name   = st.text_input("Name", value="Player B", key="o_name")
        o_rank   = st.number_input("ATP Ranking", min_value=1, max_value=2000, value=35,  key="o_rank")
        o_points = st.number_input("Ranking Points", min_value=0, max_value=20000, value=1400, key="o_pts")
        o_elo    = st.number_input("Elo Rating", min_value=1200, max_value=2800, value=2000, key="o_elo")
        o_form   = st.slider("Recent Form (0–10)", 0.0, 10.0, 5.5, 0.1, key="o_form",
                             help="Higher = better recent results")
        o_h2h    = st.number_input("H2H Wins vs Player", min_value=0, max_value=50, value=2, key="o_h2h")

        with st.expander("More stats (optional)"):
            o_height = st.number_input("Height (cm)", 150, 220, 183, key="o_height")
            o_age    = st.number_input("Age", 16, 45, 28, key="o_age")
            o_rating = st.number_input("Serve Rating", -5.0, 5.0, 0.0, 0.1, key="o_rating")
            o_hand   = st.selectbox("Dominant Hand", ["R", "L", "Unknown"], key="o_hand")

    st.divider()

    # -----------------------------------------------------------------------
    # Optional: bookmaker odds for value detection
    # -----------------------------------------------------------------------
    st.subheader("Betting Analysis (optional)")
    use_odds = st.checkbox("Enter bookmaker odds to check for value")
    p_odds, o_odds = None, None
    if use_odds:
        bc1, bc2 = st.columns(2)
        with bc1:
            p_odds = st.number_input(f"Decimal odds for {p_name}", min_value=1.01, max_value=100.0,
                                      value=1.80, step=0.01, key="p_odds")
        with bc2:
            o_odds = st.number_input(f"Decimal odds for {o_name}", min_value=1.01, max_value=100.0,
                                      value=2.10, step=0.01, key="o_odds")

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        p = dict(rank=p_rank, points=p_points, elo=p_elo, form=p_form,
                 h2h=p_h2h, height=p_height, age=p_age, rating=p_rating, hand=p_hand)
        o = dict(rank=o_rank, points=o_points, elo=o_elo, form=o_form,
                 h2h=o_h2h, height=o_height, age=o_age, rating=o_rating, hand=o_hand)
        match = dict(surface=surface, court=court, round=round_,
                     series=series, best_of=best_of)

        try:
            X = build_feature_row(p, o, match, feature_names)
            prob_p = float(model.predict_proba(X)[0, 1])
            prob_o = 1 - prob_p

            elo_diff = p_elo - o_elo
            elo_prob_p = 1 / (1 + 10 ** (-elo_diff / 400))
            elo_prob_o = 1 - elo_prob_p

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # --- Results ---
        st.subheader("Prediction")
        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                label=f"🟦 {p_name} win probability",
                value=f"{prob_p:.1%}",
                delta=f"{prob_p - elo_prob_p:+.1%} vs Elo baseline",
            )
        with r2:
            st.metric(
                label=f"🟥 {o_name} win probability",
                value=f"{prob_o:.1%}",
                delta=f"{prob_o - elo_prob_o:+.1%} vs Elo baseline",
            )

        # Confidence bar
        st.progress(prob_p, text=f"{p_name}  ←  model probability  →  {o_name}")

        # Verdict
        winner = p_name if prob_p >= 0.5 else o_name
        win_prob = max(prob_p, prob_o)
        if win_prob >= 0.70:
            confidence = "High confidence"
            colour = "green"
        elif win_prob >= 0.58:
            confidence = "Moderate confidence"
            colour = "orange"
        else:
            confidence = "Low confidence (close match)"
            colour = "red"

        st.markdown(
            f"**Verdict:** :{colour}[{confidence}] — model favours **{winner}** "
            f"({win_prob:.1%})"
        )

        # --- Elo comparison ---
        with st.expander("Elo baseline comparison"):
            e1, e2 = st.columns(2)
            e1.metric(f"{p_name} (Elo)", f"{elo_prob_p:.1%}")
            e2.metric(f"{o_name} (Elo)", f"{elo_prob_o:.1%}")
            st.caption(
                "The Elo baseline uses only rating difference. "
                "The model incorporates rank, form, H2H, surface, and more."
            )

        # --- Betting value ---
        if use_odds and p_odds and o_odds:
            st.subheader("Betting Value")
            ev_p = expected_value(prob_p, p_odds)
            ev_o = expected_value(prob_o, o_odds)
            impl_p = implied_prob(p_odds)
            impl_o = implied_prob(o_odds)
            overround = impl_p + impl_o

            b1, b2, b3 = st.columns(3)
            b1.metric(f"{p_name} EV (per £1)", f"£{ev_p:+.3f}",
                      delta="Value bet ✓" if ev_p > 0 else "No value")
            b2.metric(f"{o_name} EV (per £1)", f"£{ev_o:+.3f}",
                      delta="Value bet ✓" if ev_o > 0 else "No value")
            b3.metric("Bookmaker overround", f"{(overround - 1):.1%}",
                      help="The bookmaker's margin. Lower = fairer odds.")

            st.caption(
                "**Expected Value** = model probability × (odds − 1) − (1 − model probability). "
                "Positive EV means the model believes the true probability is higher than the "
                "bookmaker's implied probability — a potential value bet."
            )

            if ev_p > 0 or ev_o > 0:
                best = p_name if ev_p >= ev_o else o_name
                best_ev = max(ev_p, ev_o)
                best_odds = p_odds if ev_p >= ev_o else o_odds
                st.success(
                    f"**Value bet detected:** Back **{best}** at {best_odds} odds "
                    f"(EV = £{best_ev:+.3f} per £1 staked)"
                )
            else:
                st.info("No value detected at current odds — both sides are overpriced relative to model probabilities.")


if __name__ == "__main__":
    main()
