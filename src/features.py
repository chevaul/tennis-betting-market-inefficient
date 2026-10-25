"""
Feature engineering functions for tennis betting efficiency analysis.
"""

import pandas as pd
import numpy as np


def compute_implied_probability(odds):
    """
    Convert betting odds to implied probability.
    
    Parameters:
    -----------
    odds : float or array-like
        Decimal odds (e.g., 2.5)
    
    Returns:
    --------
    float or array-like
        Implied probability (between 0 and 1)
    """
    return 1 / odds


def create_rank_features(df):
    """
    Create rank-based features for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with player rank columns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional rank features
    """
    df = df.copy()
    
    if 'player1_rank' in df.columns and 'player2_rank' in df.columns:
        df['rank_diff'] = df['player1_rank'] - df['player2_rank']
        # Avoid division by zero by adding small epsilon or using np.where
        df['rank_ratio'] = np.where(
            df['player2_rank'] != 0,
            df['player1_rank'] / df['player2_rank'],
            np.nan
        )
    
    return df


def create_surface_features(df):
    """
    Create surface-based features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with surface column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with surface dummy variables
    """
    if 'surface' in df.columns:
        return pd.get_dummies(df, columns=['surface'], prefix='surface')
    return df


def create_odds_features(df, odds_cols):
    """
    Create features from betting odds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with odds columns
    odds_cols : list
        List of column names containing odds
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with implied probability features
    """
    df = df.copy()
    
    for col in odds_cols:
        if col in df.columns:
            prob_col = col.replace('odds', 'prob')
            df[prob_col] = compute_implied_probability(df[col])
    
    return df


def compute_recent_form_index(df, window_days=180, max_matches=12,surface_specific=True,
    time_decay_days=90,     # recency decay: larger = slower decay
    rank_decay_scale=200    # opponent quality decay: smaller = steeper emphasis on elite opponents
    ):
    
    """
    Compute an opponent- and time-weighted recent-form score for each player.

    Parameters
    ----------
    df : pd.DataFrame
        Requires columns: ['Date','Winner','Loser','WRank','LRank'] and optionally 'Surface'.
        Assumes one row per match (Winner vs Loser).
    window_days : int, default=180
        Lookback window in days for 'recent' matches.
    max_matches : int, default=15
        Max number of prior matches per player to use (most recent within window).
    surface_specific : bool, default=True
        If True, only use prior matches on the same surface as the current match.
    time_decay_days : int, default=90
        Exponential time-decay constant (exp(-age_days / time_decay_days)).
    rank_decay_scale : int, default=200
        Exponential opponent-quality scaling (exp(-opp_rank / rank_decay_scale)).

    Returns
    -------
    df_out : pd.DataFrame
        Adds columns: ['player1_form','player2_form','form_diff'].
        player1 is Winner, player2 is Loser.
    """
    # Basic sanity
    req = {'Date','Winner','Loser','WRank','LRank'}
    missing_req = req - set(df.columns)
    if missing_req:
        raise ValueError(f"Missing required columns: {missing_req}")

    # Work on a sorted copy
    df = df.sort_values('Date').reset_index(drop=True).copy()

    # Use .get for Surface to avoid KeyErrors
    has_surface = 'Surface' in df.columns

    form_scores = {}
    # Store tuples: (date, result(1/0), opp_rank, surface_or_None)
    player_history = {}

    for idx, row in df.iterrows():
        date = row['Date']
        # Skip rows with bad dates
        if pd.isna(date):
            form_scores[(idx, row['Winner'])] = np.nan
            form_scores[(idx, row['Loser'])] = np.nan
            continue

        winner, loser = row['Winner'], row['Loser']
        w_rank, l_rank = row['WRank'], row['LRank']
        surface = row['Surface'] if (has_surface and not pd.isna(row['Surface'])) else None

        for player, opponent, opp_rank, result in [
            (winner, loser, l_rank, 1),
            (loser, winner, w_rank, 0)
        ]:
            history = player_history.get(player, [])

            # filter by time window (strictly prior: age >= 1 day) and surface if requested
            recent = []
            for d, r, orank, s in history:
                age = (date - d).days
                if 0 < age <= window_days and (not surface_specific or surface is None or s == surface):
                    recent.append((d, r, orank, s))

            # most recent N matches
            recent = recent[-max_matches:]

            if recent:
                # compute weights: opponent quality * recency
                # guard orank NaN by treating as large rank
                wts = []
                res = []
                for d, r, orank, _ in recent:
                    orank = 10_000 if pd.isna(orank) else orank
                    age = max((date - d).days, 1)
                    w_quality = np.exp(-orank / rank_decay_scale)
                    w_time = np.exp(-age / time_decay_days)
                    wts.append(w_quality * w_time)
                    res.append(r)
                wts = np.array(wts, dtype=float)
                res = np.array(res, dtype=float)
                w_sum = wts.sum()
                form = np.sum(res * wts) / w_sum if w_sum > 0 else np.nan
            else:
                form = np.nan

            form_scores[(idx, player)] = form

        # append *current* match to both players' histories (for future rows)
        # store surface (may be None)
        player_history.setdefault(winner, []).append((date, 1, l_rank, surface))
        player_history.setdefault(loser,  []).append((date, 0, w_rank, surface))

    # Build outputs
    df['player1_form'] = [form_scores.get((i, df.loc[i, 'Winner']), np.nan) for i in df.index]
    df['player2_form'] = [form_scores.get((i, df.loc[i, 'Loser']),  np.nan) for i in df.index]
    df['form_diff'] = df['player1_form'] - df['player2_form']

    return df