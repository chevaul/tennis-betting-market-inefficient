"""
Feature engineering functions for tennis betting efficiency analysis.
"""

import pandas as pd
import numpy as np


def find_h2h(df):
    """Find h2h record between two player
    Parameter:
    df: Dataframe of match
    Return:
    h2h_record: dict, 
    df: Dataframe of match with new column of h2h record"""
    
    # Make sure DataFrame is sorted chronologically
    df = df.sort_values("Date").reset_index(drop=True)
    
    h2h_record = {}
    
    df["h2h"] = ""
    df["W_h2h"] = 0
    df["L_h2h"] = 0

    for index, row in df.iterrows():
        winner = row["Winner"]
        loser = row["Loser"]

        # Unordered pair
        player_pair = tuple(sorted([winner, loser]))

        # Initialize record if needed
        if player_pair not in h2h_record:
            h2h_record[player_pair] = {"wins_A": 0, "wins_B": 0}

        wins_A = h2h_record[player_pair]["wins_A"]
        wins_B = h2h_record[player_pair]["wins_B"]

        # Determine winner’s position (A or B)
        if player_pair[0] == winner:
            # Winner is A
            df.loc[index, "h2h"] = f"{wins_A}-{wins_B}"
            df.loc[index, "W_h2h"] = wins_A
            df.loc[index, "L_h2h"] = wins_B
            h2h_record[player_pair]["wins_A"] += 1
        else:
            # Winner is B
            df.loc[index, "h2h"] = f"{wins_B}-{wins_A}"
            df.loc[index, "W_h2h"] = wins_B
            df.loc[index, "L_h2h"] = wins_A
            h2h_record[player_pair]["wins_B"] += 1

    return h2h_record, df

