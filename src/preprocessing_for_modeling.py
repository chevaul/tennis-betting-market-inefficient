"""
Data Preprocessing for Tennis Match Prediction
Run this after loading the cleaned datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# ============================================================================
# HELPER FUNCTION: Normalized Match ID
# ============================================================================

def create_normalized_match_id(row):
    """Create same match_id for both perspectives - alphabetically sorted names"""
    date = str(row['Date'])
    p1 = str(row['player_name']).strip().replace(' ', '').replace('.', '')
    p2 = str(row['opponent_name']).strip().replace(' ', '').replace('.', '')
    names_sorted = '_'.join(sorted([p1, p2]))
    return f"{date}_{names_sorted}"

# ============================================================================
# LOAD DATA
# ============================================================================

# Choose your version
# Get script location and build path from there
script_dir = Path(__file__).parent
data_path = script_dir / '..' / 'data' / 'processed' / 'model_ready' / 'features_no_ytd.csv'

print("Loading from:", data_path.resolve())
df = pd.read_csv(data_path, low_memory=False)

print(f"Original shape: {df.shape}")
print(f"Format: Winner/Loser (will transform)\n")


def transform_to_player_perspective(df):
    """
    Transform match data to player perspective format.
    Each match creates 2 rows: one from each player's perspective.
    """
    
    context_cols = [
        'ATP', 'Location', 'Tournament', 'Date', 'Series', 'Court', 
        'Surface', 'Round', 'Best of', 'Year', 'stats_url', 'duration', 'h2h'
    ]
    context_cols = [c for c in context_cols if c in df.columns]
    
    # Create winner rows
    winner_rows = df[context_cols].copy()
    winner_rows['player_name'] = df['Winner']
    winner_rows['opponent_name'] = df['Loser']
    winner_rows['won'] = 1
    winner_rows['player_rank'] = df['WRank']
    winner_rows['player_points'] = df['WPts']
    winner_rows['player_elo'] = df['elo_winner']
    winner_rows['player_h2h'] = df['W_h2h']
    winner_rows['opponent_rank'] = df['LRank']
    winner_rows['opponent_points'] = df['LPts']
    winner_rows['opponent_elo'] = df['elo_loser']
    winner_rows['opponent_h2h'] = df['L_h2h']
    
    # Player stats
    for col in df.columns:
        if col.startswith('Winner_') and col != 'Winner':
            new_col = col.replace('Winner_', 'player_')
            winner_rows[new_col] = df[col]
        elif col.startswith('Loser_') and col != 'Loser':
            new_col = col.replace('Loser_', 'opponent_')
            winner_rows[new_col] = df[col]
        elif col.startswith('winner_') and 'ytd' in col:
            new_col = col.replace('winner_', 'player_')
            winner_rows[new_col] = df[col]
        elif col.startswith('loser_') and 'ytd' in col:
            new_col = col.replace('loser_', 'opponent_')
            winner_rows[new_col] = df[col]
    
    # Form
    if 'winner_form' in df.columns:
        winner_rows['player_form'] = df['winner_form']
        winner_rows['opponent_form'] = df['loser_form']
        winner_rows['form_diff'] = df['form_diff']
    
    # Elo probability
    if 'proba_elo' in df.columns:
        winner_rows['player_win_prob_elo'] = df['proba_elo']
    
    # Create loser rows (SWAP everything)
    loser_rows = df[context_cols].copy()
    loser_rows['player_name'] = df['Loser']
    loser_rows['opponent_name'] = df['Winner']
    loser_rows['won'] = 0
    loser_rows['player_rank'] = df['LRank']
    loser_rows['player_points'] = df['LPts']
    loser_rows['player_elo'] = df['elo_loser']
    loser_rows['player_h2h'] = df['L_h2h']
    loser_rows['opponent_rank'] = df['WRank']
    loser_rows['opponent_points'] = df['WPts']
    loser_rows['opponent_elo'] = df['elo_winner']
    loser_rows['opponent_h2h'] = df['W_h2h']
    
    for col in df.columns:
        if col.startswith('Loser_') and col != 'Loser':
            new_col = col.replace('Loser_', 'player_')
            loser_rows[new_col] = df[col]
        elif col.startswith('Winner_') and col != 'Winner':
            new_col = col.replace('Winner_', 'opponent_')
            loser_rows[new_col] = df[col]
        elif col.startswith('loser_') and 'ytd' in col:
            new_col = col.replace('loser_', 'player_')
            loser_rows[new_col] = df[col]
        elif col.startswith('winner_') and 'ytd' in col:
            new_col = col.replace('winner_', 'opponent_')
            loser_rows[new_col] = df[col]
    
    if 'loser_form' in df.columns:
        loser_rows['player_form'] = df['loser_form']
        loser_rows['opponent_form'] = df['winner_form']
        loser_rows['form_diff'] = -df['form_diff']
    
    if 'proba_elo' in df.columns:
        loser_rows['player_win_prob_elo'] = 1 - df['proba_elo']
    
    # Combine
    df_combined = pd.concat([winner_rows, loser_rows], ignore_index=True)
    
    if 'Date' in df_combined.columns:
        df_combined = df_combined.sort_values('Date').reset_index(drop=True)
    
    return df_combined


# ============================================================================
# 1a. TRANSFORM TO PLAYER PERSPECTIVE FIRST
# ============================================================================

print("="*80)
print("TRANSFORMING TO PLAYER PERSPECTIVE")
print("="*80)

df_transformed = transform_to_player_perspective(df)

print(f"✓ Transformed: {df_transformed.shape}")
print(f"✓ Now has player_name, opponent_name, won columns")

# ============================================================================
# 1b. CREATE NORMALIZED MATCH_ID (AFTER TRANSFORMATION!)
# ============================================================================

print("\n" + "="*80)
print("CREATING NORMALIZED MATCH_ID")
print("="*80)

df_transformed['match_id'] = df_transformed.apply(create_normalized_match_id, axis=1)

print(f"✓ Created match_id for {df_transformed['match_id'].nunique()} unique matches")
print(f"✓ Total rows: {len(df_transformed)}")

# Verify both perspectives have same match_id
match_counts = df_transformed['match_id'].value_counts()
both_persp = (match_counts == 2).sum()
one_persp = (match_counts == 1).sum()

print(f"\nPerspective verification:")
print(f"  Matches with 2 perspectives: {both_persp}")
print(f"  Matches with 1 perspective: {one_persp}")

if both_persp == df_transformed['match_id'].nunique():
    print("✓✓✓ PERFECT: GroupShuffleSplit will work correctly!")
else:
    print(f"⚠ Note: {one_persp} matches have only 1 perspective")

# Drop YTD columns
ytd_cols = [col for col in df_transformed.columns if 'ytd' in col.lower()]
if len(ytd_cols) > 0:
    print(f"\nDropping {len(ytd_cols)} YTD columns")
    df_model = df_transformed.drop(columns=ytd_cols)
else:
    df_model = df_transformed.copy()

print(f"After dropping YTD: {df_model.shape}")

# ============================================================================
# 1b. DROP NON-PREDICTIVE COLUMNS
# ============================================================================

drop_cols = [
    'player_name',      # Names not useful for prediction
    'opponent_name',
    'stats_url',        # Metadata
    'Date',             # Will use time-based split instead
    'ATP',              # Redundant with other context
    'Tournament',       # High cardinality, can cause overfitting
    'Location'          # High cardinality
]

# Keep only columns that exist
drop_cols = [col for col in drop_cols if col in df_model.columns]
# df_model already set from transformation, don't overwrite!

print(f"Ready for preprocessing: {df_model.shape}")

# ============================================================================
# 2. IDENTIFY COLUMN TYPES
# ============================================================================

# Target
target = 'won'

# Categorical columns
categorical_cols = [
    'Surface',          # Hard, Clay, Grass
    'Court',            # Indoor, Outdoor
    'Round',            # Round of 32, Finals, etc.
    'Series',           # ATP250, ATP500, etc.
    'player_hand',      # L, R
    'opponent_hand',    # L, R
]

# Keep only existing categorical columns
categorical_cols = [col for col in categorical_cols if col in df_model.columns]

# Numerical columns (everything else except target)
numerical_cols = [col for col in df_model.columns 
                  if col not in categorical_cols and col != target]

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {len(numerical_cols)} columns")

# ============================================================================
# 3. HANDLE MISSING VALUES
# ============================================================================

print("\n" + "="*80)
print("HANDLING CATEGORICAL VALUES")
print("="*80)

# Handle categorical special values (can do before split - no leakage)
df_model['player_hand'] = df_model['player_hand'].replace('U','Unknown')
df_model['opponent_hand'] = df_model['opponent_hand'].replace('U','Unknown')
df_model['Round'] = df_model['Round'].replace('Round Robin','4th Round')
df_model['Series'] = df_model['Series'].replace('Masters Cup','ATP500')

# Fill categorical missing (can do before split)
for col in categorical_cols:
    if df_model[col].isnull().any():
        df_model[col] = df_model[col].fillna('Unknown')

print("✓ Categorical values handled")

# For feature engineering, temporarily fill numericals with 0
# (Will refill properly after split with train medians)
print("\nTemporarily filling numerical NaNs with 0 for feature engineering...")
for col in numerical_cols:
    if df_model[col].isnull().any():
        df_model[col] = df_model[col].fillna(0)

print("✓ Temporary fills complete")





# ============================================================================
# 4. FEATURE ENGINEERING (Optional but recommended)
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Ranking difference (negative = player is lower ranked / worse)
if 'player_rank' in df_model.columns and 'opponent_rank' in df_model.columns:
    df_model['rank_diff'] = df_model['player_rank'] - df_model['opponent_rank']
    print("Created: rank_diff")

# Elo difference (positive = player has higher Elo / better)
if 'player_elo' in df_model.columns and 'opponent_elo' in df_model.columns:
    df_model['elo_diff'] = df_model['player_elo'] - df_model['opponent_elo']
    print("Created: elo_diff")

# H2H win rate (avoid division by zero)
if 'player_h2h' in df_model.columns and 'opponent_h2h' in df_model.columns:
    total_h2h = df_model['player_h2h'] + df_model['opponent_h2h']
    df_model['player_h2h_rate'] = np.where(
        total_h2h > 0,
        df_model['player_h2h'] / total_h2h,
        0.5  # No history = 50/50
    )
    print("Created: player_h2h_rate")
    
# Rating difference
if 'player_rating' in df_model.columns and 'opponent_rating' in df_model.columns:
    df_model['rating_diff'] = df_model['player_rating'] - df_model['opponent_rating']
    print("Created: rating_diff")
    
# Height difference
if 'player_height' in df_model.columns and 'opponent_height' in df_model.columns:
    df_model['height_diff'] = df_model['player_height'] - df_model['opponent_height']
    print("Created: height_diff")
    
# Point difference
if 'player_points' in df_model.columns and 'opponent_points' in df_model.columns:
    df_model['points_diff'] = df_model['player_points'] - df_model['opponent_points']
    print("Created: points_diff")

# Age difference
if 'player_age' in df_model.columns and 'opponent_age' in df_model.columns:
    df_model['age_diff'] = df_model['player_age'] - df_model['opponent_age']
    print("Created: age_diff")
    
# Try different scale factors
if 'player_rank' in df_model.columns and 'opponent_rank' in df_model.columns:
    df_model['proba_rank_200'] = 1 / (1 + 10 ** ((df_model['player_rank'] - df_model['opponent_rank']) / 200))
    print("Created: proba_rank_200")
    
# Different scale factor for form
if 'player_form' in df_model.columns and 'opponent_form' in df_model.columns:
    df_model['proba_form_50'] = 1 / (1 + 10 ** ((df_model['player_form'] - df_model['opponent_form']) / 50))
    print("Created: proba_form_50")

print(f"\nFinal shape with engineered features: {df_model.shape}")


# ============================================================================
# 5. ENCODE CATEGORICAL VARIABLES
# ============================================================================

print("\n" + "="*80)
print("ENCODING CATEGORICAL VARIABLES")
print("="*80)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

print(f"\nShape after encoding: {df_encoded.shape}")
print(f"Added {df_encoded.shape[1] - df_model.shape[1] + len(categorical_cols)} dummy variables")

# Export finalized full dataset (optional)
# df_to_export = df_encoded[df_encoded['won'] == 1]
df_model.to_csv(script_dir / '..' / 'data' / 'processed' /   'model_ready' / 'features_final.csv', index=False)
print("Exported finalized dataset to 'features_final.csv'")


# ============================================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================================

X = df_encoded.drop(columns=[target])
X_not_encoded = df_model.drop(columns=[target])
y = df_model[target]

print("\n" + "="*80)
print("TRAIN-TEST SPLIT")
print("="*80)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=X['match_id']))

# Save test features with identifiers immediately
test_features_with_id = X.iloc[test_idx][['Date', 'player_name', 'opponent_name', 'match_id', 'player_elo', 'opponent_elo', 'player_rank', 'opponent_rank']].reset_index(drop=True)
test_features_with_id['won'] = y.iloc[test_idx].values  # Get from y, not X

# Save to model_ready location
out_dir = script_dir / '..' / 'data' / 'processed' / 'model_ready'
out_dir.mkdir(parents=True, exist_ok=True)
test_features_with_id.to_csv(out_dir / 'test_identifiers.csv', index=False)
print(f"Saved test_identifiers.csv with {len(test_features_with_id)} rows")

X_train, X_test = X.iloc[train_idx].drop(columns=['match_id'] + drop_cols, axis=1), X.iloc[test_idx].drop(columns=['match_id'] + drop_cols, axis=1)
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

X_train_not_encoded = X_not_encoded.iloc[train_idx].drop(columns=['match_id'] + drop_cols, axis=1)
X_test_not_encoded = X_not_encoded.iloc[test_idx].drop(columns=['match_id'] + drop_cols, axis=1)

# ============================================================================
# FILL NUMERICAL MISSING VALUES (AFTER SPLIT - NO DATA LEAKAGE!)
# ============================================================================

print("\n" + "="*80)
print("FILLING NUMERICAL MISSING VALUES (Using Train Medians)")
print("="*80)

# Get numerical columns
numerical_cols_in_X = X_train.select_dtypes(include=[np.number]).columns

# Compute medians from TRAIN set only
train_medians = {}
for col in numerical_cols_in_X:
    if X_train[col].isnull().any():
        train_medians[col] = X_train[col].median()

# Fill train and test using TRAIN medians
for col, median_val in train_medians.items():
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)  # Use TRAIN median!
    print(f"  {col}: filled with {median_val:.2f}")

# Verify
train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()
print(f"\nRemaining missing:")
print(f"  Train: {train_missing}")
print(f"  Test: {test_missing}")

if train_missing == 0 and test_missing == 0:
    print("✓ All missing values filled (no data leakage!)")


print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train target distribution: {y_train.value_counts().to_dict()}")
print(f"Test target distribution: {y_test.value_counts().to_dict()}")


# ============================================================================
# 7. SCALE NUMERICAL FEATURES (Important for some models)
# ============================================================================

print("\n" + "="*80)
print("FEATURE SCALING")
print("="*80)

# Identify numerical columns to scale
cols_to_scale = [col for col in X_train.columns if not any(
    cat in col for cat in categorical_cols
)]

# Identify probability columns (already 0-1 scale)
proba_cols = [col for col in cols_to_scale if 'proba_' in col or col == 'player_win_prob_elo']

# OPTION A: Scale everything including probabilities (DEFAULT)
cols_to_scale_final = cols_to_scale
print(f"\nUsing OPTION A: Scaling ALL {len(cols_to_scale_final)} columns")
print("  (Probabilities will be scaled too)")

# OPTION B: Exclude probabilities from scaling (UNCOMMENT TO USE)
# cols_to_scale_final = [col for col in cols_to_scale if col not in proba_cols]
# print(f"\nUsing OPTION B: Scaling {len(cols_to_scale_final)} columns")
# print(f"  (Excluding {len(proba_cols)} probability columns)")

# Apply scaling
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[cols_to_scale_final] = scaler.fit_transform(X_train[cols_to_scale_final])
X_test_scaled[cols_to_scale_final] = scaler.transform(X_test[cols_to_scale_final])

print("\nScaling complete.")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PREPROCESSING COMPLETE")
print("="*80)

print(f"\nFinal dataset:")
print(f"  Total samples: {len(X):,}")
print(f"  Total features: {X.shape[1]:,}")
print(f"  Train samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")
print(f"  Target balance: {y.mean():.1%} won")

print("\nReady for modeling!")
print("\nYou now have:")
print("  X_train, X_test, y_train, y_test - Original scale")
print("  X_train_scaled, X_test_scaled - Scaled features")
print("\nUse original scale for tree-based models (Random Forest, XGBoost)")
print("Use scaled features for linear models (Logistic Regression, SVM)")

# ============================================================================
# SAVE PROCESSED DATA (Optional)
# ============================================================================

from pathlib import Path

script_dir = Path(__file__).parent

out_dir = script_dir / '..' / 'data' / 'processed' / 'model_ready'
out_dir.mkdir(parents=True, exist_ok=True)

# Uncomment to save
X_train.to_csv(out_dir /'X_train.csv', index=False)
X_test.to_csv(out_dir /'X_test.csv', index=False)
y_train.to_csv(out_dir /'y_train.csv', index=False)
y_test.to_csv(out_dir /'y_test.csv', index=False)
X_train_scaled.to_csv(out_dir /'X_train_scaled.csv', index=False)
X_test_scaled.to_csv(out_dir /'X_test_scaled.csv', index=False)
X_train_not_encoded.to_csv(out_dir /'X_train_not_encoded.csv', index=False)
X_test_not_encoded.to_csv(out_dir /'X_test_not_encoded.csv', index=False)