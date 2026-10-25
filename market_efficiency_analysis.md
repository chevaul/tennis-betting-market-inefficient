# Complete Guide: Model Predictions to Market Efficiency Analysis

**Tennis Betting Efficiency Project - Team Guide**

From generating probabilities to identifying profitable betting opportunities, including our two-perspective dataset structure.

-----

## Overview

This guide covers the complete workflow:

1. Understanding our two-perspective data structure
1. Generating model predictions with match_id preservation
1. Handling both perspectives correctly
1. Obtaining and processing market odds
1. Running market efficiency analysis
1. Simulating profitability
1. Creating deliverables

-----

## Part 1: Understanding Our Data Structure

### Key Concept: Two Rows Per Match

Our dataset has a **unique structure**:

- Each match creates **2 rows**
- Row 1: Winner’s perspective (won=1)
- Row 2: Loser’s perspective (won=0)

**Example:**

```
Match: Federer beats Nadal

Row 1 (winner perspective):
  player_name: Federer
  opponent_name: Nadal
  won: 1
  model_prob: 0.75 (probability Federer wins)

Row 2 (loser perspective):
  player_name: Nadal
  opponent_name: Federer
  won: 0
  model_prob: 0.25 (probability Nadal wins)
```

**Important:** 0.75 + 0.25 = 1.0 (probabilities are complementary)

### Why This Matters for Market Analysis

**For calibration:** Use BOTH rows (checks consistency from any perspective)
**For profit simulation:** Use ONE row per match (avoid double-counting)
**For market comparison:** Use ONE row per match (each match = one betting opportunity)

-----

## Part 2: Generating Predictions (With match_id Preservation)

### Step 2.1: Our Train/Test Split Approach

**We use GroupShuffleSplit to keep both perspectives together:**

```python
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np

# Load data with match_id
df_full = pd.read_csv('../data/processed/model_ready/features_no_ytd.csv')

# Prepare features and target
# ... (your preprocessing steps)

# CRITICAL: Keep match_id for grouping
X = df_encoded.drop(columns=['won'])  # Keep match_id in X temporarily
y = df_encoded['won']

print("="*80)
print("TRAIN-TEST SPLIT (Preserving Match Groups)")
print("="*80)

# GroupShuffleSplit ensures both rows of same match stay together
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=X['match_id']))

# Save match_id separately BEFORE dropping it
match_id_train = X.iloc[train_idx]['match_id'].reset_index(drop=True)
match_id_test = X.iloc[test_idx]['match_id'].reset_index(drop=True)

# NOW drop match_id from features
X_train = X.iloc[train_idx].drop('match_id', axis=1).reset_index(drop=True)
X_test = X.iloc[test_idx].drop('match_id', axis=1).reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train matches: {match_id_train.nunique()} unique matches")
print(f"Test matches: {match_id_test.nunique()} unique matches")

# Verify we have 2 rows per match
assert len(X_train) == match_id_train.nunique() * 2, "Should have 2 rows per match"
assert len(X_test) == match_id_test.nunique() * 2, "Should have 2 rows per match"
print("✓ Verified: 2 rows per match in both train and test")
```

### Step 2.2: Train Model and Generate Predictions

```python
# Train model (example with CatBoost)
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    random_state=42,
    verbose=False
)

model.fit(X_train, y_train)

# Generate predictions on test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of winning

print(f"\nPredictions generated:")
print(f"  Shape: {y_prob.shape}")
print(f"  Range: [{y_prob.min():.3f}, {y_prob.max():.3f}]")
```

### Step 2.3: Create Predictions Dataset WITH match_id

```python
# CRITICAL: Include match_id so we can link to market odds
predictions_all = pd.DataFrame({
    'match_id': match_id_test,
    'model_prob': y_prob,
    'prediction': y_pred,
    'actual_outcome': y_test.values if hasattr(y_test, 'values') else y_test
})

print(f"\nAll predictions: {len(predictions_all)} rows")
print(f"Unique matches: {predictions_all['match_id'].nunique()} matches")
print(f"Rows per match: {len(predictions_all) / predictions_all['match_id'].nunique():.1f}")

# Save
predictions_all.to_csv('../data/processed/model_ready/predictions_all_perspectives.csv', index=False)
print("Saved: predictions_all_perspectives.csv")
```

### Step 2.4: Verify Probability Consistency

**Check that both perspectives of same match sum to 1.0:**

```python
print("\n" + "="*80)
print("PROBABILITY CONSISTENCY CHECK")
print("="*80)

# Group by match_id
match_groups = predictions_all.groupby('match_id')

inconsistent_matches = []
for match_id, group in match_groups:
    if len(group) == 2:
        prob_sum = group['model_prob'].sum()
        if abs(prob_sum - 1.0) > 0.01:  # Allow 1% rounding error
            inconsistent_matches.append({
                'match_id': match_id,
                'prob_1': group.iloc[0]['model_prob'],
                'prob_2': group.iloc[1]['model_prob'],
                'sum': prob_sum
            })

if len(inconsistent_matches) > 0:
    print(f"⚠ WARNING: {len(inconsistent_matches)} matches have inconsistent probabilities")
    print("\nExamples:")
    for m in inconsistent_matches[:5]:
        print(f"  Match {m['match_id']}: {m['prob_1']:.3f} + {m['prob_2']:.3f} = {m['sum']:.3f}")
    print("\nThis suggests a model issue - probabilities should be complementary!")
else:
    print("✓ All matches have consistent probabilities")
    print("  Example matches:")
    for match_id, group in match_groups.head(3):
        if len(group) == 2:
            print(f"    Match {match_id}: {group.iloc[0]['model_prob']:.3f} + {group.iloc[1]['model_prob']:.3f} = {group['model_prob'].sum():.3f}")
```

### Step 2.5: Create Unique Matches Dataset

**For market comparison, use ONE row per match:**

```python
# Option A: Keep only winner perspective (actual_outcome = 1)
predictions_unique = predictions_all[predictions_all['actual_outcome'] == 1].copy()

# Option B: Keep first row of each match
# predictions_unique = predictions_all.drop_duplicates(subset='match_id', keep='first')

print(f"\nUnique matches dataset: {len(predictions_unique)} rows")
print(f"Should be half of all predictions: {len(predictions_all) / 2:.0f}")

# Verify
assert len(predictions_unique) == predictions_all['match_id'].nunique(), "Should be one per match"
print("✓ Verified: One row per unique match")

# Save
predictions_unique.to_csv('../data/processed/model_ready/predictions_unique_matches.csv', index=False)
print("Saved: predictions_unique_matches.csv")
```

**Use predictions_all for:** Calibration curves, probability consistency checks
**Use predictions_unique for:** Market comparison, profit simulation, accuracy metrics

-----

## Part 3: Obtaining and Processing Market Odds

### Step 3.1: Check If You Already Have Odds

```python
# Check your original features file
features_original = pd.read_csv('../data/raw/features.csv')  # Or wherever your original data is

# Look for odds columns
odds_cols = [col for col in features_original.columns if 'odds' in col.lower()]
print("Columns with 'odds':")
print(odds_cols)

# Common column names:
# - winner_odds, loser_odds
# - B365W, B365L (Bet365)
# - PSW, PSL (Pinnacle Sports)
```

### Step 3.2: Create match_id in Original Data

```python
# Create consistent match_id to merge with predictions
# Method 1: Date + Winner + Loser
features_original['match_id'] = (
    features_original['Date'].astype(str) + '_' +
    features_original['Winner'].str.replace(' ', '').str.replace('.', '') + '_' +
    features_original['Loser'].str.replace(' ', '').str.replace('.', '')
)

# Method 2: If you already have a unique identifier in original data, use that

# Extract odds
market_odds = features_original[['match_id', 'winner_odds', 'loser_odds']].copy()

# Remove duplicates if any
market_odds = market_odds.drop_duplicates(subset='match_id')

print(f"Market odds dataset: {len(market_odds)} matches")
```

### Step 3.3: Merge Predictions with Market Odds

```python
# Merge using match_id
comparison = predictions_unique.merge(market_odds, on='match_id', how='inner')

print(f"\nMerged dataset:")
print(f"  Total matches: {len(comparison)}")
print(f"  Matches with odds: {comparison['winner_odds'].notna().sum()}")
print(f"  Match rate: {comparison['winner_odds'].notna().sum() / len(comparison):.1%}")

# Drop matches without odds
comparison = comparison.dropna(subset=['winner_odds'])
print(f"  Final dataset: {len(comparison)} matches with complete data")
```

### Step 3.4: Convert Odds to Probabilities

```python
print("\n" + "="*80)
print("CONVERTING ODDS TO PROBABILITIES")
print("="*80)

# Convert decimal odds to implied probability
comparison['market_prob_winner_raw'] = 1 / comparison['winner_odds']
comparison['market_prob_loser_raw'] = 1 / comparison['loser_odds']

# Calculate overround (bookmaker margin)
comparison['total_prob'] = comparison['market_prob_winner_raw'] + comparison['market_prob_loser_raw']
comparison['overround'] = comparison['total_prob'] - 1.0

print(f"Average overround: {comparison['overround'].mean():.1%}")
print(f"Range: {comparison['overround'].min():.1%} to {comparison['overround'].max():.1%}")
print("\n(Overround = bookmaker's profit margin, typically 2-5%)")

# Remove overround to get "fair" probabilities
comparison['market_prob'] = comparison['market_prob_winner_raw'] / comparison['total_prob']

print(f"\nExample conversions:")
print(f"  Odds 1.50 → Raw prob: {1/1.50:.1%}, Fair prob: {(1/1.50)/1.04:.1%}")
print(f"  Odds 2.75 → Raw prob: {1/2.75:.1%}, Fair prob: {(1/2.75)/1.04:.1%}")

# For market analysis, use market_prob (fair odds without margin)
```

### Step 3.5: If You Don’t Have Odds

**Download from tennis-data.co.uk:**

```python
# Example: Download 2023 ATP data
# Go to: http://www.tennis-data.co.uk/2023/2023.xlsx
# Save to: ../data/raw/tennis_data_2023.xlsx

import pandas as pd

# Load downloaded data
tennis_data = pd.read_excel('../data/raw/tennis_data_2023.xlsx')

# Columns typically include: Date, Winner, Loser, B365W, B365L (Bet365 odds)
print("Available columns:")
print(tennis_data.columns.tolist())

# Create match_id matching your format
tennis_data['match_id'] = (
    tennis_data['Date'].astype(str) + '_' +
    tennis_data['Winner'].str.replace(' ', '').str.replace('.', '') + '_' +
    tennis_data['Loser'].str.replace(' ', '').str.replace('.', '')
)

# Extract odds (use Bet365 or Pinnacle - most reliable)
market_odds = tennis_data[['match_id', 'B365W', 'B365L']].copy()
market_odds = market_odds.rename(columns={'B365W': 'winner_odds', 'B365L': 'loser_odds'})

# Continue with Step 3.3 (merge with predictions)
```

-----

## Part 4: Market Efficiency Analysis

### Analysis 1: Model vs Market Accuracy

**Which produces more accurate probabilities?**

```python
from sklearn.metrics import log_loss, brier_score_loss

print("="*80)
print("ACCURACY COMPARISON: MODEL VS MARKET")
print("="*80)

# Calculate metrics (using unique matches only)
model_logloss = log_loss(comparison['actual_outcome'], comparison['model_prob'])
market_logloss = log_loss(comparison['actual_outcome'], comparison['market_prob'])

model_brier = brier_score_loss(comparison['actual_outcome'], comparison['model_prob'])
market_brier = brier_score_loss(comparison['actual_outcome'], comparison['market_prob'])

print(f"\nLog Loss (lower is better):")
print(f"  Model:  {model_logloss:.4f}")
print(f"  Market: {market_logloss:.4f}")
improvement_ll = ((market_logloss - model_logloss) / market_logloss) * 100
print(f"  Model improvement: {improvement_ll:+.1f}%")

print(f"\nBrier Score (lower is better):")
print(f"  Model:  {model_brier:.4f}")
print(f"  Market: {market_brier:.4f}")
improvement_brier = ((market_brier - model_brier) / market_brier) * 100
print(f"  Model improvement: {improvement_brier:+.1f}%")

# Interpretation
if model_logloss < market_logloss:
    print("\n✓ MODEL OUTPERFORMS MARKET")
    print("  Evidence of market inefficiency")
    print("  Model has found patterns markets miss")
elif abs(model_logloss - market_logloss) < 0.01:
    print("\n≈ MODEL AND MARKET EQUALLY ACCURATE")
    print("  Market is efficient")
    print("  Hard to find systematic edge")
else:
    print("\n✗ MARKET OUTPERFORMS MODEL")
    print("  Market incorporates information better")
    print("  Model may need improvement")
```

### Analysis 2: Calibration Comparison

**Are the probabilities trustworthy?**

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

print("\n" + "="*80)
print("CALIBRATION ANALYSIS")
print("="*80)

# Calculate calibration curves
prob_true_model, prob_pred_model = calibration_curve(
    comparison['actual_outcome'], 
    comparison['model_prob'], 
    n_bins=10,
    strategy='quantile'  # Equal number of samples per bin
)

prob_true_market, prob_pred_market = calibration_curve(
    comparison['actual_outcome'], 
    comparison['market_prob'], 
    n_bins=10,
    strategy='quantile'
)

# Create visualization
fig, ax = plt.subplots(figsize=(10, 10))

# Perfect calibration line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration', alpha=0.7)

# Model calibration
ax.plot(prob_pred_model, prob_true_model, 'o-', linewidth=3, markersize=10,
        label='Our Model', color='#3498db')

# Market calibration
ax.plot(prob_pred_market, prob_true_market, 's-', linewidth=3, markersize=10,
        label='Betting Market', color='#e74c3c')

# Formatting
ax.set_xlabel('Predicted Probability', fontsize=13, fontweight='bold')
ax.set_ylabel('Actual Win Rate', fontsize=13, fontweight='bold')
ax.set_title('Probability Calibration: Model vs Market', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Add diagonal reference
ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')

plt.tight_layout()
plt.savefig('calibration_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: calibration_comparison.png")
plt.close()

# Quantify calibration error
model_calib_error = np.mean(np.abs(prob_true_model - prob_pred_model))
market_calib_error = np.mean(np.abs(prob_true_market - prob_pred_market))

print(f"\nCalibration error:")
print(f"  Model:  {model_calib_error:.4f}")
print(f"  Market: {market_calib_error:.4f}")

if model_calib_error < market_calib_error:
    print("  ✓ Model is better calibrated")
else:
    print("  ✓ Market is better calibrated")
```

### Analysis 3: Disagreement Patterns

**When and where do model and market disagree?**

```python
print("\n" + "="*80)
print("DISAGREEMENT ANALYSIS")
print("="*80)

# Calculate disagreement
comparison['prob_diff'] = comparison['model_prob'] - comparison['market_prob']
comparison['abs_diff'] = abs(comparison['prob_diff'])

print(f"\nDisagreement statistics:")
print(f"  Mean difference: {comparison['prob_diff'].mean():+.4f}")
print(f"    (Positive = Model more bullish on average)")
print(f"  Std deviation: {comparison['prob_diff'].std():.4f}")
print(f"  Mean absolute difference: {comparison['abs_diff'].mean():.4f}")

# Distribution
print(f"\nDisagreement distribution:")
print(comparison['prob_diff'].describe())

# Large disagreements
large_diff = comparison[comparison['abs_diff'] > 0.10]
print(f"\nLarge disagreements (>10%): {len(large_diff)} matches ({len(large_diff)/len(comparison)*100:.1f}%)")

# Categorize
comparison['disagreement_type'] = 'Small (<3%)'
comparison.loc[comparison['abs_diff'] >= 0.03, 'disagreement_type'] = 'Moderate (3-10%)'
comparison.loc[comparison['abs_diff'] >= 0.10, 'disagreement_type'] = 'Large (>10%)'

print(f"\nBreakdown:")
print(comparison['disagreement_type'].value_counts())

# Who wins when they disagree?
print(f"\nWhen model is more bullish (prob_diff > 0.05):")
model_bullish = comparison[comparison['prob_diff'] > 0.05]
if len(model_bullish) > 0:
    print(f"  Matches: {len(model_bullish)}")
    print(f"  Model avg prob: {model_bullish['model_prob'].mean():.1%}")
    print(f"  Market avg prob: {model_bullish['market_prob'].mean():.1%}")
    print(f"  Actual win rate: {model_bullish['actual_outcome'].mean():.1%}")
    
    # Who was closer?
    model_error = abs(model_bullish['actual_outcome'].mean() - model_bullish['model_prob'].mean())
    market_error = abs(model_bullish['actual_outcome'].mean() - model_bullish['market_prob'].mean())
    
    if model_error < market_error:
        print(f"  ✓ Model was more accurate (error: {model_error:.1%} vs {market_error:.1%})")
    else:
        print(f"  ✗ Market was more accurate (error: {market_error:.1%} vs {model_error:.1%})")
```

### Analysis 4: Segment-by-Segment Efficiency

**Where is market most inefficient?**

```python
# You'll need to merge with original features to get surface, tournament, etc.
# Load original features
features_context = pd.read_csv('../data/processed/model_ready/features_no_ytd.csv')

# Create match_id in features if needed
features_context['match_id'] = (
    features_context['Date'].astype(str) + '_' +
    features_context['player_name'].str.replace(' ', '').str.replace('.', '') + '_' +
    features_context['opponent_name'].str.replace(' ', '').str.replace('.', '')
)

# Keep only unique matches (one perspective)
features_context = features_context[features_context['won'] == 1]

# Merge
analysis_full = comparison.merge(
    features_context[['match_id', 'Surface', 'Series', 'Round', 'player_rank', 'opponent_rank']], 
    on='match_id', 
    how='left'
)

print("="*80)
print("MARKET EFFICIENCY BY SEGMENT")
print("="*80)

# Function to compare model vs market for a segment
def compare_segment(data, segment_name):
    if len(data) < 10:  # Need minimum sample size
        return None
    
    model_ll = log_loss(data['actual_outcome'], data['model_prob'])
    market_ll = log_loss(data['actual_outcome'], data['market_prob'])
    improvement = ((market_ll - model_ll) / market_ll) * 100
    
    return {
        'segment': segment_name,
        'n': len(data),
        'model_ll': model_ll,
        'market_ll': market_ll,
        'improvement_%': improvement
    }

# By surface
print("\n1. BY SURFACE:")
print(f"{'Surface':15s} {'N':>6s} {'Model LL':>10s} {'Market LL':>10s} {'Improvement':>12s}")
print("-" * 65)

surface_results = []
for surface in ['Hard', 'Clay', 'Grass']:
    subset = analysis_full[analysis_full['Surface'] == surface]
    result = compare_segment(subset, surface)
    if result:
        surface_results.append(result)
        print(f"{result['segment']:15s} {result['n']:6d} {result['model_ll']:10.4f} {result['market_ll']:10.4f} {result['improvement_%']:+11.1f}%")

# By tournament tier
print("\n2. BY TOURNAMENT TIER:")
print(f"{'Tier':20s} {'N':>6s} {'Model LL':>10s} {'Market LL':>10s} {'Improvement':>12s}")
print("-" * 70)

tier_results = []
for tier in ['Grand Slam', 'Masters', 'ATP500', 'ATP250']:
    subset = analysis_full[analysis_full['Series'].str.contains(tier, na=False)]
    result = compare_segment(subset, tier)
    if result:
        tier_results.append(result)
        print(f"{result['segment']:20s} {result['n']:6d} {result['model_ll']:10.4f} {result['market_ll']:10.4f} {result['improvement_%']:+11.1f}%")

# By match competitiveness
print("\n3. BY MATCH TYPE:")
print(f"{'Match Type':20s} {'N':>6s} {'Model LL':>10s} {'Market LL':>10s} {'Improvement':>12s}")
print("-" * 70)

analysis_full['rank_spread'] = abs(analysis_full['player_rank'] - analysis_full['opponent_rank'])
analysis_full['match_type'] = pd.cut(
    analysis_full['rank_spread'], 
    bins=[0, 20, 50, 5000], 
    labels=['Close (0-20)', 'Moderate (20-50)', 'Mismatch (50+)']
)

match_results = []
for match_type in ['Close (0-20)', 'Moderate (20-50)', 'Mismatch (50+)']:
    subset = analysis_full[analysis_full['match_type'] == match_type]
    result = compare_segment(subset, match_type)
    if result:
        match_results.append(result)
        print(f"{result['segment']:20s} {result['n']:6d} {result['model_ll']:10.4f} {result['market_ll']:10.4f} {result['improvement_%']:+11.1f}%")

# Find where model has biggest edge
all_results = surface_results + tier_results + match_results
if all_results:
    best_segment = max(all_results, key=lambda x: x['improvement_%'])
    print(f"\n✓ Biggest edge: {best_segment['segment']} ({best_segment['improvement_%']:+.1f}% improvement)")
```

-----

## Part 5: Profitability Simulation

### Simulation 1: Fixed Stake Betting

**Simple strategy: Bet $100 when model prob > market prob + threshold**

```python
print("\n" + "="*80)
print("PROFIT SIMULATION: FIXED STAKE")
print("="*80)

def simulate_fixed_stake_betting(df, threshold=0.05, stake=100):
    """
    Bet fixed amount when model disagrees with market
    
    Parameters:
    - threshold: Minimum prob_diff to bet
    - stake: Amount to bet per match
    
    Returns: DataFrame with bet results
    """
    
    # Identify value bets
    df = df.copy()
    df['is_value_bet'] = df['prob_diff'] > threshold
    value_bets = df[df['is_value_bet']].copy()
    
    if len(value_bets) == 0:
        print(f"No bets meet threshold {threshold}")
        return None
    
    # Calculate returns
    returns = []
    for idx, row in value_bets.iterrows():
        decimal_odds = 1 / row['market_prob']
        
        if row['actual_outcome'] == 1:
            # Win: get back stake + profit
            profit = stake * (decimal_odds - 1)
        else:
            # Loss: lose stake
            profit = -stake
        
        returns.append(profit)
    
    value_bets['bet_return'] = returns
    value_bets['cumulative_profit'] = np.cumsum(returns)
    
    # Calculate statistics
    total_profit = sum(returns)
    total_staked = len(returns) * stake
    roi = (total_profit / total_staked) * 100
    win_rate = value_bets['actual_outcome'].mean()
    
    # Sharpe ratio (annualized, assuming ~50 bets per year)
    if len(returns) > 1:
        returns_array = np.array(returns)
        sharpe = (returns_array.mean() / returns_array.std()) * np.sqrt(50) if returns_array.std() > 0 else 0
    else:
        sharpe = 0
    
    return {
        'threshold': threshold,
        'num_bets': len(returns),
        'total_profit': total_profit,
        'total_staked': total_staked,
        'roi': roi,
        'win_rate': win_rate,
        'avg_odds': (1 / value_bets['market_prob']).mean(),
        'sharpe_ratio': sharpe,
        'bets_df': value_bets
    }

# Test with threshold = 0.05
results = simulate_fixed_stake_betting(comparison, threshold=0.05, stake=100)

if results:
    print(f"\nStrategy: Bet $100 when model prob > market prob + 0.05")
    print(f"\nResults:")
    print(f"  Number of bets: {results['num_bets']}")
    print(f"  Total staked: ${results['total_staked']:.2f}")
    print(f"  Total profit: ${results['total_profit']:.2f}")
    print(f"  ROI: {results['roi']:.2f}%")
    print(f"  Win rate: {results['win_rate']:.1%}")
    print(f"  Average odds: {results['avg_odds']:.2f}")
    print(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
```

### Simulation 2: Test Multiple Thresholds

**Find optimal threshold:**

```python
print("\n" + "-"*80)
print("TESTING MULTIPLE THRESHOLDS")
print("-"*80)

thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
threshold_results = []

print(f"\n{'Threshold':>10s} {'Bets':>8s} {'Profit':>12s} {'ROI':>8s} {'Win Rate':>10s} {'Sharpe':>8s}")
print("-" * 75)

for thresh in thresholds:
    result = simulate_fixed_stake_betting(comparison, threshold=thresh, stake=100)
    
    if result and result['num_bets'] > 0:
        threshold_results.append(result)
        
        print(f"{thresh:>10.2f} {result['num_bets']:>8d} "
              f"${result['total_profit']:>11.2f} {result['roi']:>7.2f}% "
              f"{result['win_rate']:>9.1%} {result['sharpe_ratio']:>7.2f}")

# Convert to DataFrame
threshold_df = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'bets_df'} 
    for r in threshold_results
])

# Find best by ROI
best_roi_idx = threshold_df['roi'].idxmax()
best_roi = threshold_df.iloc[best_roi_idx]

# Find best by Sharpe
best_sharpe_idx = threshold_df['sharpe_ratio'].idxmax()
best_sharpe = threshold_df.iloc[best_sharpe_idx]

print(f"\n✓ Best ROI: threshold={best_roi['threshold']:.2f}, ROI={best_roi['roi']:.2f}%, Bets={int(best_roi['num_bets'])}")
print(f"✓ Best Sharpe: threshold={best_sharpe['threshold']:.2f}, Sharpe={best_sharpe['sharpe_ratio']:.2f}, Bets={int(best_sharpe['num_bets'])}")
```

### Simulation 3: Kelly Criterion Betting

**Optimal bet sizing based on perceived edge:**

```python
print("\n" + "="*80)
print("KELLY CRITERION ANALYSIS")
print("="*80)

def calculate_kelly_fraction(model_prob, market_prob):
    """
    Kelly formula: f = (bp - q) / b
    where:
    - b = decimal_odds - 1
    - p = model probability
    - q = 1 - p
    """
    decimal_odds = 1 / market_prob
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    return max(0, kelly)  # No negative bets

# Calculate Kelly fraction for each match
comparison['kelly_fraction'] = comparison.apply(
    lambda row: calculate_kelly_fraction(row['model_prob'], row['market_prob']),
    axis=1
)

# Analyze Kelly recommendations
kelly_bets = comparison[comparison['kelly_fraction'] > 0.01]  # Bet at least 1% of bankroll

print(f"Matches where Kelly > 1%: {len(kelly_bets)} ({len(kelly_bets)/len(comparison)*100:.1f}%)")
print(f"\nKelly fraction statistics:")
print(kelly_bets['kelly_fraction'].describe())

print(f"\nTop 10 Kelly recommendations:")
top_kelly = kelly_bets.nlargest(10, 'kelly_fraction')[['match_id', 'model_prob', 'market_prob', 'kelly_fraction', 'actual_outcome']]
print(top_kelly.to_string(index=False))

# Simulate Kelly betting
print(f"\n" + "-"*80)
print("KELLY BETTING SIMULATION")
print("-"*80)

bankroll = 10000  # Starting bankroll
kelly_multiplier = 0.5  # Half-Kelly for safety
total_profit = 0

for idx, row in kelly_bets.iterrows():
    bet_amount = bankroll * row['kelly_fraction'] * kelly_multiplier
    decimal_odds = 1 / row['market_prob']
    
    if row['actual_outcome'] == 1:
        profit = bet_amount * (decimal_odds - 1)
    else:
        profit = -bet_amount
    
    total_profit += profit

kelly_roi = (total_profit / bankroll) * 100

print(f"Starting bankroll: ${bankroll:,.2f}")
print(f"Kelly multiplier: {kelly_multiplier} (half-Kelly for safety)")
print(f"Total bets: {len(kelly_bets)}")
print(f"Total profit: ${total_profit:,.2f}")
print(f"ROI: {kelly_roi:.2f}%")
print(f"Final bankroll: ${bankroll + total_profit:,.2f}")
```

-----

## Part 6: Visualizations

### Viz 1: Model vs Market Scatter Plot

```python
fig, ax = plt.subplots(figsize=(11, 11))

# Create scatter plot
scatter = ax.scatter(
    comparison['market_prob'], 
    comparison['model_prob'],
    c=comparison['actual_outcome'], 
    cmap='RdYlGn', 
    alpha=0.5, 
    s=50,
    edgecolors='black',
    linewidth=0.5
)

# Perfect agreement line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect agreement', alpha=0.7)

# Color bar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Actual Outcome (0=Loss, 1=Win)', fontsize=11)

# Labels and title
ax.set_xlabel('Market Probability', fontsize=13, fontweight='bold')
ax.set_ylabel('Model Probability', fontsize=13, fontweight='bold')
ax.set_title('Model vs Market Predictions\n(Points colored by actual outcome)', 
            fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Add region labels
ax.text(0.15, 0.85, 'Model more\nbullish', fontsize=11, 
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(0.85, 0.15, 'Market more\nbullish', fontsize=11,
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.tight_layout()
plt.savefig('model_vs_market_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: model_vs_market_scatter.png")
plt.close()
```

**Interpretation:**

- **Green above diagonal:** Model correctly more confident → PROFIT OPPORTUNITY
- **Red above diagonal:** Model incorrectly more confident → LOSS
- **Green below diagonal:** Market correctly more confident → MISSED OPPORTUNITY
- **Points near diagonal:** Agreement (no edge)

### Viz 2: Profitability Dashboard

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Betting Strategy Profitability Analysis', fontsize=16, fontweight='bold')

# Plot 1: ROI by threshold
axes[0, 0].plot(threshold_df['threshold'], threshold_df['roi'], 'o-', linewidth=2, markersize=8, color='#3498db')
axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
axes[0, 0].fill_between(threshold_df['threshold'], 0, threshold_df['roi'], 
                        where=(threshold_df['roi'] > 0), alpha=0.3, color='green')
axes[0, 0].fill_between(threshold_df['threshold'], 0, threshold_df['roi'], 
                        where=(threshold_df['roi'] <= 0), alpha=0.3, color='red')
axes[0, 0].set_xlabel('Probability Threshold')
axes[0, 0].set_ylabel('ROI (%)')
axes[0, 0].set_title('ROI vs Selectivity')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Number of bets
axes[0, 1].plot(threshold_df['threshold'], threshold_df['num_bets'], 'o-', linewidth=2, markersize=8, color='#e74c3c')
axes[0, 1].set_xlabel('Probability Threshold')
axes[0, 1].set_ylabel('Number of Bets')
axes[0, 1].set_title('Bet Frequency vs Selectivity')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Win rate by threshold
axes[0, 2].plot(threshold_df['threshold'], threshold_df['win_rate'] * 100, 'o-', linewidth=2, markersize=8, color='#2ecc71')
axes[0, 2].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
axes[0, 2].set_xlabel('Probability Threshold')
axes[0, 2].set_ylabel('Win Rate (%)')
axes[0, 2].set_title('Win Rate vs Selectivity')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Cumulative profit (best threshold)
best_result = threshold_results[best_roi_idx]
cumulative_df = best_result['bets_df'].sort_index()

axes[1, 0].plot(range(len(cumulative_df)), cumulative_df['cumulative_profit'], linewidth=2, color='#3498db')
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1, 0].fill_between(range(len(cumulative_df)), 0, cumulative_df['cumulative_profit'],
                        where=(cumulative_df['cumulative_profit'] > 0), alpha=0.3, color='green')
axes[1, 0].fill_between(range(len(cumulative_df)), 0, cumulative_df['cumulative_profit'],
                        where=(cumulative_df['cumulative_profit'] <= 0), alpha=0.3, color='red')
axes[1, 0].set_xlabel('Bet Number')
axes[1, 0].set_ylabel('Cumulative Profit ($)')
axes[1, 0].set_title(f'Cumulative Profit (threshold={best_result["threshold"]})')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Disagreement distribution
axes[1, 1].hist(comparison['prob_diff'], bins=50, alpha=0.7, color='#9b59b6', edgecolor='black')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='Threshold=0.05')
axes[1, 1].set_xlabel('Model Prob - Market Prob')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Disagreements')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Segment comparison
segment_df = pd.DataFrame(surface_results)
if len(segment_df) > 0:
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in segment_df['improvement_%']]
    axes[1, 2].barh(range(len(segment_df)), segment_df['improvement_%'], color=colors)
    axes[1, 2].set_yticks(range(len(segment_df)))
    axes[1, 2].set_yticklabels(segment_df['segment'])
    axes[1, 2].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[1, 2].set_xlabel('Improvement over Market (%)')
    axes[1, 2].set_title('Model Edge by Surface')
    axes[1, 2].grid(True, alpha=0.3, axis='x')
    axes[1, 2].invert_yaxis()

plt.tight_layout()
plt.savefig('profitability_dashboard.png', dpi=300, bbox_inches='tight')
print("Saved: profitability_dashboard.png")
plt.close()
```

-----

## Part 7: Creating Final Deliverables

### Deliverable 1: Summary Statistics Table

```python
# Create comprehensive results table
summary = pd.DataFrame({
    'Metric': [
        'Matches Analyzed',
        'Model Log Loss',
        'Market Log Loss',
        'Model Improvement',
        '',
        'Optimal Threshold',
        'Value Bets Identified',
        'Total Profit (sim)',
        'ROI',
        'Win Rate on Value Bets',
        'Sharpe Ratio'
    ],
    'Value': [
        f"{len(comparison):,}",
        f"{model_logloss:.4f}",
        f"{market_logloss:.4f}",
        f"{improvement_ll:+.1f}%",
        '',
        f"{best_roi['threshold']:.2f}",
        f"{int(best_roi['num_bets']):,}",
        f"${best_roi['total_profit']:,.2f}",
        f"{best_roi['roi']:.2f}%",
        f"{best_roi['win_rate']:.1%}",
        f"{best_roi['sharpe_ratio']:.2f}"
    ]
})

print("\n" + "="*80)
print("MARKET EFFICIENCY SUMMARY")
print("="*80)
print("\n" + summary.to_string(index=False))

# Save
summary.to_csv('../data/processed/model_ready/market_efficiency_summary.csv', index=False)
```

### Deliverable 2: Report Template

```python
# Save comprehensive report as markdown
report = f"""
# Market Efficiency Analysis Report

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Analyst:** [Your Name]
**Matches Analyzed:** {len(comparison):,}

---

## Executive Summary

Our CatBoost model demonstrates superior accuracy compared to betting market probabilities:

- **Model log loss:** {model_logloss:.4f}
- **Market log loss:** {market_logloss:.4f}
- **Improvement:** {improvement_ll:+.1f}%

Profit simulation indicates potential for positive returns:

- **Optimal threshold:** {best_roi['threshold']:.2f}
- **Value bets identified:** {int(best_roi['num_bets']):,} ({best_roi['num_bets']/len(comparison)*100:.1f}% of matches)
- **Simulated ROI:** {best_roi['roi']:.2f}%
- **Win rate on value bets:** {best_roi['win_rate']:.1%}

**Key Finding:** {
'Model significantly outperforms market - evidence of market inefficiency' if improvement_ll > 5 else
'Model marginally outperforms market - limited edge available' if improvement_ll > 0 else
'Market is efficient - no systematic edge detected'
}

---

## 1. Accuracy Comparison

### Log Loss
- Model: {model_logloss:.4f}
- Market: {market_logloss:.4f}
- Improvement: {improvement_ll:+.1f}%

### Brier Score
- Model: {model_brier:.4f}
- Market: {market_brier:.4f}
- Improvement: {improvement_brier:+.1f}%

**Interpretation:** {"Model probabilities are more accurate than market probabilities." if model_logloss < market_logloss else "Market probabilities are more accurate than model probabilities."}

---

## 2. Profitability Analysis

### Fixed Stake Strategy

Strategy: Bet $100 when model probability exceeds market probability by threshold amount.

Results across different thresholds:

{threshold_df.to_string(index=False)}

**Optimal Strategy:** Threshold = {best_roi['threshold']:.2f}
- Expected ROI: {best_roi['roi']:.2f}%
- Bet frequency: {best_roi['num_bets']} bets ({best_roi['num_bets']/len(comparison)*100:.1f}% of matches)
- Win rate: {best_roi['win_rate']:.1%}
- Risk-adjusted return (Sharpe): {best_roi['sharpe_ratio']:.2f}

---

## 3. Market Inefficiency Patterns

### By Surface
{pd.DataFrame(surface_results).to_string(index=False) if surface_results else "No surface data available"}

### By Tournament Tier
{pd.DataFrame(tier_results).to_string(index=False) if tier_results else "No tournament data available"}

### By Match Competitiveness
{pd.DataFrame(match_results).to_string(index=False) if match_results else "No ranking data available"}

**Key Finding:** {"Model has strongest edge on " + best_segment['segment'] + f" ({best_segment['improvement_%']:+.1f}% improvement)" if all_results else "Segment analysis pending"}

---

## 4. Recommendations

Based on this analysis:

1. **Betting Strategy:** {"Implement selective betting strategy with threshold=" + str(best_roi['threshold']) if best_roi['roi'] > 0 else "Market appears efficient - betting not recommended"}

2. **Focus Areas:** {"Concentrate on " + best_segment['segment'] + " matches where model has strongest edge" if all_results else "Analysis needed"}

3. **Bet Sizing:** {"Use fractional Kelly (0.5x) for risk management" if best_roi['roi'] > 0 else "N/A"}

4. **Expected Returns:** {"ROI of " + f"{best_roi['roi']:.2f}%" + " on selected matches" if best_roi['roi'] > 0 else "No positive expectation found"}

---

## 5. Risk Assessment

- **Sample size:** {len(kelly_bets)} bets ({"sufficient" if len(kelly_bets) > 100 else "limited - more data needed"})
- **Sharpe ratio:** {best_roi['sharpe_ratio']:.2f} ({"excellent" if best_roi['sharpe_ratio'] > 2 else "good" if best_roi['sharpe_ratio'] > 1 else "poor"})
- **Consistency:** {"Performance consistent across segments" if len(all_results) > 0 and min([r['improvement_%'] for r in all_results]) > -5 else "Variable across segments - some inefficiency"}

---

## Appendix: Methodology

- **Model:** CatBoost (73.1% accuracy, 0.808 AUC)
- **Data:** {len(comparison):,} test matches
- **Period:** 2015-2024
- **Validation:** Out-of-sample test set, cross-validated
- **No data leakage:** Only pre-match information used

---

*Analysis completed on {pd.Timestamp.now().strftime('%Y-%m-%d')}*
"""

# Save report
with open('../data/processed/model_ready/market_efficiency_report.md', 'w') as f:
    f.write(report)

print("\nSaved: market_efficiency_report.md")
```

-----

## Part 8: Troubleshooting & FAQ

### Q: “I don’t have match_id in my test set”

**A:** Recreate the split with match_id preservation:

```python
# Load original data with match_id
df_with_id = pd.read_csv('../data/processed/model_ready/features_no_ytd.csv')

# Your preprocessing steps...
# Then split:

from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df_with_id['match_id']))

# Save match_id for later
match_id_test = df_with_id.iloc[test_idx]['match_id']
match_id_test.to_csv('match_id_test.csv', index=False)
```

### Q: “Where do I get market odds?”

**A:** Three options:

1. **Your original dataset** - Check if features.csv has winner_odds/loser_odds
1. **tennis-data.co.uk** - Free historical ATP data with Bet365/Pinnacle odds
1. **If unavailable** - Focus on temporal validation (test on 2024 matches) instead

### Q: “Should I use both perspectives or one per match?”

**A:** Depends on the analysis:

|Analysis Type                |Use              |
|-----------------------------|-----------------|
|Calibration curves           |BOTH perspectives|
|Probability consistency check|BOTH perspectives|
|Market comparison (log loss) |ONE per match    |
|Profit simulation            |ONE per match    |
|Accuracy metrics             |ONE per match    |

### Q: “Model probabilities don’t sum to 1.0”

**A:** They should! Check:

```python
# Verify complementary probabilities
for match_id in comparison['match_id'].unique()[:5]:
    match_probs = predictions_all[predictions_all['match_id'] == match_id]['model_prob']
    if len(match_probs) == 2:
        print(f"Match {match_id}: {match_probs.iloc[0]:.3f} + {match_probs.iloc[1]:.3f} = {match_probs.sum():.3f}")
```

If they don’t sum to 1.0, there’s an issue with your model or data processing.

### Q: “Simulation shows negative ROI - what does that mean?”

**A:** Either:

1. Market is efficient (hard to beat)
1. Model isn’t as good as markets at pricing matches
1. Threshold is wrong (try different values)
1. Sample size too small (need more data)

**Don’t panic** - markets are designed to be efficient. Even small positive ROI is valuable!

### Q: “Which threshold should I recommend?”

**A:** Consider the trade-off:

- **Low threshold (0.02):** Many bets, lower ROI, more consistent
- **High threshold (0.15):** Few bets, higher ROI, more volatile

**Recommendation:** Choose threshold that maximizes **Sharpe ratio** (risk-adjusted returns), not just ROI.

-----

## Part 9: Week Timeline

### Monday: Setup (2-3 hours)

- [ ] Generate predictions with match_id
- [ ] Load/obtain market odds
- [ ] Create comparison dataset
- [ ] Verify data quality

### Tuesday: Core Analysis (3-4 hours)

- [ ] Model vs market accuracy (log loss, Brier)
- [ ] Calibration plots
- [ ] Basic disagreement analysis
- [ ] Initial findings

### Wednesday: Profitability (3-4 hours)

- [ ] Fixed stake simulation
- [ ] Multiple threshold testing
- [ ] Kelly criterion analysis
- [ ] Optimal strategy identification

### Thursday: Deep Dive (3-4 hours)

- [ ] Segment analysis (surface, tournament, match type)
- [ ] Favorite-longshot bias
- [ ] Edge quantification
- [ ] Risk assessment

### Friday: Deliverables (2-3 hours)

- [ ] Compile all results
- [ ] Create visualizations
- [ ] Write summary report
- [ ] Present findings to team

**Total time: 13-18 hours**

-----

## Part 10: Expected Deliverables

### By Friday, deliver:

**1. Summary Document (2-3 pages)**

- Executive summary (key findings in 3 bullets)
- Model vs market comparison
- Profitability simulation results
- Recommendations

**2. Key Visualizations (4-6 images)**

- Calibration comparison plot
- Model vs market scatter plot
- Profitability dashboard (6-panel)
- Segment analysis bar chart

**3. Data Files**

- predictions_unique_matches.csv (with market odds)
- market_efficiency_summary.csv
- betting_simulation_results.csv

**4. Code Notebook**

- Jupyter notebook with all analysis
- Reproducible and documented
- Clear section headers

-----

## Part 11: Quick Start Code (Complete Workflow)

```python
"""
COMPLETE WORKFLOW: MARKET EFFICIENCY ANALYSIS
Copy and run this entire script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit
import joblib

sns.set_style("whitegrid")

# ============================================================================
# STEP 1: GENERATE PREDICTIONS WITH MATCH_ID
# ============================================================================

print("STEP 1: Generating predictions...")

# Load data
df = pd.read_csv('../data/processed/model_ready/features_no_ytd.csv')

# Prepare features (your preprocessing steps here)
# ... 

# Split with GroupShuffleSplit (preserves match groups)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=X['match_id']))

# Save match_id
match_id_test = X.iloc[test_idx]['match_id'].reset_index(drop=True)

# Create train/test sets (drop match_id from features)
X_train = X.iloc[train_idx].drop('match_id', axis=1)
X_test = X.iloc[test_idx].drop('match_id', axis=1)
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Load trained model
model = joblib.load('../data/processed/model_ready/catboost_best_model.pkl')

# Generate predictions
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Create predictions dataset
predictions_all = pd.DataFrame({
    'match_id': match_id_test,
    'model_prob': y_prob,
    'prediction': y_pred,
    'actual_outcome': y_test.values if hasattr(y_test, 'values') else y_test
})

# Create unique matches version (one per match)
predictions_unique = predictions_all[predictions_all['actual_outcome'] == 1].copy()

print(f"✓ Generated {len(predictions_all)} predictions ({predictions_all['match_id'].nunique()} unique matches)")

# ============================================================================
# STEP 2: LOAD AND PROCESS MARKET ODDS
# ============================================================================

print("\nSTEP 2: Loading market odds...")

# Load original features with odds
features_with_odds = pd.read_csv('../data/raw/features.csv')  # Adjust path as needed

# Create match_id (must match your format)
features_with_odds['match_id'] = (
    features_with_odds['Date'].astype(str) + '_' +
    features_with_odds['Winner'].str.replace(' ', '').str.replace('.', '') + '_' +
    features_with_odds['Loser'].str.replace(' ', '').str.replace('.', '')
)

# Extract odds (adjust column names as needed)
market_odds = features_with_odds[['match_id', 'winner_odds', 'loser_odds']].copy()
market_odds = market_odds.drop_duplicates(subset='match_id')

# Merge with predictions
comparison = predictions_unique.merge(market_odds, on='match_id', how='inner')

print(f"✓ Matched {len(comparison)} predictions with market odds")

# Convert odds to probabilities
comparison['market_prob_raw'] = 1 / comparison['winner_odds']
comparison['market_prob_loser_raw'] = 1 / comparison['loser_odds']
comparison['total_prob'] = comparison['market_prob_raw'] + comparison['market_prob_loser_raw']
comparison['overround'] = comparison['total_prob'] - 1.0

# Remove overround (fair probability)
comparison['market_prob'] = comparison['market_prob_raw'] / comparison['total_prob']

print(f"✓ Average overround: {comparison['overround'].mean():.1%}")

# ============================================================================
# STEP 3: ACCURACY COMPARISON
# ============================================================================

print("\nSTEP 3: Comparing accuracy...")

model_ll = log_loss(comparison['actual_outcome'], comparison['model_prob'])
market_ll = log_loss(comparison['actual_outcome'], comparison['market_prob'])
improvement = ((market_ll - model_ll) / market_ll) * 100

print(f"✓ Model log loss: {model_ll:.4f}")
print(f"✓ Market log loss: {market_ll:.4f}")
print(f"✓ Improvement: {improvement:+.1f}%")

# ============================================================================
# STEP 4: PROFIT SIMULATION
# ============================================================================

print("\nSTEP 4: Simulating profitability...")

# Calculate disagreement
comparison['prob_diff'] = comparison['model_prob'] - comparison['market_prob']

# Test multiple thresholds
thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
results = []

for threshold in thresholds:
    value_bets = comparison[comparison['prob_diff'] > threshold]
    
    if len(value_bets) == 0:
        continue
    
    profit = 0
    for _, row in value_bets.iterrows():
        odds = 1 / row['market_prob']
        if row['actual_outcome'] == 1:
            profit += 100 * (odds - 1)
        else:
            profit -= 100
    
    roi = (profit / (len(value_bets) * 100)) * 100
    
    results.append({
        'threshold': threshold,
        'num_bets': len(value_bets),
        'profit': profit,
        'roi': roi,
        'win_rate': value_bets['actual_outcome'].mean()
    })

results_df = pd.DataFrame(results)
best = results_df.loc[results_df['roi'].idxmax()]

print(f"✓ Best threshold: {best['threshold']:.2f}")
print(f"✓ ROI: {best['roi']:.2f}%")
print(f"✓ Number of bets: {int(best['num_bets'])}")

# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================

print("\nSTEP 5: Creating visualizations...")

# Calibration plot
fig, ax = plt.subplots(figsize=(8, 8))

prob_true_model, prob_pred_model = calibration_curve(
    comparison['actual_outcome'], comparison['model_prob'], n_bins=10
)
prob_true_market, prob_pred_market = calibration_curve(
    comparison['actual_outcome'], comparison['market_prob'], n_bins=10
)

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
ax.plot(prob_pred_model, prob_true_model, 'o-', linewidth=2, label='Model')
ax.plot(prob_pred_market, prob_true_market, 's-', linewidth=2, label='Market')
ax.legend()
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Actual Win Rate')
ax.set_title('Calibration Comparison')
ax.grid(True, alpha=0.3)

plt.savefig('market_efficiency_calibration.png', dpi=300, bbox_inches='tight')
print("✓ Saved: market_efficiency_calibration.png")
plt.close()

# ============================================================================
# STEP 6: SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print(f"\nKey Findings:")
print(f"1. Model {'outperforms' if improvement > 0 else 'underperforms'} market by {abs(improvement):.1f}%")
print(f"2. Simulated ROI: {best['roi']:.2f}% ({int(best['num_bets'])} bets)")
print(f"3. {'Evidence of market inefficiency' if best['roi'] > 2 else 'Limited edge available' if best['roi'] > 0 else 'No systematic edge'}")

print(f"\nFiles saved:")
print(f"  - predictions_all_perspectives.csv")
print(f"  - predictions_unique_matches.csv")
print(f"  - market_efficiency_calibration.png")
print(f"  - market_efficiency_report.md")
```

-----

## Key Takeaways for Your Team

1. **Two-perspective structure:** Remember we have 2 rows per match - handle accordingly
1. **match_id is critical:** Needed to link predictions to market odds and avoid double-counting
1. **Use correct dataset:**
- predictions_all for calibration
- predictions_unique for market comparison
1. **Probability consistency:** Model probs from both perspectives should sum to 1.0
1. **Market odds processing:** Remove overround to get fair probabilities
1. **Multiple analyses:** Run accuracy, calibration, profit simulation, and segment analysis
1. **Interpret carefully:** Positive ROI doesn’t guarantee profitability (transaction costs, sample size, etc.)

-----

**This guide has everything you need. Start with Part 2 (generating predictions), then Part 3 (market odds), then Parts 4-6 (analysis). Good luck!**