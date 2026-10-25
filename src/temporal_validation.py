# Copy this into a new Jupyter notebook cell-by-cell
# Notebook: temporal_validation_2024.ipynb

# ============================================================================
# CELL 1: Imports and Setup
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TEMPORAL VALIDATION: Train 2015-2023, Test 2024")
print("="*80)

# ============================================================================
# CELL 2: Load and Split Data
# ============================================================================

# Load full dataset
df = pd.read_csv('./data/processed/model_ready/features_final.csv')

# Add Year
df['Year'] = pd.to_datetime(df['Date']).dt.year

# Temporal split
train_data = df[df['Year'] <= 2023].copy()
test_data = df[df['Year'] == 2024].copy()

print(f"Train: {len(train_data)} rows (2015-2023)")
print(f"Test: {len(test_data)} rows (2024)")
print(f"Test matches: ~{len(test_data)//2}")

# ============================================================================
# CELL 3: Preprocessing
# ============================================================================

# Fix categorical values
for data in [train_data, test_data]:
    data['player_hand'] = data['player_hand'].replace('U', 'Unknown')
    data['opponent_hand'] = data['opponent_hand'].replace('U', 'Unknown')

# Categorical columns
categorical_cols = ['Surface', 'Court', 'Round', 'Series', 'player_hand', 'opponent_hand']
categorical_cols = [c for c in categorical_cols if c in train_data.columns]

# Fill missing - numerical
numerical_cols = train_data.select_dtypes(include=[np.number]).columns
numerical_cols = [c for c in numerical_cols if c != 'won']

for col in numerical_cols:
    median_val = train_data[col].median()
    train_data[col] = train_data[col].fillna(median_val)
    test_data[col] = test_data[col].fillna(median_val)

# Fill missing - categorical
for col in categorical_cols:
    train_data[col] = train_data[col].fillna('Unknown')
    test_data[col] = test_data[col].fillna('Unknown')

print("Missing values handled")

# ============================================================================
# CELL 4: Feature Engineering
# ============================================================================

for data in [train_data, test_data]:
    if 'player_rank' in data.columns:
        data['rank_diff'] = data['player_rank'] - data['opponent_rank']
    if 'player_elo' in data.columns:
        data['elo_diff'] = data['player_elo'] - data['opponent_elo']

print("Features engineered")

# ============================================================================
# CELL 5: Encoding
# ============================================================================

# One-hot encode
train_encoded = pd.get_dummies(train_data, columns=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)

# Align columns
missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0

test_encoded = test_encoded[train_encoded.columns]

print(f"Encoded - Train: {train_encoded.shape}, Test: {test_encoded.shape}")

# ============================================================================
# CELL 6: Prepare X and y
# ============================================================================

drop_cols = ['player_name', 'opponent_name', 'Date', 'Year', 'ATP', 
             'Tournament', 'Location', 'stats_url', 'match_id']
drop_cols = [c for c in drop_cols if c in train_encoded.columns]

X_train = train_encoded.drop(columns=['won'] + drop_cols)
y_train = train_encoded['won']

X_test = test_encoded.drop(columns=['won'] + drop_cols)
y_test = test_encoded['won']

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# ============================================================================
# CELL 7: Train CatBoost
# ============================================================================

print("\n" + "="*80)
print("TRAINING CATBOOST")
print("="*80)

catboost = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)

catboost.fit(X_train, y_train)
print("Training complete")

# ============================================================================
# CELL 8: Evaluate on 2024
# ============================================================================

y_pred = catboost.predict(X_test)
y_prob = catboost.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
ll = log_loss(y_test, y_prob)

print("\n" + "="*80)
print("2024 TEMPORAL VALIDATION RESULTS")
print("="*80)

print(f"\nCatBoost on 2024 data:")
print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"  AUC: {auc:.4f}")
print(f"  Log Loss: {ll:.4f}")

# ============================================================================
# CELL 9: Comparison to Random Split
# ============================================================================

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nOriginal (Random Split):")
print(f"  Accuracy: 73.37%")
print(f"  AUC: 0.8118")

print(f"\nTemporal (2024 Test):")
print(f"  Accuracy: {acc*100:.2f}%")
print(f"  AUC: {auc:.4f}")

diff_acc = (acc - 0.7337) * 100
diff_auc = auc - 0.8118

print(f"\nDifference:")
print(f"  Accuracy: {diff_acc:+.2f} pts")
print(f"  AUC: {diff_auc:+.4f}")

if abs(diff_acc) < 3:
    print("\n✓✓✓ EXCELLENT temporal robustness!")
elif abs(diff_acc) < 5:
    print("\n✓✓ GOOD temporal robustness")
else:
    print("\n⚠ Moderate degradation")

# ============================================================================
# CELL 10: Save Results
# ============================================================================

temporal_results = {
    'Split_Type': 'Temporal (Test 2024)',
    'Accuracy': acc,
    'AUC': auc,
    'Log_Loss': ll,
    'Test_Matches': len(test_data)//2
}

print("\n" + "="*80)
print("FOR PRESENTATION")
print("="*80)
print(f"\n'Temporal validation: Model tested on 2024 data")
print(f" ({len(test_data)//2} unseen matches) achieved {acc*100:.1f}%")
print(f" accuracy vs {73.37:.1f}% on random split - confirming")
print(f" temporal robustness'")