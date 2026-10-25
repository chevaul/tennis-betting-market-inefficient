"""
Simple One-Page Visualization of Preprocessing
Shows the key aspects in a single figure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Assumes you've run preprocessing_for_modeling.py and have:
# X_train, X_test, y_train, y_test (original scale)
# X_train_scaled, X_test_scaled (scaled)

files_to_get = ['X_test.csv', 'y_test.csv', 'X_train.csv', 'y_train.csv','X_test_scaled.csv', 'X_train_scaled.csv']

# Get script location and build path from there
script_dir = Path(__file__).parent
data_path = script_dir / '..' / 'data' / 'processed' / 'model_ready' / 'features_no_ytd.csv'

for i in files_to_get:
    vars()[i.split('.')[0]] = pd.read_csv(script_dir / '..' / 'data' / 'processed' / 'model_ready' / i)
    
y_train_array = y_train.values.ravel() if hasattr(y_train, 'values') else np.array(y_train).ravel()
y_test_array = y_test.values.ravel() if hasattr(y_test, 'values') else np.array(y_test).ravel()

sns.set_style("whitegrid")

# Convert y to arrays
if isinstance(y_train, pd.DataFrame):
    y_train_array = y_train['won'].values
    y_test_array = y_test['won'].values
elif isinstance(y_train, pd.Series):
    y_train_array = y_train.values
    y_test_array = y_test.values
else:
    y_train_array = np.array(y_train).ravel()
    y_test_array = np.array(y_test).ravel()

# Create figure with more space for middle row
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Tennis Match Data: Preprocessing & Train/Test Split Overview', 
             fontsize=18, fontweight='bold', y=0.98)

# ============================================================================
# 1. TRAIN/TEST SPLIT (Top Left)
# ============================================================================
ax1 = plt.subplot(3, 4, 1)
sizes = [len(X_train), len(X_test)]
colors = ['#3498db', '#e74c3c']
wedges, texts, autotexts = ax1.pie(sizes, labels=['Train (80%)', 'Test (20%)'], 
                                     autopct='%1.0f%%', colors=colors, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)
ax1.set_title('Train/Test Split', fontweight='bold')

# ============================================================================
# 2. TARGET BALANCE (Top Center-Left)
# ============================================================================
ax2 = plt.subplot(3, 4, 2)

train_win = int(np.sum(y_train_array == 1))
train_loss = int(np.sum(y_train_array == 0))
test_win = int(np.sum(y_test_array == 1))
test_loss = int(np.sum(y_test_array == 0))

x = np.arange(2)
width = 0.35
ax2.bar(x - width/2, [train_loss, test_loss], width, label='Loss', color='#e74c3c')
ax2.bar(x + width/2, [train_win, test_win], width, label='Win', color='#2ecc71')
ax2.set_xticks(x)
ax2.set_xticklabels(['Train', 'Test'])
ax2.set_ylabel('Count')
ax2.set_title('Target Distribution (Balanced 50/50)', fontweight='bold')
ax2.legend()

# ============================================================================
# 3. DATASET INFO (Top Center-Right)
# ============================================================================
ax3 = plt.subplot(3, 4, 3)
info_text = f"""
Total Features: {X_train.shape[1]:,}
Total Samples: {len(X_train) + len(X_test):,}

Train: {len(X_train):,}
Test: {len(X_test):,}

Original: 38 features
Encoded: {X_train.shape[1]:,}
Engineered: 4
"""
ax3.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax3.axis('off')
ax3.set_title('Dataset Info', fontweight='bold')

# ============================================================================
# 4. FEATURE IMPORTANCE INFO (Top Right)
# ============================================================================
ax4 = plt.subplot(3, 4, 4)
key_info = """
Key Features:
- Elo ratings
- ATP rankings  
- Win probabilities
- H2H records
- Player attributes
- Form metrics

Visualized below
"""
ax4.text(0.1, 0.5, key_info, fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax4.axis('off')
ax4.set_title('Features Used', fontweight='bold')

# ============================================================================
# 5-7. SCALING EFFECT - SIDE BY SIDE (Middle Row)
# ============================================================================

possible_features = ['elo_diff', 'player_elo', 'rank_diff']
available_features = [f for f in possible_features if f in X_train.columns]

for idx, feature in enumerate(available_features[:3]):
    # Original (left)
    ax_orig = plt.subplot(3, 4, 5 + idx*2)
    orig_data = X_train[feature].dropna().values
    
    ax_orig.hist(orig_data, bins=40, color='#3498db', alpha=0.7, edgecolor='none')
    ax_orig.set_ylabel('Frequency', fontsize=9)
    ax_orig.set_xlabel('Original Value', fontsize=9)
    ax_orig.set_title(f'{feature} - Original Scale', fontweight='bold', fontsize=10)
    ax_orig.tick_params(labelsize=8)
    
    # Add stats
    ax_orig.text(0.02, 0.98, f'μ={orig_data.mean():.1f}\nσ={orig_data.std():.1f}', 
                transform=ax_orig.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
    
    # Scaled (right)
    ax_scaled = plt.subplot(3, 4, 6 + idx*2)
    scaled_data = X_train_scaled[feature].dropna().values
    
    ax_scaled.hist(scaled_data, bins=40, color='#e74c3c', alpha=0.7, edgecolor='none')
    ax_scaled.set_ylabel('Frequency', fontsize=9)
    ax_scaled.set_xlabel('Scaled Value', fontsize=9)
    ax_scaled.set_title(f'{feature} - Scaled (μ=0, σ=1)', fontweight='bold', fontsize=10)
    ax_scaled.tick_params(labelsize=8)
    
    # Add stats
    ax_scaled.text(0.02, 0.98, f'μ={scaled_data.mean():.2f}\nσ={scaled_data.std():.2f}', 
                  transform=ax_scaled.transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)

# ============================================================================
# 8. ELO VS RANK (Bottom Left)
# ============================================================================
ax8 = plt.subplot(3, 4, 9)
if 'elo_diff' in X_train.columns and 'rank_diff' in X_train.columns:
    scatter = ax8.scatter(X_train['elo_diff'], X_train['rank_diff'], 
                         c=y_train_array, cmap='RdYlGn', alpha=0.3, s=3)
    ax8.set_xlabel('Elo Difference', fontsize=9)
    ax8.set_ylabel('Rank Difference', fontsize=9)
    ax8.set_title('Elo vs Rank (colored by outcome)', fontweight='bold', fontsize=10)
    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
    ax8.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
    plt.colorbar(scatter, ax=ax8, label='Won')
    ax8.tick_params(labelsize=8)

# ============================================================================
# 9. PLAYER VS OPPONENT ELO (Bottom Center-Left)
# ============================================================================
ax9 = plt.subplot(3, 4, 10)
if 'player_elo' in X_train.columns and 'opponent_elo' in X_train.columns:
    scatter = ax9.scatter(X_train['player_elo'], X_train['opponent_elo'], 
                         c=y_train_array, cmap='RdYlGn', alpha=0.3, s=3)
    ax9.set_xlabel('Player Elo', fontsize=9)
    ax9.set_ylabel('Opponent Elo', fontsize=9)
    ax9.set_title('Player vs Opponent Elo', fontweight='bold', fontsize=10)
    ax9.plot([1400, 2200], [1400, 2200], 'k--', alpha=0.3, linewidth=0.8)
    plt.colorbar(scatter, ax=ax9, label='Won')
    ax9.tick_params(labelsize=8)

# ============================================================================
# 10. WIN PROBABILITY (Bottom Center-Right)
# ============================================================================
ax10 = plt.subplot(3, 4, 11)
if 'player_win_prob_elo' in X_train.columns:
    winners_mask = y_train_array == 1
    losers_mask = y_train_array == 0
    
    winners_prob = X_train[winners_mask]['player_win_prob_elo']
    losers_prob = X_train[losers_mask]['player_win_prob_elo']
    
    ax10.hist(winners_prob, bins=40, alpha=0.6, label='Winners', 
             color='#2ecc71', density=True)
    ax10.hist(losers_prob, bins=40, alpha=0.6, label='Losers', 
             color='#e74c3c', density=True)
    ax10.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax10.set_xlabel('Win Probability', fontsize=9)
    ax10.set_ylabel('Density', fontsize=9)
    ax10.set_title('Elo Win Probability by Outcome', fontweight='bold', fontsize=10)
    ax10.legend(fontsize=9)
    ax10.tick_params(labelsize=8)

# ============================================================================
# 11. KEY INSIGHT (Bottom Right)
# ============================================================================
ax11 = plt.subplot(3, 4, 12)
insight_text = """
KEY TAKEAWAYS:

1. Perfect 50/50 balance
   No class imbalance

2. Scaling normalizes
   All features to μ=0, σ=1

3. Clear patterns visible
   Elo/rank differences
   predict outcomes

4. Ready for modeling!
"""
ax11.text(0.1, 0.5, insight_text, fontsize=10, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2))
ax11.axis('off')
ax11.set_title('Summary', fontweight='bold')

plt.tight_layout()
plt.savefig('preprocessing_overview.png', dpi=300, bbox_inches='tight')
print("\nSaved: preprocessing_overview.png")
plt.close()

print("\nVisualization complete!")
print("The middle row now shows original (left) and scaled (right) for each feature")
