"""
Conference Visualization Generator - LSE Data Science & Quant Conference
Generates all publication-quality figures for presentation

Usage:
    python visualizations_conference.py              # Generate all
    python visualizations_conference.py calibration  # Generate specific one
    
Author: Ecem Yildirim
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss, roc_curve, auc, roc_auc_score
import sys
from pathlib import Path

# Set global style
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

# ============================================================================
# LOAD DATA (ONCE)
# ============================================================================

print("="*80)
print("CONFERENCE VISUALIZATION GENERATOR")
print("="*80)

print("\nLoading data...")
data_path = Path('./data/model_results/betting_model_comparison_unique.csv')
df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df):,} matches")

# Calculate time span (used by multiple visualizations)
dates = pd.to_datetime(df['Date'])
years_span = (dates.max() - dates.min()).days / 365.25
print(f"✓ Time span: {years_span:.1f} years ({dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')})")

out_dir = Path('./figs')

# ============================================================================
# VIZ 1: CALIBRATION COMPARISON
# ============================================================================

def viz_calibration():
    """Model vs Market Calibration - Proves probabilities are accurate"""
    print("\n" + "="*80)
    print("GENERATING: Calibration Comparison")
    print("="*80)
    
    # Calculate calibration curves
    prob_true_model, prob_pred_model = calibration_curve(
        df['y_true'], df['model_prob'], n_bins=10
    )
    prob_true_market, prob_pred_market = calibration_curve(
        df['y_true'], df['market_prob'], n_bins=10
    )
    
    # Calculate log loss
    model_logloss = log_loss(df['y_true'], df['model_prob'])
    market_logloss = log_loss(df['y_true'], df['market_prob'])
    improvement = ((market_logloss - model_logloss) / market_logloss) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot
    ax.plot([0, 1], [0, 1], 'k--', linewidth=3, label='Perfect Calibration', zorder=1)
    ax.plot(prob_pred_model, prob_true_model, 'o-', linewidth=3, markersize=12, 
            label=f'Our Model (Log Loss: {model_logloss:.4f})', color='#2E86AB', zorder=3)
    ax.plot(prob_pred_market, prob_true_market, 's-', linewidth=3, markersize=12, 
            label=f'Market (Log Loss: {market_logloss:.4f})', color='#A23B72', zorder=2)
    
    # Styling
    ax.set_xlabel('Predicted Probability', fontsize=16, fontweight='bold')
    ax.set_ylabel('Actual Win Rate', fontsize=16, fontweight='bold')
    ax.set_title('Calibration Comparison: Model vs Market\n',
                 fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Annotation
    textstr = f'Model improvement: {improvement:.1f}%\nSample size: {len(df):,} matches'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'calibration_comparison_conference.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: calibration_comparison_conference.png")
    plt.close()

# ============================================================================
# VIZ 2: ROI BY THRESHOLD ANALYSIS
# ============================================================================

def viz_threshold_analysis():
    """ROI vs Edge Threshold - Shows systematic optimization"""
    print("\n" + "="*80)
    print("GENERATING: ROI by Threshold Analysis")
    print("="*80)
    
    def calculate_metrics(edge_threshold, stake=100):
        value_bets = df[df['edge'] > edge_threshold].copy()
        if len(value_bets) == 0:
            return None
        
        value_bets['profit'] = np.where(
            value_bets['y_true'] == 1,
            (value_bets['AvgW'] - 1) * stake,
            -stake
        )
        
        total_profit = value_bets['profit'].sum()
        total_staked = len(value_bets) * stake
        roi = (total_profit / total_staked) * 100
        annual_roi = ((1 + roi/100) ** (1/years_span) - 1) * 100
        win_rate = value_bets['y_true'].mean()
        
        returns = value_bets['profit'].values / stake
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        return {
            'threshold': edge_threshold,
            'num_bets': len(value_bets),
            'roi': roi,
            'annual_roi': annual_roi,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe
        }
    
    # Test thresholds
    thresholds = np.arange(0.01, 0.25, 0.01)
    results = [calculate_metrics(t) for t in thresholds]
    results = [r for r in results if r and r['num_bets'] >= 50]
    results_df = pd.DataFrame(results)
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Betting Strategy Optimization: ROI vs Edge Threshold Analysis', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Total ROI
    ax1 = axes[0, 0]
    ax1.plot(results_df['threshold'], results_df['roi'], 'o-', linewidth=3, 
             markersize=8, color='#2E86AB', markerfacecolor='white', markeredgewidth=2)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax1.set_xlabel('Edge Threshold', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total ROI (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Total ROI (10-Year) vs Edge Threshold', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    
    # 2. Annual ROI
    ax2 = axes[0, 1]
    ax2.plot(results_df['threshold'], results_df['annual_roi'], 'o-', linewidth=3, 
             markersize=8, color='#F18F01', markerfacecolor='white', markeredgewidth=2)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Edge Threshold', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Annual ROI (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Annualized ROI (CAGR) vs Edge Threshold', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    
    # 3. Sharpe Ratio
    ax3 = axes[1, 0]
    ax3.plot(results_df['threshold'], results_df['sharpe_ratio'], 'o-', linewidth=3, 
             markersize=8, color='#6A994E', markerfacecolor='white', markeredgewidth=2)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax3.axhline(2, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Good (2.0)')
    ax3.axhline(3, color='blue', linestyle=':', linewidth=2, alpha=0.5, label='Excellent (3.0)')
    ax3.set_xlabel('Edge Threshold', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio', fontsize=14, fontweight='bold')
    ax3.set_title('Risk-Adjusted Returns vs Edge Threshold', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=11, loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=12)
    
    # 4. Number of Bets
    ax4 = axes[1, 1]
    ax4.plot(results_df['threshold'], results_df['num_bets'], 'o-', linewidth=3, 
             markersize=8, color='#BC4B51', markerfacecolor='white', markeredgewidth=2)
    ax4.set_xlabel('Edge Threshold', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Bets', fontsize=14, fontweight='bold')
    ax4.set_title('Bet Frequency vs Edge Threshold', fontsize=15, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'roi_by_threshold_conference.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: roi_by_threshold_conference.png")
    plt.close()

# ============================================================================
# VIZ 3: CUMULATIVE RETURNS OVER TIME
# ============================================================================

def viz_cumulative_returns():
    """Temporal consistency - proves not just luck"""
    print("\n" + "="*80)
    print("GENERATING: Cumulative Returns Over Time")
    print("="*80)
    
    # Conservative strategy
    conservative = df[df['edge'] > 0.05].copy().sort_values('Date')
    conservative['profit'] = np.where(
        conservative['y_true'] == 1,
        (conservative['AvgW'] - 1) * 100,
        -100
    )
    conservative['cumulative_profit'] = conservative['profit'].cumsum()
    conservative['bet_number'] = range(1, len(conservative) + 1)
    conservative['running_max'] = conservative['cumulative_profit'].cummax()
    conservative['drawdown'] = conservative['running_max'] - conservative['cumulative_profit']
    conservative['year'] = pd.to_datetime(conservative['Date']).dt.year
    
    # Aggressive strategy
    aggressive = df[df['edge'] > 0.20].copy().sort_values('Date')
    aggressive['profit'] = np.where(
        aggressive['y_true'] == 1,
        (aggressive['AvgW'] - 1) * 100,
        -100
    )
    aggressive['cumulative_profit'] = aggressive['profit'].cumsum()
    aggressive['bet_number'] = range(1, len(aggressive) + 1)
    
    # Baseline
    df_unique = df.drop_duplicates(subset='match_id', keep='first').sort_values('Date')
    baseline_profits = []
    for _, row in df_unique.iterrows():
        winner_prob = 1 / row['AvgW']
        loser_prob = 1 / row['AvgL']
        profit = 100 * (row['AvgW'] - 1) if winner_prob > loser_prob else -100
        baseline_profits.append(profit)
    
    df_unique['baseline_cumulative'] = np.cumsum(baseline_profits)
    df_unique['bet_number'] = range(1, len(df_unique) + 1)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Cumulative Returns Analysis: Temporal Consistency Over 10 Years', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 1. All strategies
    ax1 = axes[0, 0]
    ax1.plot(conservative['bet_number'], conservative['cumulative_profit'], 
             linewidth=2.5, label='Conservative (Edge > 0.05)', color='#2E86AB')
    ax1.plot(aggressive['bet_number'], aggressive['cumulative_profit'], 
             linewidth=2.5, label='Aggressive (Edge > 0.20)', color='#6A994E')
    ax1.plot(df_unique['bet_number'], df_unique['baseline_cumulative'], 
             linewidth=2.5, label='Baseline (Favorites)', color='#BC4B51', alpha=0.7)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_xlabel('Bet Number', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Profit ($)', fontsize=13, fontweight='bold')
    ax1.set_title('Cumulative Profit: All Strategies', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax1.tick_params(labelsize=11)
    
    # 2. Conservative detail
    ax2 = axes[0, 1]
    ax2.plot(conservative['bet_number'], conservative['cumulative_profit'], 
             linewidth=3, color='#2E86AB')
    ax2.fill_between(conservative['bet_number'], 0, conservative['cumulative_profit'], 
                     alpha=0.3, color='#2E86AB')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Bet Number', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Profit ($)', fontsize=13, fontweight='bold')
    ax2.set_title(f'Conservative Strategy (Edge > 0.05)\n{len(conservative):,} bets, '
                  f'{conservative["y_true"].mean():.1%} win rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax2.tick_params(labelsize=11)
    
    final_profit = conservative['cumulative_profit'].iloc[-1]
    ax2.text(0.98, 0.98, f'Final: ${final_profit:,.0f}\nROI: 33.5%', 
             transform=ax2.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 3. Drawdown analysis
    ax3 = axes[1, 0]
    ax3.fill_between(conservative['bet_number'], 0, -conservative['drawdown'], 
                     alpha=0.5, color='#BC4B51', label='Drawdown')
    ax3.plot(conservative['bet_number'], -conservative['drawdown'], 
             linewidth=2, color='#BC4B51')
    ax3.set_xlabel('Bet Number', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Drawdown ($)', fontsize=13, fontweight='bold')
    ax3.set_title(f'Maximum Drawdown Analysis (Conservative)\n'
                  f'Max drawdown: ${conservative["drawdown"].max():,.0f}', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)
    ax3.tick_params(labelsize=11)
    
    # 4. Year-by-year
    ax4 = axes[1, 1]
    yearly_returns = conservative.groupby('year')['profit'].sum()
    colors = ['#2E86AB' if profit > 0 else '#BC4B51' for profit in yearly_returns.values]
    bars = ax4.bar(yearly_returns.index, yearly_returns.values, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.axhline(0, color='black', linestyle='-', linewidth=2)
    ax4.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Annual Profit ($)', fontsize=13, fontweight='bold')
    ax4.set_title('Year-by-Year Performance (Conservative)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(labelsize=11)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height/1000:.0f}K',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')
    
    positive_years = (yearly_returns > 0).sum()
    total_years = len(yearly_returns)
    ax4.text(0.98, 0.98, f'Positive years: {positive_years}/{total_years} ({positive_years/total_years:.0%})', 
             transform=ax4.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_dir / 'cumulative_returns_conference.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: cumulative_returns_conference.png")
    plt.close()

# ============================================================================
# VIZ 3: EDGE DISTRIBUTION
# ============================================================================

def viz_edge_distribution():
    """Edge distribution & market disagreement analysis"""
    print("\n" + "="*80)
    print("GENERATING: Edge Distribution Analysis")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Model Edge Distribution & Market Disagreement Analysis', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Edge histogram
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(df['edge'], bins=60, alpha=0.7, edgecolor='black')
    
    for i, patch in enumerate(patches):
        if bins[i] < -0.05:
            patch.set_facecolor('#BC4B51')
        elif bins[i] > 0.05:
            patch.set_facecolor('#6A994E')
        else:
            patch.set_facecolor('#95B8D1')
    
    ax1.axvline(0, color='red', linestyle='--', linewidth=3, label='No Edge', alpha=0.7)
    ax1.axvline(0.05, color='green', linestyle='--', linewidth=3, label='5% Threshold', alpha=0.7)
    ax1.axvline(0.20, color='darkgreen', linestyle='--', linewidth=3, label='20% Threshold', alpha=0.7)
    ax1.set_xlabel('Edge (Model Prob - Market Prob)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax1.set_title('Distribution of Model Edge', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(labelsize=12)
    
    # Stats box
    textstr = f'Mean: {df["edge"].mean():.3f}\nMedian: {df["edge"].median():.3f}\nStd: {df["edge"].std():.3f}'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontweight='bold')
    
    # 2. Model vs Market scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['market_prob'], df['model_prob'], 
                         c=df['edge'], cmap='RdYlGn', alpha=0.4, s=15,
                         vmin=-0.3, vmax=0.3)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=3, label='Perfect Agreement', alpha=0.7)
    ax2.set_xlabel('Market Probability', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Model Probability', fontsize=14, fontweight='bold')
    ax2.set_title('Model vs Market Predictions\n(Color = Edge)', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Edge', fontsize=12, fontweight='bold')
    
    # 3. Edge by outcome
    ax3 = axes[1, 0]
    winners = df[df['y_true'] == 1]['edge']
    losers = df[df['y_true'] == 0]['edge']
    
    ax3.hist(winners, bins=50, alpha=0.6, label='Winners', color='#2ecc71', density=True)
    ax3.hist(losers, bins=50, alpha=0.6, label='Losers', color='#e74c3c', density=True)
    ax3.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax3.axvline(winners.mean(), color='#2ecc71', linestyle=':', linewidth=2)
    ax3.axvline(losers.mean(), color='#e74c3c', linestyle=':', linewidth=2)
    ax3.set_xlabel('Edge', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax3.set_title('Edge Distribution by Actual Outcome', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=12)
    
    # 4. Value bet frequency
    ax4 = axes[1, 1]
    thresholds = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    percentages = [(df['edge'] > t).sum() / len(df) * 100 for t in thresholds]
    bets = [(df['edge'] > t).sum() for t in thresholds]
    
    colors_bars = ['#6A994E' if pct > 10 else '#F18F01' if pct > 5 else '#BC4B51' 
                   for pct in percentages]
    bars = ax4.bar(range(len(thresholds)), percentages, color=colors_bars, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(range(len(thresholds)))
    ax4.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax4.set_xlabel('Edge Threshold', fontsize=14, fontweight='bold')
    ax4.set_ylabel('% of Matches', fontsize=14, fontweight='bold')
    ax4.set_title('Value Bet Frequency by Threshold', fontsize=15, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(labelsize=12)
    
    for i, (bar, pct, num) in enumerate(zip(bars, percentages, bets)):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{pct:.1f}%\n({num:,})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'cumulative_returns_conference.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: cumulative_returns_conference.png")
    plt.close()

# ============================================================================
# VIZ 4: EDGE DISTRIBUTION (STANDALONE)
# ============================================================================

def viz_edge_standalone():
    """Focused edge distribution visualization"""
    print("\n" + "="*80)
    print("GENERATING: Edge Distribution (Standalone)")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create histogram
    n, bins, patches = ax.hist(df['edge'], bins=60, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Color code
    for i, patch in enumerate(patches):
        if bins[i] < -0.05:
            patch.set_facecolor('#BC4B51')
        elif bins[i] > 0.05:
            patch.set_facecolor('#6A994E')
        else:
            patch.set_facecolor('#95B8D1')
    
    # Reference lines
    ax.axvline(0, color='red', linestyle='--', linewidth=3, label='No Edge', alpha=0.8, zorder=10)
    ax.axvline(0.05, color='green', linestyle='--', linewidth=3, 
              label='Conservative Threshold (5%)', alpha=0.8, zorder=10)
    ax.axvline(0.20, color='darkgreen', linestyle='--', linewidth=3, 
              label='Aggressive Threshold (20%)', alpha=0.8, zorder=10)
    
    ax.set_xlabel('Edge (Model Probability - Market Probability)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax.set_title('Distribution of Model Edge: Where We Find Value vs Markets', 
                 fontsize=18, fontweight='bold', pad=15)
    ax.legend(fontsize=13, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=13)
    
    # Statistics box
    stats_text = f"""Edge Statistics:
Mean: {df['edge'].mean():.4f} (slight positive bias)
Median: {df['edge'].median():.4f}
Std Dev: {df['edge'].std():.4f}

Value Opportunities:
Edge > 5%: {(df['edge'] > 0.05).sum():,} matches ({(df['edge'] > 0.05).mean()*100:.1f}%)
Edge > 20%: {(df['edge'] > 0.20).sum():,} matches ({(df['edge'] > 0.20).mean()*100:.1f}%)"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold', family='monospace')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'edge_distribution_conference.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: edge_distribution_conference.png")
    plt.close()

# ============================================================================
# VIZ 5: PERFORMANCE DASHBOARD
# ============================================================================

def viz_performance_dashboard():
    """Comprehensive model performance metrics"""
    print("\n" + "="*80)
    print("GENERATING: Model Performance Dashboard")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Model Performance Dashboard: Comprehensive Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. ROC Curves
    ax1 = plt.subplot(2, 3, 1)
    fpr_model, tpr_model, _ = roc_curve(df['y_true'], df['model_prob'])
    roc_auc_model = auc(fpr_model, tpr_model)
    fpr_market, tpr_market, _ = roc_curve(df['y_true'], df['market_prob'])
    roc_auc_market = auc(fpr_market, tpr_market)
    
    ax1.plot(fpr_model, tpr_model, linewidth=3, label=f'Model (AUC = {roc_auc_model:.3f})', color='#2E86AB')
    ax1.plot(fpr_market, tpr_market, linewidth=3, label=f'Market (AUC = {roc_auc_market:.3f})', color='#BC4B51')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random', alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax1.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # 2. Accuracy metrics
    ax2 = plt.subplot(2, 3, 2)
    metrics = ['Accuracy', 'AUC', 'Log Loss\n(lower better)']
    model_values = [
        (df['model_prob'] > 0.5).eq(df['y_true']).mean() * 100,
        roc_auc_model * 100,
        log_loss(df['y_true'], df['model_prob']) * 100
    ]
    market_values = [
        (df['market_prob'] > 0.5).eq(df['y_true']).mean() * 100,
        roc_auc_market * 100,
        log_loss(df['y_true'], df['market_prob']) * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax2.bar(x - width/2, model_values, width, label='Our Model', 
                    color='#2E86AB', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, market_values, width, label='Market', 
                    color='#BC4B51', edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax2.set_title('Predictive Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(labelsize=11)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Strategy comparison
    ax3 = plt.subplot(2, 3, 3)
    strategies = ['Baseline\n(Favorites)', 'Conservative\n(Edge > 0.05)', 'Aggressive\n(Edge > 0.20)']
    annual_rois = [-0.62, 2.98, 4.90]
    sharpe_ratios = [-2.80, 3.90, 6.92]
    
    x = np.arange(len(strategies))
    width = 0.35
    bars1 = ax3.bar(x - width/2, annual_rois, width, label='Annual ROI (%)', 
                    color=['#BC4B51', '#6A994E', '#2E86AB'], edgecolor='black', linewidth=1.5)
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, sharpe_ratios, width, label='Sharpe Ratio', 
                         color=['#E74C3C', '#2ECC71', '#3498DB'], edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax3.axhline(0, color='black', linestyle='-', linewidth=2)
    ax3.set_ylabel('Annual ROI (%)', fontsize=13, fontweight='bold')
    ax3_twin.set_ylabel('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax3.set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, fontsize=10, fontweight='bold')
    
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
    
    # 4. Edge by outcome (violin)
    ax4 = plt.subplot(2, 3, 4)
    data_to_plot = [
        df[df['y_true'] == 1]['edge'].values,
        df[df['y_true'] == 0]['edge'].values
    ]
    parts = ax4.violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True, widths=0.7)
    
    for pc, color in zip(parts['bodies'], ['#2ecc71', '#e74c3c']):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    ax4.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Actual Winners', 'Actual Losers'], fontsize=11, fontweight='bold')
    ax4.set_ylabel('Edge', fontsize=13, fontweight='bold')
    ax4.set_title('Edge Distribution by Match Outcome', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(labelsize=11)
    
    # 5. Win rate by edge bin
    ax5 = plt.subplot(2, 3, 5)
    edge_bins = pd.cut(df['edge'], bins=np.linspace(-0.5, 0.5, 21))
    edge_grouped = df.groupby(edge_bins, observed=True).agg({
        'y_true': ['mean', 'count']
    }).reset_index()
    edge_grouped.columns = ['edge_bin', 'win_rate', 'count']
    edge_grouped = edge_grouped[edge_grouped['count'] >= 20]
    
    bin_midpoints = [interval.mid for interval in edge_grouped['edge_bin']]
    colors_line = ['#BC4B51' if wr < 0.5 else '#6A994E' for wr in edge_grouped['win_rate']]
    
    ax5.scatter(bin_midpoints, edge_grouped['win_rate'] * 100, 
               s=edge_grouped['count']*2, c=colors_line, alpha=0.7, edgecolor='black', linewidth=1)
    ax5.plot(bin_midpoints, edge_grouped['win_rate'] * 100, 
            linewidth=2, color='#2E86AB', alpha=0.7)
    ax5.axhline(50, color='black', linestyle='--', linewidth=2, alpha=0.5, label='50% (Random)')
    ax5.axvline(0, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax5.set_xlabel('Edge', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Win Rate (%)', fontsize=13, fontweight='bold')
    ax5.set_title('Win Rate by Edge\n(Bubble size = # matches)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=11)
    
    # 6. Summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_data = [
        ['Metric', 'Model', 'Market', 'Improvement'],
        ['Log Loss', f'{log_loss(df["y_true"], df["model_prob"]):.4f}', 
         f'{log_loss(df["y_true"], df["market_prob"]):.4f}', '+9.9%'],
        ['Brier Score', f'{brier_score_loss(df["y_true"], df["model_prob"]):.4f}', 
         f'{brier_score_loss(df["y_true"], df["market_prob"]):.4f}', 
         f'+{((brier_score_loss(df["y_true"], df["market_prob"]) - brier_score_loss(df["y_true"], df["model_prob"])) / brier_score_loss(df["y_true"], df["market_prob"]) * 100):.1f}%'],
        ['AUC', f'{roc_auc_model:.4f}', f'{roc_auc_market:.4f}', 
         f'+{((roc_auc_model - roc_auc_market) / roc_auc_market * 100):.1f}%'],
        ['', '', '', ''],
        ['Strategy', 'Annual ROI', 'Sharpe', 'Edge'],
        ['Conservative', '2.98%', '3.90', '+3.60pts'],
        ['Aggressive', '4.90%', '6.92', '+5.52pts'],
    ]
    
    table = ax6.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style headers
    for j in range(4):
        table[(0, j)].set_facecolor('#2E86AB')
        table[(0, j)].set_text_props(weight='bold', color='white', fontsize=12)
        table[(5, j)].set_facecolor('#2E86AB')
        table[(5, j)].set_text_props(weight='bold', color='white', fontsize=12)
    
    ax6.set_title('Summary Statistics', fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'model_performance_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_performance_dashboard.png")
    plt.close()

# ============================================================================
# VIZ 6: TEMPORAL ROBUSTNESS
# ============================================================================

def viz_temporal_robustness():
    """Year-by-year performance consistency"""
    print("\n" + "="*80)
    print("GENERATING: Temporal Robustness Analysis")
    print("="*80)
    
    df['year'] = pd.to_datetime(df['Date']).dt.year
    
    # Calculate yearly stats
    yearly_stats = []
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        if len(year_data) > 50:
            stats = {
                'year': year,
                'matches': len(year_data),
                'accuracy': (year_data['model_prob'] > 0.5).eq(year_data['y_true']).mean() * 100,
                'auc': roc_auc_score(year_data['y_true'], year_data['model_prob']),
                'log_loss': log_loss(year_data['y_true'], year_data['model_prob']),
                'mean_edge': year_data['edge'].mean(),
                'positive_edge_pct': (year_data['edge'] > 0.05).mean() * 100
            }
            yearly_stats.append(stats)
    
    yearly_df = pd.DataFrame(yearly_stats)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Temporal Robustness: Model Performance Over Time (2015-2024)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Accuracy over time
    ax1 = axes[0, 0]
    ax1.plot(yearly_df['year'], yearly_df['accuracy'], 'o-', linewidth=3, 
             markersize=10, color='#2E86AB', markerfacecolor='white', markeredgewidth=2)
    ax1.axhline(yearly_df['accuracy'].mean(), color='green', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Mean: {yearly_df["accuracy"].mean():.1f}%')
    ax1.fill_between(yearly_df['year'], 
                    yearly_df['accuracy'].mean() - yearly_df['accuracy'].std(),
                    yearly_df['accuracy'].mean() + yearly_df['accuracy'].std(),
                    alpha=0.2, color='green', label='±1 Std Dev')
    ax1.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Model Accuracy by Year', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    ax1.set_ylim(65, 80)
    
    # 2. AUC over time
    ax2 = axes[0, 1]
    ax2.plot(yearly_df['year'], yearly_df['auc'], 'o-', linewidth=3, 
             markersize=10, color='#6A994E', markerfacecolor='white', markeredgewidth=2)
    ax2.axhline(yearly_df['auc'].mean(), color='blue', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Mean: {yearly_df["auc"].mean():.3f}')
    ax2.fill_between(yearly_df['year'], 
                    yearly_df['auc'].mean() - yearly_df['auc'].std(),
                    yearly_df['auc'].mean() + yearly_df['auc'].std(),
                    alpha=0.2, color='blue', label='±1 Std Dev')
    ax2.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax2.set_ylabel('AUC', fontsize=14, fontweight='bold')
    ax2.set_title('Model AUC by Year', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(0.75, 0.90)
    
    # 3. Mean edge over time
    ax3 = axes[1, 0]
    ax3.plot(yearly_df['year'], yearly_df['mean_edge']*100, 'o-', linewidth=3, 
             markersize=10, color='#F18F01', markerfacecolor='white', markeredgewidth=2)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No Edge')
    ax3.axhline(yearly_df['mean_edge'].mean()*100, color='green', linestyle=':', 
               linewidth=2, alpha=0.7, label=f'Mean: {yearly_df["mean_edge"].mean()*100:.2f}%')
    ax3.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mean Edge (%)', fontsize=14, fontweight='bold')
    ax3.set_title('Average Model Edge by Year', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=12)
    
    # 4. Sample size & value bets
    ax4 = axes[1, 1]
    bars = ax4.bar(yearly_df['year'], yearly_df['matches'], alpha=0.6, 
                  color='#95B8D1', edgecolor='black', linewidth=1.5, label='Total Matches')
    ax4.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Matches', fontsize=14, fontweight='bold')
    ax4.tick_params(labelsize=12)
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(yearly_df['year'], yearly_df['positive_edge_pct'], 'o-', 
                 linewidth=3, markersize=10, color='#6A994E', 
                 markerfacecolor='white', markeredgewidth=2, label='% Value Bets')
    ax4_twin.set_ylabel('Value Bets (%)', fontsize=14, fontweight='bold', color='#6A994E')
    ax4_twin.tick_params(axis='y', labelcolor='#6A994E', labelsize=12)
    ax4.set_title('Sample Size & Value Bet Frequency by Year', fontsize=15, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'temporal_robustness_conference.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: temporal_robustness_conference.png")
    plt.close()

# ============================================================================
# VIZ 7: FEATURE IMPORTANCE (CATBOOST)
# ============================================================================

def viz_feature_importance():
    """Beautiful CatBoost feature importance for conference"""
    print("\n" + "="*80)
    print("GENERATING: CatBoost Feature Importance")
    print("="*80)
    
    # Feature importance data (update with your actual CatBoost values)
    feature_importance = {
        'ATP Points Difference': 10.79,
        'Elo Ratings Difference': 8.99,
        'Elo-based Probability of Winning': 8.91,
        'Form-based Probability of Winning': 7.96,
        'ATP Ratings Difference': 6.28,
        'Player Elo Rating': 3.65,
        'Player Age': 3.42,
        'Opponent Elo Rating': 3.40,
        'Player Form': 3.36,
        'Opponent Age': 3.32,
    }
    
    df_importance = pd.DataFrame(list(feature_importance.items()), 
                                 columns=['Feature', 'Importance'])
    df_importance = df_importance.sort_values('Importance', ascending=True)
    df_plot = df_importance.tail(10)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color gradient (top features highlighted)
    colors = []
    for i in range(len(df_plot)):
        if i >= len(df_plot) - 3:
            colors.append('#2E86AB')  # Top 3: Dark blue
        elif i >= len(df_plot) - 5:
            colors.append('#7198A8')  # Top 4-5: Blue
        else:
            colors.append('#95B8D1')  # Rest: Light blue
    
    # Horizontal bars
    bars = ax.barh(df_plot['Feature'], df_plot['Importance'], 
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Value labels on bars
    for bar, value in zip(bars, df_plot['Importance']):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_yticklabels(df_plot['Feature'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Importance (%)', fontsize=15, fontweight='bold')
    ax.set_title('CatBoost Feature Importance: Top 10 Predictive Features\n' +
                 '(Prediction Values Change Method)',
                 fontsize=17, fontweight='bold', pad=20)
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='Top 3 Features', alpha=0.85),
        Patch(facecolor="#7198A8", edgecolor='#7198A8', label='Top 4-5 Features', alpha=0.85),
        Patch(facecolor='#95B8D1', edgecolor='#95B8D1', label='Other Important', alpha=0.85)
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'catboost_feature_importance_conference.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: catboost_feature_importance_conference.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Check if specific visualization requested
    if len(sys.argv) > 1:
        viz_name = sys.argv[1].lower()
        viz_map = {
            'calibration': viz_calibration,
            'threshold': viz_threshold_analysis,
            'cumulative': viz_cumulative_returns,
            'edge': viz_edge_standalone,
            'dashboard': viz_performance_dashboard,
            'temporal': viz_temporal_robustness,
            'features': viz_feature_importance
        }
        
        if viz_name in viz_map:
            viz_map[viz_name]()
        else:
            print(f"\n❌ Unknown visualization: {viz_name}")
            print(f"Available: {', '.join(viz_map.keys())}")
    
    else:
        # Generate all visualizations
        print("\nGenerating all visualizations...")
        print("="*80)
        
        viz_calibration()
        viz_threshold_analysis()
        viz_cumulative_returns()
        viz_edge_standalone()
        viz_performance_dashboard()
        viz_temporal_robustness()
        viz_feature_importance()
        
        print("\n" + "="*80)
        print("ALL VISUALIZATIONS COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("  ✓ calibration_comparison_conference.png")
        print("  ✓ roi_by_threshold_conference.png")
        print("  ✓ cumulative_returns_conference.png")
        print("  ✓ edge_distribution_conference.png")
        print("  ✓ model_performance_dashboard.png")
        print("  ✓ temporal_robustness_conference.png")
        print("  ✓ catboost_feature_importance_conference.png")
        
        
        print("="*80)