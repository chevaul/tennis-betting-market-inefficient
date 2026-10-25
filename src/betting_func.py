import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_normalized_match_id_winner_loser(row):
    """Create normalized match_id from Winner/Loser format"""
    date = str(row['Date'])
    w = str(row['Winner']).strip().replace(' ', '').replace('.', '')
    l = str(row['Loser']).strip().replace(' ', '').replace('.', '')
    # Sort alphabetically - consistent ID
    names_sorted = '_'.join(sorted([w, l]))
    return f"{date}_{names_sorted}"


def create_normalized_match_id_player(row):
    """For player_name/opponent_name format"""
    dt = pd.to_datetime(row['Date'], errors='coerce')
    date = dt.strftime('%Y-%m-%d') if pd.notna(dt) else str(row['Date'])[:10]
    p1 = str(row['player_name']).strip().replace(' ', '').replace('.', '')
    p2 = str(row['opponent_name']).strip().replace(' ', '').replace('.', '')
    names_sorted = '_'.join(sorted([p1, p2]))
    return f"{date}_{names_sorted}"
def create_normalized_match_id_winner(row):
    """For Winner/Loser format"""
    date = str(row['Date'])
    w = str(row['Winner']).strip().replace(' ', '').replace('.', '')
    l = str(row['Loser']).strip().replace(' ', '').replace('.', '')
    names_sorted = '_'.join(sorted([w, l]))
    return f"{date}_{names_sorted}"


def simulate_fixed_stake_betting(comparison, threshold=0.05, stake=100, use_best_odds=False):
    """
    Fixed stake betting with comprehensive statistics
    """
    # USE UNIQUE MATCHES ONLY (one bet per match)
    comparison_unique = comparison.drop_duplicates(subset='match_id', keep='first')
    
    # Select matches where edge exceeds threshold
    bets = comparison_unique[comparison_unique['edge'] > threshold].copy()
    
    if len(bets) == 0:
        return {
            'threshold': threshold,
            'num_bets': 0,
            'total_profit': 0,
            'total_staked': 0,
            'roi': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    # Calculate profits
    odds_col = 'best_decimal_odds' if use_best_odds else 'decimal_odds'
    bets['profit'] = np.where(
        bets['actual_outcome'] == 1,
        stake * (bets[odds_col] - 1),
        -stake
    )
    
    # Calculate cumulative metrics
    bets['cumulative_profit'] = bets['profit'].cumsum()
    bets['cumulative_staked'] = stake * (np.arange(len(bets)) + 1)
    
    # Calculate statistics
    total_profit = bets['profit'].sum()
    total_staked = stake * len(bets)
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
    win_rate = (bets['actual_outcome'] == 1).mean()
    
    # Sharpe ratio (annualized, assuming daily betting)
    returns = bets['profit'] / stake
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Maximum drawdown
    cummax = bets['cumulative_profit'].cummax()
    drawdown = cummax - bets['cumulative_profit']
    max_drawdown = drawdown.max()
    
    return {
        'threshold': threshold,
        'num_bets': len(bets),
        'total_profit': total_profit,
        'total_staked': total_staked,
        'roi': roi,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_odds': bets[odds_col].mean() if len(bets) > 0 else 0,
        'bets_df': bets
    }


# =====================================
# SIMULATION 2: KELLY CRITERION
# =====================================

def calculate_kelly_fraction(model_prob, market_prob):
    """
    Calculate Kelly fraction with comprehensive safety checks
    """
    # Input validation
    if pd.isna(model_prob) or pd.isna(market_prob):
        return 0
    
    # Probability range validation
    if market_prob <= 0.01 or market_prob >= 0.99:
        return 0
    if model_prob <= 0 or model_prob >= 1:
        return 0
    
    # No bet if no positive edge
    if model_prob <= market_prob:
        return 0
    
    # Calculate Kelly with caps
    decimal_odds = min(1 / market_prob, 100)  # Cap at 100:1
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Cap Kelly fraction at 25% (very aggressive already)
    kelly = max(0, min(kelly, 0.25))
    
    return kelly

def simulate_kelly_betting(comparison, bankroll=10000, kelly_multiplier=0.25, min_kelly=0.01):
    """
    Kelly Criterion betting with proper bankroll management
    """
    # Calculate Kelly fractions
    comparison = comparison.copy()
    comparison['kelly_fraction'] = comparison.apply(
        lambda row: calculate_kelly_fraction(row['model_prob'], row['market_prob']),
        axis=1
    )
    
    # Filter for minimum Kelly
    kelly_bets = comparison[comparison['kelly_fraction'] >= min_kelly].copy()
    
    if len(kelly_bets) == 0:
        return {
            'num_bets': 0,
            'starting_bankroll': bankroll,
            'final_bankroll': bankroll,
            'total_profit': 0,
            'roi': 0,
            'kelly_multiplier': kelly_multiplier,
            'max_drawdown': 0,
            'max_drawdown_pct': 0
        }
    
    # Simulate sequential betting
    current_bankroll = bankroll
    bankroll_history = [bankroll]
    bet_history = []
    
    for idx, row in kelly_bets.iterrows():
        # Stop if bankrupt
        if current_bankroll <= 0:
            break
        
        # Calculate bet size (with additional safety caps)
        kelly_frac = row['kelly_fraction'] * kelly_multiplier
        kelly_frac = min(kelly_frac, 0.1)  # Max 10% after multiplier
        
        bet_amount = current_bankroll * kelly_frac
        bet_amount = min(bet_amount, current_bankroll * 0.1)  # Double-check
        
        # Calculate outcome
        decimal_odds = min(row['decimal_odds'], 100)
        
        if row['actual_outcome'] == 1:
            profit = bet_amount * (decimal_odds - 1)
        else:
            profit = -bet_amount
        
        # Update bankroll
        current_bankroll += profit
        bankroll_history.append(current_bankroll)
        
        bet_history.append({
            'bet_num': len(bet_history) + 1,
            'bet_amount': bet_amount,
            'kelly_fraction': row['kelly_fraction'],
            'profit': profit,
            'bankroll': current_bankroll
        })
    
    # Calculate metrics
    total_profit = current_bankroll - bankroll
    roi = (total_profit / bankroll) * 100
    
    # Maximum drawdown
    bankroll_array = np.array(bankroll_history)
    cummax = np.maximum.accumulate(bankroll_array)
    drawdown = cummax - bankroll_array
    max_drawdown = drawdown.max()
    max_drawdown_pct = (max_drawdown / bankroll) * 100
    
    return {
        'num_bets': len(bet_history),
        'starting_bankroll': bankroll,
        'final_bankroll': current_bankroll,
        'total_profit': total_profit,
        'roi': roi,
        'kelly_multiplier': kelly_multiplier,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'bankroll_history': bankroll_history,
        'bet_history': pd.DataFrame(bet_history) if bet_history else pd.DataFrame()
    }

# =====================================
# MAIN ANALYSIS FUNCTION
# =====================================

def run_complete_betting_analysis(df, tournament=None):
    """
    Run all three betting simulations with visualizations
    """
    if tournament:
        df = df[df['Tournament'] == tournament].copy()

    comparison = df.copy()

    # Initialize variables for later use
    best_roi = None
    best_sharpe = None
    
    # ========================================
    # SIMULATION 1: FIXED STAKE ANALYSIS
    # ========================================
    print("\n" + "="*80)
    print("SIMULATION 1: FIXED STAKE BETTING - MULTIPLE THRESHOLDS")
    print("="*80)
    
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    threshold_results = []
    
    print(f"\n{'Threshold':>10s} {'Bets':>8s} {'Profit':>12s} {'ROI':>8s} {'Win Rate':>10s} {'Sharpe':>8s} {'Max DD':>10s}")
    print("-" * 85)
    
    for thresh in thresholds:
        result = simulate_fixed_stake_betting(comparison, threshold=thresh, stake=100)
        
        if result['num_bets'] > 0:
            threshold_results.append(result)
            print(f"{thresh:>10.2f} {result['num_bets']:>8d} "
                  f"${result['total_profit']:>11.2f} {result['roi']:>7.2f}% "
                  f"{result['win_rate']:>9.1%} {result['sharpe_ratio']:>7.2f} "
                  f"${result['max_drawdown']:>9.2f}")
    
    # Find optimal thresholds
    threshold_df = None
    if threshold_results:
        threshold_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != 'bets_df'} 
            for r in threshold_results
        ])
        
        best_roi_idx = threshold_df['roi'].idxmax()
        best_roi = threshold_df.iloc[best_roi_idx]
        
        # Filter for minimum number of bets (at least 20)
        viable_thresholds = threshold_df[threshold_df['num_bets'] >= 20]
        
        if not viable_thresholds.empty:
            best_sharpe_idx = viable_thresholds['sharpe_ratio'].idxmax()
            best_sharpe = viable_thresholds.iloc[best_sharpe_idx]
        else:
            best_sharpe = best_roi
        
        print(f"\n✓ Best ROI: threshold={best_roi['threshold']:.2f}, "
              f"ROI={best_roi['roi']:.2f}%, Bets={int(best_roi['num_bets'])}")
        print(f"✓ Best Sharpe: threshold={best_sharpe['threshold']:.2f}, "
              f"Sharpe={best_sharpe['sharpe_ratio']:.2f}, Bets={int(best_sharpe['num_bets'])}")
    
    # ========================================
    # SIMULATION 2: KELLY CRITERION
    # ========================================
    print("\n" + "="*80)
    print("SIMULATION 2: KELLY CRITERION ANALYSIS")
    print("="*80)
    
    # Calculate Kelly statistics
    comparison['kelly_fraction'] = comparison.apply(
        lambda row: calculate_kelly_fraction(row['model_prob'], row['market_prob']),
        axis=1
    )
    
    kelly_positive = comparison[comparison['kelly_fraction'] > 0]
    
    print(f"\nKelly Fraction Statistics:")
    print(f"  Matches with positive Kelly: {len(kelly_positive)} ({len(kelly_positive)/len(comparison)*100:.1f}%)")
    
    # Calculate median for later use
    kelly_median = kelly_positive['kelly_fraction'].median() if len(kelly_positive) > 0 else 0
    
    if len(kelly_positive) > 0:
        print(f"  Mean Kelly fraction: {kelly_positive['kelly_fraction'].mean():.4f}")
        print(f"  Median Kelly fraction: {kelly_median:.4f}")
        print(f"  95th percentile: {kelly_positive['kelly_fraction'].quantile(0.95):.4f}")
        
        print(f"\nTop 5 Kelly Recommendations:")
        top_kelly = kelly_positive.nlargest(5, 'kelly_fraction')[
            ['model_prob', 'market_prob', 'edge', 'kelly_fraction', 'actual_outcome']
        ].round(3)
        print(top_kelly.to_string())
    
    # Test different Kelly multipliers
    print("\n" + "-"*80)
    print("KELLY BETTING SIMULATION (Starting bankroll: $10,000)")
    print("-"*80)
    
    kelly_multipliers = [0.1, 0.25, 0.5, 0.75, 1.0]
    kelly_results = []
    
    print(f"\n{'Kelly Mult':>12s} {'Bets':>8s} {'Final Bank':>12s} {'Profit':>12s} {'ROI':>8s} {'Max DD %':>10s}")
    print("-" * 75)
    
    for mult in kelly_multipliers:
        result = simulate_kelly_betting(comparison, bankroll=10000, kelly_multiplier=mult)
        
        if result['num_bets'] > 0:
            kelly_results.append(result)
            
            # Check for unrealistic results
            if abs(result['roi']) > 10000:
                print(f"{mult:>12.2f} {result['num_bets']:>8d} "
                      f"ERROR: Unrealistic ROI detected")
            else:
                print(f"{mult:>12.2f} {result['num_bets']:>8d} "
                      f"${result['final_bankroll']:>11,.2f} "
                      f"${result['total_profit']:>11,.2f} "
                      f"{result['roi']:>7.2f}% "
                      f"{result['max_drawdown_pct']:>9.2f}%")
    
    # ========================================
    # SIMULATION 3: OPTIMAL THRESHOLD FINDING
    # ========================================
    print("\n" + "="*80)
    print("SIMULATION 3: OPTIMAL THRESHOLD ANALYSIS")
    print("="*80)
    
    # Fine-grained threshold search
    fine_thresholds = np.linspace(0, 0.2, 41)  # 0 to 20% in 0.5% steps
    fine_results = []
    
    for thresh in fine_thresholds:
        result = simulate_fixed_stake_betting(comparison, threshold=thresh, stake=100)
        if result['num_bets'] >= 10:  # Minimum 10 bets for statistical significance
            fine_results.append({
                'threshold': thresh,
                'roi': result['roi'],
                'sharpe': result['sharpe_ratio'],
                'num_bets': result['num_bets']
            })
    
    if fine_results:
        fine_df = pd.DataFrame(fine_results)
        
        # Find optimal by different criteria
        optimal_roi = fine_df.loc[fine_df['roi'].idxmax()]
        optimal_sharpe = fine_df.loc[fine_df['sharpe'].idxmax()]
        
        # Find balanced optimal (good ROI with reasonable number of bets)
        fine_df['score'] = fine_df['roi'] * np.log(fine_df['num_bets'] + 1)
        optimal_balanced = fine_df.loc[fine_df['score'].idxmax()]
        
        print(f"\nOptimal Thresholds:")
        print(f"  Maximum ROI: threshold={optimal_roi['threshold']:.3f}, "
              f"ROI={optimal_roi['roi']:.2f}%, Bets={int(optimal_roi['num_bets'])}")
        print(f"  Best Risk-Adjusted: threshold={optimal_sharpe['threshold']:.3f}, "
              f"Sharpe={optimal_sharpe['sharpe']:.2f}, Bets={int(optimal_sharpe['num_bets'])}")
        print(f"  Balanced (ROI×Volume): threshold={optimal_balanced['threshold']:.3f}, "
              f"ROI={optimal_balanced['roi']:.2f}%, Bets={int(optimal_balanced['num_bets'])}")
    
    # ========================================
    # VISUALIZATION
    # ========================================
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Edge Distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(comparison['edge'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', label='No Edge', linewidth=2)
    if best_sharpe is not None:
        ax1.axvline(x=best_sharpe['threshold'], color='green', linestyle='-', 
                   label=f"Optimal Threshold ({best_sharpe['threshold']:.2f})", linewidth=2)
    ax1.set_xlabel('Edge (Model - Market Probability)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Betting Edge')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model vs Market
    ax2 = plt.subplot(3, 3, 2)
    scatter = ax2.scatter(comparison['market_prob'], comparison['model_prob'], 
                         c=comparison['actual_outcome'], cmap='RdYlGn', alpha=0.5)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No Edge')
    ax2.set_xlabel('Market Implied Probability')
    ax2.set_ylabel('Model Probability')
    ax2.set_title('Model vs Market Probabilities\n(Green=Won, Red=Lost)')
    plt.colorbar(scatter, ax=ax2, label='Actual Outcome')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ROI by Threshold
    if threshold_df is not None:
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(threshold_df['threshold'], threshold_df['roi'], 'b-', linewidth=2, label='ROI')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(threshold_df['threshold'], threshold_df['num_bets'], 'r--', linewidth=2, label='Num Bets')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('ROI (%)', color='b')
        ax3_twin.set_ylabel('Number of Bets', color='r')
        ax3.set_title('ROI and Bet Volume by Threshold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
    
    # Plot 4: Sharpe Ratio by Threshold
    if threshold_df is not None:
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(threshold_df['threshold'], threshold_df['sharpe_ratio'], 'g-', linewidth=2)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_title('Risk-Adjusted Returns by Threshold')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good (>1)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Plot 5: Kelly Distribution
    ax5 = plt.subplot(3, 3, 5)
    if len(kelly_positive) > 0:
        ax5.hist(kelly_positive['kelly_fraction'], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax5.axvline(kelly_median, color='red', linestyle='--', 
                   label=f'Median: {kelly_median:.3f}', linewidth=2)
        ax5.set_xlabel('Kelly Fraction')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Kelly Fraction Distribution (>0)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Bankroll Evolution (Kelly)
    ax6 = plt.subplot(3, 3, 6)
    for result in kelly_results[:3]:  # Show first 3 multipliers
        if result['num_bets'] > 0 and abs(result['roi']) < 10000:
            ax6.plot(result['bankroll_history'], 
                    label=f"Kelly {result['kelly_multiplier']:.2f} (ROI: {result['roi']:.1f}%)", 
                    linewidth=2, alpha=0.7)
    ax6.axhline(y=10000, color='black', linestyle='--', alpha=0.5, label='Initial Bankroll')
    ax6.set_xlabel('Bet Number')
    ax6.set_ylabel('Bankroll ($)')
    ax6.set_title('Bankroll Evolution - Kelly Betting')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Win Rate by Edge
    ax7 = plt.subplot(3, 3, 7)
    edge_bins = pd.cut(comparison['edge'], bins=10)
    win_rate_by_edge = comparison.groupby(edge_bins)['actual_outcome'].agg(['mean', 'count'])
    win_rate_by_edge = win_rate_by_edge[win_rate_by_edge['count'] >= 5]  # Min 5 samples
    
    if not win_rate_by_edge.empty:
        edge_centers = [interval.mid for interval in win_rate_by_edge.index]
        ax7.bar(range(len(edge_centers)), win_rate_by_edge['mean'], alpha=0.7)
        ax7.set_xticks(range(len(edge_centers)))
        ax7.set_xticklabels([f'{x:.2f}' for x in edge_centers], rotation=45)
        ax7.set_xlabel('Edge Range (Midpoint)')
        ax7.set_ylabel('Win Rate')
        ax7.set_title('Actual Win Rate by Edge Level')
        ax7.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
    
    # Plot 8: Calibration Plot
    ax8 = plt.subplot(3, 3, 8)
    prob_bins = pd.cut(comparison['model_prob'], bins=10)
    calibration = comparison.groupby(prob_bins)['actual_outcome'].agg(['mean', 'count'])
    calibration = calibration[calibration['count'] >= 5]
    
    if not calibration.empty:
        prob_centers = [interval.mid for interval in calibration.index]
        ax8.scatter(prob_centers, calibration['mean'], s=calibration['count']*10, alpha=0.6)
        ax8.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Calibration')
        ax8.set_xlabel('Model Probability (Binned)')
        ax8.set_ylabel('Actual Win Rate')
        ax8.set_title('Model Calibration\n(Size = Sample Count)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # Plot 9: Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Prepare summary values with safe formatting
    best_sharpe_threshold = best_sharpe['threshold'] if best_sharpe is not None else 0
    best_sharpe_roi = best_sharpe['roi'] if best_sharpe is not None else 0
    best_sharpe_ratio = best_sharpe['sharpe_ratio'] if best_sharpe is not None else 0
    best_sharpe_bets = int(best_sharpe['num_bets']) if best_sharpe is not None else 0
    
    summary_text = f"""
    BETTING STRATEGY SUMMARY
    {'='*40}
    
    Dataset:
    • Total matches: {len(comparison)}
    • Positive edge rate: {(comparison['edge'] > 0).mean():.1%}
    • Average edge: {comparison['edge'].mean():.3f}
    
    Best Fixed Stake (Sharpe):
    • Threshold: {best_sharpe_threshold:.2f}
    • ROI: {best_sharpe_roi:.1f}%
    • Sharpe: {best_sharpe_ratio:.2f}
    • Bets: {best_sharpe_bets}
    
    Kelly Criterion:
    • Positive Kelly matches: {len(kelly_positive)}
    • Median Kelly fraction: {kelly_median:.3f}
    • Recommended: 25% Kelly multiplier
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('Comprehensive Betting Strategy Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    return {
        'comparison': comparison,
        'threshold_results': threshold_df,
        'kelly_results': kelly_results,
        'best_sharpe': best_sharpe,
        'best_roi': best_roi
    }