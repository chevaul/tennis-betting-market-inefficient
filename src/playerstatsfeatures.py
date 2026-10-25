import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

def analyze_handedness_probabilities(df):
    """
    Analyze win probabilities based on player handedness matchups.
    
    Parameters:
    df: DataFrame with columns 'Winner_hand' and 'Loser_hand'
        Values should be 'L' for left-handed or 'R' for right-handed
    
    Returns:
    Dictionary containing probabilities and statistical test results
    """
    
    # Create contingency table for all matchups
    # Rows: Winner's hand, Columns: Loser's hand
    contingency = pd.crosstab(df['Winner_hand'], df['Loser_hand'])
    print("Contingency Table (Winner Hand × Loser Hand):")
    print(contingency)
    print("\n")
    
    # Calculate conditional probabilities
    results = {}
    
    # P(Right-hander wins | opponent is left-handed)
    r_vs_l_wins = len(df[(df['Winner_hand'] == 'R') & (df['Loser_hand'] == 'L')])
    l_vs_r_wins = len(df[(df['Winner_hand'] == 'L') & (df['Loser_hand'] == 'R')])
    r_vs_l_total = r_vs_l_wins + l_vs_r_wins
    
    if r_vs_l_total > 0:
        results['P(R wins | opp is L)'] = r_vs_l_wins / r_vs_l_total
        results['P(L wins | opp is R)'] = l_vs_r_wins / r_vs_l_total
    
    # P(Right-hander wins | opponent is right-handed)
    r_vs_r_total = len(df[(df['Winner_hand'] == 'R') & (df['Loser_hand'] == 'R')]) * 2
    if r_vs_r_total > 0:
        results['P(R wins | opp is R)'] = 0.5  # By definition in mirror matchups
    
    # P(Left-hander wins | opponent is left-handed)
    l_vs_l_total = len(df[(df['Winner_hand'] == 'L') & (df['Loser_hand'] == 'L')]) * 2
    if l_vs_l_total > 0:
        results['P(L wins | opp is L)'] = 0.5  # By definition in mirror matchups
    
    # Print probability results
    print("Conditional Win Probabilities:")
    print("-" * 50)
    for key, value in results.items():
        print(f"{key}: {value:.4f} ({value*100:.2f}%)")
    print("\n")
    
    # Chi-square test for independence
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print("Statistical Significance Test:")
    print("-" * 50)
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    
    if p_value < 0.05:
        print("\nResult: Handedness matchup IS statistically significant (p < 0.05)")
    else:
        print("\nResult: Handedness matchup is NOT statistically significant (p >= 0.05)")
    
    print("\nExpected frequencies under independence:")
    print(pd.DataFrame(expected, 
                      index=contingency.index, 
                      columns=contingency.columns))
    
    return results, chi2, p_value

def analyze_handedness_by_surface(df):
    """
    Analyze win rates by handedness across different surface types.
    
    Parameters:
    df: DataFrame with columns 'Winner_hand', 'Loser_hand', and 'Surface'
    
    Returns:
    DataFrame with win rates by surface and handedness
    """
    
    results = []
    
    surfaces = df['Surface'].unique()
    
    for surface in surfaces:
        surface_df = df[df['Surface'] == surface]
        
        # R vs L matchups
        r_vs_l = len(surface_df[(surface_df['Winner_hand'] == 'R') & (surface_df['Loser_hand'] == 'L')])
        l_vs_r = len(surface_df[(surface_df['Winner_hand'] == 'L') & (surface_df['Loser_hand'] == 'R')])
        total_mixed = r_vs_l + l_vs_r
        
        # R vs R matchups (count each match once, winner is always R)
        r_vs_r = len(surface_df[(surface_df['Winner_hand'] == 'R') & (surface_df['Loser_hand'] == 'R')])
        
        # L vs L matchups (count each match once, winner is always L)
        l_vs_l = len(surface_df[(surface_df['Winner_hand'] == 'L') & (surface_df['Loser_hand'] == 'L')])
        
        results.append({
            'Surface': surface,
            'R_wins_vs_L': r_vs_l,
            'L_wins_vs_R': l_vs_r,
            'Total_mixed_matchups': total_mixed,
            'R_win_rate_vs_L': r_vs_l / total_mixed if total_mixed > 0 else None,
            'L_win_rate_vs_R': l_vs_r / total_mixed if total_mixed > 0 else None,
            'R_vs_R_matches': r_vs_r,
            'L_vs_L_matches': l_vs_l,
            'Total_matches': len(surface_df)
        })
    
    results_df = pd.DataFrame(results)
    return results_df


def analyze_handedness_by_tournament(df, top_n=10):
    """
    Analyze win rates by handedness for top tournaments.
    
    Parameters:
    df: DataFrame with columns 'Winner_hand', 'Loser_hand', and 'Tournament'
    top_n: Number of top tournaments to analyze (by match count)
    
    Returns:
    DataFrame with win rates by tournament and handedness
    """
    
    # Get top N tournaments by match count
    top_tournaments = df['Tournament'].value_counts().head(top_n).index
    
    results = []
    
    for tournament in top_tournaments:
        tour_df = df[df['Tournament'] == tournament]
        
        # Get surface for this tournament (assuming consistent)
        surface = tour_df['Surface'].mode()[0] if 'Surface' in tour_df.columns else 'Unknown'
        
        # R vs L matchups
        r_vs_l = len(tour_df[(tour_df['Winner_hand'] == 'R') & (tour_df['Loser_hand'] == 'L')])
        l_vs_r = len(tour_df[(tour_df['Winner_hand'] == 'L') & (tour_df['Loser_hand'] == 'R')])
        total_mixed = r_vs_l + l_vs_r
        
        # Same-handedness matches
        r_vs_r = len(tour_df[(tour_df['Winner_hand'] == 'R') & (tour_df['Loser_hand'] == 'R')])
        l_vs_l = len(tour_df[(tour_df['Winner_hand'] == 'L') & (tour_df['Loser_hand'] == 'L')])
        
        results.append({
            'Tournament': tournament,
            'Surface': surface,
            'R_wins_vs_L': r_vs_l,
            'L_wins_vs_R': l_vs_r,
            'Total_mixed_matchups': total_mixed,
            'R_win_rate_vs_L': r_vs_l / total_mixed if total_mixed > 0 else None,
            'L_win_rate_vs_R': l_vs_r / total_mixed if total_mixed > 0 else None,
            'R_vs_R_matches': r_vs_r,
            'L_vs_L_matches': l_vs_l,
            'Total_matches': len(tour_df)
        })
    
    results_df = pd.DataFrame(results)
    return results_df

def statistical_tests_by_surface(df):
    """
    Perform chi-square tests for each surface.
    """
    print("Statistical Significance Tests by Surface:")
    print("=" * 70)
    
    surfaces = df['Surface'].unique()
    
    for surface in surfaces:
        surface_df = df[df['Surface'] == surface]
        
        # Create contingency table
        contingency = pd.crosstab(surface_df['Winner_hand'], surface_df['Loser_hand'])
        
        if contingency.size >= 4:  # Need at least 2x2 table
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            print(f"\n{surface}:")
            print(f"  Chi-square: {chi2:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
            print(f"  Sample size: {len(surface_df)} matches")
        else:
            print(f"\n{surface}: Insufficient data for statistical test")

def plot_surface_analysis(surface_df):
    """
    Create visualizations for surface analysis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Win rates by surface
    surfaces = surface_df['Surface']
    r_rates = surface_df['R_win_rate_vs_L'] * 100
    l_rates = surface_df['L_win_rate_vs_R'] * 100
    
    x = np.arange(len(surfaces))
    width = 0.35
    
    axes[0].bar(x - width/2, r_rates, width, label='R wins vs L', alpha=0.8)
    axes[0].bar(x + width/2, l_rates, width, label='L wins vs R', alpha=0.8)
    axes[0].set_xlabel('Surface')
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_title('Win Rates by Surface (R vs L matchups)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(surfaces)
    axes[0].legend()
    axes[0].axhline(y=50, color='r', linestyle='--', alpha=0.3, label='50% baseline')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Match counts by surface
    match_counts = surface_df[['Surface', 'Total_mixed_matchups', 'R_vs_R_matches', 'L_vs_L_matches']]
    match_counts.set_index('Surface').plot(kind='bar', stacked=True, ax=axes[1], alpha=0.8)
    axes[1].set_xlabel('Surface')
    axes[1].set_ylabel('Number of Matches')
    axes[1].set_title('Match Distribution by Surface and Matchup Type')
    axes[1].legend(['R vs L', 'R vs R', 'L vs L'])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_tournament_analysis(tournament_df):
    """
    Create visualizations for tournament analysis.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by L win rate
    tournament_df = tournament_df.sort_values('L_win_rate_vs_R', ascending=False)
    
    tournaments = tournament_df['Tournament']
    r_rates = tournament_df['R_win_rate_vs_L'] * 100
    l_rates = tournament_df['L_win_rate_vs_R'] * 100
    
    x = np.arange(len(tournaments))
    width = 0.35
    
    ax.bar(x - width/2, r_rates, width, label='R wins vs L', alpha=0.8)
    ax.bar(x + width/2, l_rates, width, label='L wins vs R', alpha=0.8)
    ax.set_xlabel('Tournament')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rates by Tournament (R vs L matchups)')
    ax.set_xticks(x)
    ax.set_xticklabels(tournaments, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def full_handedness_analysis(df):
    """
    Run complete analysis of handedness by surface and tournament.
    """
    print("=" * 70)
    print("HANDEDNESS ANALYSIS BY SURFACE")
    print("=" * 70)
    
    surface_results = analyze_handedness_by_surface(df)
    print("\nSurface Analysis Results:")
    print(surface_results.to_string(index=False))
    
    print("\n")
    statistical_tests_by_surface(df)
    
    print("\n" + "=" * 70)
    print("HANDEDNESS ANALYSIS BY TOURNAMENT (Top 10)")
    print("=" * 70)
    
    tournament_results = analyze_handedness_by_tournament(df, top_n=10)
    print("\nTournament Analysis Results:")
    print(tournament_results[['Tournament', 'Surface', 'R_win_rate_vs_L', 
                              'L_win_rate_vs_R', 'Total_matches']].to_string(index=False))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_surface_analysis(surface_results)
    plot_tournament_analysis(tournament_results)
    
    return surface_results, tournament_results

def calculate_differences(df):
    """
    Calculate age and height differences between winner and loser.
    Creates new columns: Age_diff, Height_diff
    Positive values = winner has advantage (older/taller)
    """
    df = df.copy()
    
    # Age difference (Winner age - Loser age)
    if 'Winner_age' in df.columns and 'Loser_age' in df.columns:
        df['Age_diff'] = df['Winner_age'] - df['Loser_age']
    
    # Height difference (Winner height - Loser height)
    if 'Winner_height' in df.columns and 'Loser_height' in df.columns:
        df['Height_diff'] = df['Winner_height'] - df['Loser_height']
    
    return df


def descriptive_statistics(df):
    """
    Print descriptive statistics for age and height differences.
    """
    print("=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    
    if 'Age_diff' in df.columns:
        print("\nAge Difference (Winner - Loser):")
        print(f"  Mean: {df['Age_diff'].mean():.2f} years")
        print(f"  Median: {df['Age_diff'].median():.2f} years")
        print(f"  Std Dev: {df['Age_diff'].std():.2f} years")
        print(f"  Min: {df['Age_diff'].min():.2f} years")
        print(f"  Max: {df['Age_diff'].max():.2f} years")
        
        # Check if winners are generally younger or older
        younger_wins = (df['Age_diff'] < 0).sum()
        older_wins = (df['Age_diff'] > 0).sum()
        same_age = (df['Age_diff'] == 0).sum()
        
        print(f"\n  Younger player wins: {younger_wins} ({younger_wins/len(df)*100:.1f}%)")
        print(f"  Older player wins: {older_wins} ({older_wins/len(df)*100:.1f}%)")
        print(f"  Same age: {same_age} ({same_age/len(df)*100:.1f}%)")
    
    if 'Height_diff' in df.columns:
        print("\nHeight Difference (Winner - Loser):")
        print(f"  Mean: {df['Height_diff'].mean():.2f} cm")
        print(f"  Median: {df['Height_diff'].median():.2f} cm")
        print(f"  Std Dev: {df['Height_diff'].std():.2f} cm")
        print(f"  Min: {df['Height_diff'].min():.2f} cm")
        print(f"  Max: {df['Height_diff'].max():.2f} cm")
        
        # Check if winners are generally taller or shorter
        shorter_wins = (df['Height_diff'] < 0).sum()
        taller_wins = (df['Height_diff'] > 0).sum()
        same_height = (df['Height_diff'] == 0).sum()
        
        print(f"\n  Shorter player wins: {shorter_wins} ({shorter_wins/len(df)*100:.1f}%)")
        print(f"  Taller player wins: {taller_wins} ({taller_wins/len(df)*100:.1f}%)")
        print(f"  Same height: {same_height} ({same_height/len(df)*100:.1f}%)")


def statistical_tests(df):
    """
    Perform statistical significance tests.
    """
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)
    
    # T-test: Is the mean difference significantly different from 0?
    if 'Age_diff' in df.columns:
        age_diff_clean = df['Age_diff'].dropna()
        t_stat, p_value = stats.ttest_1samp(age_diff_clean, 0)
        
        print("\nAge Difference T-Test (H0: mean difference = 0):")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        
        if p_value < 0.05:
            direction = "older" if age_diff_clean.mean() > 0 else "younger"
            print(f"  → Winners tend to be {direction} than losers")
    
    if 'Height_diff' in df.columns:
        height_diff_clean = df['Height_diff'].dropna()
        t_stat, p_value = stats.ttest_1samp(height_diff_clean, 0)
        
        print("\nHeight Difference T-Test (H0: mean difference = 0):")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        
        if p_value < 0.05:
            direction = "taller" if height_diff_clean.mean() > 0 else "shorter"
            print(f"  → Winners tend to be {direction} than losers")
    
    # Chi-square test: Does being taller/older affect win probability?
    if 'Age_diff' in df.columns:
        age_categories = pd.cut(df['Age_diff'], 
                                bins=[-np.inf, -2, 2, np.inf], 
                                labels=['Younger', 'Similar', 'Older'])
        age_contingency = pd.crosstab(age_categories, [1]*len(df))
        
        print("\nAge Category Distribution:")
        print(age_categories.value_counts().sort_index())


def analyze_by_bins(df):
    """
    Analyze win rates by binned age and height differences.
    """
    print("\n" + "=" * 70)
    print("WIN RATE ANALYSIS BY DIFFERENCE BINS")
    print("=" * 70)
    
    results = []
    
    if 'Age_diff' in df.columns:
        # Create age bins
        df['Age_bin'] = pd.cut(df['Age_diff'], 
                               bins=[-np.inf, -5, -2, 2, 5, np.inf],
                               labels=['Much younger (-5+)', 'Younger (-2 to -5)', 
                                      'Similar (±2)', 'Older (2 to 5)', 'Much older (5+)'])
        
        print("\nWin Rate by Age Difference:")
        age_counts = df['Age_bin'].value_counts().sort_index()
        
        for category in age_counts.index:
            count = age_counts[category]
            pct = count / len(df) * 100
            print(f"  {category}: {count} matches ({pct:.1f}%)")
    
    if 'Height_diff' in df.columns:
        # Create height bins
        df['Height_bin'] = pd.cut(df['Height_diff'], 
                                  bins=[-np.inf, -5, -2, 2, 5, np.inf],
                                  labels=['Much shorter (-5+ cm)', 'Shorter (-2 to -5 cm)', 
                                         'Similar (±2 cm)', 'Taller (2 to 5 cm)', 'Much taller (5+ cm)'])
        
        print("\nWin Rate by Height Difference:")
        height_counts = df['Height_bin'].value_counts().sort_index()
        
        for category in height_counts.index:
            count = height_counts[category]
            pct = count / len(df) * 100
            print(f"  {category}: {count} matches ({pct:.1f}%)")


def analyze_by_surface(df):
    """
    Analyze age and height effects across different surfaces.
    """
    if 'Surface' not in df.columns:
        return
    
    print("\n" + "=" * 70)
    print("ANALYSIS BY SURFACE")
    print("=" * 70)
    
    surfaces = df['Surface'].unique()
    
    for surface in surfaces:
        surface_df = df[df['Surface'] == surface]
        
        print(f"\n{surface} ({len(surface_df)} matches):")
        
        if 'Age_diff' in df.columns:
            mean_age = surface_df['Age_diff'].mean()
            print(f"  Mean age diff: {mean_age:.2f} years")
            
        if 'Height_diff' in df.columns:
            mean_height = surface_df['Height_diff'].mean()
            print(f"  Mean height diff: {mean_height:.2f} cm")


def plot_distributions(df):
    """
    Create visualizations for age and height differences.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Age difference histogram
    if 'Age_diff' in df.columns:
        axes[0, 0].hist(df['Age_diff'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
        axes[0, 0].axvline(x=df['Age_diff'].mean(), color='green', linestyle='--', 
                          linewidth=2, label=f'Mean: {df["Age_diff"].mean():.2f}')
        axes[0, 0].set_xlabel('Age Difference (Winner - Loser) [years]')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Age Differences')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
    
    # Height difference histogram
    if 'Height_diff' in df.columns:
        axes[0, 1].hist(df['Height_diff'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
        axes[0, 1].axvline(x=df['Height_diff'].mean(), color='green', linestyle='--', 
                          linewidth=2, label=f'Mean: {df["Height_diff"].mean():.2f}')
        axes[0, 1].set_xlabel('Height Difference (Winner - Loser) [cm]')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Height Differences')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
    
    # Box plots by surface
    if 'Surface' in df.columns and 'Age_diff' in df.columns:
        df.boxplot(column='Age_diff', by='Surface', ax=axes[1, 0])
        axes[1, 0].set_title('Age Difference by Surface')
        axes[1, 0].set_xlabel('Surface')
        axes[1, 0].set_ylabel('Age Difference (years)')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.sca(axes[1, 0])
        plt.xticks(rotation=45)
    
    if 'Surface' in df.columns and 'Height_diff' in df.columns:
        df.boxplot(column='Height_diff', by='Surface', ax=axes[1, 1])
        axes[1, 1].set_title('Height Difference by Surface')
        axes[1, 1].set_xlabel('Surface')
        axes[1, 1].set_ylabel('Height Difference (cm)')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.sca(axes[1, 1])
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def full_age_height_analysis(df):
    """
    Run complete analysis of age and height differences.
    """
    # Calculate differences
    df = calculate_differences(df)
    
    # Run all analyses
    descriptive_statistics(df)
    statistical_tests(df)
    analyze_by_bins(df)
    analyze_by_surface(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_distributions(df)
    
    return df

# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('your_tennis_data.csv')
    
    # Required columns: Winner_age, Loser_age, Winner_height, Loser_height
    # Optional: Surface, Tournament
    
    # Run analysis
    df_with_diffs = full_age_height_analysis(df)