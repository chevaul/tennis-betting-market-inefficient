"""
Utility functions for tennis betting efficiency analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns


def load_tennis_data(filepath):
    """
    Load tennis data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()


def check_missing_data(df):
    """
    Check for missing data in DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns:
    --------
    pd.DataFrame
        Summary of missing data by column
    """
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_pct': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    return missing[missing['missing_count'] > 0].sort_values('missing_count', ascending=False)


def safe_has(df, *cols):
    """
    Check whether all specified columns exist in a pandas DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to check for column existence.

    *cols : str
        One or more column names to verify in the DataFrame.

    Returns:
    --------
    bool
        True if all specified columns exist in the DataFrame,
        False otherwise.
    """
    return all(c in df.columns for c in cols)

def plot_distribution(data, column, title=None, figsize=(10, 6)):
    """
    Plot distribution of a column.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame
    column : str
        Column name to plot
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if data[column].dtype in ['object', 'category']:
        data[column].value_counts().plot(kind='bar', ax=ax)
    else:
        data[column].hist(bins=30, ax=ax)
    
    ax.set_title(title or f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    
    return fig


def plot_calibration(y_true, y_pred, n_bins=10, title='Calibration Plot', figsize=(10, 6)):
    """
    Create calibration plot.
    
    Parameters:
    -----------
    y_true : array-like
        True binary outcomes
    y_pred : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for calibration
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=figsize)
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred, n_bins=n_bins
    )
    
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def summarize_dataframe(df):
    """
    Provide summary statistics for DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns:
    --------
    None
    """
    print(f"Shape: {df.shape}")
    print(f"\nColumn Types:\n{df.dtypes}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nSummary Statistics:\n{df.describe()}")
