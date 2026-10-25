"""
Modeling utilities for tennis betting efficiency analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve


def calculate_calibration(y_true, y_pred, n_bins=10):
    """
    Calculate calibration metrics for probability predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True binary outcomes (0 or 1)
    y_pred : array-like
        Predicted probabilities (between 0 and 1)
    n_bins : int
        Number of bins for calibration curve
    
    Returns:
    --------
    tuple
        (fraction_of_positives, mean_predicted_value)
    """
    return calibration_curve(y_true, y_pred, n_bins=n_bins)


def calculate_log_loss(y_true, y_pred):
    """
    Calculate log loss (cross-entropy) for probability predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True binary outcomes (0 or 1)
    y_pred : array-like
        Predicted probabilities (between 0 and 1)
    
    Returns:
    --------
    float
        Log loss value
    """
    return log_loss(y_true, y_pred)


def calculate_brier_score(y_true, y_pred):
    """
    Calculate Brier score for probability predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True binary outcomes (0 or 1)
    y_pred : array-like
        Predicted probabilities (between 0 and 1)
    
    Returns:
    --------
    float
        Brier score value
    """
    return brier_score_loss(y_true, y_pred)


def expected_value_analysis(implied_prob, true_prob, stake=1):
    """
    Calculate expected value of a bet.
    
    Parameters:
    -----------
    implied_prob : float or array-like
        Implied probability from odds (must be > 0)
    true_prob : float or array-like
        True probability estimate
    stake : float
        Bet stake amount
    
    Returns:
    --------
    float or array-like
        Expected value of the bet
    """
    # Ensure implied_prob is not zero to avoid division by zero
    if isinstance(implied_prob, (int, float)):
        if implied_prob <= 0:
            raise ValueError("implied_prob must be greater than 0")
        odds = 1 / implied_prob
    else:
        # Handle array-like inputs
        implied_prob = np.asarray(implied_prob)
        if np.any(implied_prob <= 0):
            raise ValueError("All implied_prob values must be greater than 0")
        odds = 1 / implied_prob
    
    expected_return = true_prob * odds * stake
    expected_value = expected_return - stake
    return expected_value


class MarketEfficiencyAnalyzer:
    """
    Analyze market efficiency in sports betting.
    """
    
    def __init__(self):
        self.results = {}
    
    def fit(self, implied_probs, outcomes):
        """
        Fit analyzer on implied probabilities and outcomes.
        
        Parameters:
        -----------
        implied_probs : array-like
            Implied probabilities from betting odds
        outcomes : array-like
            Actual binary outcomes (0 or 1)
        """
        self.results['log_loss'] = calculate_log_loss(outcomes, implied_probs)
        self.results['brier_score'] = calculate_brier_score(outcomes, implied_probs)
        self.results['calibration'] = calculate_calibration(outcomes, implied_probs)
        
        return self
    
    def get_results(self):
        """
        Get analysis results.
        
        Returns:
        --------
        dict
            Dictionary containing analysis results
        """
        return self.results
