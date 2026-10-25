# utils/kelly_calculator.py

def calculate_kelly_fraction(model_prob, market_prob, kelly_fraction=0.25):
    """
    Calculate Kelly criterion bet size
    
    Args:
        model_prob: Your model's win probability (0-1)
        market_prob: Market implied probability (0-1)
        kelly_fraction: Kelly multiplier (0.25 = quarter Kelly, conservative)
    
    Returns:
        Recommended bet size as decimal (e.g., 0.05 = 5% of bankroll)
    """
    
    # No bet if no positive edge
    if model_prob <= market_prob:
        return 0.0
    
    # Convert market prob to decimal odds
    decimal_odds = min(1 / market_prob, 100)  # Cap at 100:1
    b = decimal_odds - 1  # Net odds
    p = model_prob
    q = 1 - p
    
    # Kelly formula: (bp - q) / b
    kelly = (b * p - q) / b
    
    # Cap Kelly at 25% (even full Kelly shouldn't exceed this)
    kelly = max(0, min(kelly, 0.25))
    
    # Apply fractional Kelly
    return kelly * kelly_fraction


def get_bet_recommendation(edge, kelly_size):
    """
    Get betting recommendation based on edge and Kelly size
    
    Returns: (recommendation, color, emoji)
    """
    
    if edge > 0.15:
        return "STRONG VALUE BET", "success", "✅"
    elif edge > 0.05:
        return "VALUE BET", "info", "✓"
    elif edge > 0:
        return "SLIGHT EDGE", "warning", "⚠️"
    else:
        return "NO BET", "error", "❌"


def calculate_expected_value(model_prob, decimal_odds, stake=100):
    """
    Calculate expected value of a bet
    
    Args:
        model_prob: Win probability according to model
        decimal_odds: Decimal odds (e.g., 2.5)
        stake: Bet amount in dollars
    
    Returns:
        Expected profit/loss
    """
    
    # Expected return if win
    win_return = stake * (decimal_odds - 1)
    
    # Expected loss if lose
    loss = -stake
    
    # Expected value
    ev = model_prob * win_return + (1 - model_prob) * loss
    
    return ev
