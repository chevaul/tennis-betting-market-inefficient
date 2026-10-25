# Tennis Match Prediction Model - Executive Summary

## What We Built

A machine learning system for ATP tennis prediction achieving professional-grade performance: 73.1% accuracy with strong probability calibration for sports betting applications.

---

## Key Results

| Metric             | Our Model (CatBoost) | Benchmark         | Improvement |
| ------------------ | -------------------- | ----------------- | ----------- |
| **Accuracy** | 73.1%                | 60% (favorites)   | +13.1 pts  |
| **AUC**      | 0.808                | 0.706 (Elo alone) | +10.2 pts   |
| **Log Loss** | 0.532                | 0.622 (baseline)  | -14.5%      |

**Translation:** We correctly predict ~7 out of 10 matches, with strong probability calibration for risk assessment.

---

## What Makes This Different

### 1. Comprehensive Feature Engineering

* Elo ratings with probability transformations
* ATP rankings and year-to-date performance statistics
* Head-to-head records and recent form metrics
* Surface-specific adjustments

### 2. Advanced Modeling

* **CatBoost** - best performer (73.1% accuracy, 0.808 AUC)
* XGBoost with hyperparameter tuning (48 configurations tested)
* Regularized logistic regression for interpretability
* Ensemble methods for robustness
* Cross-validated to prevent overfitting

### 3. Clean Data Pipeline

* 24,875 ATP matches (2015-2024)
* Two-perspective transformation (49,750 training samples)
* Removed all data leakage (only pre-match information)
* Handled multicollinearity and missing values

---

## Potential Applications

### For Sports Betting

* **Probability calibration:** Model probabilities are well-aligned with actual outcomes (0.53 log loss)
* **Value identification:** Strong AUC (0.808) enables comparison against bookmaker odds
* **Risk management:** Confidence scores differentiate sure bets from coin flips

### Market Inefficiency Detection

* Compare model probabilities to market odds
* Identify mispriced matches
* Quantify edge size for bet sizing (Kelly criterion)

---

## What We're Currently Working On

1. **Betting simulation:** Backtesting against historical odds to measure profitability
2. **Feature importance:** Understanding which factors bookmakers may underweight
3. **Temporal validation:** Testing on most recent matches (2024-2025)
4. **Market comparison:** Analyzing where model disagrees most with betting markets

---

## Technical Highlights (For the Curious)

**Models Tested:**

* Baseline Elo probability
* Logistic Regression (with/without regularization)
* Random Forest
* Gradient Boosting
* XGBoost
* CatBoost (best performer)

**Top Predictive Features:**

1. Points difference (12.54% importance)
2. Elo difference (11.28%)
3. Elo-based win probability (9.04%)
4. Rank-based probability (6.52%)
5. Form difference (6.35%)

**Data Quality:**

* No data leakage (verified via multicollinearity analysis)
* Balanced classes (50/50 wins/losses)
* Robust train/test split with cross-validation

---

## Next Steps

We're interested in exploring:

* Real-world betting strategy development
* Live prediction deployment
* Market inefficiency analysis
* Collaboration with domain experts

---

## Contact

**Ecem Yildirim** | Project Lead 
General Course, LSE | Computer Science & Economics, Georgetown University
Tennis Betting Markets Efficiency Project
ecemyildirim900@gmail.com | [linkedin.com/in/yildirimecem](https://www.linkedin.com/in/yildirimecem/)

**Available Materials:**

* Full technical documentation
* Jupyter notebooks (EDA, modeling, baseline features)
* Visualization suite (model performance, feature analysis)
* Detailed methodology and results

---

*This project demonstrates strong predictive modeling and clean data science practices. The model achieves professional-grade performance with transparent, reproducible methodology.*
