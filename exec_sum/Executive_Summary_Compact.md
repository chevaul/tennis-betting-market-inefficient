# Tennis Match Prediction Model

**73.1% accuracy | 0.808 AUC | Professional-grade performance for sports betting applications**

---

## Key Results

| Metric   | CatBoost Model | Benchmark        | Improvement |
| -------- | -------------- | ---------------- | ----------- |
| Accuracy | 73.1%          | 60% (favorites)  | +13.1 pts   |
| AUC      | 0.808          | 0.706 (Elo)      | +10.2 pts   |
| Log Loss | 0.532          | 0.622 (baseline) | -14.5%      |

**Bottom line:** Correctly predicts 7/10 matches with well-calibrated probabilities for risk assessment.

---

## Methodology

**Data:** 24,875 ATP matches (2015-2024) → 49,750 training samples (two-perspective transformation)

**Features:** Elo ratings, ATP rankings, points, form metrics, head-to-head records, surface adjustments

**Models Tested:** Baseline Elo, Logistic Regression, Random Forest, Gradient Boosting, XGBoost, CatBoost

**Validation:** Cross-validated, proper train/test split, no data leakage

**Top Features:** Points difference (12.5%), Elo difference (11.3%), Elo probability (9.0%), Rank probability (6.5%), Form (6.4%)

---

## Applications for Sports Betting

**Probability Calibration:** 0.532 log loss indicates model probabilities accurately reflect true win rates - critical for comparing against bookmaker odds

**Value Identification:** 0.808 AUC demonstrates strong ability to separate winners from losers, enabling detection of mispriced markets

**Risk Management:** Model provides confidence scores, allowing differentiation between high-confidence predictions and coin-flip scenarios

---

## Current Work

1. Backtesting against historical betting odds to measure profitability
2. Temporal validation on 2024-2025 matches
3. Market comparison analysis (where model disagrees with bookmakers)
4. Feature importance deep-dive (what factors are underweighted in markets)

---

## Technical Details

**Models:** Tested 7 approaches including regularized linear models and tree-based ensembles. CatBoost achieved best performance across all metrics.

**Data Quality:** Rigorous preprocessing with multicollinearity analysis (VIF), proper handling of missing values, and verification of no post-match data leakage.

**Robustness:** Performance consistent across cross-validation folds, suggesting strong generalization to unseen matches.

---

## Next Steps

Interested in exploring: real-world betting strategy development, live prediction deployment, market inefficiency analysis, collaboration with domain experts.

---

**Ecem Yildirim** | General Course, LSE | Computer Science & Economics, Georgetown

ecemyildirim900@gmail.com | [linkedin.com/in/yildirimecem](https://www.linkedin.com/in/yildirimecem/)

*Full technical documentation, notebooks, and visualizations available upon request.*
