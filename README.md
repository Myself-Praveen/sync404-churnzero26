# ChurnZero: Cost-Optimized Churn Prediction Architecture

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)
![Hackathon](https://img.shields.io/badge/Status-Championship_Ready-success.svg)

## 📌 Project Overview
**Team Sync404**'s official submission for the ChurnZero '26 Hackathon. 

Churn prediction is inherently a financial optimization problem, not just a classification task. Standard metrics like accuracy treat all mistakes equally—but in retail banking, a False Negative (missing a churning customer) costs exponentially more than a False Positive (wasting a retention offer). 

We architected a fully leakage-free, cost-sensitive machine learning pipeline that minimizes the total business loss by deploying a calibrated stacked ensemble.

### 🏆 Key Achievements
- **Validation PR-AUC:** `0.9999` (Defended via raw-feature depth-3 baseline).
- **Asymmetric Cost Minimization:** Reduced total portfolio loss to Rs. 53,500.
- **True Positive Rate:** `99.9%` retention capture rate.
- **Algorithmic Fairness:** Built-in dynamic threshold equalization across protected demographics.
- **Variance Reduction:** Seed averaging dropped prediction uncertainty to an incredibly stable `0.0004`.

---

## 🏗️ Architecture & Methodology

Our end-to-end pipeline is entirely automated in `ChurnZero_Code.py`.

1. **Domain-Driven Feature Engineering:** Created 40+ custom financial and behavioral signals (e.g., `flight_risk_index`, `frustration_score`, `emi_burden_ratio`) to maximize rank separability between loyalists and churners.
2. **Leakage-Free Validation:** Strict 5-fold Stratified CV. Target-mean encodings are dynamically calculated *inside* the folds to prevent data leakage.
3. **Stacked Ensemble Engine:** Blended 5 diverse Gradient Boosting models (`HistGradientBoosting`, `LGBM`, `XGBoost`, `CatBoost`) using a Logistic Regression meta-learner.
4. **Probability Calibration (Platt Scaling):** Applied Sigmoid calibration on Out-Of-Fold probabilities to convert rank-order scores into true statistical likelihoods.
5. **Cost-Sensitive Thresholding:** Optimized the decision boundary (Threshold = `0.0090`) to explicitly minimize a predefined cost matrix (FN = Rs. 40,000 | FP = Rs. 500).

---

## 📊 Global Explainability (SHAP)
Using `TreeExplainer`, we extracted exact, game-theoretic feature attributions. The model confirms that digital friction and poor customer service are stronger churn indicators than demographics:
1. `total_digital_logins`
2. `unresolved_complaint_count`
3. `balance_decline_percentage`
4. `complaint_resolution_time`
5. `mobile_app_login_count`

*(Note: The pipeline also provides local SHAP force values for every individual prediction to empower the retention team with actionable intelligence).*

---

## 🚀 How to Run

### Prerequisites
Make sure you have Python 3.13+ installed. Install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Execution
Simply place the dataset files (`ChurnZero_dataset_v1.csv` and `ChurnZero_test_v1.csv`) in the same directory as the script and run:
```bash
python ChurnZero_Code.py
```

### Outputs
The script is an end-to-end pipeline that will generate:
1. Console output with EDA, baseline PR-AUC, fold-level metrics, fairness audit, and threshold math.
2. `ChurnZero_Sync404_Predictions.csv`: The final predictions for the test set.
3. Serialized model artifacts (`.pkl` files) containing the entire state of the ensemble, preprocessor, and calibration engines.

---
*Built by Team Sync404 for ChurnZero '26.*
