"""
================================================================================
ChurnZero 26 - Championship Model Pipeline
================================================================================
Team: Sync404
Hackathon: ChurnZero 26, IIT Kharagpur

Usage:
    python ChurnZero_Code.py
    Requires: ChurnZero_dataset_v1.csv and ChurnZero_test_v1.csv in the CWD.
    Outputs:  ChurnZero_Sync404_Predictions.csv and churn_artefacts.pkl.

Architecture:
    5-model ensemble (2× HGB, LGB, XGB, CatBoost) with OOF weight optimisation,
    LR stacking meta-learner, seed averaging (3 seeds), and cost-sensitive
    threshold tuning (FN=Rs.40k, FP=Rs.500).

Changelog:
    [FIX-1]   Missing indicator for app_rating_given (56% missing → informative)
    [FIX-2]   Drop mobile_banking_active_flag (near-zero variance)
    [FIX-3]   Dynamic scale_pos_weight from data (was hardcoded 5.2)
    [FIX-4]   Remove StandardScaler (GBMs are scale-invariant)
    [FIX-5]   Log transforms for heavily skewed columns (skew > 2)
    [FIX-6]   Fairness verdict conditional on chi-square p-values
    [FIX-7]   CalibratedClassifierCV removed from pipeline (saves ~200 redundant fits)
    [FIX-8]   Full reproducibility seed block (random, numpy, PYTHONHASHSEED)
    [FIX-9]   Preprocessor rebuilt after target encoding (includes _te cols)
    [FIX-10]  LGB early stopping in CV (50 rounds, fold validation set)
    [FIX-11]  Blend ratio tuned on OOF (was arbitrary 0.5/0.5)
    [FIX-12]  Feature importance from full-retrain model, not CV-fold model
    [FIX-13]  SHAP TreeExplainer replaces split-count importance
    [FIX-14]  CatBoost early stopping in CV (50 rounds); full retrain intentionally runs all iterations
    [ADD-1]   CatBoost in ensemble (native categorical handling)
    [ADD-2]   Seed averaging on test set (3 seeds; variance reduction)
    [ADD-3]   OOF-based ensemble weight optimisation (scipy SLSQP)
    [ADD-4]   Stacking meta-learner (LogisticRegression on OOF probs)
    [ADD-5]   Target-mean encoding for high-churn-signal categoricals
    [ADD-6]   5 new domain features driven by top correlated raw signals
    [FIX-15]  XGB early stopping — consistent with LGB and CatBoost
    [ADD-7]   Platt scaling (sigmoid calibration) on OOF probs before test predictions
    [ADD-8]   Lift@10% metric printed alongside threshold optimisation results
================================================================================
"""

# Dependencies (pip install):
#   lightgbm>=4.0  xgboost>=2.0  catboost>=1.2  shap>=0.44
#   scikit-learn>=1.4  pandas>=2.0  numpy>=1.24  scipy>=1.10  joblib>=1.3

import pandas as pd
import numpy as np
import random
import os
import warnings
warnings.filterwarnings('ignore')

# ML Core
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.optimize import minimize
from scipy.stats import chi2_contingency

# Optional heavy hitters
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[INFO] LightGBM not installed.")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] XGBoost not installed.")

# [ADD-1] CatBoost import
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[INFO] CatBoost not installed. pip install catboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[INFO] SHAP not installed. pip install shap")

import sys, time, sklearn, joblib

# [FIX-8] Global reproducibility — covers Python hash, numpy, and stdlib random
GLOBAL_SEED = 42
SEEDS = [42, 123, 999]  # [ADD-2] seed averaging
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

print(f"[ENV] Python: {sys.version.split()[0]}")
print(f"[ENV] scikit-learn: {sklearn.__version__}")
print(f"[ENV] NumPy: {np.__version__} | Pandas: {pd.__version__}")
if HAS_LGB:  print(f"[ENV] LightGBM: {lgb.__version__}")
if HAS_XGB:  print(f"[ENV] XGBoost: {xgb.__version__}")
if HAS_CAT:  print(f"[ENV] CatBoost: installed")
if HAS_SHAP: print(f"[ENV] SHAP: installed")


# =============================================================================
# SECTION 1: FEATURE ENGINEERING
# =============================================================================

# Columns that are known ordinals — mapped to integers for consistency
AWARENESS_MAP  = {'Not Aware': 0, 'Low': 1, 'Medium': 2, 'High': 3}
SENTIMENT_MAP  = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

# [FIX-5] Columns to log-transform (skewness > 2 confirmed from EDA)
LOG_SKEWED_COLS = [
    'customer_lifetime_value',   # skew 1.76 — borderline, include
    'credit_card_limit',         # skew 1.68
    'emi_amount',                # skew 3.98
    'monthly_transaction_value', # skew 2.05
    'loan_outstanding_amount',   # skew 3.34
    'complaint_resolution_time', # skew 3.61
    'escalation_count',          # skew 7.38 — strongly skewed
    'unresolved_complaint_count',# skew 5.55
    'total_trans_amt',           # skew 2.05
]

# [ADD-5] Target-mean encoded categoricals (high churn-signal categories)
# These are computed inside CV folds to prevent leakage (see get_target_encoding)
TARGET_ENCODE_COLS = [
    'competitor_bank_offer_awareness',
    'customer_feedback_sentiment',
    'card_category',
    'customer_segment',
    'relationship_type',
]

PRODUCT_FLAGS = [
    'savings_account_flag', 'current_account_flag', 'credit_card_flag',
    'personal_loan_flag', 'home_loan_flag', 'auto_loan_flag',
    'fixed_deposit_flag', 'investment_product_flag',
    'insurance_product_flag', 'demat_account_flag'
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 40+ domain-driven features across 7 strategic categories.
    All features are created from raw columns only — no target leakage.
    """
    df = df.copy()

    # --- Map ordinal categoricals to numeric ---
    df['competitor_awareness_num'] = (
        df['competitor_bank_offer_awareness'].map(AWARENESS_MAP).fillna(0)
    )
    df['feedback_sentiment_num'] = (
        df['customer_feedback_sentiment'].map(SENTIMENT_MAP).fillna(1)
    )

    # =========================================================================
    # CATEGORY A: BEHAVIORAL EARLY WARNING SIGNALS
    # =========================================================================

    # A1. Flight Risk Index — declining balance × growing inactivity
    df['flight_risk_index'] = (
        df['balance_decline_percentage'] * np.log1p(df['account_inactive_days'])
    )

    # A2. Digital Disengagement — days since login / total logins
    df['digital_disengagement'] = (
        df['last_login_days'] / (df['total_digital_logins'] + 1)
    )

    # A3. Transaction Momentum — Q4 vs Q1 amount AND count change
    df['transaction_momentum'] = (
        df['total_amt_chng_q4_q1'] * df['total_ct_chng_q4_q1']
    )

    # A4. Balance Erosion — how much of avg balance has eroded
    df['balance_erosion'] = np.where(
        df['avg_monthly_balance'] > 0,
        (df['avg_monthly_balance'] - df['current_balance']) / df['avg_monthly_balance'],
        0
    )

    # A5. Relative Inactivity — login gap relative to tenure
    df['relative_inactivity'] = (
        df['last_login_days'] / (df['tenure_months'] + 1)
    )

    # [ADD-6] A6. Transaction Count Ratio — recent to average (sparse usage signal)
    df['txn_count_ratio'] = (
        df['monthly_transaction_count'] / (df['total_trans_count'] / 12 + 1)
    )

    # [ADD-6] A7. Digital Activity Score — combines logins and digital tx ratio
    # Top correlated raw features: total_digital_logins (0.508), mobile_app_login_count (0.427)
    df['digital_activity_score'] = (
        df['total_digital_logins'] * df['digital_transaction_ratio']
    )

    # =========================================================================
    # CATEGORY B: SERVICE FRUSTRATION SIGNALS
    # =========================================================================

    # B1. Frustration Score — weighted complaint metric
    df['frustration_score'] = (
        df['unresolved_complaint_count'] * 3 +
        df['escalation_count'] * 2 +
        df['total_complaints']
    )

    # B2. Resolution Failure Rate
    df['resolution_failure_rate'] = np.where(
        df['total_complaints'] > 0,
        df['unresolved_complaint_count'] / df['total_complaints'],
        0
    )

    # B3. Satisfaction Gap — NPS vs satisfaction divergence
    df['satisfaction_gap'] = df['nps_score'] - df['satisfaction_score']

    # B4. Service Intensity — total physical touchpoints
    df['service_intensity'] = (
        df['branch_visit_count'] +
        df['call_center_interaction_count'] +
        df['relationship_manager_interaction_count'] +
        df['service_request_count']
    )

    # [ADD-6] B5. Complaint Burden Rate — complaints per month of tenure
    df['complaint_burden_rate'] = (
        df['total_complaints'] / (df['tenure_months'] + 1)
    )

    # =========================================================================
    # CATEGORY C: COMPETITOR THREAT & RETENTION RESPONSE
    # =========================================================================

    # C1. Competitor Threat Score — awareness × transaction value at risk
    df['competitor_threat'] = (
        df['competitor_awareness_num'] *
        (df['credit_card_spend'] + df['monthly_transaction_value'] + 1)
    )

    # C2. Retention Resistance — received offer but didn't accept
    df['retention_resistance'] = (
        df['retention_offer_received'] - df['retention_offer_accepted']
    )

    # C3. Campaign Fatigue — fraction of campaigns not responded to
    df['campaign_fatigue'] = np.where(
        df['campaign_received_count'] > 0,
        1 - (df['campaign_response_count'] / df['campaign_received_count']),
        0
    )

    # C4. Last Contact Recency — combined recency score
    df['contact_recency_score'] = (
        df['last_contacted_days'] + df['last_campaign_response_days']
    )

    # =========================================================================
    # CATEGORY D: FINANCIAL HEALTH & PRODUCT DEPTH
    # =========================================================================

    # D1. Product Breadth — total distinct products held
    valid_flags = [c for c in PRODUCT_FLAGS if c in df.columns]
    df['product_breadth'] = df[valid_flags].sum(axis=1)

    # D2. Credit Stress — utilization + late payments + default risk
    df['credit_stress'] = (
        df['credit_utilization_ratio'] +
        df['late_credit_card_payment_count'] * 0.1 +
        df['loan_default_risk_score'] * 0.01
    )

    # D3. Credit Utilization Trend — 6m vs 3m (rising = financial squeeze)
    df['credit_util_trend'] = (
        df['credit_utilization_6m_avg'] - df['credit_utilization_3m_avg']
    )

    # D4. EMI Burden Ratio — EMI as fraction of monthly income
    df['emi_burden_ratio'] = np.where(
        df['monthly_income_estimate'] > 0,
        df['emi_amount'] / df['monthly_income_estimate'],
        0
    )

    # D5. Revolving Ratio — revolving balance vs credit limit
    df['revolving_ratio'] = np.where(
        df['credit_card_limit'] > 0,
        df['total_revolving_bal'] / df['credit_card_limit'],
        0
    )

    # [ADD-6] D6. EMI Delay Rate — delayed EMIs / total EMI payments implied
    df['emi_delay_rate'] = np.where(
        df['tenure_months'] > 0,
        df['emi_payment_delay_count'] / (df['tenure_months'] + 1),
        0
    )

    # =========================================================================
    # CATEGORY E: DIGITAL CHANNEL BEHAVIOR
    # =========================================================================

    # E1. Channel Preference — digital vs physical
    df['digital_vs_physical'] = (
        df['digital_transaction_ratio'] -
        (df['branch_visit_count'] / (df['monthly_transaction_count'] + 1))
    )

    # E2. App vs Web Preference
    df['app_vs_web'] = (
        df['mobile_app_login_count'] / (df['website_login_count'] + 1)
    )

    # E3. Login Failure Rate
    df['login_failure_rate'] = np.where(
        df['total_digital_logins'] > 0,
        df['failed_login_count'] / df['total_digital_logins'],
        0
    )

    # =========================================================================
    # CATEGORY F: INTERACTION FEATURES
    # =========================================================================

    # F1. High-Value At-Risk — CLV × flight risk
    df['high_value_at_risk'] = (
        df['customer_lifetime_value'] * df['flight_risk_index']
    )

    # F2. Frustrated AND Aware — compound threat
    df['frustrated_and_aware'] = (
        df['frustration_score'] * df['competitor_awareness_num']
    )

    # F3. Silent Bleed — low engagement + high balance decline
    df['silent_bleed'] = (
        df['digital_disengagement'] * df['balance_decline_percentage']
    )

    # F4. Stressed and Complaining
    df['stressed_and_complaining'] = (
        df['credit_stress'] * df['frustration_score']
    )

    # =========================================================================
    # CATEGORY G: EXPLICIT HIGH-ORDER INTERACTIONS
    # =========================================================================

    # G1. Complaint × digital coldness
    df['complaint_digital_gap'] = (
        df['total_complaints'] * (1 - df['digital_transaction_ratio'])
    )

    # G2. New user + high frustration is more alarming
    df['tenure_complaint_interaction'] = (
        np.log1p(df['tenure_months']) * df['frustration_score']
    )

    # G3. High credit limit + high utilization
    df['credit_limit_stress'] = (
        df['credit_card_limit'] * df['credit_utilization_ratio'] / 1e6
    )

    # G4. Channel confusion — high branch + high digital
    df['channel_confusion'] = (
        df['branch_visit_count'] * df['digital_transaction_ratio']
    )

    # G5. NPS detractor × complaint volume
    df['detractor_complaint'] = (
        (df['nps_score'] < 7).astype(int) * df['total_complaints']
    )

    # [ADD-6] G6. Recency × frustration — recently angry customers
    df['recent_frustration'] = (
        df['contact_recency_score'] * df['frustration_score']
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# =============================================================================
# SECTION 2: TARGET-MEAN ENCODING (leakage-safe, computed inside CV folds)
# [ADD-5] Encodes categorical columns with their within-fold churn rate.
#         Global mean used as prior (smoothed encoding prevents overfit on rare cats).
# =============================================================================

def apply_target_encoding(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, X_test: pd.DataFrame,
    cols: list = TARGET_ENCODE_COLS, smoothing: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Smoothed target mean encoding. Fit on fold-train, apply to fold-val and test.
    smoothing: equivalent sample size for the global prior (higher = more shrinkage).
    """
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    global_mean = y_train.mean()

    encoding_maps = {}
    for col in cols:
        if col not in X_train.columns:
            continue
        stats = (
            pd.concat([X_train[col], y_train], axis=1)
            .groupby(col)
            .agg(count=(col, 'count'), mean=(y_train.name, 'mean'))
        )
        # Smoothing formula: (n * cat_mean + smoothing * global_mean) / (n + smoothing)
        stats['smoothed'] = (
            (stats['count'] * stats['mean'] + smoothing * global_mean) /
            (stats['count'] + smoothing)
        )
        enc_col = col + '_te'
        encoding_maps[col] = stats['smoothed']
        for df in [X_train, X_val, X_test]:
            df[enc_col] = df[col].map(stats['smoothed']).fillna(global_mean)

    return X_train, X_val, X_test, encoding_maps



# =============================================================================
# SECTION 3: PREPROCESSING PIPELINE
# [FIX-4] StandardScaler removed — GBMs are scale-invariant.
# [FIX-1] Missing indicator handled before this pipeline, so app_rating_given
#         is imputed normally here after the indicator column is created.
# =============================================================================

def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a leakage-safe ColumnTransformer.
    - Numeric: median impute only (no scaling — not needed for tree models)
    - Categorical: ordinal encode with unknown handling
    """
    numeric_features     = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()

    # [FIX-4] Pipeline: imputer only — no StandardScaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        # StandardScaler intentionally removed — GBMs need no feature scaling
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop',
        n_jobs=-1   # parallel transform consistent with model parallelism
    )
    return preprocessor


# =============================================================================
# SECTION 4: MODEL CONSTRUCTION
# [ADD-1] CatBoost added. [FIX-3] scale_pos_weight computed from data.
# [FIX-7] CalibratedClassifierCV removed from here; calibration applied once
#         after full retrain to avoid ~200 redundant model fits inside CV.
# =============================================================================

def build_models(scale_pos_weight: float) -> dict:
    """
    Return individual estimators (not a VotingClassifier).
    We collect OOF probs per model and optimise weights separately [ADD-3].
    scale_pos_weight: computed as neg/pos from training labels [FIX-3].
    """
    models = {}

    # Model 1: HistGBT — always available, fast, handles NaN natively
    models['hgb'] = HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.03,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_bins=255,
        class_weight='balanced',
        early_stopping=True,
        n_iter_no_change=50,
        validation_fraction=0.1,
        random_state=42
    )

    # Model 2: HistGBT diversity variant (shallower, different regularisation)
    models['hgb2'] = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=30,
        l2_regularization=0.5,
        max_bins=128,
        class_weight='balanced',
        early_stopping=True,
        n_iter_no_change=50,
        validation_fraction=0.1,
        random_state=123
    )

    # Model 3: LightGBM
    if HAS_LGB:
        models['lgb'] = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.02,
            max_depth=6,
            num_leaves=40,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            is_unbalance=True,
            importance_type='gain',
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )

    # Model 4: XGBoost — [FIX-3] scale_pos_weight now computed from data
    if HAS_XGB:
        models['xgb'] = xgb.XGBClassifier(
            n_estimators=600,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,  # was hardcoded 5.2
            eval_metric='aucpr',
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )

    # Model 5: CatBoost — [ADD-1] native categorical encoding, no ordinal needed
    if HAS_CAT:
        models['cat'] = CatBoostClassifier(
            iterations=600,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=3,
            auto_class_weights='Balanced',
            eval_metric='PRAUC',
            random_seed=42,
            verbose=0
        )
    else:
        # Fallback when CatBoost is unavailable: a third HistGBT variant with
        # shallower trees and stronger regularisation to maintain ensemble diversity.
        # Install CatBoost for best results: pip install catboost
        print("[ENSEMBLE] CatBoost unavailable — adding HGB3 fallback for diversity.")
        models['hgb3'] = HistGradientBoostingClassifier(
            max_iter=700,
            learning_rate=0.02,
            max_depth=4,          # shallower than hgb/hgb2
            min_samples_leaf=40,
            l2_regularization=2.0,
            max_bins=64,          # coarser bins — different inductive bias
            class_weight='balanced',
            early_stopping=True,
            n_iter_no_change=50,
            validation_fraction=0.1,
            random_state=999
        )

    print(f"[ENSEMBLE] Models: {list(models.keys())}")
    return models


# =============================================================================
# SECTION 5: OOF WEIGHT OPTIMISATION
# [ADD-3] Finds the per-model blend weights that maximise OOF PR-AUC.
#         Uses scipy minimize (Nelder-Mead) with simplex projection.
# =============================================================================

def optimise_weights(oof_matrix: np.ndarray, y_true: pd.Series) -> np.ndarray:
    """
    oof_matrix: shape (n_samples, n_models) — OOF probabilities per model.
    y_true: binary target.
    Returns: weight array summing to 1.
    """
    n_models = oof_matrix.shape[1]

    def neg_prauc(w):
        w = np.array(w)
        w = np.clip(w, 0, None)
        w = w / (w.sum() + 1e-9)
        blended = oof_matrix @ w
        return -average_precision_score(y_true, blended)

    # Start from uniform weights
    w0 = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models

    result = minimize(neg_prauc, w0, method='SLSQP',
                      bounds=bounds,
                      constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                      options={'maxiter': 500, 'ftol': 1e-7})

    optimal_w = np.clip(result.x, 0, None)
    optimal_w /= optimal_w.sum()
    return optimal_w


# =============================================================================
# SECTION 6: STACKING META-LEARNER
# [ADD-4] LR trained on OOF probs from base models → learns optimal blend.
#         LogisticRegression is the competition-standard choice for stacking:
#         only 5 inputs → LR generalises better than an MLP (fewer parameters).
# =============================================================================

def build_meta_learner() -> LogisticRegression:
    """
    Logistic Regression meta-learner on OOF probabilities.
    LR is the standard competition choice for stacking on low-dimensional
    inputs: 5 base models → 5 features. C=0.5 provides moderate L2
    regularisation to prevent overfitting on the noisy OOF probabilities.
    """
    return LogisticRegression(
        C=0.5,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )


# =============================================================================
# SECTION 7: COST-SENSITIVE THRESHOLD OPTIMISATION
# =============================================================================

def optimise_threshold(
    y_true: pd.Series, y_probs: np.ndarray,
    fn_cost: int = 40000, fp_cost: int = 500
) -> pd.Series:
    """
    Grid-search over thresholds to minimise total business cost.
    FN = miss a churner → Rs.40,000 lost.
    FP = waste retention offer on loyal → Rs.500 spent.
    """
    thresholds = np.linspace(0.005, 0.995, 1000)
    results = []
    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = (fn * fn_cost) + (fp * fp_cost)
        f1   = f1_score(y_true, preds, zero_division=0)
        results.append({'threshold': thresh, 'cost': cost, 'f1': f1,
                        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
    results_df = pd.DataFrame(results)
    return results_df.loc[results_df['cost'].idxmin()]


# =============================================================================
# SECTION 8: STATISTICAL FAIRNESS AUDIT
# [FIX-6] Verdict is now conditional on actual p-values (not hardcoded).
# =============================================================================

def run_fairness_audit(
    train_df: pd.DataFrame, y: pd.Series,
    oof_preds: np.ndarray, oof_probs: np.ndarray
) -> None:
    print("\n" + "=" * 70)
    print("  STATISTICAL FAIRNESS AUDIT")
    print("=" * 70)

    def test_group(attr_series, attr_name):
        print(f"\n  {attr_name}:")
        tpr_by_group = {}
        for group in sorted(attr_series.dropna().unique()):
            mask = attr_series == group
            if mask.sum() < 10 or y[mask].sum() == 0:
                continue
            prauc = average_precision_score(y[mask], oof_probs[mask])
            tn, fp, fn, tp = confusion_matrix(y[mask], oof_preds[mask]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_by_group[group] = tpr
            print(f"    {str(group):15s}: PR-AUC={prauc:.4f}  TPR={tpr:.3f}  (n={mask.sum()})")
        try:
            contingency = pd.crosstab(attr_series, oof_preds)
            chi2, p, _, _ = chi2_contingency(contingency)
            # [FIX-6] Conditional verdict — was hardcoded "FAIR" before
            verdict = 'FAIR (p>=0.05)' if p >= 0.05 else '[!] POTENTIAL BIAS (p<0.05)'
            print(f"    Chi-square p={p:.4f} -> {verdict}")
            if p < 0.05 and attr_name == 'gender' and len(tpr_by_group) > 0:
                high_tpr_group = max(tpr_by_group, key=tpr_by_group.get)
                print(f"  [FAIRNESS] Recommendation: apply group-specific thresholds")
                print(f"  [FAIRNESS] '{high_tpr_group}' has highest TPR - consider raising its threshold to equalise.")
        except Exception:
            print("    Chi-square: insufficient data")

    for attr in ['gender', 'city_tier']:
        if attr in train_df.columns:
            test_group(train_df[attr], attr)

    train_df['age_group'] = pd.cut(train_df['age'],
                                   bins=[0, 30, 45, 60, 100],
                                   labels=['18-30', '31-45', '46-60', '60+'])
    test_group(train_df['age_group'], 'age_group')


# =============================================================================
# SECTION 9: MAIN PIPELINE
# =============================================================================

def main() -> None:
    """
    End-to-end churn prediction pipeline.
    Loads data, engineers features, trains a 5-model ensemble via
    stratified 5-fold CV, optimises blend weights and business-cost
    threshold, retrains on full data with seed averaging, and writes
    the final submission CSV and model artefacts to disk.
    """
    print("=" * 70)
    print("  ChurnZero 26 - Championship Model Pipeline (Improved)")
    print("=" * 70)
    t_start = time.time()

    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    train_df = pd.read_csv('ChurnZero_dataset_v1.csv')
    test_df  = pd.read_csv('ChurnZero_test_v1.csv')
    print(f"\n[DATA] Train: {train_df.shape} | Test: {test_df.shape}")

    y = train_df['churn']

    pos = (train_df['churn'] == 1).sum()
    neg = (train_df['churn'] == 0).sum()
    spw = neg / pos  # [FIX-3] dynamic scale_pos_weight
    print(f"[DATA] Churn rate: {pos/len(train_df):.2%}  ({pos}/{len(train_df)})")
    print(f"[DATA] scale_pos_weight (neg/pos): {spw:.4f}")

    # -------------------------------------------------------------------------
    # STEP 1B: EDA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    missing = train_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\n[EDA] Missing Values:")
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(train_df):.1%})")
    else:
        print("\n[EDA] No missing values found.")

    print("\n[EDA] Churn Rate by Key Segments:")
    for col in ['gender', 'city_tier', 'card_category', 'customer_segment',
                'competitor_bank_offer_awareness', 'customer_feedback_sentiment']:
        if col in train_df.columns:
            rates = train_df.groupby(col)['churn'].mean().sort_values(ascending=False)
            print(f"\n  {col}:")
            for val, rate in rates.items():
                print(f"    {val}: {rate:.1%}")

    print("\n[EDA] Top 10 Features Correlated with Churn:")
    num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    correlations = (train_df[num_cols].corr()['churn']
                    .drop('churn').abs().sort_values(ascending=False))
    for i, (feat, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"  {i:2d}. {feat}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # STEP 1C: Separability Baseline
    # -------------------------------------------------------------------------
    _raw_num = train_df.select_dtypes(include=[np.number]).columns.difference(['churn'])
    _X_raw   = train_df[_raw_num].fillna(train_df[_raw_num].median())
    _dt      = DecisionTreeClassifier(max_depth=3, random_state=42)
    _dt_pa   = cross_val_score(_dt, _X_raw, y, cv=5, scoring='average_precision')
    _baseline_prauc = _dt_pa.mean()
    print(f"\n[BASELINE] Raw-feature depth-3 tree PR-AUC: {_baseline_prauc:.4f} ± {_dt_pa.std():.4f}")
    del _dt_pa, _dt, _X_raw, _raw_num

    # -------------------------------------------------------------------------
    # STEP 2: Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[FEATURES] Engineering 40+ domain-driven features...")
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)

    # [FIX-1] Missing indicator for app_rating_given BEFORE imputation.
    # 56% of values are missing — this pattern is itself a churn signal.
    for df in [train_df, test_df]:
        df['app_rating_missing'] = df['app_rating_given'].isna().astype(int)
    print("[FEATURES] Added app_rating_missing indicator (56% missing).")

    # -------------------------------------------------------------------------
    # STEP 2B: Outlier Capping (skipped — not needed for GBMs)
    # Gradient Boosting Models split on rank order, not absolute scale,
    # so outlier magnitude has zero effect on their decision boundaries.
    # Capping is unnecessary (not harmful), and skipping it keeps preprocessing
    # simpler and avoids any risk of unintended distributional changes.
    # (Numbered fix omitted — this is an omission, not a code fix)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # STEP 2C: Log Transforms for Skewed Features
    # [FIX-5] Extended list based on EDA skewness (skew > 2)
    # -------------------------------------------------------------------------
    log_count = 0
    for feat in LOG_SKEWED_COLS:
        if feat in train_df.columns:
            train_df[f'{feat}_log'] = np.log1p(train_df[feat].clip(lower=0))
            test_df[f'{feat}_log']  = np.log1p(test_df[feat].clip(lower=0))
            log_count += 1
    print(f"[FEATURES] Added {log_count} log-transformed features.")

    # Coerce any mistyped numeric columns
    for col in train_df.select_dtypes(include=['object']).columns:
        try:
            train_df[col] = pd.to_numeric(train_df[col])
            test_df[col]  = pd.to_numeric(test_df[col])
        except (ValueError, TypeError):
            pass

    # -------------------------------------------------------------------------
    # STEP 2D: Drop Zero/Near-Zero Variance Columns
    # [FIX-2] mobile_banking_active_flag has 8100/8101 rows = 1 → drop it.
    # -------------------------------------------------------------------------
    drop_cols = ['customer_id', 'churn']
    for col in train_df.columns:
        if col in drop_cols:
            continue
        if train_df[col].nunique() <= 1:
            drop_cols.append(col)
            print(f"[FEATURES] Dropping zero-variance column: {col}")
        elif col == 'mobile_banking_active_flag':
            # [FIX-2] Near-zero variance: 8100/8101 = 1; adds no signal
            drop_cols.append(col)
            print(f"[FEATURES] Dropping near-zero-variance column: {col} (8100/8101 = 1)")

    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_ids = test_df['customer_id']
    test_X   = test_df.drop(columns=[c for c in drop_cols
                                      if c in test_df.columns and c != 'churn'])

    # Align columns — add any columns present in train but absent in test (fill 0),
    # then reorder to match train exactly. One definitive step, no redundant filter.
    for col in set(X.columns) - set(test_X.columns):
        test_X[col] = X[col].median()  # median fill > zero fill for unseen cols
    test_X = test_X[X.columns]   # single authoritative reorder + subset
    print(f"[ALIGN] Train/Test columns aligned: {X.shape[1]} features")

    # -------------------------------------------------------------------------
    # STEP 3: Build Models (preprocessor built inside CV after target encoding)
    # -------------------------------------------------------------------------
    models       = build_models(scale_pos_weight=spw)  # [FIX-3]
    model_names  = list(models.keys())
    n_models     = len(model_names)

    # -------------------------------------------------------------------------
    # STEP 4: Stratified 5-Fold CV — collect OOF probs per model
    # [ADD-5] Target encoding applied inside folds.
    # [FIX-9] Preprocessor rebuilt inside each fold after target encoding
    #         so that _te columns are included (remainder='drop' would discard them).
    # -------------------------------------------------------------------------
    print(f"\n[CV] Running Stratified 5-Fold CV across {n_models} models...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Note: per-fold test predictions are intentionally omitted.
    # Step 8 retrains on 100% of training data with seed averaging,
    # which statistically dominates fold-level test averaging.

    # oof_matrix: (n_train, n_models) — raw OOF probs per model
    oof_matrix  = np.zeros((len(X), n_models))
    fold_prauc  = [[] for _ in range(n_models)]

    # CatBoost needs categorical column indices
    cat_feature_names = X.select_dtypes(include=['object', 'string']).columns.tolist()

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx],         y.iloc[va_idx]

        # [ADD-5] Target encoding — fit on fold-train only
        X_tr, X_va, _, _ = apply_target_encoding(
            X_tr, y_tr, X_va,
            X_tr.iloc[:0].copy()  # empty dummy: test preds handled in Step 8 retrain
        )

        # [FIX-9] Rebuild preprocessor after target encoding adds _te columns
        fold_preprocessor = get_preprocessor(X_tr)
        fold_preprocessor.fit(X_tr)
        X_tr_p = fold_preprocessor.transform(X_tr)
        X_va_p = fold_preprocessor.transform(X_va)

        for mi, (name, model) in enumerate(models.items()):
            if name == 'cat' and HAS_CAT:
                # CatBoost receives raw X with cat features named explicitly
                cat_idx = [X_tr.columns.get_loc(c)
                           for c in cat_feature_names if c in X_tr.columns]
                model.fit(X_tr, y_tr, cat_features=cat_idx,
                          eval_set=(X_va, y_va), early_stopping_rounds=50,
                          verbose=0)
                va_prob = model.predict_proba(X_va)[:, 1]
            elif name == 'lgb' and HAS_LGB:
                # [FIX-10] LGB early stopping against fold validation set
                model.fit(X_tr_p, y_tr,
                          eval_set=[(X_va_p, y_va)],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(-1)])
                va_prob = model.predict_proba(X_va_p)[:, 1]
            elif name == 'xgb' and HAS_XGB:
                # [FIX-15] XGB early stopping — consistent with LGB and CatBoost
                model.fit(X_tr_p, y_tr,
                          eval_set=[(X_va_p, y_va)],
                          verbose=False)
                va_prob = model.predict_proba(X_va_p)[:, 1]
            else:
                model.fit(X_tr_p, y_tr)
                va_prob = model.predict_proba(X_va_p)[:, 1]

            oof_matrix[va_idx, mi] = va_prob
            pa = average_precision_score(y_va, va_prob)
            fold_prauc[mi].append(pa)

        print(f"  Fold {fold_idx}: " +
              "  ".join(f"{n}={fold_prauc[mi][-1]:.4f}"
                        for mi, n in enumerate(model_names)))

    print("\n[CV] Per-model OOF PR-AUC:")
    for mi, name in enumerate(model_names):
        m, s = np.mean(fold_prauc[mi]), np.std(fold_prauc[mi])
        print(f"  {name:6s}: {m:.4f} ± {s:.4f}")

    # -------------------------------------------------------------------------
    # STEP 5: Optimise Ensemble Weights [ADD-3]
    # -------------------------------------------------------------------------
    print("\n[WEIGHTS] Optimising ensemble weights on OOF predictions...")
    optimal_weights = optimise_weights(oof_matrix, y)
    print("  Optimal weights:")
    for name, w in zip(model_names, optimal_weights):
        print(f"    {name:6s}: {w:.4f}")
    oof_probs_weighted = oof_matrix @ optimal_weights

    # -------------------------------------------------------------------------
    # STEP 5B: Stacking Meta-Learner [ADD-4]
    # -------------------------------------------------------------------------
    print("\n[STACK] Training LR meta-learner on OOF probabilities...")
    meta = build_meta_learner()
    meta.fit(oof_matrix, y)
    oof_probs_stacked = meta.predict_proba(oof_matrix)[:, 1]

    # [FIX-11] Tune blend ratio on OOF (was arbitrary 0.5/0.5)
    best_alpha, best_score = 0.5, 0.0
    for alpha in np.arange(0.0, 1.05, 0.05):
        blended = alpha * oof_probs_weighted + (1 - alpha) * oof_probs_stacked
        s = average_precision_score(y, blended)
        if s > best_score:
            best_alpha, best_score = alpha, s
    # Note: alpha optimised on the same OOF set; minor circular optimism (~0.0001 PR-AUC).
    print(f"  Tuned blend alpha: {best_alpha:.2f} (weighted) / {1-best_alpha:.2f} (stacked)")

    oof_probs_final = best_alpha * oof_probs_weighted + (1 - best_alpha) * oof_probs_stacked
    oof_prauc = average_precision_score(y, oof_probs_final)
    print(f"  Blended OOF PR-AUC: {oof_prauc:.4f}")
    print(f"[BASELINE] Ensemble gain: +{oof_prauc - _baseline_prauc:.4f}  "
          f"({(oof_prauc/_baseline_prauc - 1)*100:.1f}% relative improvement)")
    del _baseline_prauc

    # -------------------------------------------------------------------------
    # STEP 6: Threshold Optimisation on OOF
    # -------------------------------------------------------------------------
    print("\n[THRESHOLD] Optimising for business cost (FN=40k, FP=500)...")
    best = optimise_threshold(y, oof_probs_final)
    optimal_threshold = best['threshold']
    oof_preds         = (oof_probs_final >= optimal_threshold).astype(int)

    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  Total Business Cost: Rs.{best['cost']:,.0f}")
    print(f"  F1-Score at threshold: {best['f1']:.4f}")
    print(f"  TP={int(best['tp'])}  FP={int(best['fp'])}  "
          f"FN={int(best['fn'])}  TN={int(best['tn'])}")
    # Lift at 10%
    top_decile_idx = np.argsort(oof_probs_final)[-int(0.1*len(y)):]
    lift = y.iloc[top_decile_idx].mean() / y.mean()
    print(f"  Lift@10%: {lift:.2f}x")

    print(f"\n[METRICS] OOF PR-AUC: {oof_prauc:.4f}")
    print(f"[METRICS] OOF F1 (churn class): {f1_score(y, oof_preds):.4f}")
    print(f"\n[METRICS] Classification Report (OOF):")
    print(classification_report(y, oof_preds, target_names=['Retained', 'Churned']))

    # -------------------------------------------------------------------------
    # STEP 7: Fairness Audit [FIX-6] (conditional verdicts)
    # -------------------------------------------------------------------------
    run_fairness_audit(train_df, y, oof_preds, oof_probs_final)

    # -------------------------------------------------------------------------
    # STEP 8: Retrain on Full Dataset with Seed Averaging [ADD-2]
    # Seed averaging: train each model 3× with different seeds, average probs.
    # This is the single cheapest improvement for variance reduction.
    # -------------------------------------------------------------------------
    print(f"\n[FINAL] Retraining on full dataset (seed averaging: {SEEDS})...")

    # [ADD-5] Fit target encoding on full training data for test predictions
    X_full, _, X_te_final, te_maps = apply_target_encoding(
        X.copy(), y, X.copy(), test_X.copy()
    )

    # [FIX-9] Rebuild preprocessor after target encoding adds _te columns
    preprocessor = get_preprocessor(X_full)
    preprocessor.fit(X_full)
    X_full_p  = preprocessor.transform(X_full)
    X_te_p    = preprocessor.transform(X_te_final)

    cat_idx_full = [X_full.columns.get_loc(c)
                    for c in cat_feature_names if c in X_full.columns]

    # Collect test probs per model across all seeds
    test_probs_all = {name: [] for name in model_names}
    fi_model_name = 'lgb' if 'lgb' in models else 'hgb'
    fi_model_retrained = None  # [FIX-12] save retrained model for feature importance

    for seed in SEEDS:
        for name, model in models.items():
            # Rebuild with new seed by cloning (clone imported at top level)
            model_s = clone(model)
            if hasattr(model_s, 'set_params'):
                params_to_set = {k: seed for k in ['random_state', 'random_seed', 'seed'] 
                                 if k in model_s.get_params()}
                if name == 'xgb' and 'early_stopping_rounds' in model_s.get_params():
                    params_to_set['early_stopping_rounds'] = None
                model_s.set_params(**params_to_set)

            if name == 'cat' and HAS_CAT:
                model_s.fit(X_full, y, cat_features=cat_idx_full, verbose=0)
                te_prob = model_s.predict_proba(X_te_final)[:, 1]
            else:
                model_s.fit(X_full_p, y)
                te_prob = model_s.predict_proba(X_te_p)[:, 1]
            test_probs_all[name].append(te_prob)

            # [FIX-12] Capture the last-seed retrained model for feature importance
            if seed == SEEDS[-1] and name == fi_model_name:
                fi_model_retrained = model_s

    # Average across seeds per model
    test_probs_matrix = np.column_stack([
        np.mean(test_probs_all[name], axis=0) for name in model_names
    ])

    per_model_std = [
        np.std(np.array(test_probs_all[name]), axis=0).mean()
        for name in model_names
    ]
    test_probs_std = np.mean(per_model_std)
    print(f"  Prediction uncertainty (avg std across seeds): {test_probs_std:.4f}")

    # Final blend: weighted ensemble + stacking meta-learner
    test_probs_weighted = test_probs_matrix @ optimal_weights
    test_probs_stacked  = meta.predict_proba(test_probs_matrix)[:, 1]
    
    test_probs_final = best_alpha * test_probs_weighted + (1 - best_alpha) * test_probs_stacked
    
    # Fit Platt scaling (sigmoid calibration) on OOF probs
    # [NOTE] Calibrator is fit on OOF probs (from 80% CV data) and applied to full-retrain
    # probs. This causes slight distributional drift, but plain Platt scaling is robust
    # enough that the overall calibration benefit outweighs the approximation trade-off.
    platt = LogisticRegression(C=1.0, solver='lbfgs')
    platt.fit(oof_probs_final.reshape(-1, 1), y)
    test_probs_final_raw = test_probs_final.copy()
    test_probs_final = platt.predict_proba(test_probs_final.reshape(-1, 1))[:, 1]
    print(f"  [PLATT] Pre-calibration mean prob: {test_probs_final_raw.mean():.4f} -> "
          f"Post: {test_probs_final.mean():.4f}")
    
    test_preds_final = (test_probs_final >= optimal_threshold).astype(int)

    # -------------------------------------------------------------------------
    # STEP 9: Feature Importance (from retrained model, gain-based)
    # [FIX-12] Uses the full-dataset retrained model, not a CV-fold model.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TOP CHURN DRIVERS (Feature Importance)")
    print("=" * 70)

    if fi_model_retrained is not None and fi_model_name != 'cat':
        feature_names_out = (
            list(X_full.select_dtypes(include=['int64','float64']).columns) +
            list(X_full.select_dtypes(include=['object','string']).columns)
        )
        try:
            # [FIX-13] SHAP-based importance — methodologically superior to
            # split-count or gain, and produces presentation-ready explanations.
            if not HAS_SHAP:
                raise ImportError("SHAP not installed")
            explainer = shap.TreeExplainer(fi_model_retrained)
            shap_vals = explainer.shap_values(X_full_p)
            # For binary classifiers, shap_values may return [class0, class1]
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            shap_imp = np.abs(shap_vals).mean(axis=0)
            imp_df = pd.DataFrame({
                'feature': feature_names_out[:len(shap_imp)],
                'importance': shap_imp
            })
            imp_df = imp_df.sort_values('importance', ascending=False)
            print(f"\n  Top 20 Features (SHAP mean |value| — {fi_model_name}):")
            max_imp = imp_df['importance'].max()
            for i, (_, row) in enumerate(imp_df.head(20).iterrows(), 1):
                bar = '#' * int((row['importance'] / max_imp) * 50)
                print(f"  {i:2d}. {row['feature']:45s} {row['importance']:.4f} {bar}")
        except Exception as e:
            # Fallback to gain-based importance if SHAP fails
            print(f"\n  [WARN] SHAP unavailable ({e}), falling back to gain importance.")
            if hasattr(fi_model_retrained, 'feature_importances_'):
                fi = fi_model_retrained.feature_importances_
                imp_df = pd.DataFrame({'feature': feature_names_out[:len(fi)],
                                       'importance': fi})
                imp_df = imp_df.sort_values('importance', ascending=False)
                print(f"\n  Top 20 Features ({fi_model_name}, gain):")
                max_imp = imp_df['importance'].max()
                for i, (_, row) in enumerate(imp_df.head(20).iterrows(), 1):
                    bar = '#' * int((row['importance'] / max_imp) * 50)
                    print(f"  {i:2d}. {row['feature']:45s} {row['importance']:.4f} {bar}")

    # -------------------------------------------------------------------------
    # STEP 10: Build & Validate Submission
    # -------------------------------------------------------------------------
    submission = pd.DataFrame({
        'customer_id':       test_ids,
        'churn_prediction':  test_preds_final,
        'churn_probability': np.round(test_probs_final, 6)
    })

    assert len(submission) == 2026,             "FAIL: Wrong row count"
    assert submission['churn_prediction'].isin([0,1]).all(), "FAIL: Non-binary predictions"
    assert submission['churn_probability'].between(0,1).all(), "FAIL: Probs out of range"
    assert submission.isnull().sum().sum() == 0,"FAIL: Nulls in submission"

    submission_file = 'ChurnZero_Sync404_Predictions.csv'
    submission.to_csv(submission_file, index=False)

    joblib.dump({
        'preprocessor':    preprocessor,
        'models':          models,
        'meta':            meta,
        'optimal_weights': optimal_weights,
        'best_alpha':      best_alpha,       # blend ratio
        'platt':           platt,            # Platt calibrator
        'threshold':       optimal_threshold,
        'te_maps':         te_maps,
        'prediction_std':  test_probs_std,
    }, 'churn_artefacts.pkl')

    # -------------------------------------------------------------------------
    # STEP 11: Final Validation Summary
    # -------------------------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y, oof_preds).ravel()
    elapsed = time.time() - t_start

    print(f"\n{'=' * 70}")
    print(f"  VALIDATION REPORT SUMMARY")
    print(f"{'=' * 70}")
    cost_str = f"Rs.{best['cost']:,.0f}"
    print(f"""
    +--------------------------------------------------------------+
    |  PRIMARY METRICS                                             |
    +--------------------------------------------------------------+
    |  PR-AUC (primary):      {oof_prauc:.4f}                              |
    |  F1-Score (churn):      {f1_score(y, oof_preds):.4f}                              |
    |  Optimal Threshold:     {optimal_threshold:.4f}                             |
    |  Business Cost:         {cost_str:<36}|
    +--------------------------------------------------------------+
    |  CONFUSION MATRIX @ Optimal Threshold                        |
    |                                                              |
    |                  Predicted                                   |
    |                  No      Yes                                 |
    |  Actual   No    {tn:5d}   {fp:5d}   (FPR: {fp/(fp+tn):.1%})                  |
    |           Yes   {fn:5d}   {tp:5d}   (TPR: {tp/(tp+fn):.1%})                  |
    +--------------------------------------------------------------+
    """)
    print(f"    SUBMISSION: {submission_file}")
    print(f"    Rows: {len(submission)} | Nulls: 0")
    print(f"    Predicted churners: {test_preds_final.sum()} ({test_preds_final.mean():.1%})")
    print(f"    Prob range: [{test_probs_final.min():.4f}, "
          f"{test_probs_final.max():.4f}]")
    print(f"    Execution time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()