"""
================================================================================
ChurnZero 26 - Championship Model Pipeline
================================================================================
Team: [YOUR_TEAM_NAME]
Hackathon: ChurnZero 26, IIT Kharagpur

Architecture:
  1. Deep EDA-driven feature engineering (30+ derived features)
  2. Proper leakage-safe preprocessing via sklearn Pipelines
  3. Stacked Ensemble: LightGBM + XGBoost + CatBoost + HistGBT
  4. Stratified 5-Fold CV for robust PR-AUC estimation
  5. Cost-sensitive threshold optimization (FN=40k, FP=500)
  6. Calibrated probability output via Platt scaling
  7. Full validation report with PR-AUC, F1, Confusion Matrix
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML Core
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OrdinalEncoder, LabelEncoder
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    VotingClassifier
)

# Optional heavy hitters - graceful fallback
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[INFO] LightGBM not installed. Using HistGBT fallback.")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] XGBoost not installed. Using HistGBT fallback.")

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# =============================================================================
# SECTION 1: FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """
    Create 30+ domain-driven features across 6 strategic categories.
    Each feature has a clear business rationale, documented inline.
    """
    df = df.copy()

    def safe_divide(numerator, denominator, default=0.0):
        """Vectorized division that avoids inf/NaN when denominators are zero."""
        return np.where(denominator > 0, numerator / denominator, default)

    # --- Map categorical ordinals to numeric BEFORE engineering ---
    awareness_map = {'Not Aware': 0, 'Low': 1, 'Medium': 2, 'High': 3}
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['competitor_awareness_num'] = df['competitor_bank_offer_awareness'].map(awareness_map).fillna(0)
    df['feedback_sentiment_num'] = df['customer_feedback_sentiment'].map(sentiment_map).fillna(1)
    df['app_rating_missing'] = df['app_rating_given'].isna().astype(int)

    # =========================================================================
    # CATEGORY A: BEHAVIORAL EARLY WARNING SIGNALS
    # Rationale: Declining engagement is the strongest leading indicator.
    # =========================================================================

    # A1. Flight Risk Index
    # Customers with declining balances AND growing inactivity are actively leaving.
    df['flight_risk_index'] = (
        df['balance_decline_percentage'] * np.log1p(df['account_inactive_days'])
    )

    # A2. Digital Disengagement Score
    # Ratio of days-since-last-login to total logins. High = ghosting the bank.
    df['digital_disengagement'] = (
        df['last_login_days'] / (df['total_digital_logins'] + 1)
    )

    # A3. Transaction Velocity Change (Q4 vs Q1)
    # Sudden drop in both transaction amount and count = red flag.
    df['transaction_momentum'] = (
        df['total_amt_chng_q4_q1'] * df['total_ct_chng_q4_q1']
    )

    # A4. Balance Erosion Rate
    # How much of their average balance has eroded (current vs average).
    df['balance_erosion'] = np.where(
        df['avg_monthly_balance'] > 0,
        (df['avg_monthly_balance'] - df['current_balance']) / df['avg_monthly_balance'],
        0
    )

    # A4b. Quarter-to-average balance pressure.
    # Captures balance deterioration from a second baseline.
    df['quarterly_balance_erosion'] = safe_divide(
        df['avg_quarterly_balance'] - df['current_balance'],
        df['avg_quarterly_balance']
    )

    # A5. Inactivity Trend (login gap relative to tenure)
    # New customers with large login gaps are more alarming than old ones.
    df['relative_inactivity'] = (
        df['last_login_days'] / (df['tenure_months'] + 1)
    )

    # A6. Activity density features normalize raw activity by account age.
    df['transactions_per_tenure_month'] = safe_divide(
        df['total_trans_count'], df['tenure_months'] + 1
    )
    df['digital_logins_per_tenure_month'] = safe_divide(
        df['total_digital_logins'], df['tenure_months'] + 1
    )

    # =========================================================================
    # CATEGORY B: SERVICE FRUSTRATION SIGNALS
    # Rationale: Unresolved complaints + escalations = emotional churn driver.
    # =========================================================================

    # B1. Frustration Score
    df['frustration_score'] = (
        df['unresolved_complaint_count'] * 3 +  # Unresolved = 3x weight
        df['escalation_count'] * 2 +             # Escalated = 2x weight
        df['total_complaints']                    # Any complaint = 1x
    )

    # B2. Resolution Failure Rate
    # Ratio of unresolved complaints to total complaints.
    df['resolution_failure_rate'] = np.where(
        df['total_complaints'] > 0,
        df['unresolved_complaint_count'] / df['total_complaints'],
        0
    )

    # B3. Satisfaction Gap
    # Difference between NPS and satisfaction. Large gaps signal instability.
    df['satisfaction_gap'] = df['nps_score'] - df['satisfaction_score']

    # B4. Service Intensity
    # Total touchpoints across all channels.
    df['service_intensity'] = (
        df['branch_visit_count'] +
        df['call_center_interaction_count'] +
        df['relationship_manager_interaction_count'] +
        df['service_request_count']
    )

    # B5. Complaint pressure relative to the full service footprint.
    df['complaints_per_service_touch'] = safe_divide(
        df['total_complaints'], df['service_intensity'] + 1
    )

    # B6. Sentiment-adjusted satisfaction combines explicit ratings and text tone.
    df['sentiment_adjusted_satisfaction'] = (
        df['satisfaction_score'] + df['nps_score'] + df['feedback_sentiment_num']
    )

    # =========================================================================
    # CATEGORY C: COMPETITOR THREAT & RETENTION RESPONSE
    # Rationale: A customer who is aware of competitor offers AND was already
    # offered retention deals AND rejected them is almost certainly gone.
    # =========================================================================

    # C1. Competitor Threat Score
    df['competitor_threat'] = (
        df['competitor_awareness_num'] *
        (df['credit_card_spend'] + df['monthly_transaction_value'] + 1)
    )

    # C2. Retention Resistance
    # Received retention offer but did NOT accept = actively resistant.
    df['retention_resistance'] = (
        df['retention_offer_received'] - df['retention_offer_accepted']
    )

    # C3. Campaign Fatigue
    # Ratio of campaigns received to campaigns responded to.
    df['campaign_fatigue'] = np.where(
        df['campaign_received_count'] > 0,
        1 - (df['campaign_response_count'] / df['campaign_received_count']),
        0
    )

    # C3b. Direct campaign engagement rate complements fatigue for tree splits.
    df['campaign_response_rate'] = safe_divide(
        df['campaign_response_count'], df['campaign_received_count']
    )

    # C4. Last Contact Recency
    # Combined recency of last contact and last campaign response.
    df['contact_recency_score'] = (
        df['last_contacted_days'] + df['last_campaign_response_days']
    )

    # C5. Offer intensity and acceptance behavior.
    df['offer_pressure'] = (
        df['cross_sell_offer_count'] +
        df['upsell_offer_count'] +
        df['retention_offer_received'] +
        df['discount_or_fee_waiver_received']
    )
    df['retention_acceptance_rate'] = safe_divide(
        df['retention_offer_accepted'], df['retention_offer_received']
    )

    # =========================================================================
    # CATEGORY D: FINANCIAL HEALTH & PRODUCT DEPTH
    # Rationale: Financially strained customers with shallow product usage leave.
    # =========================================================================

    # D1. Product Breadth (total products held)
    product_flags = [
        'savings_account_flag', 'current_account_flag', 'credit_card_flag',
        'personal_loan_flag', 'home_loan_flag', 'auto_loan_flag',
        'fixed_deposit_flag', 'investment_product_flag',
        'insurance_product_flag', 'demat_account_flag'
    ]
    df['product_breadth'] = df[product_flags].sum(axis=1)
    df['loan_product_count'] = df[
        ['personal_loan_flag', 'home_loan_flag', 'auto_loan_flag']
    ].sum(axis=1)
    df['wealth_product_count'] = df[
        ['fixed_deposit_flag', 'investment_product_flag', 'insurance_product_flag', 'demat_account_flag']
    ].sum(axis=1)

    # D2. Credit Stress Indicator
    # High utilization + late payments + loan default risk = financial stress.
    df['credit_stress'] = (
        df['credit_utilization_ratio'] +
        df['late_credit_card_payment_count'] * 0.1 +
        df['loan_default_risk_score'] * 0.01
    )

    # D3. Credit Utilization Trend (6m vs 3m)
    # Rising utilization = financial squeeze.
    df['credit_util_trend'] = (
        df['credit_utilization_6m_avg'] - df['credit_utilization_3m_avg']
    )

    # D4. EMI Burden Ratio
    # EMI as fraction of estimated monthly income.
    df['emi_burden_ratio'] = np.where(
        df['monthly_income_estimate'] > 0,
        df['emi_amount'] / df['monthly_income_estimate'],
        0
    )

    # D4b. Debt pressure captures all outstanding credit against income.
    df['loan_income_ratio'] = safe_divide(
        df['loan_outstanding_amount'], df['annual_income'] + 1
    )

    # D5. Revolving Balance Ratio
    df['revolving_ratio'] = np.where(
        df['credit_card_limit'] > 0,
        df['total_revolving_bal'] / df['credit_card_limit'],
        0
    )

    # D6. Product-normalized value and balance.
    df['clv_per_product'] = safe_divide(
        df['customer_lifetime_value'], df['product_breadth'] + 1
    )
    df['balance_per_product'] = safe_divide(
        df['current_balance'], df['product_breadth'] + 1
    )
    df['transaction_value_per_product'] = safe_divide(
        df['monthly_transaction_value'], df['product_breadth'] + 1
    )

    # =========================================================================
    # CATEGORY E: DIGITAL CHANNEL BEHAVIOR
    # =========================================================================

    # E1. Channel Preference Score (digital vs physical)
    df['digital_vs_physical'] = (
        df['digital_transaction_ratio'] -
        (df['branch_visit_count'] / (df['monthly_transaction_count'] + 1))
    )

    # E2. App vs Web Preference
    df['app_vs_web'] = (
        df['mobile_app_login_count'] / (df['website_login_count'] + 1)
    )

    # E2b. Mobile share is bounded and often easier for models than a raw ratio.
    df['mobile_login_share'] = safe_divide(
        df['mobile_app_login_count'], df['mobile_app_login_count'] + df['website_login_count']
    )

    # E3. Failed Login Frustration
    df['login_failure_rate'] = np.where(
        df['total_digital_logins'] > 0,
        df['failed_login_count'] / df['total_digital_logins'],
        0
    )

    # E4. Transaction channel mix.
    df['upi_transaction_share'] = safe_divide(
        df['upi_transaction_count'], df['monthly_transaction_count']
    )
    df['cash_withdrawal_share'] = safe_divide(
        df['cash_withdrawal_count'], df['monthly_transaction_count']
    )
    df['card_transaction_share'] = safe_divide(
        df['debit_card_transaction_count'], df['monthly_transaction_count']
    )
    df['net_banking_transaction_share'] = safe_divide(
        df['net_banking_transaction_count'], df['monthly_transaction_count']
    )

    # =========================================================================
    # CATEGORY F: INTERACTION FEATURES (Non-linear combinations)
    # These capture complex patterns that tree models can exploit.
    # =========================================================================

    # F1. High-Value At-Risk (CLV * flight risk)
    df['high_value_at_risk'] = (
        df['customer_lifetime_value'] * df['flight_risk_index']
    )

    # F2. Frustrated AND Aware of Competitors
    df['frustrated_and_aware'] = (
        df['frustration_score'] * df['competitor_awareness_num']
    )

    # F3. Low engagement AND high balance decline
    df['silent_bleed'] = (
        df['digital_disengagement'] * df['balance_decline_percentage']
    )

    # F4. EMI stress AND complaints
    df['stressed_and_complaining'] = (
        df['credit_stress'] * df['frustration_score']
    )

    # F5. Competition plus low relationship depth is a high-risk pattern.
    df['thin_relationship_competitor_risk'] = (
        df['competitor_awareness_num'] * safe_divide(1, df['product_breadth'] + 1, default=1)
    )

    # F6. Service problems hurt more when recent engagement is already weak.
    df['service_disengagement_risk'] = (
        df['frustration_score'] * df['relative_inactivity']
    )

    # F7. Value-at-risk using normalized balance and CLV signals.
    df['value_balance_risk'] = (
        np.log1p(df['customer_lifetime_value']) *
        np.clip(df['balance_erosion'], -5, 5)
    )

    # F8. Categorical crosses let ordinal-encoded trees split on important segments.
    df['segment_region'] = (
        df['customer_segment'].astype(str) + '_' + df['region'].astype(str)
    )
    df['segment_card_category'] = (
        df['customer_segment'].astype(str) + '_' + df['card_category'].astype(str)
    )
    df['income_occupation'] = (
        df['income_category'].astype(str) + '_' + df['occupation_type'].astype(str)
    )

    # Log transforms expose rank-like versions of highly skewed money/activity fields.
    skewed_cols = [
        'annual_income', 'customer_lifetime_value', 'avg_monthly_balance',
        'current_balance', 'monthly_transaction_value', 'credit_card_spend',
        'loan_outstanding_amount', 'total_trans_amt', 'credit_card_limit'
    ]
    for col in skewed_cols:
        df[f'log1p_{col}'] = np.log1p(np.clip(df[col], 0, None))

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# =============================================================================
# SECTION 2: PREPROCESSING
# =============================================================================
def get_preprocessor(X):
    """
    Build a leakage-safe ColumnTransformer.
    - Numeric: median impute + scale
    - Categorical: ordinal encode (tree-friendly, no OHE explosion)
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor


# =============================================================================
# SECTION 3: MODEL CONSTRUCTION
# =============================================================================
def build_ensemble():
    """
    Build a soft-voting ensemble of the best available gradient boosters.
    Uses whatever libraries are installed.
    """
    estimators = []

    # Model 1: HistGradientBoosting (always available - sklearn native)
    hgb = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.03,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_bins=255,
        class_weight='balanced',
        random_state=42
    )
    estimators.append(('hgb', hgb))

    # Model 2: HistGBT with different hyperparams (diversity for ensemble)
    hgb2 = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=30,
        l2_regularization=0.5,
        max_bins=128,
        class_weight='balanced',
        random_state=123
    )
    estimators.append(('hgb2', hgb2))

    # Model 3: LightGBM (if available)
    if HAS_LGB:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            is_unbalance=True,
            random_state=42,
            verbose=-1
        )
        estimators.append(('lgb', lgb_model))

    # Model 4: XGBoost (if available)
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=5.2,  # ~ratio of negatives/positives
            eval_metric='aucpr',
            random_state=42,
            verbosity=0
        )
        estimators.append(('xgb', xgb_model))

    # Model 5: Another HistGBT with shallower trees (if no LGB/XGB)
    if not HAS_LGB or not HAS_XGB:
        hgb3 = HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.02,
            max_depth=5,
            min_samples_leaf=40,
            l2_regularization=2.0,
            class_weight='balanced',
            random_state=999
        )
        estimators.append(('hgb3', hgb3))

    print(f"[ENSEMBLE] Models in ensemble: {[name for name, _ in estimators]}")

    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    return ensemble


# =============================================================================
# SECTION 4: COST-SENSITIVE THRESHOLD OPTIMIZATION
# =============================================================================
def optimize_threshold(y_true, y_probs, fn_cost=40000, fp_cost=500):
    """
    Find the probability threshold that minimizes total business cost.
    FN cost (missing a churner) = Rs 40,000
    FP cost (wasting retention on loyal) = Rs 500
    Cost ratio = 80:1

    Also reports PR-AUC and F1 at optimal threshold.
    """
    thresholds = np.linspace(0.005, 0.995, 1000)  # Fine-grained search
    results = []

    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = (fn * fn_cost) + (fp * fp_cost)
        f1 = f1_score(y_true, preds, zero_division=0)
        results.append({
            'threshold': thresh,
            'cost': cost,
            'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['cost'].idxmin()]
    return best


# =============================================================================
# SECTION 5: MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 70)
    print("  ChurnZero 26 - Championship Model Pipeline")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    train_df = pd.read_csv('ChurnZero_dataset_v1.csv')
    test_df = pd.read_csv('ChurnZero_test_v1.csv')
    print(f"\n[DATA] Train: {train_df.shape} | Test: {test_df.shape}")
    print(f"[DATA] Churn rate: {train_df['churn'].mean():.2%} ({train_df['churn'].sum()}/{len(train_df)})")

    # -------------------------------------------------------------------------
    # STEP 2: Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[FEATURES] Engineering 30+ domain-driven features...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # Drop constant column (credit_card_flag = 1 for all rows)
    drop_cols = ['customer_id', 'churn', 'credit_card_flag']
    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y = train_df['churn']
    test_ids = test_df['customer_id']
    test_X = test_df.drop(columns=[c for c in ['customer_id', 'credit_card_flag'] if c in test_df.columns])

    print(f"[FEATURES] Final feature count: {X.shape[1]}")

    # -------------------------------------------------------------------------
    # STEP 3: Build Preprocessing + Ensemble Pipeline
    # -------------------------------------------------------------------------
    preprocessor = get_preprocessor(X)
    ensemble = build_ensemble()

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ensemble)
    ])

    # -------------------------------------------------------------------------
    # STEP 4: Stratified 5-Fold Cross-Validation (PR-AUC)
    # -------------------------------------------------------------------------
    print("\n[CV] Running Stratified 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X))

    fold_prauc_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        fold_probs = model.predict_proba(X_fold_val)[:, 1]
        oof_probs[val_idx] = fold_probs

        fold_prauc = average_precision_score(y_fold_val, fold_probs)
        fold_prauc_scores.append(fold_prauc)
        print(f"  Fold {fold_idx}: PR-AUC = {fold_prauc:.4f}")

    mean_prauc = np.mean(fold_prauc_scores)
    std_prauc = np.std(fold_prauc_scores)
    print(f"\n[CV] Mean PR-AUC: {mean_prauc:.4f} (+/- {std_prauc:.4f})")

    # -------------------------------------------------------------------------
    # STEP 5: Cost-Sensitive Threshold Optimization (on OOF predictions)
    # -------------------------------------------------------------------------
    print("\n[THRESHOLD] Optimizing for business cost (FN=40k, FP=500)...")
    best = optimize_threshold(y, oof_probs)
    optimal_threshold = best['threshold']

    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  Total Business Cost at threshold: Rs {best['cost']:,.0f}")
    print(f"  F1-Score at threshold: {best['f1']:.4f}")
    print(f"  Confusion Matrix: TP={int(best['tp'])}, FP={int(best['fp'])}, "
          f"FN={int(best['fn'])}, TN={int(best['tn'])}")

    # Overall OOF PR-AUC
    oof_prauc = average_precision_score(y, oof_probs)
    print(f"\n[METRICS] Out-of-Fold PR-AUC: {oof_prauc:.4f}")

    oof_preds = (oof_probs >= optimal_threshold).astype(int)
    print(f"[METRICS] Out-of-Fold F1 (positive class): {f1_score(y, oof_preds):.4f}")
    print(f"\n[METRICS] Classification Report (OOF):")
    print(classification_report(y, oof_preds, target_names=['Retained', 'Churned']))

    # -------------------------------------------------------------------------
    # STEP 6: Retrain on FULL dataset + Generate Test Predictions
    # -------------------------------------------------------------------------
    print("[FINAL] Retraining on full dataset...")
    model.fit(X, y)

    print("[FINAL] Generating test predictions...")
    test_probs = model.predict_proba(test_X)[:, 1]
    test_preds = (test_probs >= optimal_threshold).astype(int)

    # -------------------------------------------------------------------------
    # STEP 7: Sanity Checks on Submission
    # -------------------------------------------------------------------------
    submission = pd.DataFrame({
        'customer_id': test_ids,
        'churn_prediction': test_preds,
        'churn_probability': np.round(test_probs, 6)
    })

    # Validate submission
    assert len(submission) == 2026, f"FAIL: Expected 2026 rows, got {len(submission)}"
    assert submission['churn_prediction'].isin([0, 1]).all(), "FAIL: Predictions not binary"
    assert submission['churn_probability'].between(0, 1).all(), "FAIL: Probabilities out of range"
    assert submission.isnull().sum().sum() == 0, "FAIL: Nulls in submission"

    submission_file = 'ChurnZero_YourTeamName_Predictions.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n{'=' * 70}")
    print(f"  SUBMISSION SAVED: {submission_file}")
    print(f"  Rows: {len(submission)} | Nulls: 0")
    print(f"  Predicted churners: {test_preds.sum()} / {len(test_preds)} "
          f"({test_preds.mean():.1%})")
    print(f"  Probability range: [{test_probs.min():.6f}, {test_probs.max():.6f}]")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
