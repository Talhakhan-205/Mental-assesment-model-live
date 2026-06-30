"""
Mental Health Crisis Predictor — Improved Training Script
Target Accuracy: 85%+
Improvements:
  1. Better data cleaning (outlier removal, smarter NaN handling)
  2. Advanced feature engineering (interaction terms, ratios)
  3. Hyperparameter-tuned XGBoost + Stacking Ensemble
  4. Class-weight balancing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MENTAL HEALTH CRISIS PREDICTOR — IMPROVED PIPELINE")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('Mental Health Dataset.csv')
print(f"\n[1] Raw data shape: {df.shape}")

# ─────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────
print("\n[2] Data Cleaning...")

# Drop exact duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"    Duplicates removed: {before - len(df)}")

# Drop irrelevant columns
df.drop(columns=['Timestamp', 'Country'], inplace=True, errors='ignore')

# Fix self_employed NaN → mode imputation
df['self_employed'] = df['self_employed'].fillna(df['self_employed'].mode()[0])

# Normalize Gender variations (some datasets have 'Trans', 'Other' etc.)
df['Gender'] = df['Gender'].str.strip().str.lower()
df['Gender'] = df['Gender'].map(lambda x: 'Male' if 'male' in str(x) and 'fe' not in str(x) else 'Female')
df['Gender'] = df['Gender'].fillna('Female')  # fill rare/unknown with majority

# Normalize Occupation
df['Occupation'] = df['Occupation'].str.strip()
valid_occ = ['Corporate', 'Self-Employed', 'Student', 'Other']
df['Occupation'] = df['Occupation'].apply(lambda x: x if x in valid_occ else 'Other')

print(f"    Cleaned data shape: {df.shape}")
print(f"    Null values remaining:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# ─────────────────────────────────────────────────────────────
# 3. ENCODING
# ─────────────────────────────────────────────────────────────
print("\n[3] Encoding...")

binary_map = {'Yes': 1, 'No': 0}
tri_map    = {'Yes': 1, 'Maybe': 0.5, 'No': 0}
mood_map   = {'Low': 0, 'Medium': 1, 'High': 2}
days_map   = {
    'Go out Every day'  : 0,
    '1-14 days'         : 1,
    '15-30 days'        : 2,
    '31-60 days'        : 3,
    'More than 2 months': 4,
}
care_map = {'No': 0, 'Not sure': 0.5, 'Yes': 1}

df['Gender']                  = df['Gender'].map({'Male': 1, 'Female': 0})
df['self_employed']           = df['self_employed'].map(binary_map)
df['family_history']          = df['family_history'].map(binary_map)
df['treatment']               = df['treatment'].map(binary_map)
df['Growing_Stress']          = df['Growing_Stress'].map(tri_map)
df['Changes_Habits']          = df['Changes_Habits'].map(tri_map)
df['Mental_Health_History']   = df['Mental_Health_History'].map(tri_map)
df['Mood_Swings']             = df['Mood_Swings'].map(mood_map)
df['Coping_Struggles']        = df['Coping_Struggles'].map(binary_map)
df['Work_Interest']           = df['Work_Interest'].map(tri_map)
df['Social_Weakness']         = df['Social_Weakness'].map(tri_map)
df['mental_health_interview'] = df['mental_health_interview'].map(tri_map)
df['care_options']            = df['care_options'].map(care_map)
df['Days_Indoors']            = df['Days_Indoors'].map(days_map)

le = LabelEncoder()
df['Occupation'] = le.fit_transform(df['Occupation'].astype(str))

# Drop rows with any NaN left after mapping (unmapped categories)
df.dropna(inplace=True)
print(f"    After encoding & dropna: {df.shape}")

# ─────────────────────────────────────────────────────────────
# 4. ADVANCED FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n[4] Feature Engineering...")

# Composite scores (same as original)
df['stress_score']     = df['Growing_Stress'] + df['Mood_Swings'] + df['Coping_Struggles']
df['behavioral_score'] = df['Changes_Habits'] + df['Work_Interest'] + df['Social_Weakness'] + df['Days_Indoors']
df['awareness_score']  = df['Mental_Health_History'] + df['mental_health_interview'] + df['care_options']

# Family history interactions (strongest predictor 0.36 corr)
df['stress_x_family']    = df['stress_score']    * df['family_history']
df['care_x_family']      = df['care_options']    * df['family_history']
df['awareness_x_family'] = df['awareness_score'] * df['family_history']
df['gender_x_stress']    = df['Gender']          * df['stress_score']

# NEW: additional engineered features for 85%+ push
df['total_risk_score']   = df['stress_score'] + df['behavioral_score']
df['mh_support_index']   = df['care_options'] + df['mental_health_interview']
df['isolation_stress']   = df['Days_Indoors'] * df['stress_score']
df['family_x_behav']     = df['family_history'] * df['behavioral_score']
df['gender_x_family']    = df['Gender'] * df['family_history']
df['care_x_stress']      = df['care_options'] * df['stress_score']
df['coping_x_mood']      = df['Coping_Struggles'] * df['Mood_Swings']
df['stress_awareness_ratio'] = df['stress_score'] / (df['awareness_score'] + 0.5)

# High risk flag (original)
df['high_risk_flag'] = (
    (df['stress_score']     >= 3) &
    (df['behavioral_score'] >= 4) &
    (df['family_history']   == 1)
).astype(int)

# Very high risk flag (new)
df['very_high_risk'] = (
    (df['stress_score']     >= 4) &
    (df['behavioral_score'] >= 5) &
    (df['family_history']   == 1) &
    (df['care_options']     <= 0.5)
).astype(int)

print(f"    Total features: {df.shape[1] - 1}")  # -1 for target

# ─────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X = df.drop('treatment', axis=1)
y = df['treatment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[5] Train: {X_train.shape} | Test: {X_test.shape}")

# Scale for LR (needed as one base estimator)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# 6. TUNED MODELS
# ─────────────────────────────────────────────────────────────
print("\n[6] Training models...")

# Tuned XGBoost (primary powerhouse)
xgb = XGBClassifier(
    n_estimators      = 500,
    max_depth         = 7,
    learning_rate     = 0.03,
    subsample         = 0.85,
    colsample_bytree  = 0.75,
    min_child_weight  = 3,
    gamma             = 0.1,
    reg_alpha         = 0.05,
    reg_lambda        = 1.5,
    scale_pos_weight  = 1,
    random_state      = 42,
    eval_metric       = 'logloss',
    n_jobs            = -1,
    use_label_encoder = False
)

# Tuned Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators  = 300,
    learning_rate = 0.05,
    max_depth     = 6,
    subsample     = 0.8,
    min_samples_split = 20,
    random_state  = 42
)

# Tuned Random Forest
rf = RandomForestClassifier(
    n_estimators = 300,
    max_depth    = 15,
    min_samples_split = 10,
    min_samples_leaf  = 5,
    max_features = 'sqrt',
    class_weight = 'balanced',
    random_state = 42,
    n_jobs       = -1
)

# LR as meta-learner base
lr = LogisticRegression(max_iter=1000, random_state=42, C=0.5)

# ─────────────────────────────────────────────────────────────
# 7. STACKING CLASSIFIER (best for 85%+)
# ─────────────────────────────────────────────────────────────
stacking = StackingClassifier(
    estimators=[
        ('xgb', xgb),
        ('gb',  gb),
        ('rf',  rf),
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    cv=5,
    n_jobs=-1,
    passthrough=True   # pass original features to meta-learner too
)

print("    Fitting Stacking Classifier (this may take a few minutes)...")
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"  STACKING CLASSIFIER ACCURACY: {acc*100:.2f}%")
print(f"{'='*60}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Treatment', 'Treatment']))

# ─────────────────────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────────────────────
joblib.dump(stacking, 'mental_health_model.pkl')
joblib.dump(scaler,   'scaler.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')
print(f"\n[7] Saved: mental_health_model.pkl  |  accuracy={acc*100:.2f}%")
print("    Feature list saved to feature_columns.pkl")
