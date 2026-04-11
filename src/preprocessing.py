"""
Preprocessing functions: cleaning, encoding, imputation, scaling.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils import (
    group_icd9, AGE_MAP, MED_MAP, A1C_MAP, GLU_MAP,
    MEDICATION_COLS, DECEASED_DISCHARGE_IDS, print_step
)


def drop_ids(df):
    """Drop identifier columns that have no predictive value."""
    df = df.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore')
    print_step('Drop IDs', df)
    return df


def drop_high_missing(df, cols=None):
    """Drop columns with >40% missing values."""
    if cols is None:
        cols = ['weight', 'payer_code', 'medical_specialty']
    df = df.drop(columns=[c for c in cols if c in df.columns])
    print_step('Drop high-missing columns', df)
    return df


def drop_nzv_medications(df, nzv_cols):
    """Drop near-zero variance medication columns."""
    df = df.drop(columns=[c for c in nzv_cols if c in df.columns])
    print_step('Drop near-zero variance meds', df)
    return df


def remove_deceased(df):
    """Remove rows where patient expired or was sent to hospice."""
    before = len(df)
    df = df[~df['discharge_disposition_id'].isin(DECEASED_DISCHARGE_IDS)]
    print(f"  Removed {before - len(df)} deceased/hospice rows")
    print_step('Remove deceased/hospice', df)
    return df


def remove_invalid_gender(df):
    """Remove rows with Unknown/Invalid gender."""
    before = len(df)
    df = df[df['gender'] != 'Unknown/Invalid']
    print(f"  Removed {before - len(df)} invalid gender rows")
    print_step('Remove invalid gender', df)
    return df


def impute_missing(df):
    """Impute remaining missing values."""
    if 'race' in df.columns and df['race'].isnull().any():
        mode_val = df['race'].mode()[0]
        df['race'] = df['race'].fillna(mode_val)
        print(f"  Imputed race with mode: '{mode_val}'")

    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna('Missing')
            print(f"  Imputed {col} with 'Missing'")

    print_step('Impute missing', df)
    return df


def encode_age(df):
    """Convert age brackets to numeric midpoints."""
    df['age'] = df['age'].map(AGE_MAP)
    print_step('Encode age', df)
    return df


def encode_medications(df):
    """Ordinal-encode medication dosage columns that remain in the dataset."""
    med_cols_present = [c for c in MEDICATION_COLS if c in df.columns]
    for col in med_cols_present:
        df[col] = df[col].map(MED_MAP).fillna(0).astype(int)
    print_step(f'Encode medications ({len(med_cols_present)} cols)', df)
    return df


def encode_lab_results(df):
    """Ordinal-encode A1Cresult and max_glu_serum."""
    if 'A1Cresult' in df.columns:
        df['A1Cresult'] = df['A1Cresult'].map(A1C_MAP).fillna(0).astype(int)
    if 'max_glu_serum' in df.columns:
        df['max_glu_serum'] = df['max_glu_serum'].map(GLU_MAP).fillna(0).astype(int)
    print_step('Encode lab results', df)
    return df


def encode_binaries(df):
    """Binary-encode gender, change, diabetesMed."""
    binary_maps = {
        'gender': {'Male': 1, 'Female': 0},
        'change': {'Ch': 1, 'No': 0},
        'diabetesMed': {'Yes': 1, 'No': 0}
    }
    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    print_step('Encode binaries', df)
    return df


def group_diagnoses(df):
    """Group ICD-9 codes into clinical categories and one-hot encode."""
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col + '_group'] = df[col].apply(group_icd9)
            df = df.drop(columns=[col])

    diag_group_cols = [c for c in df.columns if c.endswith('_group')]
    df = pd.get_dummies(df, columns=diag_group_cols, prefix_sep='_', dtype=int)
    print_step('Group & encode diagnoses', df)
    return df


def one_hot_encode_categoricals(df):
    """One-hot encode remaining categorical columns."""
    cat_cols = []
    for col in ['race', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        if col in df.columns:
            df[col] = df[col].astype(str)
            cat_cols.append(col)

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, prefix_sep='_', dtype=int)
    print_step('One-hot encode categoricals', df)
    return df


def cap_outliers(df, cols, lower_pct=0.01, upper_pct=0.99):
    """Cap numeric features at the 1st and 99th percentile."""
    for col in cols:
        if col in df.columns:
            lo = df[col].quantile(lower_pct)
            hi = df[col].quantile(upper_pct)
            df[col] = df[col].clip(lower=lo, upper=hi)
    print_step('Cap outliers', df)
    return df


def scale_features(X_train, X_test, numeric_cols):
    """
    Standardize numeric features. Fit on train, transform both.
    Returns scaled copies and the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    cols_to_scale = [c for c in numeric_cols if c in X_train.columns]

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled, scaler
