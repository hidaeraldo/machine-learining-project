"""
Feature engineering: new feature creation, binning, transforms, variance filtering.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

from src.utils import print_step


def create_visit_features(df):
    """Create total_visits_prior from outpatient + emergency + inpatient counts."""
    df['total_visits_prior'] = (
        df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    )
    print("  Created: total_visits_prior")
    return df


def create_med_change_feature(df, med_cols):
    """
    Count the number of medications that were changed (Up or Down).
    Assumes medications are already ordinal-encoded: Down=2, Up=3.
    """
    present = [c for c in med_cols if c in df.columns]
    df['n_med_changes'] = df[present].apply(
        lambda row: ((row == 2) | (row == 3)).sum(), axis=1
    )
    print("  Created: n_med_changes")
    return df


def create_test_flags(df):
    """Create binary flags for whether lab tests were performed."""
    if 'A1Cresult' in df.columns:
        df['a1c_tested'] = (df['A1Cresult'] != 0).astype(int)
        print("  Created: a1c_tested")
    if 'max_glu_serum' in df.columns:
        df['glu_tested'] = (df['max_glu_serum'] != 0).astype(int)
        print("  Created: glu_tested")
    return df


def create_service_utilization(df):
    """Create service_utilization = num_lab_procedures + num_procedures."""
    df['service_utilization'] = df['num_lab_procedures'] + df['num_procedures']
    print("  Created: service_utilization")
    return df


def bin_time_in_hospital(df):
    """Bin time_in_hospital into short/medium/long categories."""
    df['stay_length'] = pd.cut(
        df['time_in_hospital'],
        bins=[0, 3, 7, float('inf')],
        labels=['short', 'medium', 'long']
    )
    df = pd.get_dummies(df, columns=['stay_length'], prefix='stay', dtype=int)
    print("  Created: stay_short, stay_medium, stay_long")
    return df


def log_transform_skewed(df, cols, skew_threshold=2.0):
    """Apply log1p transform to heavily skewed numeric features."""
    transformed = []
    for col in cols:
        if col in df.columns:
            skew = df[col].skew()
            if abs(skew) > skew_threshold:
                df[col + '_log'] = np.log1p(df[col])
                transformed.append(col)
    if transformed:
        print(f"  Log-transformed: {transformed}")
    return df


def remove_low_variance(df, threshold=0.01):
    """Drop features with variance below the threshold."""
    numeric_df = df.select_dtypes(include=[np.number])
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_df)

    low_var_cols = numeric_df.columns[~selector.get_support()].tolist()
    if low_var_cols:
        df = df.drop(columns=low_var_cols)
        print(f"  Dropped {len(low_var_cols)} low-variance features: {low_var_cols[:10]}...")
    else:
        print("  No low-variance features found")
    print_step('Variance filter', df)
    return df
