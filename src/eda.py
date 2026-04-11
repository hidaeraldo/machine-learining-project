"""
EDA helper functions: plotting, statistical summaries, outlier detection.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def missing_summary(df):
    """Return a DataFrame of missing counts and percentages, sorted descending."""
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    summary = pd.DataFrame({'missing_count': missing, 'missing_pct': pct})
    summary = summary[summary['missing_count'] > 0].sort_values(
        'missing_pct', ascending=False
    )
    return summary


def plot_missing_bar(summary, threshold=40, figsize=(10, 5)):
    """Bar chart of missing percentages with a threshold line."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(summary)), summary['missing_pct'], color='steelblue')
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'{threshold}% threshold')
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary.index, rotation=45, ha='right')
    ax.set_ylabel('Missing %')
    ax.set_title('Missing Values by Column')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_target_distribution(series, title='Target Distribution'):
    """Bar chart for target variable distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = series.value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values, color=['#2ecc71', '#e74c3c'])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f'{val}\n({val/len(series)*100:.1f}%)', ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_numeric_histograms(df, cols, bins=30, figsize=(16, 12)):
    """Grid of histograms for numeric features."""
    n = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=bins, color='steelblue', edgecolor='white')
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel('Count')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Numeric Feature Distributions', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def compute_skewness(df, numeric_cols):
    """Compute skewness for numeric columns, flag high skew."""
    skew_vals = df[numeric_cols].skew().sort_values(ascending=False)
    skew_df = pd.DataFrame({
        'skewness': skew_vals,
        'abs_skewness': skew_vals.abs(),
        'high_skew': skew_vals.abs() > 1
    })
    return skew_df


def plot_categorical_bars(df, cols, figsize=(16, 10)):
    """Bar plots for categorical features."""
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(cols):
        counts = df[col].value_counts()
        if len(counts) > 15:
            counts = counts.head(15)
        axes[i].barh(counts.index.astype(str), counts.values, color='steelblue')
        axes[i].set_title(col, fontsize=10)
        axes[i].invert_yaxis()
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Categorical Feature Distributions', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def medication_variance_analysis(df, med_cols):
    """Compute % of 'No' for each medication column and visualize."""
    no_pct = {}
    for col in med_cols:
        if col in df.columns:
            no_pct[col] = (df[col] == 'No').sum() / len(df) * 100

    no_pct_series = pd.Series(no_pct).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c' if v > 97 else 'steelblue' for v in no_pct_series.values]
    ax.barh(no_pct_series.index, no_pct_series.values, color=colors)
    ax.axvline(x=97, color='red', linestyle='--', label='97% threshold')
    ax.set_xlabel('% of rows with value "No"')
    ax.set_title('Medication Columns: Near-Zero Variance Analysis')
    ax.legend()
    plt.tight_layout()
    plt.show()

    nzv_cols = [col for col, pct in no_pct.items() if pct > 97]
    return no_pct_series, nzv_cols


def plot_bivariate_boxplots(df, features, target, figsize=(16, 12)):
    """Boxplots of numeric features grouped by binary target."""
    n = len(features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(features):
        sns.boxplot(x=target, y=col, data=df, ax=axes[i], palette='Set2')
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Numeric Features vs Readmission (0=No, 1=Yes)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_stacked_proportions(df, cat_col, target, top_n=10, figsize=(10, 5)):
    """Stacked bar chart showing readmission rate by categorical feature."""
    ct = pd.crosstab(df[cat_col], df[target], normalize='index')
    if len(ct) > top_n:
        top_cats = df[cat_col].value_counts().head(top_n).index
        ct = ct.loc[ct.index.isin(top_cats)]
    ct.columns = ['Not Readmitted', 'Readmitted <30d']
    ct.sort_values('Readmitted <30d', ascending=True).plot(
        kind='barh', stacked=True, figsize=figsize,
        color=['#2ecc71', '#e74c3c']
    )
    plt.title(f'Readmission Rate by {cat_col}')
    plt.xlabel('Proportion')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_scatter(df, x_col, y_col, target, figsize=(8, 6)):
    """Scatter plot of two numeric features colored by target."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = {0: '#2ecc71', 1: '#e74c3c'}
    labels = {0: 'Not readmitted <30d', 1: 'Readmitted <30d'}
    for cls in [0, 1]:
        mask = df[target] == cls
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                   c=colors[cls], label=labels[cls], alpha=0.3, s=10)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col} (colored by readmission)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(df, numeric_cols):
    """
    IQR-based outlier detection.
    Returns a summary DataFrame with Q1, Q3, IQR, bounds, outlier counts.
    """
    results = []
    for col in numeric_cols:
        data = df[col].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = ((data < lower) | (data > upper)).sum()
        results.append({
            'feature': col, 'Q1': q1, 'Q3': q3, 'IQR': iqr,
            'lower_bound': lower, 'upper_bound': upper,
            'outlier_count': outlier_count,
            'outlier_pct': outlier_count / len(data) * 100
        })
    return pd.DataFrame(results).sort_values('outlier_pct', ascending=False)


def plot_outlier_boxplots(df, cols, figsize=(14, 6)):
    """Boxplots for the most outlier-heavy features."""
    fig, axes = plt.subplots(1, len(cols), figsize=figsize)
    if len(cols) == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        axes[i].boxplot(df[col].dropna(), vert=True)
        axes[i].set_title(col)
    plt.suptitle('Outlier-Heavy Features (IQR Method)', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, figsize=(14, 12)):
    """Full correlation heatmap for numeric columns."""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                annot=False, fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    return corr


def correlation_with_target(corr_matrix, target_col):
    """Get correlations with the target column, sorted by absolute value."""
    target_corr = corr_matrix[target_col].drop(target_col)
    return target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)


def find_multicollinear(corr_matrix, threshold=0.95):
    """Find pairs of features with |correlation| > threshold."""
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))
    return pairs
