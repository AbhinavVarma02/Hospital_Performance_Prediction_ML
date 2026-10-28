"""Coursework workflow reproduction, separate from facility-aware evaluation."""
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from hospital_ml.data import LEGACY_COLUMNS, historical_cleaning, write_json


def run_historical(raw, output_dir, original_cleaned_path=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned, steps = historical_cleaning(raw)
    comparison = None
    if original_cleaned_path and Path(original_cleaned_path).exists():
        original = pd.read_csv(original_cleaned_path)[LEGACY_COLUMNS]
        comparison = cleaned.equals(original)
    features = ['BED_LIC', 'BED_AVL', 'BED_STF', 'DAY_PER']
    legacy_matrix = cleaned[features].copy()
    for col in ['FAC_NAME', 'COUNTY', 'Variable']:
        legacy_matrix[col + '_ENC'] = LabelEncoder().fit_transform(cleaned[col])
    # Retain the original seven-feature matrix for coursework output validation.
    X_legacy_scaled = StandardScaler().fit_transform(legacy_matrix)
    model = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(X_legacy_scaled)
    pca = PCA(n_components=2, svd_solver='full').fit(X_legacy_scaled)
    cleaned['legacy_cluster'] = model.labels_
    cleaned.groupby('legacy_cluster')[['BED_LIC', 'BED_AVL', 'BED_STF', 'Amount']].mean().to_csv(
        output_dir / 'replayed_cluster_means.csv')
    summary = {
        'status': 'historical methodology replay on fingerprinted source; not improved validation',
        'cleaning_steps': steps,
        'cleaned_rows': len(cleaned),
        'values_match_original_shared_cleaned_eight_columns': comparison,
        'matrix_columns_actually_used': list(legacy_matrix),
        'excluded_amount_from_original_matrix': True,
        'k': 3, 'random_state': 42, 'n_init': 'auto', 'pca_svd_solver': 'full',
        'cluster_sizes': {str(k): int(v) for k, v in pd.Series(model.labels_).value_counts().sort_index().items()},
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'exact_saved_cluster_output_match_claimed': False,
        'limitations': ['Original library versions and executable order were not recorded.',
                       'Original code cells have null execution counters but retained outputs.',
                       'Global scaling and ordinal facility/county/variable codes are historical defects.',
                       'Amount profile averages mix units and are retained only for audit, not interpretation.'],
    }
    write_json(output_dir / 'replay_summary.json', summary)
    return summary
