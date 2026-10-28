"""Facility-aware K-Means capacity segmentation; PCA is only a display projection.

The raw data are long reporting rows with mixed Amount units. Improved clustering
therefore fits one capacity vector per facility/report period, then assigns those
labels back to every source reporting row. No cluster is an efficiency ranking.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPORT_KEYS = ["FAC_NO", "BEG_DATE", "END_DATE"]
CAPACITY_COLUMNS = ["BED_LIC", "BED_AVL", "BED_STF", "DAY_PER"]
CLUSTER_FEATURES = ["log_licensed_beds", "available_share", "staffed_share"]
STABILITY_SEEDS = (11, 29, 71)
BOOTSTRAP_SEEDS = (101, 203, 307, 409, 503)
SELECTION_RULE = (
    "Choose highest development silhouette for K=2..8 among candidates with "
    "mean facility-bootstrap ARI >= 0.80 and minimum development cluster share "
    ">= 0.02. If none qualify, use highest development silhouette and explicitly "
    "mark the stability/size requirement unsupported. Test facilities never select K."
)


class CapacityClipper(TransformerMixin, BaseEstimator):
    """Learn 1st/99th feature quantiles from fitting facilities only.

    This retains all facilities, including large facilities, and limits extreme
    feature influence without global IQR row deletion. Missing values stay missing
    until the subsequent training-median imputer.
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.n_features_in_ = array.shape[1]
        if np.any(np.isnan(array).all(axis=0)):
            raise ValueError("A capacity feature is entirely missing in fitting data.")
        self.lower_bounds_ = np.nanquantile(array, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(array, self.upper, axis=0)
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float)
        return np.clip(array, self.lower_bounds_, self.upper_bounds_)


def capacity_preprocessor():
    return Pipeline([
        ("clip", CapacityClipper()),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])


def build_capacity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validate one capacity vector per reporting period before aggregation."""
    required = REPORT_KEYS + CAPACITY_COLUMNS
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing capacity columns: {missing}")
    if df[REPORT_KEYS].isna().any().any():
        raise ValueError("Facility/report keys must be present for capacity grouping.")
    grouped = df.groupby(REPORT_KEYS, sort=True, dropna=False)
    conflicting = grouped[CAPACITY_COLUMNS].nunique(dropna=False).gt(1)
    if conflicting.any().any():
        examples = conflicting[conflicting.any(axis=1)].index.tolist()[:5]
        raise ValueError(f"Conflicting capacities within facility/report periods: {examples}")
    table = grouped[CAPACITY_COLUMNS].first().reset_index()
    table["reporting_rows"] = grouped.size().to_numpy()
    for col in CAPACITY_COLUMNS:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    beds = table[["BED_LIC", "BED_AVL", "BED_STF"]].where(
        table[["BED_LIC", "BED_AVL", "BED_STF"]] >= 0
    )
    table["log_licensed_beds"] = np.log1p(beds["BED_LIC"])
    table["available_share"] = beds["BED_AVL"].div(beds["BED_LIC"].where(beds["BED_LIC"] > 0))
    table["staffed_share"] = beds["BED_STF"].div(beds["BED_AVL"].where(beds["BED_AVL"] > 0))
    table[CLUSTER_FEATURES] = table[CLUSTER_FEATURES].replace([np.inf, -np.inf], np.nan)
    return table


def _kmeans(k, seed):
    return KMeans(n_clusters=k, n_init=20, random_state=seed, max_iter=300)


def _silhouette(X, labels, seed):
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return None
    # Bound pairwise-distance memory; this is a fixed seeded sample, not all rows.
    return float(silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=seed))


def _bootstrap_facility_indices(table, seed):
    groups = {key: rows.to_numpy() for key, rows in table.groupby("FAC_NO").groups.items()}
    facilities = np.asarray(sorted(groups))
    draws = np.random.default_rng(seed).choice(facilities, len(facilities), replace=True)
    return np.concatenate([groups[key] for key in draws])


def select_k(development: pd.DataFrame, seed=42):
    """Select K without test facilities; refit preprocessing in each bootstrap."""
    development = development.reset_index(drop=True)
    if len(development) < 10:
        raise ValueError("At least ten development facility/report periods are required.")
    reference_preprocessor = capacity_preprocessor()
    reference_matrix = reference_preprocessor.fit_transform(development[CLUSTER_FEATURES])
    rows = []
    for k in range(2, min(8, len(development) - 1) + 1):
        reference_model = _kmeans(k, seed).fit(reference_matrix)
        reference_labels = reference_model.labels_
        if len(np.unique(reference_labels)) != k:
            continue
        seed_labels = [_kmeans(k, s).fit_predict(reference_matrix) for s in STABILITY_SEEDS]
        seed_aris = [adjusted_rand_score(a, b) for a, b in combinations([reference_labels] + seed_labels, 2)]
        bootstrap_aris = []
        for bootstrap_seed in BOOTSTRAP_SEEDS:
            indices = _bootstrap_facility_indices(development, bootstrap_seed)
            preprocessor = capacity_preprocessor()
            resampled_matrix = preprocessor.fit_transform(development.iloc[indices][CLUSTER_FEATURES])
            model = _kmeans(k, seed).fit(resampled_matrix)
            common_reference_labels = model.predict(preprocessor.transform(development[CLUSTER_FEATURES]))
            bootstrap_aris.append(adjusted_rand_score(reference_labels, common_reference_labels))
        counts = np.bincount(reference_labels, minlength=k)
        silhouette = _silhouette(reference_matrix, reference_labels, seed)
        rows.append({
            "k": k,
            "development_silhouette": silhouette,
            "development_inertia": float(reference_model.inertia_),
            "min_cluster_fraction": float(counts.min() / counts.sum()),
            "seed_ari_mean": float(np.mean(seed_aris)),
            "seed_ari_min": float(np.min(seed_aris)),
            "facility_bootstrap_ari_mean": float(np.mean(bootstrap_aris)),
            "facility_bootstrap_ari_std": float(np.std(bootstrap_aris)),
            "facility_bootstrap_ari_min": float(np.min(bootstrap_aris)),
            "facility_bootstrap_ari_values": json.dumps([float(v) for v in bootstrap_aris]),
        })
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No nondegenerate K-Means candidates; inspect feature diversity.")
    eligible = candidates[
        (candidates.facility_bootstrap_ari_mean >= 0.80)
        & (candidates.min_cluster_fraction >= 0.02)
    ]
    selection_supported = not eligible.empty
    pool = eligible if selection_supported else candidates
    selected = pool.sort_values(["development_silhouette", "k"], ascending=[False, True]).iloc[0]
    return candidates, int(selected.k), selection_supported


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_clustering(df: pd.DataFrame, output_dir, seed=42) -> dict:
    """Fit development capacity groups, evaluate once on held-out facilities."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_capacity_table(df)
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_index, test_index = next(split.split(table, groups=table["FAC_NO"]))
    train_facilities = set(table.iloc[train_index]["FAC_NO"])
    test_facilities = set(table.iloc[test_index]["FAC_NO"])
    assert train_facilities.isdisjoint(test_facilities)
    candidates, selected_k, selection_supported = select_k(table.iloc[train_index], seed=seed)
    candidates.to_csv(output_dir / "k_selection.csv", index=False)

    preprocessor = capacity_preprocessor().fit(table.iloc[train_index][CLUSTER_FEATURES])
    # K-Means and PCA consume exactly the same feature matrix and column order.
    X_cluster_scaled = preprocessor.transform(table[CLUSTER_FEATURES])
    kmeans = _kmeans(selected_k, seed).fit(X_cluster_scaled[train_index])
    labels = kmeans.predict(X_cluster_scaled)
    pca = PCA(n_components=2, svd_solver="full").fit(X_cluster_scaled[train_index])
    pca_coordinates = pca.transform(X_cluster_scaled)
    table["resource_cluster"] = labels
    table["split"] = "development"
    table.loc[test_index, "split"] = "heldout_facility"
    table["pca_1"] = pca_coordinates[:, 0]
    table["pca_2"] = pca_coordinates[:, 1]
    table.to_csv(output_dir / "facility_period_assignments.csv", index=False)

    assignments = df[REPORT_KEYS].merge(
        table[REPORT_KEYS + ["resource_cluster", "split"]],
        on=REPORT_KEYS, how="left", validate="many_to_one", sort=False,
    )
    assert len(assignments) == len(df) and assignments.resource_cluster.notna().all()
    # Large row assignments are generated locally and ignored by Git.
    assignments.to_csv(output_dir / "reporting_row_assignments.csv.gz", index=False, compression="gzip")
    profile_columns = ["BED_LIC", "BED_AVL", "BED_STF", "available_share", "staffed_share"]
    profile = table.groupby(["split", "resource_cluster"])[profile_columns].agg(
        ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    )
    profile.columns = [f"{feature}_{stat.replace('<lambda_0>', 'q25').replace('<lambda_1>', 'q75')}" for feature, stat in profile.columns]
    profile = profile.join(table.groupby(["split", "resource_cluster"]).agg(
        facility_periods=("FAC_NO", "size"), unique_facilities=("FAC_NO", "nunique"),
        reporting_rows=("reporting_rows", "sum"),
    ))
    profile.to_csv(output_dir / "cluster_profiles.csv")
    loadings = pd.DataFrame(pca.components_.T, index=CLUSTER_FEATURES, columns=["PC1", "PC2"])
    loadings.to_csv(output_dir / "pca_loadings.csv", index_label="feature")
    pd.DataFrame(kmeans.cluster_centers_, columns=CLUSTER_FEATURES).to_csv(
        output_dir / "scaled_centroids.csv", index_label="resource_cluster"
    )
    # Development-only sensitivity to the prespecified outlier treatment.
    unclipped = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    unclipped_matrix = unclipped.fit_transform(table.iloc[train_index][CLUSTER_FEATURES])
    unclipped_labels = _kmeans(selected_k, seed).fit_predict(unclipped_matrix)
    unclipped_ari = adjusted_rand_score(labels[train_index], unclipped_labels)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(candidates.k, candidates.development_silhouette, marker="o")
    axes[0].axvline(selected_k, color="gray", linestyle="--", label=f"Selected K={selected_k}")
    axes[0].set(xlabel="K", ylabel="Development silhouette", title="K selected on development facilities")
    axes[0].legend()
    for cluster_id in range(selected_k):
        mask = labels == cluster_id
        axes[1].scatter(pca_coordinates[mask, 0], pca_coordinates[mask, 1],
                        s=9, color=plt.get_cmap("tab10")(cluster_id), alpha=0.5,
                        label=f"Cluster {cluster_id}")
    axes[1].set(xlabel="PC1", ylabel="PC2", title="Capacity patterns; PCA display only")
    axes[1].legend(title="Arbitrary IDs", markerscale=2)
    figure.tight_layout()
    figure.savefig(output_dir / "capacity_clusters.png", dpi=150)
    plt.close(figure)
    summary = {
        "analysis_unit": "facility/report period; assigned back to long reporting rows",
        "source_reporting_rows": int(len(df)),
        "facility_periods": int(len(table)),
        "development_periods": int(len(train_index)),
        "heldout_periods": int(len(test_index)),
        "development_facilities": int(len(train_facilities)),
        "heldout_facilities": int(len(test_facilities)),
        "facility_overlap": len(train_facilities & test_facilities),
        "features": CLUSTER_FEATURES,
        "excluded_from_distance": ["FAC_NO", "FAC_NAME", "COUNTY", "Variable", "Amount", "DAY_PER"],
        "selected_k": selected_k,
        "selection_rule": SELECTION_RULE,
        "selection_stability_size_supported": bool(selection_supported),
        "selection_requirement_warning": None if selection_supported else "No candidate met the prespecified stability/size requirements; descriptive grouping is provisional.",
        "selected_development_silhouette": float(candidates.loc[candidates.k == selected_k, "development_silhouette"].iloc[0]),
        "heldout_silhouette": _silhouette(X_cluster_scaled[test_index], labels[test_index], seed),
        "heldout_cluster_counts": {str(k): int(v) for k, v in pd.Series(labels[test_index]).value_counts().sort_index().items()},
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "preprocessing_fitted_on": "development facilities only; each bootstrap refits all preprocessing",
        "clipping_quantiles": [0.01, 0.99],
        "development_ari_without_clipping_at_selected_k": float(unclipped_ari),
        "development_silhouette_without_clipping_at_selected_k": _silhouette(unclipped_matrix, unclipped_labels, seed),
        "feature_missing_periods": {name: int(table[name].isna().sum()) for name in CLUSTER_FEATURES},
        "development_imputation_medians": preprocessor.named_steps["impute"].statistics_.tolist(),
        "development_scaler_means": preprocessor.named_steps["scale"].mean_.tolist(),
        "development_scaler_scales": preprocessor.named_steps["scale"].scale_.tolist(),
        "clipping_lower_bounds": preprocessor.named_steps["clip"].lower_bounds_.tolist(),
        "clipping_upper_bounds": preprocessor.named_steps["clip"].upper_bounds_.tolist(),
        "negative_bed_cells_treated_as_missing": int((table[["BED_LIC", "BED_AVL", "BED_STF"]] < 0).sum().sum()),
        "available_above_licensed_periods": int((table.BED_AVL > table.BED_LIC).sum()),
        "staffed_above_available_periods": int((table.BED_STF > table.BED_AVL).sum()),
        "random_seed": seed,
        "seed_stability_seeds": list(STABILITY_SEEDS),
        "facility_bootstrap_seeds": list(BOOTSTRAP_SEEDS),
        "interpretation": "Capacity/resource-availability patterns, not hospital efficiency, clinical quality, staffing headcount, or causal optimization. Amount is excluded because Variable mixes units.",
        "limitations": [
            "Facility-periods from the same facility are correlated; facility-separated diagnostics and resampling reduce this dependence but do not establish temporal forecasting.",
            "Each facility-period has equal fitting weight; facilities with more reporting periods contribute more vectors. Long Variable rows do not multiply fitting weight.",
            "Five facility bootstraps provide sensitivity evidence, not a confidence interval or proof of a natural population partition.",
            "K=2..8 is a prespecified finite search; silhouette favors some geometries and must be read with stability and profiles.",
            "All periods receive the frozen development model; PCA and cluster IDs carry no ordering or efficiency meaning.",
            "Capacity-only features cannot distinguish financial resource intensity or patient case mix.",
            "Licensed beds are end-of-period counts; available and staffed beds are period averages. Their ratios are descriptive capacity shares and can exceed one, without representing simultaneous utilization.",
        ],
    }
    _write_json(output_dir / "clustering_summary.json", summary)
    return summary
