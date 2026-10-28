"""Checks for information boundaries, independent of model quality."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from hospital_ml.common import facility_split
from hospital_ml import supervised


def reporting_frame(facilities=30, periods=4):
    rows = []
    for facility in range(facilities):
        for period in range(periods):
            beds = 10 + facility * 5 + period
            rows.append({
                "FAC_NO": f"facility_{facility:03d}", "FAC_NAME": "excluded identity",
                "BEG_DATE": f"01JAN{2012 + period}", "END_DATE": f"31DEC{2012 + period}",
                "BED_LIC": float(beds), "BED_AVL": float(beds - 2), "BED_STF": float(beds - 4),
                "DAY_PER": 365.0, "COUNTY": "A" if facility % 2 else "B",
                "Variable": "TOT_OP_EXP", "Amount": float(beds * 10000 + period * 300),
            })
    return pd.DataFrame(rows)


def test_feature_matrix_excludes_outcome_and_identity_and_handles_zero_denominator():
    frame = reporting_frame()
    frame.loc[0, ["BED_LIC", "BED_AVL"]] = 0
    features = supervised.expense_features(frame)
    assert list(features) == supervised.NUMERIC_FEATURES + supervised.CATEGORICAL_FEATURES
    assert np.isnan(features.loc[0, "AVAILABLE_SHARE"])
    assert np.isnan(features.loc[0, "STAFFED_SHARE"])
    changed = frame.copy()
    changed["Amount"] = 1e30
    changed["FAC_NO"] = "arbitrary replacement"
    pd.testing.assert_frame_equal(features, supervised.expense_features(changed))


def test_preprocessing_learns_only_training_values_and_accepts_unseen_counties():
    frame = reporting_frame()
    training = supervised.expense_features(frame.iloc[:20])
    heldout = supervised.expense_features(frame.iloc[20:25])
    heldout.loc[:, "COUNTY"] = "Unseen County"
    heldout.loc[:, "BED_LIC"] = 1e15
    fitted = supervised.make_preprocessor().fit(training)
    clipper = fitted.named_transformers_["numeric"].named_steps["clip"]
    boundaries = clipper.upper_bounds_.copy()
    result = fitted.transform(heldout)
    assert np.isfinite(result).all()
    assert result.shape[0] == len(heldout)
    np.testing.assert_array_equal(clipper.upper_bounds_, boundaries)
    assert clipper.upper_bounds_[0] < 1e15
    county_count = len(fitted.named_transformers_["categorical"].named_steps["encode"].categories_[0])
    np.testing.assert_array_equal(result[:, -county_count:], 0)


def test_classification_threshold_is_fitted_per_training_fold():
    frame = reporting_frame(8, 1)
    X = supervised.expense_features(frame)
    expense = np.array([0, 1, 2, 3, 100, 101, 102, 103], dtype=float)
    groups = frame["FAC_NO"].to_numpy()
    folds = [(np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])),
             (np.array([1, 3, 5, 7]), np.array([0, 2, 4, 6]))]
    rows = supervised._cv_classification(
        X, expense, groups, folds,
        {"dummy": supervised._pipeline(DummyClassifier(strategy="most_frequent"))},
    )
    assert rows["training_expense_median"].tolist() == [51.0, 52.0]
    assert rows["positive_prevalence"].tolist() == [0.5, 0.5]


def test_holdout_outcomes_cannot_change_model_selection_or_threshold(tmp_path, monkeypatch):
    frame = reporting_frame()
    # Add an incompatible reporting measure to verify explicit target selection.
    other = frame.iloc[:5].copy()
    other["Variable"] = "DAY_TOT"
    other["Amount"] = -1e20
    frame = pd.concat([frame, other], ignore_index=True)
    monkeypatch.setattr(supervised, "_regression_candidates", lambda seed: {
        "dummy_mean": supervised._pipeline(DummyRegressor(strategy="mean")),
        "dummy_median": supervised._pipeline(DummyRegressor(strategy="median")),
        "ridge": supervised._pipeline(Ridge(alpha=10)),
    })
    monkeypatch.setattr(supervised, "_classification_candidates", lambda seed: {
        "dummy_most_frequent": supervised._pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": supervised._pipeline(LogisticRegression(max_iter=1000)),
    })
    first = supervised.run_supervised(frame, tmp_path / "first")
    target = frame.loc[frame["Variable"] == "TOT_OP_EXP"].reset_index(drop=True)
    _, holdout = facility_split(target, test_size=0.2, seed=42)
    changed = frame.copy()
    changed.loc[holdout, "Amount"] *= 1000
    second = supervised.run_supervised(changed, tmp_path / "second")
    assert first["eligible_reporting_rows"] == 120
    assert first["facilities"]["overlap"] == 0
    assert first["development_training_expense_threshold"] == second["development_training_expense_threshold"]
    assert first["regression_selection"] == second["regression_selection"]
    assert first["classification_selection"] == second["classification_selection"]
    assert first["regression_cv_mean"] == second["regression_cv_mean"]
    assert first["classification_cv_mean"] == second["classification_cv_mean"]
    choice = first["regression_selection"]["selected_model"]
    assert first["regression_holdout"][choice]["mae"] != second["regression_holdout"][choice]["mae"]
    assert (tmp_path / "first" / "supervised_metrics.json").is_file()


def test_inadequate_target_data_fails_explicitly(tmp_path):
    with pytest.raises(ValueError, match="at least ten facilities"):
        supervised.run_supervised(reporting_frame(5), tmp_path)
