"""Facility-held-out baselines for one comparable reporting measure.

The target is reported total operating expense, never clinical quality, demand,
readmissions, or efficiency. A reporting-period amount is not a forecast.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .common import QuantileClipper, facility_split


NUMERIC_FEATURES = ["BED_LIC", "AVAILABLE_SHARE", "STAFFED_SHARE", "DAY_PER"]
CATEGORICAL_FEATURES = ["COUNTY"]


def expense_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic features; identities and outcome are deliberately absent."""
    result = frame[["BED_LIC", "DAY_PER", "COUNTY"]].copy()
    licensed = pd.to_numeric(frame["BED_LIC"], errors="coerce")
    available = pd.to_numeric(frame["BED_AVL"], errors="coerce")
    staffed = pd.to_numeric(frame["BED_STF"], errors="coerce")
    # A zero denominator conveys no defined share; imputation is learned later.
    result["AVAILABLE_SHARE"] = available.where(available >= 0) / licensed.where(licensed > 0)
    result["STAFFED_SHARE"] = staffed.where(staffed >= 0) / available.where(available > 0)
    result["BED_LIC"] = licensed.where(licensed >= 0)
    result["DAY_PER"] = pd.to_numeric(result["DAY_PER"], errors="coerce").where(
        pd.to_numeric(result["DAY_PER"], errors="coerce") > 0
    )
    result[NUMERIC_FEATURES] = result[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    # Use numpy NaN rather than pandas.NA for sklearn 1.6's categorical imputer.
    result["COUNTY"] = result["COUNTY"].astype(object).where(result["COUNTY"].notna(), np.nan)
    return result[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def make_preprocessor(clip: bool = True) -> ColumnTransformer:
    """All empirical quantities are fitted inside the pipeline/CV fold."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("clip", QuantileClipper(lower=0.01, upper=0.99) if clip else "passthrough"),
            ("scale", RobustScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def _pipeline(model, clip: bool = True) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor(clip)), ("model", model)])


def _regression_candidates(seed: int) -> dict:
    return {
        "dummy_mean": _pipeline(DummyRegressor(strategy="mean")),
        "dummy_median": _pipeline(DummyRegressor(strategy="median")),
        "ridge": _pipeline(Ridge(alpha=10.0)),
        "random_forest": _pipeline(
            RandomForestRegressor(
                n_estimators=80, max_depth=12, min_samples_leaf=5, random_state=seed, n_jobs=1
            )
        ),
    }


def _classification_candidates(seed: int) -> dict:
    return {
        "dummy_most_frequent": _pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": _pipeline(LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        "random_forest": _pipeline(
            RandomForestClassifier(
                n_estimators=80, max_depth=12, min_samples_leaf=5, random_state=seed, n_jobs=1
            )
        ),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, probability) -> dict:
    has_both = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if has_both else None,
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)) if has_both else None,
        "pr_auc_average_precision": float(average_precision_score(y_true, probability)) if has_both else None,
        "positive_prevalence": float(np.mean(y_true)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def _probability(fitted, X) -> np.ndarray:
    classes = fitted.named_steps["model"].classes_
    probability = fitted.predict_proba(X)
    return probability[:, np.flatnonzero(classes == 1)[0]] if 1 in classes else np.zeros(len(X))


def _assert_disjoint(frame: pd.DataFrame, train, test) -> None:
    if set(frame.iloc[train]["FAC_NO"]) & set(frame.iloc[test]["FAC_NO"]):
        raise AssertionError("Facility leakage: train and validation/test facilities overlap")


def _cv_regression(X, y, groups, folds, candidates) -> pd.DataFrame:
    rows = []
    for fold, (train, validation) in enumerate(folds, start=1):
        if set(groups[train]) & set(groups[validation]):
            raise AssertionError("Facility overlap within CV")
        for name, candidate in candidates.items():
            fitted = clone(candidate).fit(X.iloc[train], y[train])
            rows.append({"fold": fold, "model": name, **regression_metrics(y[validation], fitted.predict(X.iloc[validation]))})
    return pd.DataFrame(rows)


def _cv_classification(X, expense, groups, folds, candidates) -> pd.DataFrame:
    rows = []
    for fold, (train, validation) in enumerate(folds, start=1):
        if set(groups[train]) & set(groups[validation]):
            raise AssertionError("Facility overlap within CV")
        threshold = float(np.median(expense[train]))
        train_label = (expense[train] > threshold).astype(int)
        validation_label = (expense[validation] > threshold).astype(int)
        if len(np.unique(train_label)) < 2 or len(np.unique(validation_label)) < 2:
            raise ValueError("Expense classification needs both classes in every grouped CV split")
        for name, candidate in candidates.items():
            fitted = clone(candidate).fit(X.iloc[train], train_label)
            metrics = classification_metrics(validation_label, fitted.predict(X.iloc[validation]), _probability(fitted, X.iloc[validation]))
            metrics.pop("confusion_matrix_labels_0_1")
            rows.append({"fold": fold, "model": name, "training_expense_median": threshold, **metrics})
    return pd.DataFrame(rows)


def facility_bootstrap_intervals(y, prediction, groups, *, probability=None, seed=42, repetitions=200) -> dict:
    """Percentile intervals resample whole facilities, conditional on fitted model.

    Multiple observations from the same reporting facility are never independently
    resampled. These intervals do not include model-training uncertainty.
    """
    y, prediction, groups = np.asarray(y), np.asarray(prediction), np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return {"method": "unavailable: fewer than two held-out facilities"}
    indices = {facility: np.flatnonzero(groups == facility) for facility in unique}
    rng = np.random.default_rng(seed)
    metrics = []
    for _ in range(repetitions):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([indices[facility] for facility in selected])
        if probability is None:
            metric = regression_metrics(y[sample], prediction[sample])
        else:
            if len(np.unique(y[sample])) < 2:
                continue
            metric = classification_metrics(y[sample], prediction[sample], np.asarray(probability)[sample])
            metric.pop("confusion_matrix_labels_0_1")
        metrics.append(metric)
    values = pd.DataFrame(metrics)
    return {
        "method": "95% percentile bootstrap of held-out facilities; conditional on fitted model",
        "requested_repetitions": repetitions,
        "valid_repetitions": len(metrics),
        "intervals": {column: [float(values[column].quantile(0.025)), float(values[column].quantile(0.975))] for column in values},
    }


def run_supervised(df: pd.DataFrame, output_dir, seed: int = 42, target_variable: str = "TOT_OP_EXP") -> dict:
    """Run prespecified facility CV, freeze choices, then open one holdout."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = df.loc[df["Variable"].astype(str).str.strip() == target_variable].copy()
    source_rows = len(frame)
    numeric_target = pd.to_numeric(frame["Amount"], errors="coerce")
    eligible = np.isfinite(numeric_target) & frame["FAC_NO"].notna()
    removed_missing_target_or_facility = int((~eligible).sum())
    frame = frame.loc[eligible].reset_index(drop=True)
    frame["Amount"] = numeric_target.loc[eligible].to_numpy()
    if len(frame) == 0 or frame["FAC_NO"].nunique() < 10:
        raise ValueError(f"Need at least ten facilities with finite {target_variable} values")
    development, holdout = facility_split(frame, test_size=0.2, seed=seed)
    _assert_disjoint(frame, development, holdout)
    X = expense_features(frame)
    y = frame["Amount"].to_numpy(dtype=float)
    groups = frame["FAC_NO"].to_numpy()
    X_dev, y_dev, dev_groups = X.iloc[development].reset_index(drop=True), y[development], groups[development]
    folds = list(GroupKFold(n_splits=3).split(X_dev, y_dev, groups=dev_groups))
    regression = _regression_candidates(seed)
    classification = _classification_candidates(seed)
    cv_regression = _cv_regression(X_dev, y_dev, dev_groups, folds, regression)
    cv_classification = _cv_classification(X_dev, y_dev, dev_groups, folds, classification)
    regression_choice = str(cv_regression.groupby("model")["mae"].mean().idxmin())
    classification_choice = str(cv_classification.groupby("model")["balanced_accuracy"].mean().idxmax())
    cv_regression.to_csv(output_dir / "regression_cv.csv", index=False)
    cv_classification.to_csv(output_dir / "classification_cv.csv", index=False)

    # Outlier sensitivity uses development folds only; it cannot select by test.
    # Hold the selected estimator fixed, compare feature clipping vs no clipping.
    reg_unclipped = clone(regression[regression_choice]).set_params(preprocess__numeric__clip="passthrough")
    cls_unclipped = clone(classification[classification_choice]).set_params(preprocess__numeric__clip="passthrough")
    sensitivity_reg = _cv_regression(X_dev, y_dev, dev_groups, folds, {"selected_without_clipping": reg_unclipped})
    sensitivity_cls = _cv_classification(X_dev, y_dev, dev_groups, folds, {"selected_without_clipping": cls_unclipped})
    sensitivity_reg.to_csv(output_dir / "regression_outlier_sensitivity_cv.csv", index=False)
    sensitivity_cls.to_csv(output_dir / "classification_outlier_sensitivity_cv.csv", index=False)

    # The fixed holdout is consulted only after choices and thresholds are frozen.
    X_test, y_test = X.iloc[holdout], y[holdout]
    threshold = float(np.median(y_dev))
    class_dev, class_test = (y_dev > threshold).astype(int), (y_test > threshold).astype(int)
    regression_holdout, classification_holdout = {}, {}
    selected_predictions, selected_class_predictions, selected_probability = None, None, None
    for name in dict.fromkeys(["dummy_mean", "dummy_median", regression_choice]):
        fitted = clone(regression[name]).fit(X_dev, y_dev)
        prediction = fitted.predict(X_test)
        regression_holdout[name] = regression_metrics(y_test, prediction)
        regression_holdout[name]["facility_macro_mae"] = float(
            pd.DataFrame({"facility": groups[holdout], "absolute_error": np.abs(y_test - prediction)})
            .groupby("facility")["absolute_error"].mean().mean()
        )
        if name == regression_choice:
            selected_predictions = prediction
    for name in dict.fromkeys(["dummy_most_frequent", classification_choice]):
        fitted = clone(classification[name]).fit(X_dev, class_dev)
        prediction, probability = fitted.predict(X_test), _probability(fitted, X_test)
        classification_holdout[name] = classification_metrics(class_test, prediction, probability)
        classification_holdout[name]["facility_macro_accuracy"] = float(
            pd.DataFrame({"facility": groups[holdout], "correct": prediction == class_test})
            .groupby("facility")["correct"].mean().mean()
        )
        if name == classification_choice:
            selected_class_predictions, selected_probability = prediction, probability

    report = {
        "target_variable": target_variable,
        "target_description": "Reported total operating expenses" if target_variable == "TOT_OP_EXP" else f"Reported Amount for {target_variable}",
        "target_units": "Source-reported dollars; reporting-period totals, not inflation adjusted" if target_variable == "TOT_OP_EXP" else "Units defined by the selected reporting Variable",
        "classification_definition": "Amount above the development-training median for this reporting Variable; exploratory expense category",
        "seed": seed,
        "source_reporting_rows_for_target": source_rows,
        "eligible_reporting_rows": len(frame),
        "excluded_missing_or_nonfinite_target_or_missing_facility": removed_missing_target_or_facility,
        "negative_target_rows_retained": int((y < 0).sum()),
        "zero_target_rows_retained": int((y == 0).sum()),
        "rows": {"development": len(development), "holdout": len(holdout)},
        "facilities": {"development": int(len(np.unique(dev_groups))), "holdout": int(len(np.unique(groups[holdout]))), "overlap": 0},
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "validation": "20% facility holdout; three GroupKFold development folds; every fitted preprocessing statistic and expense threshold learned in training only",
        "feature_outliers": "1st/99th percentile clipping learned within each training fold; target values retained; no-clipping sensitivity uses development folds only",
        "regression_selection": {"criterion": "minimum mean development-fold MAE, including dummy baselines", "selected_model": regression_choice},
        "classification_selection": {"criterion": "maximum mean development-fold balanced accuracy, including dummy baseline", "selected_model": classification_choice},
        "development_training_expense_threshold": threshold,
        "classification_cv_training_thresholds": cv_classification.groupby("fold")["training_expense_median"].first().to_dict(),
        "regression_cv_mean": cv_regression.groupby("model")[["mae", "rmse", "r2", "median_absolute_error"]].mean().to_dict(orient="index"),
        "classification_cv_mean": cv_classification.groupby("model")[["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc_average_precision"]].mean().to_dict(orient="index"),
        "development_sensitivity_without_clipping": {"regression_mean_mae": float(sensitivity_reg["mae"].mean()), "classification_mean_balanced_accuracy": float(sensitivity_cls["balanced_accuracy"].mean()), "used_for_selection": False},
        "regression_holdout": regression_holdout,
        "classification_holdout": classification_holdout,
        "regression_selected_model_ci": facility_bootstrap_intervals(y_test, selected_predictions, groups[holdout], seed=seed),
        "classification_selected_model_ci": facility_bootstrap_intervals(class_test, selected_class_predictions, groups[holdout], probability=selected_probability, seed=seed),
        "limitations": [
            "Expense is a financial reporting measure, not a verified clinical or efficiency outcome.",
            "Facility holdout estimates transfer to unseen facilities in this historical extract; no temporal forecasting claim is made.",
            "Report periods and nominal dollar values differ; DAY_PER is included but there is no inflation or case-mix adjustment.",
            "The expense-category cutoff is an exploratory training median and has no clinical or policy interpretation.",
            "Facility bootstrap intervals are conditional on the fitted model and omit training and source-reporting uncertainty.",
            "Primary metrics and model selection weight reporting rows equally; facilities with more report periods have more influence. Facility-macro MAE/accuracy provide an equal-facility holdout sensitivity.",
            "Licensed beds are end-of-period counts; available and staffed beds are period averages. Their ratios are descriptive capacity features, not simultaneous utilization measures, and can exceed one.",
        ],
    }
    (output_dir / "supervised_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(regression_holdout).T.rename_axis("model").to_csv(output_dir / "regression_holdout.csv")
    flat_classification = {name: {key: value for key, value in metrics.items() if key != "confusion_matrix_labels_0_1"} for name, metrics in classification_holdout.items()}
    pd.DataFrame(flat_classification).T.rename_axis("model").to_csv(output_dir / "classification_holdout.csv")
    return report
