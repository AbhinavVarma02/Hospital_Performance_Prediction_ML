import numpy as np
import pandas as pd
import pytest

from hospital_ml.clustering import (
    CAPACITY_COLUMNS, CLUSTER_FEATURES, CapacityClipper,
    build_capacity_table, capacity_preprocessor,
)


def reporting_rows():
    return pd.DataFrame({
        "FAC_NO": ["a", "a", "b", "b"],
        "BEG_DATE": ["01JAN2012"] * 4, "END_DATE": ["31DEC2012"] * 4,
        "BED_LIC": [100, 100, 200, 200], "BED_AVL": [90, 90, 180, 180],
        "BED_STF": [45, 45, 90, 90], "DAY_PER": [366] * 4,
        "Variable": ["TOT_OP_EXP", "OCC_AVL"] * 2,
        "Amount": [1e8, .5, 2e8, .6],
    })


def test_capacity_unit_preserves_long_rows_without_mixing_amount_units():
    df = reporting_rows()
    table = build_capacity_table(df)
    assert len(table) == 2
    assert table.reporting_rows.sum() == len(df)
    assert table.available_share.tolist() == [.9, .9]
    assert table.staffed_share.tolist() == [.5, .5]
    changed = df.assign(Amount=[-1e15, 9e15, 7, 8], Variable="anything")
    pd.testing.assert_frame_equal(build_capacity_table(changed), table)


def test_conflicting_capacities_are_rejected_not_silently_averaged():
    df = reporting_rows()
    df.loc[1, "BED_STF"] = 20
    with pytest.raises(ValueError, match="Conflicting capacities"):
        build_capacity_table(df)


def test_zero_denominator_and_negative_beds_become_missing():
    df = reporting_rows()
    df.loc[:1, ["BED_LIC", "BED_AVL"]] = 0
    df.loc[2:, "BED_STF"] = -1
    table = build_capacity_table(df)
    assert table.loc[0, ["available_share", "staffed_share"]].isna().all()
    assert np.isnan(table.loc[1, "staffed_share"])
    assert "Amount" not in CLUSTER_FEATURES


def test_preprocessing_does_not_learn_holdout_extremes():
    train = np.array([[1., .3, .1], [2., .5, .2], [3., .7, .3], [4., .9, .4]])
    pipeline = capacity_preprocessor().fit(train)
    before = pipeline.named_steps["clip"].upper_bounds_.copy()
    transformed = pipeline.transform([[1e10, np.nan, .8]])
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(before, pipeline.named_steps["clip"].upper_bounds_)
    assert transformed[0, 0] < 2
