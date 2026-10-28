import numpy as np
import pandas as pd
from hospital_ml.common import QuantileClipper, facility_split
from hospital_ml.data import prepare_data


def raw_fixture():
    return pd.DataFrame({
        'FAC_NO': ['001', '001', '002', '003', '004'], 'FAC_NAME': ['A', 'A', 'B', 'C', 'D'],
        'BEG_DATE': ['01JAN2012', '01JAN2013', '01JAN2012', '01JAN2012', '01JAN2012'],
        'END_DATE': ['31DEC2012', '31DEC2013', '31DEC2012', '31DEC2012', '31DEC2012'],
        'COUNTY': ['X', 'X', None, 'Y', 'Z'], 'DAY_PER': [366, 365, 366, 366, 366],
        'BED_LIC': ['100', '100', '*', '5000', '-1'], 'BED_AVL': [100, 100, 20, 4000, 50],
        'BED_STF': [60, 60, 10, 1000, 20], 'Variable': ['TOT_OP_EXP'] * 5,
        'Amount': [0, 0, -100, 1e12, 200], 'EXTRA_FULL_SCHEMA_FIELD': [1, 1, 1, 1, 1]})


def test_dedup_preserves_different_reporting_periods_and_missing_predictors():
    source = raw_fixture()
    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    cleaned, audit = prepare_data(source)
    assert len(cleaned) == 5
    assert audit['exact_full_schema_duplicates_removed'] == 1
    assert cleaned.iloc[2].BED_LIC != cleaned.iloc[2].BED_LIC
    assert cleaned.iloc[4].BED_LIC != cleaned.iloc[4].BED_LIC
    assert cleaned.Amount.tolist() == [0, 0, -100, 1e12, 200]
    assert cleaned.iloc[0].FAC_NO == '001'


def test_clipping_uses_fit_data_and_retains_test_outliers():
    clip = QuantileClipper(lower=0, upper=1).fit([[0], [10]])
    assert clip.transform([[1000], [-900]]).tolist() == [[10], [0]]
    assert clip.upper_bounds_.tolist() == [10]


def test_split_keeps_every_facility_in_one_partition():
    source = raw_fixture()
    train, test = facility_split(source)
    assert set(source.iloc[train].FAC_NO).isdisjoint(source.iloc[test].FAC_NO)
    assert len(train) + len(test) == len(source)
