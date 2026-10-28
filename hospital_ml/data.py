"""Data preparation and a separate copy of the coursework cleaning workflow."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path('data/raw/hospital_2012_2017.csv')
REQUIRED = ['FAC_NO', 'FAC_NAME', 'BEG_DATE', 'END_DATE', 'COUNTY', 'DAY_PER',
            'BED_LIC', 'BED_AVL', 'BED_STF', 'Variable', 'Amount']
LEGACY_COLUMNS = ['FAC_NAME', 'BED_LIC', 'BED_AVL', 'BED_STF', 'COUNTY',
                  'DAY_PER', 'Variable', 'Amount']
NUMERIC = ['DAY_PER', 'BED_LIC', 'BED_AVL', 'BED_STF', 'Amount']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_raw(path=RAW_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} is missing. Run python scripts/fetch_data.py first.')
    frame = pd.read_csv(path, low_memory=False, dtype={'FAC_NO': 'string'})
    absent = sorted(set(REQUIRED) - set(frame))
    if absent:
        raise ValueError(f'Missing required columns: {absent}')
    return frame


def prepare_data(raw):
    """Preserve report identity and missing predictors without learned cleaning.

    Deduplication precedes projection so equal values in different years survive.
    Invalid numeric predictors become missing, for train-only pipeline imputation.
    No target-based filtering, quantiles, encodings, or scaling occur here.
    """
    full = raw.drop_duplicates().copy()
    frame = full[REQUIRED].copy()
    audit = {'raw_rows': len(raw), 'raw_columns': len(raw.columns),
             'exact_full_schema_duplicates_removed': len(raw) - len(full)}
    for col in ['FAC_NO', 'FAC_NAME', 'COUNTY', 'Variable', 'BEG_DATE', 'END_DATE']:
        frame[col] = frame[col].astype('string').str.strip().replace('', pd.NA)
    invalid_keys = frame[['FAC_NO', 'BEG_DATE', 'END_DATE', 'Variable']].isna().any(axis=1)
    audit['missing_identity_rows_excluded'] = int(invalid_keys.sum())
    frame = frame.loc[~invalid_keys].copy()
    invalid_dates = {}
    for col in ['BEG_DATE', 'END_DATE']:
        dates = pd.to_datetime(frame[col], format='%d%b%Y', errors='coerce')
        invalid_dates[col] = int(dates.isna().sum())
    if any(invalid_dates.values()):
        raise ValueError(f'Unparseable report dates: {invalid_dates}')
    audit['numeric_coercions_to_missing'] = {}
    for col in NUMERIC:
        values = pd.to_numeric(frame[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        audit['numeric_coercions_to_missing'][col] = int((frame[col].notna() & values.isna()).sum())
        frame[col] = values.astype(float)
    # Negative capacity/duration is invalid by definition; ratios are not censored.
    audit['invalid_nonnegative_predictors_to_missing'] = {}
    for col in NUMERIC[:-1]:
        invalid = frame[col].lt(0) | (frame[col].eq(0) if col == 'DAY_PER' else False)
        audit['invalid_nonnegative_predictors_to_missing'][col] = int(invalid.sum())
        frame.loc[invalid, col] = np.nan
    frame['COUNTY'] = frame['COUNTY'].astype(object).where(frame['COUNTY'].notna(), np.nan)
    audit.update({'prepared_reporting_rows': len(frame), 'facilities': frame.FAC_NO.nunique(),
                  'facility_reporting_periods': len(frame[['FAC_NO', 'BEG_DATE', 'END_DATE']].drop_duplicates()),
                  'variables': frame.Variable.nunique(),
                  'missing_values': {col: int(frame[col].isna().sum()) for col in frame}})
    return frame.reset_index(drop=True), audit


def historical_cleaning(raw):
    """Run the coursework cleaning rules separately from model evaluation.

    Projected deduplication and global IQR filters reproduce the original method.
    Updated models use prepare_data and fit preprocessing within training folds.
    """
    df = raw[LEGACY_COLUMNS].dropna().copy()
    steps = [{'step': 'raw', 'rows': len(raw)}, {'step': 'drop_missing_selected_columns', 'rows': len(df)}]
    df = df.drop_duplicates()
    steps.append({'step': 'historical_projected_deduplication', 'rows': len(df)})
    for col in ['BED_LIC', 'BED_AVL', 'BED_STF', 'Amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['BED_LIC', 'BED_AVL', 'BED_STF', 'Amount'])
    steps.append({'step': 'drop_non_numeric_values', 'rows': len(df)})
    for col in ['BED_LIC', 'BED_AVL', 'BED_STF']:
        q1, q3 = df[col].quantile([0.25, 0.75])
        before = len(df)
        df = df[df[col].between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))]
        steps.append({'step': f'historical_sequential_iqr_{col}', 'rows': len(df), 'removed': before - len(df)})
    return df.reset_index(drop=True), steps


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
