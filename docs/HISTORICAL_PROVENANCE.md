# Historical provenance and preservation

The original project was developed collaboratively by **Abhinav Varma and Devika Rani Sanaboyina** for UMBC DATA 602. The three original notebooks and two submitted PDFs remain unchanged. Improved work is added separately; historical outputs are not overwritten with results from a different method.

## Immutable source

The starting revision is [`1f00286346c6a4549059bf804994c3765a73a156`](https://github.com/AbhinavVarma02/Hospital_Performance_Prediction_ML/tree/1f00286346c6a4549059bf804994c3765a73a156), dated May 22, 2025. Notebook and PDF files were uploaded in [`5927fc52526033aa2ea81cb5ea1bb0e0e878012b`](https://github.com/AbhinavVarma02/Hospital_Performance_Prediction_ML/commit/5927fc52526033aa2ea81cb5ea1bb0e0e878012b). Git history also preserves both collaborators' README contributions. Commit authorship records who uploaded a file; it does not apportion the underlying collaborative analysis.

| Original artifact | SHA-256 at the starting revision |
| --- | --- |
| `602_Regression_Model_Abhinav&Devika.ipynb` | `0bdac7d3d4196029edf26449cb4c342b8fcc6b5f70e8d80bef18af1d2a9bbcd4` |
| `602_Clustering_&_Classification_Model_Abhinav&Devika.ipynb` | `004c9eda8655183ad76fea0691ec44c3164d6e65be9c1f029afc04427599474e` |
| `602_Classification_Model_Abhinav&Devika.ipynb` | `b10dc78bd13cc705517bafaa85e4d0deedeb9ca6bc0d80ae9c1d84bede85b75a` |
| `602 executive  summary -  Devika  & Abhinav.pdf` | `92a961f77faef12ba0052c4a2ff3a9c872fef164d4ae41ebc7d2307d2d74a556` |
| `DATA 602 FINAL PROJECT PPT[1].pdf` | `3b5ac7998c98418dd5336ea134440d6b4e02ca1a8baa8a89113e2d3138161e4a` |
| Original `README.md` (subsequently updated) | `0fb68a852c93dc28bc2e9bf002dc710912a70a22f39548c4dea6f2061fa79e6b` |

The README hash identifies the historical README in Git, not the updated working README. The original notebooks/PDFs can be compared directly with the starting revision:

```bash
git diff 1f00286346c6a4549059bf804994c3765a73a156 -- '*.ipynb' '*.pdf'
```

When running this after adding improved notebooks, inspect the five original paths specifically: new notebook additions are expected.

## Evidence for the existing résumé claims

The submitted statements are preserved here verbatim:

> Profiled capacity/spending patterns in 578K hospital report rows using Python, Pandas and CRISP-DM.
>
> Segmented 513K+ cleaned hospital records into resource-intensity groups using K-Means and PCA.

The row-count evidence is in saved notebook outputs at the immutable revision. Cell numbers below are **one-based positions among all cells**, including Markdown; they are not execution counters.

| Evidence | Original location | Saved output |
| --- | --- | --- |
| Raw reporting rows | Regression notebook, cell 2; raw classification notebook, cell 2 | `(578350, 35)` |
| Rows after dropping nulls in eight selected columns | Regression notebook, cell 5 | `(577920, 8)` |
| Rows after projected-column duplicate removal | Regression notebook, cell 6 | 24,898 duplicates; `(553022, 8)` |
| Rows after numeric coercion and sequential bed-IQR filtering | Regression notebook, cell 12 | `(513363, 8)` |
| Loaded cleaned data supplied to clustering | Clustering notebook, cell 5 | `(513363, 11)` |
| K-Means and PCA application | Clustering notebook, cell 8 | Three cluster summary rows and a PCA plot |

Permanent source views: [regression notebook](https://github.com/AbhinavVarma02/Hospital_Performance_Prediction_ML/blob/1f00286346c6a4549059bf804994c3765a73a156/602_Regression_Model_Abhinav%26Devika.ipynb), [clustering notebook](https://github.com/AbhinavVarma02/Hospital_Performance_Prediction_ML/blob/1f00286346c6a4549059bf804994c3765a73a156/602_Clustering_%26_Classification_Model_Abhinav%26Devika.ipynb), [raw classification notebook](https://github.com/AbhinavVarma02/Hospital_Performance_Prediction_ML/blob/1f00286346c6a4549059bf804994c3765a73a156/602_Classification_Model_Abhinav%26Devika.ipynb).

These are **reporting rows**, with repeated facilities, reporting periods and `Variable` entries. They are not counts of distinct hospitals, patients or independent observations. The 578K statement rounds the saved 578,350 rows; 513K+ refers to the saved 513,363-row historical cleaned dataset. No improved run is required to have either count, and no rows are fabricated, duplicated or removed to reach a résumé number.

The historical source shows Python/Pandas profiling, preprocessing, modeling and evaluation. CRISP-DM is explicitly documented in the original executive summary (page 2) and presentation (page 4). Be prepared to explain the phases actually followed, including that deployment was not established by these notebooks.

## Limits of the historical evidence

Every original code cell has a null `execution_count` despite saved outputs. The notebooks therefore preserve historical code and results but do not establish a clean, ordered execution. The original CSVs were downloaded from Google Drive, were not committed, and did not have recorded content hashes. A fresh data download must be identified separately from the original saved outputs.

The original cleaning outputs do not log the number removed by numeric coercion. A fresh replay now measures that stage explicitly: 553,022 rows become 542,702 rows, a reduction of 10,320. The subsequent bed filters remove 11,267, 5,590 and 12,482 rows respectively, yielding 513,363 rows. These are calculated observations, not hard-coded quotas; see the replay evidence below.

The historical clustering code creates a four-column `X_cluster_scaled` matrix (`BED_LIC`, `BED_AVL`, `BED_STF`, `Amount`) but actually passes the earlier seven-column `X_scaled` matrix to **both** K-Means and PCA. That earlier matrix contains bed columns, `DAY_PER`, and ordinal encodings of facility, county and `Variable`, and excludes `Amount`. Thus the historical output is evidence of K-Means/PCA segmentation of the cleaned reporting rows; it is not evidence that the intended four-column resource matrix was used. The improved notebooks correct this explicitly and report their own results.

Historical interpretation and evaluation also need qualification:

- `BED_STF` means average staffed beds, not employee headcount. `BED_AVL` is average available beds, whereas `BED_LIC` is licensed beds at period end. Large bed counts do not establish higher efficiency, better care or rural status. The [official dictionary](https://data.chhs.ca.gov/dataset/ea0c8ca9-023e-46a3-b95b-b9d4ab8ec195/resource/842c8bcb-bfac-4aa9-9de8-b11eb049f6de/download/hafdselectedfiledoc2004.pdf) defines these on PDF page 5 (printed page 3).
- `Amount` is indexed by `Variable`; it must not be treated as dollars, workload or readmission scores indiscriminately across all reporting measures. The executive summary also calls the rows “hospital discharges”; the official file and measure structure establish that these are reporting rows instead.
- Historical cluster labels such as “Efficiency_Cluster” are unvalidated descriptive labels. The data do not establish efficiency rankings, causal benefits or clinical outcomes.
- Scaling/encoding before row-wise splitting exposes preprocessing to test data, and repeated facilities may occur in both splits. Historical predictive metrics are not reliable estimates for unseen facilities.
- The “with outliers” and “without outliers” classifications change features and cleaning as well as outlier handling, so their difference does not isolate an outlier-treatment effect.
- The original raw classification Markdown states 54.20% logistic-regression accuracy, while its saved classification output states 0.60. Preserve both as historical artifacts; do not quote the conflicting prose as a new result.
- RMSE divided by the test-set maximum is not an adequate measure of practical reliability. Historical performance claims should not be carried forward as validated operational results.

## Fresh data and historical cleaning replay

The original raw Google Drive download was unavailable during this work. The same named 2012–2017 reporting extract was retrieved from the [official California Health and Human Services data resource](https://data.chhs.ca.gov/dataset/ea0c8ca9-023e-46a3-b95b-b9d4ab8ec195/resource/81602900-aa04-4fcb-b70b-19bc0553f79b/download/selected-file_2012_2017_machinereadable.csv); its [resource landing page](https://data.ca.gov/@department-of-health-care-access-and-information/hospital-annual-financial-data-selected-data-pivot-tables/r/50114563-deae-4fcb-8900-9533f9d59f8f) identifies the hospital annual financial data source. It contains 578,350 rows and 35 columns. Its SHA-256 is `0325bca518c17678a4e7bc85325e89fe3a8bc7c8bfcc1e0653caafbeae0a7a91`.

The historical cleaned file was separately retrieved from the original notebook's Google Drive file ID, `1PCU0ibbltz_Ir4xWLV-ewzphm0kR1_Ta`. It has 513,363 rows and SHA-256 `0ac8790edaea1ff4e0de5257c186319832da963143903a235684f22d410d4fbc`.

Replaying the original cleaning logic on the official raw extract reproduces **513,363 rows**, and all eight original selected columns match the downloaded historical cleaned data. The row-by-row value comparison excludes the three historical encoded columns. This establishes agreement for the retained analytical values; it does not recover an unknown historical raw-file hash or reproduce the original model metrics.

The committed [cleaning replay report](../reports/historical/cleaning_replay.json) records the measured stages, both input hashes and `values_match_downloaded_cleaned: true`. The separate [improved data audit](../reports/data/audit.json) identifies 472 facilities, 2,690 facility/report periods, 228 reporting measures and no exact full-schema duplicates. This also explains why the improved analysis retains all 578,350 reporting rows: the original deduplication dropped columns identifying report periods before deciding which rows were duplicates. [The source manifest](../data/source_manifest.json) pins the retrieved sources, including the official dictionary.

## Improved analysis and result separation

The improved clustering uses **2,690 facility/report-period capacity vectors**, with grouping keys `FAC_NO`, `BEG_DATE` and `END_DATE`. It avoids giving a facility period repeated weight simply because its capacity appears beside many reporting measures. Every source reporting row receives its period's cluster assignment afterward. Its distance features are log licensed beds, available/licensed beds and staffed/available beds; `Amount`, facility identity, county and `Variable` are excluded from the distance matrix. K-Means and PCA use the same transformed capacity matrix. This is a revised analytical unit and feature definition, not an attempt to preserve the original cluster labels or row-count threshold.

The improved supervised tasks select the prespecified `Variable == 'TOT_OP_EXP'` measure, **total operating expenses**, yielding 2,690 reporting-period observations before task eligibility checks. The official dictionary defines this measure on PDF page 9 (printed page 7). Regression predicts that reported amount; classification identifies an amount above the training-fold median for that measure. It is an exploratory expense category, not a demand forecast, efficiency label or clinical outcome. Facility-separated development validation and a held-out facility set evaluate the improved method. Training-only imputation, outlier treatment, scaling and categorical encoding replace globally fitted preprocessing.

Fresh model results belong to the improved notebooks and generated reports, with their own data fingerprint, row counts, selected features, split definition and random seeds. The repository's run instructions and current results document identify what was executed successfully. The new task definitions and splits mean their scores should not be compared directly with the original row-wise metrics as a measure of algorithmic improvement. Historical model outputs remain preserved separately.
