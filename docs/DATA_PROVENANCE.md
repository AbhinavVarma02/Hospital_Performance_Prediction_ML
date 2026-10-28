# Data provenance

The original project was developed collaboratively by **Abhinav Varma and Devika Rani Sanaboyina** for UMBC DATA 602. The three coursework notebooks and two submitted PDFs remain unchanged. Updated workflows and reports are stored separately.

## Coursework artifacts

The starting revision is [`1f00286346c6a4549059bf804994c3765a73a156`](https://github.com/AbhinavVarma02/Hospital_Performance_Prediction_ML/tree/1f00286346c6a4549059bf804994c3765a73a156), dated May 22, 2025. The notebooks and PDFs were uploaded in [`5927fc52526033aa2ea81cb5ea1bb0e0e878012b`](https://github.com/AbhinavVarma02/Hospital_Performance_Prediction_ML/commit/5927fc52526033aa2ea81cb5ea1bb0e0e878012b). Git history retains the collaborators' README contributions. Upload authorship does not apportion the collaborative analysis.

| Original artifact | SHA-256 |
| --- | --- |
| `602_Regression_Model_Abhinav&Devika.ipynb` | `0bdac7d3d4196029edf26449cb4c342b8fcc6b5f70e8d80bef18af1d2a9bbcd4` |
| `602_Clustering_&_Classification_Model_Abhinav&Devika.ipynb` | `004c9eda8655183ad76fea0691ec44c3164d6e65be9c1f029afc04427599474e` |
| `602_Classification_Model_Abhinav&Devika.ipynb` | `b10dc78bd13cc705517bafaa85e4d0deedeb9ca6bc0d80ae9c1d84bede85b75a` |
| `602 executive  summary -  Devika  & Abhinav.pdf` | `92a961f77faef12ba0052c4a2ff3a9c872fef164d4ae41ebc7d2307d2d74a556` |
| `DATA 602 FINAL PROJECT PPT[1].pdf` | `3b5ac7998c98418dd5336ea134440d6b4e02ca1a8baa8a89113e2d3138161e4a` |

The original README remains available in Git at the starting revision; its SHA-256 is `0fb68a852c93dc28bc2e9bf002dc710912a70a22f39548c4dea6f2061fa79e6b`. The working README documents the current workflows.

Permanent source views: [regression](https://github.com/AbhinavVarma02/Hospital_Performance_Prediction_ML/blob/1f00286346c6a4549059bf804994c3765a73a156/602_Regression_Model_Abhinav%26Devika.ipynb), [clustering/classification](https://github.com/AbhinavVarma02/Hospital_Performance_Prediction_ML/blob/1f00286346c6a4549059bf804994c3765a73a156/602_Clustering_%26_Classification_Model_Abhinav%26Devika.ipynb), and [raw classification](https://github.com/AbhinavVarma02/Hospital_Performance_Prediction_ML/blob/1f00286346c6a4549059bf804994c3765a73a156/602_Classification_Model_Abhinav%26Devika.ipynb).

`python scripts/verify_artifacts.py` checks the five original file fingerprints and the saved execution evidence for the added notebooks. Original outputs remain in place, with text copies under `reports/historical/`.

## Source data

The source is California HCAI/OSHPD hospital annual financial reporting data for 2012–2017. The [official resource](https://data.chhs.ca.gov/dataset/ea0c8ca9-023e-46a3-b95b-b9d4ab8ec195/resource/81602900-aa04-4fcb-b70b-19bc0553f79b/download/selected-file_2012_2017_machinereadable.csv) and [resource landing page](https://data.ca.gov/@department-of-health-care-access-and-information/hospital-annual-financial-data-selected-data-pivot-tables/r/50114563-deae-4fcb-8900-9533f9d59f8f) identify the retrieved extract. It contains 578,350 rows and 35 columns, covering 472 facilities and 2,690 facility reporting periods.

A long-format row represents a reporting measure for a facility period. Rows are not individual patients, discharges, or independent hospitals. There are 228 `Variable` values in the prepared data and no exact full-schema duplicate rows.

The [source manifest](../data/source_manifest.json) records retrieval date, URLs, input dimensions, and SHA-256 fingerprints. The raw extract fingerprint is `0325bca518c17678a4e7bc85325e89fe3a8bc7c8bfcc1e0653caafbeae0a7a91`. The optional shared coursework cleaned file, referenced by Drive ID `1PCU0ibbltz_Ir4xWLV-ewzphm0kR1_Ta`, has fingerprint `0ac8790edaea1ff4e0de5257c186319832da963143903a235684f22d410d4fbc`.

The original raw Drive link was unavailable during retrieval, so the updated workflow uses the official resource. Original notebooks did not record input hashes or library versions. Their saved outputs therefore remain separate from the newly fingerprinted sources and executions.

## Measurement definitions

The [official dictionary](https://data.chhs.ca.gov/dataset/ea0c8ca9-023e-46a3-b95b-b9d4ab8ec195/resource/842c8bcb-bfac-4aa9-9de8-b11eb049f6de/download/hafdselectedfiledoc2004.pdf) defines:

| Column or measure | Definition | Dictionary location |
| --- | --- | --- |
| `BED_LIC` | Licensed beds at the end of the reporting period | PDF page 5, printed page 3 |
| `BED_AVL` | Average available beds | PDF page 5, printed page 3 |
| `BED_STF` | Average staffed beds | PDF page 5, printed page 3 |
| `TOT_OP_EXP` | Total operating expenses, P8 C1 L200 | PDF page 9, printed page 7 |

Staffed beds are not employee headcount. Available/licensed and staffed/available ratios are descriptive capacity features, not validated efficiency or simultaneous utilization measures. They can exceed one because the input reporting summaries differ.

`Amount` takes its unit from `Variable`. Pooled amounts cannot be interpreted consistently as dollars, workload, or readmissions. The updated supervised analysis uses only `TOT_OP_EXP`; capacity clustering excludes `Amount`.

## Data preparation

The [data audit](../reports/data/audit.json) records type conversions, missing values, reporting identities, and source dimensions. The updated preparation keeps report dates, removes only full-schema duplicates before projection, and converts invalid numeric predictors to missing. Pipeline imputation and other learned preprocessing occur inside training partitions. No global target thresholds or predictor quantiles are fitted during preparation.

The separate coursework workflow retains its original selected-column cleaning sequence: missing-value removal, projected deduplication, numeric coercion, and sequential bed-IQR filtering. It produces 513,363 cleaned reporting rows. The [coursework cleaning report](../reports/historical/cleaning_replay.json) records these stages. That globally filtered dataset is not an input to updated model validation.

The capacity analysis uses `FAC_NO`, `BEG_DATE`, and `END_DATE` to construct one capacity vector per reporting period. It validates that capacity and duration values agree within each group. Clustering uses log licensed beds and two availability shares. Supervised tasks use one reported expense observation per period. The [methods](METHODS.md) document feature construction, facility separation, and evaluation.

## Coursework limitations

Every original code cell has a null execution counter despite retained outputs. The files preserve code and results without establishing a clean sequential execution or a recoverable library environment.

The original clustering constructed a four-column `X_cluster_scaled` matrix but fitted K-Means and PCA to the earlier seven-column `X_scaled`, which contained ordinal facility, county, and measure encodings and excluded `Amount`. The updated workflow uses one explicit capacity matrix consistently.

The original predictive workflows fitted scaling and encoding before row-wise splitting; repeated facilities could occur in both partitions. Their metrics do not estimate transfer to unseen facilities. The original outlier comparisons also change features and cleaning, so they do not isolate outlier handling. Some original narrative descriptions conflict with saved outputs or measure definitions. These remain preserved as coursework artifacts; updated results use the documented task definitions and validation.

The new results are stored separately with source fingerprints, features, split definitions, seeds, and execution records. Different tasks and analytical units prevent direct comparison with the original scores as evidence of model improvement.
