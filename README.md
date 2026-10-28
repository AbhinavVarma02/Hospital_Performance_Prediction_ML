# Hospital Performance & Resource Optimization

**Original project collaborators: Abhinav Varma & Devika Rani Sanaboyina, UMBC DATA 602.**

Traditional machine learning analysis of California hospital financial reporting data using Python, Pandas, K-Means, PCA, regression, and classification. This revision improves validation and reproducibility while preserving all three original notebooks and both original PDFs byte for byte.

## Historical scope, preserved and reproduced

The existing résumé statements remain historical descriptions of the collaborative work:

> Profiled capacity/spending patterns in 578K hospital report rows using Python, Pandas and CRISP-DM.
>
> Segmented 513K+ cleaned hospital records into resource-intensity groups using K-Means and PCA.

The fingerprinted official extract contains **578,350 reporting rows**. Replaying the original cleaning produces **513,363 rows**, with an **exact eight-column value match** to the original shared cleaned CSV. These are observed counts; no data were altered to reach them. Rows are reporting measures across periods, not distinct hospitals. The extract covers **472 facilities and 2,690 reporting periods**.

The original clustering used the seven-column `X_scaled` despite constructing a different `X_cluster_scaled`. Its original output and methodology remain visible. Historical resource-intensity language should be explained as exploratory capacity/resource grouping, without an efficiency or clinical-quality claim. [Historical provenance](docs/HISTORICAL_PROVENANCE.md) contains commit links, hashes, original evidence, and the replay trail. [Interview guide](docs/INTERVIEW_GUIDE.md) explains the corrections candidly.

## Reproduced notebooks

All four new notebooks have saved outputs from fresh, sequential kernel executions. They can also run independently after fetching the input.

| Notebook | Purpose |
| --- | --- |
| [01_data_audit](notebooks/01_data_audit.ipynb) | Source fingerprint, missingness, reporting units, capacity EDA, historical cleaning reconciliation |
| [02_capacity_clustering](notebooks/02_capacity_clustering.ipynb) | One vector per facility period; K selection, silhouette, bootstrap stability, shared K-Means/PCA matrix, descriptive profiles |
| [03_supervised_evaluation](notebooks/03_supervised_evaluation.ipynb) | Expense regression and classification; grouped CV, dummy baselines, held-out metrics and facility-bootstrap intervals |
| [04_historical_replay](notebooks/04_historical_replay.ipynb) | Isolated original cleaning and actual seven-feature K-Means/PCA replay, separate from improved evaluation |

## What changed

- **Consistent matrix:** improved K-Means and PCA both consume the same `X_cluster_scaled` and feature order. PCA is a visualization, not evidence of efficiency.
- **Facility separation:** 377 development and 95 held-out facilities, with zero overlap. Supervised model selection uses three development GroupKFold splits. Learned imputation, clipping, scaling, encoding, and classification cutoffs stay inside training partitions.
- **K validation:** compare K=2..8 on development facilities using silhouette, initialization ARI, and five whole-facility bootstrap refits. Apply a declared stability/size selection rule before inspecting the holdout.
- **Defensible features:** preserve report dates; deduplicate only full original rows; retain missing predictors for training-fit imputation; avoid ordinal identity codes. Use licensed beds plus descriptive availability shares to reduce repeated weighting of correlated bed counts. COUNTY is one-hot encoded for supervised models.
- **Comparable targets:** `Amount` mixes dollars, counts, hours, and ratios. Improved clustering excludes it; supervised evaluation uses only `TOT_OP_EXP`, total operating expenses. Classification denotes above-training-median reported expense, not demand or efficiency.
- **Honest evaluation:** dummy baselines, MAE/RMSE/R²/median absolute error, balanced accuracy/macro F1/ROC-AUC/average precision, confusion matrices, prevalence, equal-facility sensitivity, and whole-facility bootstrap intervals. No normalization by the largest outcome to imply tiny errors.
- **Outlier sensitivity:** retain evaluation rows and target extremes. Predictor clipping is learned inside training; development-only comparisons without clipping are reported, including cases where clipping performs worse.

## Executed results

The improved analysis uses **2,690 facility-period capacity vectors**, then maps labels to all 578,350 long reporting rows. This is separate from the historical 513,363-row segmentation.

| Improved evaluation | Result |
| --- | --- |
| Selected K | 3 |
| Development / held-out silhouette | 0.407 / 0.445 |
| Mean facility-bootstrap ARI at K=3 | 0.965 |
| Expense regression holdout MAE | $100.2M vs $150.9M median dummy baseline |
| Expense regression holdout RMSE / R² | $166.5M / 0.319 |
| Expense classification balanced accuracy | 0.790 vs 0.500 dummy baseline |
| Expense classification ROC-AUC / average precision | 0.850 / 0.812; positive prevalence 0.513 |

[Full results and limitations](docs/RESULTS.md) link to the machine-readable artifacts. Historical scores use different targets, rows, and validation; they are not directly comparable. These results describe this historical extract, without establishing clinical utility, hospital efficiency, or future forecasting performance.

## Reproduce locally

Tested with Python **3.11.17** on Linux. Create an isolated environment and run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/fetch_data.py --historical-cleaned
python scripts/execute_notebooks.py
python scripts/verify_artifacts.py --with-data
python -m pytest -q
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. `requirements.txt` lists direct scientific dependencies; `requirements-lock.txt` records every package in the tested environment. Notebooks contain no package-install or download cells, no Colab-specific paths, and no dependency on prior notebook state.

The raw CSV (~205 MB) and original shared cleaned CSV (~47 MB) are deliberately excluded from Git. [Source manifest](data/source_manifest.json) records URLs, retrieval date, hashes, and observed dimensions. The original raw Drive link was unavailable, so the replay uses the official California resource. Fetching verifies the recorded fingerprint and rejects changed source bytes instead of silently substituting data. If the optional shared cleaned link becomes unavailable, fetch only the official raw input; the historical notebook reports that independent cleaned-file comparison is unavailable.

Compact metrics, profiles, assignments by facility period, and figures are tracked under [reports](reports). Large reporting-row assignments are generated locally and ignored. [Execution manifest](reports/execution_manifest.json) records versions, notebook hashes, cell counts, and execution status. GitHub Actions runs offline boundary/integrity tests; its optional manual reproduction job also downloads the fingerprinted raw file and executes all new notebooks.

## Original collaborative artifacts

The original opening order remains regression → clustering/classification → classification with outliers. These files and their saved results remain unchanged:

- `602_Regression_Model_Abhinav&Devika.ipynb`
- `602_Clustering_&_Classification_Model_Abhinav&Devika.ipynb`
- `602_Classification_Model_Abhinav&Devika.ipynb`
- `602 executive  summary -  Devika  & Abhinav.pdf`
- `DATA 602 FINAL PROJECT PPT[1].pdf`

Original supervised results are preserved rather than rerun as current validated findings. The new regression/classification notebook replaces their evaluation methodology; the historical replay reproduces cleaning and K-Means/PCA. Original code cells retain outputs despite null execution counters and unrecorded library versions, so exact historical per-row labels and execution-state reproduction are not asserted.

## References

- [California HCAI hospital selected financial data](https://data.ca.gov/@department-of-health-care-access-and-information/hospital-annual-financial-data-selected-data-pivot-tables/r/50114563-deae-4fcb-8900-9533f9d59f8f) and [official data dictionary](https://data.chhs.ca.gov/dataset/ea0c8ca9-023e-46a3-b95b-b9d4ab8ec195/resource/842c8bcb-bfac-4aa9-9de8-b11eb049f6de/download/hafdselectedfiledoc2004.pdf).
- Scikit-learn: [preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html), [group-separated splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html), and [silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html).
