# Hospital Performance & Resource Optimization

**Project collaborators: Abhinav Varma & Devika Rani Sanaboyina, UMBC DATA 602.**

Traditional machine learning analysis of California hospital financial reporting data using Python, Pandas, K-Means, PCA, regression, and classification. The updated workflows add facility-aware validation, clearer feature definitions, cluster stability analysis, and baseline comparisons. The original coursework notebooks, reports, and collaborator contributions are preserved alongside the updated analysis.

## Data

The 2012–2017 California HCAI/OSHPD extract contains **578,350 reporting rows**, **472 facilities**, and **2,690 facility reporting periods**. Each long-format row represents one reporting measure for a facility period. `Amount` can represent dollars, counts, hours, or ratios depending on `Variable`.

Capacity clustering uses one vector per facility period, then assigns its cluster to the associated reporting rows. Supervised models use only `TOT_OP_EXP`, total operating expenses, to keep the target units consistent. See [data provenance](docs/DATA_PROVENANCE.md) for source versions, definitions, and original artifacts, and [methods](docs/METHODS.md) for the modeling decisions.

## Notebooks

The four added notebooks include saved outputs from fresh, sequential kernel executions. Each can run independently after fetching the input.

| Notebook | Purpose |
| --- | --- |
| [01_data_audit](notebooks/01_data_audit.ipynb) | Source fingerprints, missingness, reporting units, and capacity exploration |
| [02_capacity_clustering](notebooks/02_capacity_clustering.ipynb) | Facility-period clustering, K selection, silhouette, stability, PCA, and profiles |
| [03_supervised_evaluation](notebooks/03_supervised_evaluation.ipynb) | Expense regression and classification with grouped validation and baselines |
| [04_historical_replay](notebooks/04_historical_replay.ipynb) | Separate execution of the original cleaning and seven-feature K-Means/PCA workflow |

## Methods

- **Consistent feature inputs:** K-Means and PCA consume the same `X_cluster_scaled` matrix and column order. PCA is used for visualization.
- **Facility-aware validation:** 377 development facilities and 95 held-out facilities have no overlap. Supervised model selection uses three development GroupKFold splits. Imputation, clipping, scaling, encoding, and classification thresholds are fitted within training partitions.
- **Cluster selection:** K=2..8 is evaluated on development facilities using silhouette, initialization agreement, and five whole-facility bootstrap refits. A declared stability and cluster-size rule selects K before held-out evaluation.
- **Feature preparation:** Report identifiers and dates are retained. Only exact full-schema duplicates are removed. Missing predictors remain available for training-fit imputation. Licensed beds and descriptive availability shares reduce repeated weighting of correlated bed counts; COUNTY uses one-hot encoding for supervised models.
- **Comparable targets:** Clustering excludes mixed-unit `Amount`. Regression predicts reported operating expenses, while classification identifies values above a training-derived expense median.
- **Model evaluation:** Dummy baselines accompany MAE, RMSE, R², median absolute error, balanced accuracy, macro F1, ROC-AUC, average precision, prevalence, and confusion matrices. Whole-facility bootstrap intervals and equal-facility metrics describe uncertainty and weighting sensitivity.
- **Outlier sensitivity:** Evaluation rows and target extremes are retained. Predictor clipping is learned from training data; development-only comparisons without clipping are reported.

## Results

| Evaluation | Result |
| --- | --- |
| Selected K | 3 |
| Development / held-out silhouette | 0.407 / 0.445 |
| Mean facility-bootstrap ARI at K=3 | 0.965 |
| Expense regression holdout MAE | $100.2M vs $150.9M median dummy baseline |
| Expense regression holdout RMSE / R² | $166.5M / 0.319 |
| Expense classification balanced accuracy | 0.790 vs 0.500 dummy baseline |
| Expense classification ROC-AUC / average precision | 0.850 / 0.812; positive prevalence 0.513 |

[Results and limitations](docs/RESULTS.md) provide the full context and links to machine-readable reports. Cluster profiles describe capacity and resource availability; they do not establish hospital efficiency, clinical quality, or causal optimization benefits. Facility holdout evaluates unseen facilities in this extract, not future forecasting performance. Scores from the original coursework use different targets and validation and are not directly comparable.

## Run locally

Tested with Python **3.11.17** on Linux. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/fetch_data.py --historical-cleaned
python scripts/execute_notebooks.py
python scripts/verify_artifacts.py --with-data
python -m pytest -q
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. `requirements.txt` lists direct dependencies; `requirements-lock.txt` records the tested environment. Notebooks contain no installation or download cells, Colab-specific paths, or dependency on prior notebook state.

The raw CSV (~205 MB) and shared coursework cleaned CSV (~47 MB) are excluded from Git. The [source manifest](data/source_manifest.json) records URLs, retrieval date, fingerprints, and dimensions. Fetching verifies the recorded source fingerprint. The original raw Drive link was unavailable, so the workflow uses the official California resource. If the optional cleaned-file link is unavailable, fetch only the raw source; the original-workflow notebook reports that the cleaned-file comparison is unavailable.

Compact metrics, profiles, facility-period assignments, and figures are tracked under [reports](reports). Large reporting-row assignments are generated locally and ignored. The [execution manifest](reports/execution_manifest.json) records versions, notebook hashes, code-cell counts, and execution status. GitHub Actions runs offline validation; its optional manual job downloads the raw source and executes the added notebooks.

## Original coursework

The original opening order is regression → clustering/classification → classification with outliers. These notebooks, PDFs, and their saved results remain unchanged:

- `602_Regression_Model_Abhinav&Devika.ipynb`
- `602_Clustering_&_Classification_Model_Abhinav&Devika.ipynb`
- `602_Classification_Model_Abhinav&Devika.ipynb`
- `602 executive  summary -  Devika  & Abhinav.pdf`
- `DATA 602 FINAL PROJECT PPT[1].pdf`

The original supervised results remain separate from the updated evaluation. Original code cells retain outputs despite null execution counters and unrecorded library versions; their saved execution state is therefore not fully recoverable. The original workflow is retained for methodological comparison.

## References

- [California HCAI hospital selected financial data](https://data.ca.gov/@department-of-health-care-access-and-information/hospital-annual-financial-data-selected-data-pivot-tables/r/50114563-deae-4fcb-8900-9533f9d59f8f) and [official data dictionary](https://data.chhs.ca.gov/dataset/ea0c8ca9-023e-46a3-b95b-b9d4ab8ec195/resource/842c8bcb-bfac-4aa9-9de8-b11eb049f6de/download/hafdselectedfiledoc2004.pdf).
- Scikit-learn: [preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html), [group-separated splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html), and [silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html).
