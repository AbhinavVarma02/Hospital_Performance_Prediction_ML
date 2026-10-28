# Interview guide

This guide explains the project and its methodological improvements. It is not a claim that the historical metrics were reproduced unchanged. Credit the original analysis to **Abhinav Varma and Devika Rani Sanaboyina**, and distinguish the original course work from the later validation work.

## A concise account

“We explored hospital capacity and reporting patterns in a 578,350-row dataset using Python and Pandas. The rows represent facility-period-measure reports, so multiple rows belong to the same hospital. Our original cleaning workflow produced 513,363 rows and used K-Means with PCA to explore groups. I preserved that work, then audited the methodology: I corrected the feature matrix, separated facilities during evaluation, moved learned preprocessing inside training, evaluated K quantitatively, and replaced efficiency claims with capacity and resource-pattern descriptions.”

Use the [historical provenance](HISTORICAL_PROVENANCE.md) to show the exact saved outputs, starting commit and unchanged original files. The [fresh historical cleaning replay](../reports/historical/cleaning_replay.json) also reproduces 513,363 rows from the official 578,350-row extract and confirms that its eight analytical columns match the downloaded original cleaned data. This reproduces the cleaning, not the original model scores. Describe fresh model results using the executed improved notebooks and current reports. A changed sample size or lower held-out score is an expected consequence of a more defensible design, not a reason to alter data.

## Explain the existing résumé bullets

**“Profiled capacity/spending patterns in 578K hospital report rows using Python, Pandas and CRISP-DM.”** The saved raw shape is 578,350 × 35, independently matched by the recovered official extract. “Hospital report rows” is essential: this is not 578K hospitals. The improved audit counts 472 facilities and 2,690 facility/report periods. Discuss capacity columns and monetary measures by their actual definitions; `Amount` across all `Variable` values does not share one unit. CRISP-DM is described in the original executive summary (page 2) and presentation (page 4); explain the actual business-understanding, data-understanding, preparation, modeling and evaluation steps. Discuss the absent deployment phase as a limitation rather than implying a production system existed.

**“Segmented 513K+ cleaned hospital records into resource-intensity groups using K-Means and PCA.”** The historical cleaned shape is 513,363 rows, and the clustering notebook retains a K-Means/PCA result on those rows. Explain that these are cleaned reporting records, not independent hospitals. Be candid that an audit found the historical code passed the broader `X_scaled` matrix instead of the intended `X_cluster_scaled`; the improved analysis uses a clearly defined matrix consistently. Resource-intensity/capacity is a descriptive interpretation, not a validated efficiency ranking. Do not claim the historical three-cluster solution had quantitative K validation; that evidence belongs to the new analysis.

## Explain the improved analytical units

The new capacity analysis fits one vector per facility/report period rather than treating repeated measure rows as independent capacity observations. It uses `log1p(BED_LIC)`, `BED_AVL / BED_LIC` and `BED_STF / BED_AVL`. This retains a hospital-size feature while reducing the repeated influence of three strongly related bed counts. It then assigns the resulting period cluster back to every source reporting row. Missing or zero ratio denominators become missing, with imputation fitted only on development data. No `Amount` values are mixed into capacity distances.

The [official definitions](https://data.chhs.ca.gov/dataset/ea0c8ca9-023e-46a3-b95b-b9d4ab8ec195/resource/842c8bcb-bfac-4aa9-9de8-b11eb049f6de/download/hafdselectedfiledoc2004.pdf) are licensed beds **at period end**, available beds **on average**, and staffed beds **on average** (PDF page 5, printed page 3). Consequently, the available/licensed ratio combines different time summaries and may exceed one. Call the ratios descriptive capacity shares; do not present them as a clinically validated utilization or efficiency rate. `BED_STF` is not a count of employees.

The new regression/classification examples use the prespecified `TOT_OP_EXP` measure, total operating expenses (dictionary PDF page 9, printed page 7). That gives one comparable expense observation per facility/report period rather than mixing financial, utilization and other units in one outcome. Regression estimates the reported period expense from contemporaneous capacity, duration and county predictors. Classification asks whether it exceeds a training-derived median expense. The cutoff is learned afresh within each cross-validation training fold, then from all development data for the final holdout. It does not define “high demand,” “good efficiency” or future outcomes.

## Questions to prepare

| Interview question | Defensible explanation |
| --- | --- |
| What did you predict? | The historical models pooled `Amount` across measures and used a global median-derived label. The improved tasks predict reported total operating expenses (`TOT_OP_EXP`) and a training-median expense category. The amount covers its reporting period and is not inflation adjusted. It does not predict readmission, future demand or clinical quality. |
| Why keep the old notebooks? | They preserve collaborative authorship and historical results. Corrections are separately reviewable, so readers can distinguish what was originally done from what was subsequently validated. |
| What was the matrix bug? | A resource matrix was prepared, but K-Means and PCA used an earlier matrix containing ordinal category codes. A single explicit matrix prevents unintended features and keeps PCA aligned with the clustering geometry. |
| Why are facility codes excluded from predictors? | They identify entities and can encourage memorization. They are useful as grouping keys for validation, not as arbitrary numeric distances. Facility names are also excluded from distance calculations. |
| Why split by facility? | Rows from the same facility share capacity, location and reporting history. Row-wise splits let related observations appear on both sides. Grouped splits ask whether performance transfers to facilities not used for fitting. |
| Does a facility split prove forecasting performance? | No. It tests transfer to unseen facilities. A separate chronological design is needed for future-period forecasting, and use of contemporaneous capacity features should be disclosed. |
| Where are preprocessing parameters fitted? | Imputation statistics, clipping bounds, scaling and category vocabularies are learned only on the training portion of each fold. Validation and test data are transformed using those fitted parameters. Target-derived classification cutoffs must also be learned within training. |
| Why change categorical encoding? | Arbitrary integer codes give categories a false order and distance. One-hot encoding handles nominal predictors; unknown categories need an explicit safe transform path. |
| Why not remove all outliers? | Extreme resource values can be real large hospitals. Use transparent transformations or training-fitted treatment, retain untouched evaluation targets, and disclose the effects. Historical global IQR filtering is retained only as a historical method. |
| Why does `Variable` matter? | `Amount` can represent different measures and units. Pooling it without measure semantics makes regression errors and cluster interpretations misleading. The new supervised tasks fix one expense measure; the capacity clustering excludes `Amount` entirely. |
| How was K chosen? | K=2 through 8 are compared on development facilities. The prespecified rule maximizes development silhouette among candidates with mean facility-bootstrap adjusted Rand index at least 0.80 and minimum development cluster share at least 0.02. If no candidate qualifies, the fallback is explicitly provisional. Cite the executed candidate table and actual stability values. Held-out facilities do not select K; a visible separation in two PCA components alone does not validate K. |
| What does stability measure? | Agreement across repeated initializations or facility resamples on a common evaluation set, using a permutation-invariant measure such as adjusted Rand index. It assesses sensitivity of the grouping, not clinical validity. |
| What is PCA doing? | It supplies a lower-dimensional view of the same transformed feature space used by K-Means. Report explained variance and loadings. A two-dimensional plot may omit relevant variation. |
| How do you interpret clusters? | Compare original-unit medians/distributions and cluster membership sizes. Use neutral descriptions such as larger-capacity or lower-resource groups. Cluster numbers are arbitrary and do not imply rankings. |
| What are the supervised baselines? | Compare regression to training-derived constant and relevant simple baselines, and classification to dummy prediction. Use the same held-out facilities and task definition so the comparison is meaningful. |
| Which metrics matter? | Regression: MAE, RMSE and R² with units and baseline comparison. Classification: class balance, balanced accuracy, macro F1, confusion matrix and appropriate probability metrics where both classes exist. Explain grouped validation variability and avoid using accuracy alone. |
| What if corrected results are worse? | Report them. A harder validation split and removal of leakage often reduce apparent performance; the result is more informative about the stated generalization task. Do not compare a historical row-wise score directly to a new facility-held-out score as if only the algorithm changed. |
| Is this a resource-allocation recommendation? | It is an exploratory traditional-ML analysis. Acting on clusters would require better outcome definitions, case-mix adjustment, domain review and prospective validation. |

## Demonstration sequence

1. Show the historical raw/cleaned shapes and original K-Means/PCA cell at the pinned Git revision.
2. Show the original matrix mismatch and the corrected explicit feature construction.
3. Show data fingerprints, the cleaning replay and retained grouping keys in `notebooks/01_data_audit.ipynb` and `notebooks/04_historical_replay.ipynb`.
4. Show the candidate-K validation table and stability results in `notebooks/02_capacity_clustering.ipynb` before discussing the selected clusters.
5. Show original-unit cluster profiles and PCA variance/loadings in `reports/clustering/` to explain the groups.
6. Show supervised baselines beside grouped validation and held-out metrics in `notebooks/03_supervised_evaluation.ipynb` and `reports/supervised/`.
7. End with limitations: repeated reporting rows, measure semantics, possible reporting changes, noncausal interpretation, and the difference between unseen-facility generalization and future forecasting.

Quote only outputs actually present in a completed run. If a data source is unavailable, explain that limitation and show the preserved historical outputs separately; synthetic fixtures or smoke tests do not reproduce the real data analysis.
