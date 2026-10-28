# Executed results and limitations

These results were regenerated on 2026-10-28 with Python 3.11.17, Pandas 2.2.3, NumPy 2.2.6, and scikit-learn 1.6.1. All four new notebooks executed in fresh kernels, in sequential order, with zero error outputs. [Execution manifest](../reports/execution_manifest.json) records versions, code-cell counts, and file fingerprints. The original three notebooks and two PDFs remain byte-identical to source commit `1f00286346c6a4549059bf804994c3765a73a156`.

## Historical count reconciliation

[Cleaning replay](../reports/historical/cleaning_replay.json) records this observed trail:

| Step | Reporting rows |
| --- | ---: |
| Raw official extract | 578,350 |
| Selected-column missing-value removal | 577,920 |
| Original projected deduplication | 553,022 |
| Numeric coercion and missing-number removal | 542,702 |
| Sequential BED_LIC IQR filter | 531,435 |
| Sequential BED_AVL IQR filter | 525,845 |
| Sequential BED_STF IQR filter | 513,363 |

The final eight analysis columns match the original shared cleaned CSV exactly in values and row order. There is no sampling or trimming to a desired count. Original saved results are preserved in their notebooks and [extracted text outputs](../reports/historical/original_saved_text_outputs.json). [Historical replay](../reports/historical/replay_summary.json) separately records the actual seven-feature K-Means/PCA computation; its displayed cluster means also agree with the historical printed precision, without claiming recovery of historical per-row labels or library state.

Historical cleaning removed duplicates after discarding facility IDs/dates and learned IQR cutoffs globally. It remains isolated from improved training and evaluation. The improved dataset instead retains all 578,350 reporting rows, including missing predictors; there are no exact full-schema duplicate rows.

## Capacity clusters

[Selection evidence](../reports/clustering/k_selection.csv) compares K=2..8 using development facilities only. With the declared stability/size rule, K=3 is selected. Development silhouette is 0.4070, held-out silhouette 0.4447, mean K=3 facility-bootstrap ARI 0.9647, and minimum bootstrap ARI 0.9464. Initialization ARI is 1.0 for K=3. K=2 also has initialization ARI 1.0 but mean facility-bootstrap ARI only 0.3644, showing why seed stability alone is insufficient.

One capacity vector represents each of 2,690 facility reporting periods; 377 facilities/2,154 periods form development and 95 facilities/536 periods form the holdout, with no facility overlap. The frozen development model assigns labels back to all 578,350 long reporting rows. All feature fitting, K selection, and PCA fitting exclude held-out facilities. Both K-Means and PCA consume the same three-column `X_cluster_scaled`.

The first two PCs explain 74.2% of development feature variance. This projection loses information and does not validate efficiency. [Profiles](../reports/clustering/cluster_profiles.csv), [loadings](../reports/clustering/pca_loadings.csv), and [scaled centroids](../reports/clustering/scaled_centroids.csv) support interpretation:

| Arbitrary cluster ID | Development median licensed beds | Median available/licensed | Median staffed/available | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 0 | 214 | 1.00 | 0.59 | Larger capacity with a lower staffed share |
| 1 | 46 | 1.00 | 1.00 | Smaller capacity with a high staffed share |
| 2 | 258 | 0.76 | 0.68 | Lower available share across a broad capacity range |

A facility can appear in different clusters in different periods; unique-facility counts across cluster profiles are not additive. These are descriptive reporting patterns. Capacity features do not identify financial efficiency, quality, adequate staffing, causality, or optimization gains.

No-clipping sensitivity at frozen K=3 gives development ARI 0.9915 against the clipped solution. Only five facility bootstraps were used, providing sensitivity evidence rather than a precise uncertainty interval. Silhouette evaluates geometry, not operational validity. Each facility-period has equal fitting weight; facilities with more years contribute more vectors.

## Reported-expense regression and classification

The prespecified target `TOT_OP_EXP` means total operating expenses in nominal source dollars. It has 2,690 eligible facility-period rows, including 191 zeros and no negative values. No target extremes are removed. The same 377/95 facility partition is used. Three development GroupKFold splits choose the regression model by mean MAE and classifier by mean balanced accuracy, including dummy baselines. Random Forest is selected for both fixed candidate sets.

The final expense classifier threshold is **$86,796,390.50**, learned from development targets only. CV training medians differ by fold ($93,163,400; $87,170,321; $81,132,287), so labels never use validation or test medians. Positive held-out prevalence is 0.513. This category is exploratory expense level, without a demand/efficiency interpretation.

| Held-out metric | Selected Random Forest | Relevant dummy baseline |
| --- | ---: | ---: |
| Expense MAE | $100.22M | $150.86M (median dummy) |
| Expense RMSE | $166.52M | $222.87M (median dummy) |
| Expense R² | 0.319 | -0.219 (median dummy) |
| Median absolute error | $49.77M | $83.35M |
| Balanced accuracy | 0.790 | 0.500 |
| Macro F1 | 0.789 | 0.327 |
| ROC-AUC | 0.850 | 0.500 |
| PR average precision | 0.812 | 0.513 (prevalence) |

[Full supervised metrics](../reports/supervised/supervised_metrics.json) include mean and median regression dummies, fold-level results, all holdout metrics, prevalence, confusion matrices, sensitivity, and confidence intervals. The selected classifier confusion matrix is `[[211,50],[63,212]]`, rows=true class and columns=predicted class, both in 0/1 order. Accuracy is 0.789, precision 0.809, and recall 0.771.

Two hundred whole-facility bootstrap resamples give conditional 95% percentile intervals: MAE **$76.7M–$125.8M**, R² **-0.110–0.540**, and balanced accuracy **0.700–0.868**. The R² interval includes negative values, so generalization strength remains uncertain. Intervals preserve within-facility repeated reports but omit training and reporting-source uncertainty. Equal-facility holdout sensitivities are MAE $95.8M and classifier accuracy 0.792.

Predictor clipping is a prespecified robustness choice, not an established performance improvement. In development-only sensitivity, the selected regressor's mean MAE is **$172.8M with clipping vs $154.6M without**; clipping is worse. Classifier balanced accuracy is 0.806 vs 0.804. The unclipped results are disclosed without changing the frozen primary procedure after seeing held-out outcomes. A future revision could select treatment inside development CV and obtain a fresh external/temporal test.

## Limits of the conclusions

- `Amount` mixes units by `Variable`; improved models never pool it across measures. The original mixed-unit scores are preserved without comparison to this changed task.
- Licensed beds are end-of-period counts; available/staffed beds are period averages. Ratios are descriptive, can exceed one, and are not validated simultaneous utilization measures.
- The zero-expense reports are retained as observed. Their accounting meaning needs source review before operational use.
- The data are historical California reporting records, without inflation, case-mix, patient-outcome, ownership-group, or hospital-system adjustments. Separate facility IDs may still belong to a common system.
- Facility separation estimates transfer to unseen IDs in the same historical extract. Same-period predictors, variable reporting lengths, and nominal dollars prevent a future forecasting claim.
- Sample size is 472 facilities, not 578,350 independent hospitals. Repeated years and related facilities limit independent evidence.
- Improved metrics do not validate efficiency rankings, clinical decisions, cost-saving recommendations, or deployment readiness. The project remains an interpretable coursework extension using traditional ML.
