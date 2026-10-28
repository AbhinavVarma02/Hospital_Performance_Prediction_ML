# Methods

The original analysis was developed by **Abhinav Varma and Devika Rani Sanaboyina** for UMBC DATA 602. The updated workflows retain that work and extend its preprocessing, validation, clustering, and model evaluation. The original executive summary and presentation describe the CRISP-DM framework; this extension focuses on data understanding, preparation, modeling, and evaluation.

## Reporting units and targets

The source is a long-format financial reporting dataset. Each row represents a `Variable` for one facility reporting period, so many rows share capacity measurements. The updated capacity analysis fits one vector per `FAC_NO`, `BEG_DATE`, and `END_DATE`, then assigns that period's cluster to its reporting rows. Each period has equal fitting weight; facilities with more periods contribute more vectors.

`Amount` has different units across `Variable` values. Capacity clustering excludes it. The supervised tasks select `TOT_OP_EXP`, total operating expenses, so errors refer to a single comparable financial measure in nominal source dollars. Regression estimates the reported period expense using contemporaneous capacity, duration, and county. Classification asks whether expense exceeds the training-derived median for this measure.

Neither task predicts readmission, future demand, or clinical quality. The classification threshold has no clinical or policy meaning, and the same-period predictors do not establish forecasting performance.

## Feature construction

The capacity matrix contains:

| Feature | Purpose |
| --- | --- |
| `log1p(BED_LIC)` | Represent licensed capacity with reduced skew |
| `BED_AVL / BED_LIC` | Describe available capacity relative to licensed beds |
| `BED_STF / BED_AVL` | Describe staffed capacity relative to available beds |

The [official dictionary](https://data.chhs.ca.gov/dataset/ea0c8ca9-023e-46a3-b95b-b9d4ab8ec195/resource/842c8bcb-bfac-4aa9-9de8-b11eb049f6de/download/hafdselectedfiledoc2004.pdf) defines licensed beds at period end and available/staffed beds as period averages (PDF page 5, printed page 3). The ratios therefore combine different reporting summaries, can exceed one, and are descriptive capacity shares rather than simultaneous utilization measures. `BED_STF` counts staffed beds, not employees.

The bed measures are strongly correlated. A size measure plus availability shares limits the repeated influence that three related raw bed counts would otherwise receive. Facility identifiers and names remain grouping keys, not predictors or numeric distances. Clustering also excludes COUNTY and `Variable`. Supervised predictors add report duration (`DAY_PER`) and one-hot-encoded COUNTY; unknown county categories are ignored safely during transformation.

The original clustering constructed `X_cluster_scaled` but passed an earlier seven-column `X_scaled` matrix, including ordinal categorical codes, to K-Means and PCA. The updated analysis constructs one explicit matrix with a fixed column order and passes it to both algorithms.

## Preprocessing

Full-schema deduplication occurs before selecting analytical columns so that equal values in different reporting periods are retained. Report dates and facility identities remain available for validation. Invalid or suppressed numeric predictors become missing; invalid negative capacity values are treated as missing. Zero ratio denominators also produce missing values.

Learned preprocessing stays within the relevant training partition. Numeric imputation uses training medians. Feature clipping uses training 1st/99th percentiles, followed by scaling appropriate to the model workflow. Supervised preprocessing includes missing indicators and robust scaling; nominal county values use a training-fit one-hot vocabulary. Validation and held-out rows are transformed using the fitted parameters.

Large resource values may represent real facilities. The updated workflow retains evaluation rows and target extremes rather than applying global IQR deletion. Development-only sensitivity compares the selected model with and without predictor clipping; the results are reported even when clipping performs worse. Original global cleaning rules remain isolated in the coursework workflow.

## Facility-aware validation

The holdout contains 20% of FAC_NO groups: 377 development facilities and 95 held-out facilities, with no overlap. Repeated records from a facility stay in one partition. This evaluates transfer to unseen facility IDs within the reporting extract.

Supervised model selection uses three development GroupKFold splits. Every learned preprocessing statistic is refitted in each fold. The classification expense cutoff is also recomputed from each fold's training targets, then learned from all development targets for final evaluation. Validation and test targets do not determine training labels.

Facility separation does not establish future-period generalization. Facilities may also share ownership or hospital systems despite having different IDs. A separate chronological or external validation design would be needed for forecasting or operational use.

## K-Means and PCA

K=2 through 8 are compared on development facilities. The declared rule selects the highest development silhouette among candidates with mean facility-bootstrap adjusted Rand index (ARI) at least 0.80 and minimum development cluster share at least 0.02. If no candidate meets those conditions, the fallback is flagged as provisional. Silhouette uses a fixed seeded sample of up to 2,000 periods to bound pairwise-distance memory.

Stability is measured across repeated initializations and five whole-facility bootstrap resamples. Each bootstrap refits preprocessing and K-Means, then predicts on a common development reference set. ARI permits label permutations and measures sensitivity of the partition; it does not establish operational validity. Five resamples provide a sensitivity check rather than a precise confidence interval.

After selection, the development-fitted K-Means model assigns clusters to all periods. PCA is fitted on the same transformed development feature matrix, then used to display both partitions. Explained variance and loadings describe what the two-dimensional projection retains. Visible separation alone does not validate K, and cluster IDs carry no ordering.

Profiles report original-unit medians, interquartile ranges, period counts, and facility counts. They describe capacity/resource availability without efficiency rankings, staffing adequacy claims, causal conclusions, or allocation recommendations. A facility may change cluster across periods, so facility counts in separate profiles are not additive.

## Regression and classification

| Task | Fixed candidates | Development selection criterion |
| --- | --- | --- |
| Regression | Mean/median dummy, Ridge, bounded Random Forest | Mean fold MAE |
| Classification | Most-frequent dummy, logistic regression, bounded Random Forest | Mean fold balanced accuracy |

The candidates use fixed configurations. The holdout is evaluated after model choices and the final classification cutoff are frozen. Baselines and selected models use the same held-out facilities and target definition.

Regression reports MAE, RMSE, R², and median absolute error in the target's units. RMSE is not normalized by the largest observed outcome. Classification reports prevalence, accuracy, balanced accuracy, macro F1, precision, recall, confusion matrices, ROC-AUC, and average precision. Average precision is interpreted alongside positive prevalence.

Primary metrics weight reporting periods equally. Equal-facility mean absolute error and accuracy provide a weighting sensitivity. Two hundred whole-facility bootstrap resamples produce conditional percentile intervals for the selected model; correlated reports from a facility are resampled together. These intervals omit model-training and source-reporting uncertainty.

## Execution and interpretation

The added notebooks execute independently with pinned inputs and dependencies. The execution manifest records notebook fingerprints, code-cell counts, and error status. Tests check preprocessing boundaries, facility separation, outcome isolation, original artifact integrity, and saved notebook execution evidence. Data-source fingerprints identify the retrieved versions.

The original notebooks and reports remain unchanged, with their results separate from the updated analyses. Different target definitions, feature matrices, data units, and splits prevent direct score comparisons as evidence of algorithmic improvement. See [results](RESULTS.md) for outcomes and limitations, and [data provenance](DATA_PROVENANCE.md) for source definitions and coursework artifacts.
