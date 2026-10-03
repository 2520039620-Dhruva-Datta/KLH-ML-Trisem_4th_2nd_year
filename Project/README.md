# Lightweight IoT Intrusion Detection Through Feature Optimization and Explainable Analytics

A university Machine Learning project building a leakage-safe, lightweight, and explainable pipeline
for binary IoT network-intrusion detection (`Normal` vs. `Anomaly`).

## Dataset

- **Name:** IoTID20 / IoT Network Intrusion Dataset
- **Source:** https://sites.google.com/view/iot-network-intrusion-dataset/home
- **Paper:** I. Ullah and Q. H. Mahmoud, *"A Scheme for Generating a Dataset for Anomalous Activity
  Detection in IoT Networks,"* DOI: 10.1007/978-3-030-47358-7_52
- **File used:** `data/Final IoT Network Intrusion Dataset.csv` (625,783 rows x 86 columns raw)
- **Target:** `Label` -> binary (`Normal = 0`, `Anomaly = 1`).

## Excluded / forbidden predictors

Never used as model inputs: `Label` (target), `Cat`/`Sub_Cat` (alternate targets - would leak the
answer), `Flow_ID`/`Src_IP`/`Dst_IP`/`Timestamp` (high-cardinality identifiers - memorization risk).
Ten constant columns are also removed (verified from training data; full list in
`report_support/preprocessing_summary.json`).

## Folder structure

```
Project/
├── data/Final IoT Network Intrusion Dataset.csv   (authoritative raw dataset)
├── scripts/            canonical `# %%`-cell-marked .py source for every notebook, plus:
│   ├── common.py                shared leakage-safe preprocessing/eval utilities
│   ├── build_notebooks.py       converts scripts/*.py -> notebooks/*.ipynb (nbformat)
│   ├── generate_flow_diagram.py, generate_manifests.py, generate_literature_survey.py,
│   │   generate_conclusion_support.py, generate_report.py   report-support + DOCX/PDF generation
├── notebooks/           11 executed notebooks, numbered 01-11 (see below)
├── processed_data/      leakage-safe train/validation/test matrices (scaled + tree-oriented)
├── feature_selection/   combined feature ranking + 10/15/20/30/All feature-count lists
├── models/               preprocessing_pipeline.joblib + one frozen model per candidate
├── metrics/              development and final (test-set) evaluation metrics
├── predictions/          final_predictions.csv (untouched test set, winning model)
├── figures/              31 report-quality figures
├── report_support/       small JSON/CSV/MD files the report is generated FROM (never hand-typed)
├── report/               Final_Project_Report.docx / .pdf
├── archive_previous_attempt/   the earlier single-consolidated-notebook deliverable (preserved)
├── requirements.txt, .gitignore, PROJECT_PROGRESS.md
└── README.md
```

## Notebook execution order and purpose

| # | Notebook | Purpose |
|---|---|---|
| 01 | Dataset_Exploration | Raw dataset audit, dtype/missing/duplicate/target statistics |
| 02 | EDA | Class balance, distributions, correlation (descriptive only) |
| 03 | Preprocessing | **The** leakage-safe pipeline: final-test isolation, train-only imputation/scaling |
| 04 | Feature_Selection_Optimization | ANOVA + MI + RF + XGB consensus ranking; 10/15/20/30/All subsets |
| 05 | Gradient_Descent_Binary_Classification | From-scratch logistic regression (educational) |
| 06 | Logistic_Regression | Main linear candidate, feature-count + hyperparameter tuning |
| 07 | Regularisation_Ridge_Lasso_ElasticNet | L1/L2/ElasticNet sparsity study |
| 08 | Random_Forest | Main ensemble candidate, OOB + permutation importance |
| 09 | XGBoost | Main boosted-tree candidate, correct imbalance handling |
| 10 | Explainability_SHAP | SHAP (or documented fallback) for the strongest candidate |
| 11 | Final_Model_Comparison_and_Predictions | **The only** notebook touching the final test set |

Each of 06/08/09 independently tunes and **freezes** its own best model (fit on training data,
selected via validation data only). Notebook 11 loads all three frozen models and evaluates each
exactly once on the untouched test set - no further tuning happens there.

## Environment setup

Uses the existing `.MLvenv` (Python 3.14, Windows). To reproduce from scratch:

```
python -m venv .MLvenv
.MLvenv\Scripts\activate
pip install -r requirements.txt
```

`.MLvenv/` and `.venv/` are excluded from version control (see `.gitignore`).

## Running the pipeline

From the project root:

```
.MLvenv\Scripts\python.exe scripts\build_notebooks.py      REM builds notebooks/*.ipynb from scripts/*.py
```

Then execute notebooks 01 through 11 **in order** (each later notebook depends on files saved by
earlier ones):

```
.MLvenv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace ^
  --ExecutePreprocessor.timeout=900 notebooks\01_Dataset_Exploration.ipynb
REM ... repeat for 02 through 11, in order
```

or open each interactively in Jupyter/VS Code and "Restart Kernel and Run All", in order.

To regenerate the report-support files and final report after the notebooks have run:

```
.MLvenv\Scripts\python.exe scripts\generate_flow_diagram.py
.MLvenv\Scripts\python.exe scripts\generate_manifests.py
.MLvenv\Scripts\python.exe scripts\generate_conclusion_support.py
.MLvenv\Scripts\python.exe scripts\generate_literature_survey.py
.MLvenv\Scripts\python.exe scripts\generate_report.py
```

## Reproducibility and leakage prevention

- `random_state=42` everywhere a random process occurs.
- The final test set is isolated **before** any learned transformation is fit (notebook 03), and is
  not loaded again until notebook 11's single, final evaluation.
- Feature ranking, feature-count selection, model comparison, and hyperparameter tuning all use
  training/validation data only - verified explicitly in `report_support/leakage_audit.json`.
- Every notebook's important outputs are saved to disk (`processed_data/`, `metrics/`, `models/`,
  `figures/`) so later stages never silently recompute or drift from earlier results.

## Known environment-specific findings

- **SHAP** depends on `numba`'s native runtime, which this machine's Windows Application Control
  policy sometimes blocks on first access per session; the code degrades gracefully to native +
  permutation importance if that happens on a given run (see `notebooks/10_Explainability_SHAP.ipynb`).
- **scikit-learn's `liblinear` solver scales very poorly** on this dataset's sample count (one fit
  took 568s vs. 13.5s for `lbfgs`); notebook 06 uses `lbfgs` only, and notebook 07's L1/ElasticNet
  (which require `saga`, since `lbfgs` doesn't support those penalties) use a bounded 15,000-row
  training subsample. Full details in `PROJECT_PROGRESS.md`.

## Results (final, untouched test set)

See `metrics/final_model_comparison.csv` / `metrics/final_test_metrics.json` for the authoritative
numbers, and `report/Final_Project_Report.docx`/`.pdf` for the full write-up. Summary: **Random
Forest** (10 of 72 features) won with test Macro-F1 ≈ 0.998 and ROC-AUC ≈ 0.9996; XGBoost was a close
second at a fraction of the model size; Logistic Regression trailed, indicating a non-linear decision
boundary.

## Limitations

See `report_support/limitations.md` (class imbalance, single-dataset evaluation,
environment-dependent SHAP availability, bounded hyperparameter search, single fixed split, no
real-device deployment testing).

## Missing reference materials

`ML_Project_Report_Template.docx` and the `P3_GD`/`P3_SLR`/`P3_MLR`/`P4_1`/`P4_2`/`P5_regularisation`/
`P7_Random_Forest` practical notebooks referenced in the project's build instructions were not found
anywhere in this project directory. The report follows the section structure specified directly in
those instructions instead, and student/guide/institution fields are marked `NOT AVAILABLE` in
`report_support/project_metadata.json` rather than invented.
