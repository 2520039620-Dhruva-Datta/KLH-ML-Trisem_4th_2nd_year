# Project Progress — Lightweight IoT Intrusion Detection

Last updated: 2026-09-16

## AUDIT COMPLETE

### File classification

| File/Folder | Classification | Notes |
|---|---|---|
| `data/Final IoT Network Intrusion Dataset.csv` | REQUIRED AND CORRECT | Authoritative dataset. 625,783 rows x 86 cols raw (verified by code). |
| `notebooks/01_Dataset_Exploration.ipynb`, `02_EDA_.ipynb`, `Final_preprocessing_IoT_Updated.ipynb` | USEFUL REFERENCE ONLY (superseded) | Original supporting notebooks from an earlier session. Numbers cross-checked and consistent. Left in place under `notebooks/` as historical reference; the new canonical `01`/`02`/`03` notebooks below supersede them for the graded deliverable. |
| `archive_previous_attempt/notebooks/Final_IoT_Intrusion_Detection_Project.ipynb` | PREVIOUS CORRECT IoT WORK — REUSED | A complete, executed, single-notebook submission built and verified in this same session (3x clean re-executions, real SHAP, real metrics). Superseded by the new 11-notebook structure per the user's explicit instruction, but its validated logic/numbers are the source of truth being redistributed across the new scripts. |
| `archive_previous_attempt/report/*.docx/.pdf` | PREVIOUS CORRECT WORK — ARCHIVED | Report generated from the single-notebook pipeline; superseded by the new template-driven report. |
| `ML_Project_Report_Template.docx` | **MISSING — NOT PROVIDED** | Referenced repeatedly in the master prompt but does not exist anywhere in the project directory or the wider course folder (verified by recursive search twice, across two turns). Report will be built using the exact section list given in the prompt itself; template-specific items (exact title-page layout, guide/institution identity) are marked `NOT AVAILABLE — REASON: template file not provided`. |
| `P3_GD (3).ipynb`, `P3_GD (4).ipynb`, `P3_SLR.ipynb`, `P3_MLR (1).ipynb`, `P4_1...ipynb`, `P4_2...ipynb`, `P5_regularisation...ipynb`, `P7_Random_Forest...ipynb` | **MISSING — NOT PROVIDED** | None found anywhere in the project or course directory. Scripts 05/06/07/08 are written from the master prompt's explicit function lists instead (which are detailed enough to implement directly). |
| `X_train_processed1.csv` etc. (old Placement data) | NOT FOUND | Not present in this project directory — nothing to archive/forbid; confirmed no Placement data is reachable from here. |
| `.MLvenv/` | REQUIRED AND CORRECT | Python 3.14 venv with all needed packages (numpy, pandas, scikit-learn, xgboost, shap, matplotlib, seaborn, joblib, jupyter/nbconvert, python-docx, reportlab). `pyarrow` installed but its native DLL is blocked by this machine's Application Control policy — not used. `jupytext` not installed; per the prompt's own fallback, notebook generation uses `nbformat` directly instead. |

### Known environment facts (carried over from the earlier single-notebook build, still valid)
- Raw dataset: 625,783 rows x 86 columns (verified).
- Exact duplicates: 164,087 (26.22%) -> 461,696 rows after removal (verified).
- Raw ordinary missing values: 0. Raw infinite values: 736 (`Flow_Byts/s`, `Flow_Pkts/s`).
- **New data-quality finding**: 1 malformed value in `Dst_Port` (`"19 00"`) forces that column to string dtype; coerced to numeric with the bad value treated as missing. The source CSV's file-modified timestamp is *after* the timestamp on this session's earlier cached preprocessing outputs, explaining the discrepancy with the 71-feature figure quoted in older material (now 72, with a 3rd missing-indicator column for `Dst_Port`).
- Class distribution after cleaning: Anomaly 423,098 (91.64%), Normal 38,598 (8.36%).
- 10 constant columns removed (same list verified twice): `Fwd_PSH_Flags, Fwd_URG_Flags, Fwd_Byts/b_Avg, Fwd_Pkts/b_Avg, Fwd_Blk_Rate_Avg, Bwd_Byts/b_Avg, Bwd_Pkts/b_Avg, Bwd_Blk_Rate_Avg, Init_Fwd_Win_Byts, Fwd_Seg_Size_Min`.
- SHAP works in this environment (numba's native DLL block resolved after the OS finished background-verifying it) but is environment-dependent — every SHAP-using script must fail gracefully.
- **Known joblib/loky gotcha**: `RandomizedSearchCV(n_jobs=-1)` wrapping an estimator that ALSO sets `n_jobs=-1` deadlocks inside an ipykernel process on this Windows machine (confirmed: hung 30 min, worked fine as a plain script). Fix used throughout: search `n_jobs=1`, estimator `n_jobs=-1`.

## Plan / stage tracker

- [x] AUDIT COMPLETE
- [x] EXPLORATION COMPLETE (script 01 + notebook 01) — 26 cells, 0 errors, `metrics/dataset_statistics.json` saved
- [x] EDA COMPLETE (script 02 + notebook 02) — 22 cells, 0 errors, 7 figures saved
- [x] PREPROCESSING COMPLETE (script 03 + notebook 03) — 23 cells, 0 errors, all 9 `processed_data/*.csv` + `models/preprocessing_pipeline.joblib` + `report_support/preprocessing_summary.json` saved
- [x] FEATURE SELECTION COMPLETE (script 04 + notebook 04) — 21 cells, 0 errors, `feature_selection/*` (ranking + 5 feature-count lists) saved
- [x] GRADIENT DESCENT COMPLETE (script 05 + notebook 05) — 17 cells, 0 errors, `metrics/gradient_descent_metrics.csv` saved. Mini-Batch GD beat Batch GD on validation Macro-F1 (0.779 vs 0.699).
- [x] LOGISTIC REGRESSION COMPLETE (script 06 + notebook 06) — 8 code cells, 0 errors, `models/logistic_regression.joblib` + `metrics/logistic_regression_metrics.csv` saved
- [x] REGULARISATION COMPLETE (script 07 + notebook 07) — 11 code cells, 0 errors, `metrics/regularisation_metrics.csv` saved. L1 sparsity behaves exactly as expected (93% sparse at C=0.001 -> ~7% at C=1.0).
- [x] RANDOM FOREST COMPLETE (script 08 + notebook 08) — 0 errors, 6 figures, `models/random_forest.joblib` saved. Best subset: top-10 features (Macro-F1=0.9973 on validation, beating "All").
- [x] XGBOOST COMPLETE (script 09 + notebook 09) — 0 errors, 4 figures, `models/xgboost.joblib` saved. `scale_pos_weight` correctly computed and shown to be wrong-direction (0.088), `compute_sample_weight` used instead. Best subset: All 72 features.
- [x] SHAP COMPLETE (script 10 + notebook 10) — 0 errors. SHAP **succeeded** this run (numba's Application-Control block resolved again); real `shap_summary.png` + `shap_importance.png` saved. Strongest candidate confirmed: Random Forest (validation Macro-F1=0.9986).
- [x] FINAL TEST COMPLETE (script 11 + notebook 11) — 0 errors. `metrics/feature_count_comparison.csv` + figure built. **Winner: Random Forest** (10 features) — test Macro-F1=0.9978, ROC-AUC=0.9996, PR-AUC=0.9999, model size=16.16MB. XGBoost close second (Macro-F1=0.9960, only 0.92MB). Logistic Regression trails (Macro-F1=0.797) confirming a non-linear decision boundary. `predictions/final_predictions.csv` saved.
- [x] NOTEBOOK EXECUTION COMPLETE — **all 11 notebooks executed, 0 errors, every cell has real output.** (Pending: one final clean restart-and-rerun-all pass across all 11 for reproducibility confirmation, before final submission.)
- [x] REPORT SUPPORT COMPLETE — all files generated: `preprocessing_summary.json`, `selected_features_summary.csv`, `explainability_summary.json`, `project_metadata.json`, `dataset_description.json`, `experimental_setup.json`, `leakage_audit.json` (all PASS), `notebook_manifest.csv`, `artifact_manifest.csv` (64 rows), `figure_manifest.csv` (31/31 present), `limitations.md`, `future_work.md`, `lightweight_comparison.csv`, `conclusion_evidence.json`, `model_performance_report.csv`, and `literature_survey.csv` (**10 papers, 2024-2026, every Title/Authors/Journal/Volume/Issue/Pages/DOI independently verified against the Crossref API — none invented**).
- [x] REPORT COMPLETE — `report/Final_Project_Report.docx` (112 paragraphs, 7 tables, 10 images) and `report/Final_Project_Report.pdf` (valid PDF, verified header) generated by `scripts/generate_report.py`, reading exclusively from `metrics/`, `report_support/`, `figures/` — no hand-typed numbers. Sections: Abstract, 1 Introduction (1.1-1.5), 2 Literature Survey, 3 Methodology (3.1-3.9), 4 Results and Discussion (4.1-4.3), 5 Conclusion, 6 Future Work, 7 References.
- [x] README.md and requirements.txt rewritten for the new 11-notebook structure.
- [x] FINAL AUDIT COMPLETE — Placement/CGPA content scan clean (one benign contrastive mention in the archived supporting notebook: "saved as `Label`, never `PlacementStatus`"); no hard-coded `dhruv`/`C:\Users` paths in `scripts/`. **Final restart-and-rerun-all pass across all 11 notebooks, in order, completed: 0 errors, 0 unexecuted cells, across all 11 (102 total code cells).** Final locked model result reproduced bit-for-bit identical to the prior run (Random Forest, test Macro-F1=0.9978025872762715) — confirms full determinism. All report_support files, the flow diagram, and the final DOCX/PDF report were regenerated against this clean run so everything is in sync.

## FINAL FILE INVENTORY

| Folder | Contents | Status |
|---|---|---|
| `notebooks/01-11` | 11 executed notebooks | EXECUTED |
| `scripts/*.py` | 11 canonical cell-marked sources + `common.py`, `build_notebooks.py`, 5 report-generation scripts | GENERATED |
| `processed_data/` | 10 files (scaled + tree-oriented train/val/test + raw-demo copy), 557MB | GENERATED |
| `feature_selection/` | 6 files (ranking + 5 feature-count lists) | GENERATED |
| `models/` | 4 joblib files (pipeline + 3 frozen models), 18MB | GENERATED |
| `metrics/` | 9 files (dev + final test metrics) | GENERATED |
| `predictions/` | `final_predictions.csv` | GENERATED |
| `figures/` | 31 figures, 1.3MB | GENERATED |
| `report_support/` | 17 files (all report inputs, including 10 Crossref-verified literature citations) | GENERATED |
| `report/` | `Final_Project_Report.docx` + `.pdf` | GENERATED |
| `archive_previous_attempt/` | prior single-notebook deliverable (notebook + report + outputs) | ARCHIVED |
| `notebooks/01_Dataset_Exploration.ipynb` etc. (original 3 supporting notebooks) | left in `notebooks/` as historical reference | REFERENCE |
| `ML_Project_Report_Template.docx`, `P3_*/P4_*/P5_*/P7_*` practicals | never found in project or course directory | NOT AVAILABLE (documented) |
| `README.md`, `requirements.txt`, `.gitignore` | rewritten for the new structure | COMPLETE |

**PROJECT COMPLETE.** All checklist items from the master prompt are satisfied except those explicitly
marked `NOT AVAILABLE — REASON` (student/guide/institution identity fields, which depended on the
never-provided `ML_Project_Report_Template.docx`).

## Exact next step (resume point)

1. Check background task `bmrppsh4x` (nbconvert executing 06 then 07) for completion; verify both 0-error.
2. Execute notebook 08 (`notebooks/08_Random_Forest.ipynb`, already built from script), then 09.
3. Write script 10 (`10_explainability_shap.py`) — load the frozen model with the best validation Macro-F1 across 06/08/09's `final_metrics` (need to compare after 06/08/09 are all done), run SHAP TreeExplainer (RF/XGB) or fall back gracefully; save `figures/shap_summary.png`, `figures/shap_importance.png`.
4. Write script 11 (`11_final_model_comparison_and_predictions.py`) — load all 3 frozen models + `X_test_processed`/`X_test_tree`/`y_test`, evaluate ONCE each (no tuning), build `metrics/final_model_comparison.csv`, `metrics/final_test_metrics.json` (best model only), `predictions/final_predictions.csv`; also consolidate `metrics/feature_count_comparison.csv` from the three `metrics/{logistic_regression,random_forest,xgboost}_metrics.csv` files.
5. Build+execute notebooks 10 and 11.
6. Generate remaining `report_support/*` files (see master prompt Sections on report support — dataset_description.json, experimental_setup.json, literature_survey.csv [needs real WebSearch], leakage_audit.json, manifests, lightweight_comparison.csv, conclusion_evidence.json, limitations.md, future_work.md).
7. Write `scripts/generate_flow_diagram.py` -> `figures/proposed_system_flow.png`.
8. Write `scripts/generate_report.py` -> `report/Final_Project_Report.docx` + `.pdf` (reuse the render_docx/render_pdf pattern already proven to work from the earlier single-notebook deliverable).
9. Update `README.md`, `requirements.txt` (add any newly-used packages), final `.gitignore` check.
10. Final full re-run of all 11 notebooks in order (restart-kernel equivalent) to confirm end-to-end reproducibility, then final audit + completion report.

## Known issues

- `ML_Project_Report_Template.docx` and all `P3/P4/P5/P7` practical notebooks are not present anywhere in the project — proceeding per the master prompt's own "continue without it, mark NOT AVAILABLE" rule (confirmed with the user).
- `jupytext` not installed; using `nbformat` directly for `# %%` -> `.ipynb` conversion instead (permitted fallback).
- Literature survey requires real, verified web searches (not yet done) — no DOI will be invented; unverifiable fields will be marked "Not verified".
- This machine's Windows Application Control policy intermittently blocks native DLLs (scipy/sklearn/numba) on first access per session/day — always transient, resolves on retry within 1-3 attempts. Hit again at the start of this resumed session; resolved after retry.
- **Session interruption risk observed directly**: a background `nbconvert --execute` run (06+07) was killed mid-notebook when the prior session ended, with no completion marker. Recovery approach: always re-run the *whole* notebook via `--execute --inplace` from scratch (never assume partial in-place output is trustworthy) after any interruption, and check `execution_count`/error-cell status via nbformat before trusting a notebook is done.
- Design decision (differs from the archived single-notebook attempt): each of notebooks 06/08/09 freezes its OWN best model fit on **training data only** (not train+validation combined) after its own dev-only tuning; notebook 11 purely loads and evaluates these three frozen models on test, once, with no further fitting — matching the master prompt's explicit `load_frozen_models()` / "do not tune here" language.
- **Important performance finding**: on this machine, scikit-learn's `liblinear` solver scales very poorly with training-set size for this dataset (a single fit at C=100 on the full 295,484-row training set took **568 seconds**, vs. 13.5s for `lbfgs`). `saga` (needed for L1/ElasticNet, since `lbfgs` only supports L2) also did not fully converge even at `max_iter=3000` on a 30,000-row subsample (87-145s/fit). Fix applied: notebook 06 uses `lbfgs` only (drops `liblinear` from its solver search — L1/ElasticNet aren't needed there anyway); notebook 07 fits L1/ElasticNet with `saga` on a reproducible 15,000-row stratified training subsample with `max_iter=1000` (documented as a deliberate, bounded tradeoff), while L2/Ridge still uses the full training set with `lbfgs`. Each penalty type also got its own notebook cell (not one combined cell) so a slow fit can't blow the whole notebook's budget. If RF/XGBoost (08/09) show similar unexpected slowness, apply the same subsample-for-tuning pattern (already used for their bounded manual grids, which fit far fewer/faster configs than LR's).
