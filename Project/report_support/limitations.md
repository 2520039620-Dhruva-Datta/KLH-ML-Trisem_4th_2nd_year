# Limitations

- **Class imbalance**: Anomaly outnumbers Normal roughly 11:1 after cleaning; all models rely on
  balanced class/sample weighting and macro-averaged metrics rather than raw accuracy.
- **Single dataset, single benchmark**: all results are specific to IoTID20; generalization to
  other IoT deployments, device types, or attack families is not evaluated here.
- **SHAP availability is environment-dependent**: it depends on `numba`'s native runtime, which is
  subject to this machine's Windows Application Control policy. The pipeline is written to fall
  back to native and permutation importance if SHAP is unavailable on a given run.
- **Bounded hyperparameter search**: Random Forest and XGBoost use small, hand-picked grids (not
  exhaustive search or `RandomizedSearchCV`), chosen deliberately to avoid a joblib/loky
  nested-parallelism deadlock observed in this Jupyter-kernel environment on Windows.
- **L1/ElasticNet tuning subsample**: `saga`-solver logistic regression fits did not fully converge
  even at `max_iter=3000` on this data, and scaled very poorly with sample count (see
  `PROJECT_PROGRESS.md`); L1/ElasticNet experiments in notebook 07 therefore use a bounded, capped
  15,000-row training subsample rather than the full training set.
- **Single train/validation/test split**: results reflect one fixed split (`random_state=42`), not
  a k-fold cross-validated estimate.
- **No real-time or edge-device deployment testing**: model size and inference time are measured
  on the development machine, not on constrained IoT hardware.
- **Anomaly is not sub-typed here**: the binary target bundles several distinct attack categories
  (`Cat`/`Sub_Cat`) into one label; this project does not evaluate attack-type classification.
