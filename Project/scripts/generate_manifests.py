"""Generates report_support/ files that don't depend on the final test results:
project_metadata.json, dataset_description.json, experimental_setup.json,
leakage_audit.json, notebook_manifest.csv, artifact_manifest.csv, figure_manifest.csv,
limitations.md, future_work.md.
"""
import json
import platform
import sys
from pathlib import Path

import pandas as pd

_candidates = [Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts", Path.cwd().parent]
_scripts_dir = next((c for c in _candidates if (c / "common.py").exists()), None)
sys.path.insert(0, str(_scripts_dir))
from common import ROOT, REPORT_SUPPORT_DIR, METRICS_DIR, PROCESSED_DIR

# ---------------------------------------------------------------- project_metadata.json
project_metadata = {
    "project_title": "Lightweight IoT Intrusion Detection Through Feature Optimization and Explainable Analytics",
    "course": "Machine Learning",
    "degree": "Bachelor of Technology",
    "academic_year": "2026-27",
    "student_names": "NOT AVAILABLE - REASON: not supplied in the project brief or any project file; "
                      "ML_Project_Report_Template.docx (which would normally carry this) was not provided.",
    "roll_numbers": "NOT AVAILABLE - REASON: same as student_names.",
    "guide": "NOT AVAILABLE - REASON: ML_Project_Report_Template.docx (which would carry verified "
             "guide/institution details) was not provided anywhere in the project directory.",
    "guide_designation": "NOT AVAILABLE - REASON: same as guide.",
    "department": "NOT AVAILABLE - REASON: same as guide.",
    "institution": "KLH University (from the project's folder path); not independently verified against a template.",
}
with open(REPORT_SUPPORT_DIR / "project_metadata.json", "w") as f:
    json.dump(project_metadata, f, indent=2)
print("Saved: report_support/project_metadata.json")

# ---------------------------------------------------------------- dataset_description.json
dataset_statistics = json.loads((METRICS_DIR / "dataset_statistics.json").read_text())
dataset_description = {
    "dataset_name": "IoTID20 / IoT Network Intrusion Dataset",
    "dataset_relative_path": "data/Final IoT Network Intrusion Dataset.csv",
    "dataset_source": "https://sites.google.com/view/iot-network-intrusion-dataset/home",
    "dataset_paper": "I. Ullah and Q. H. Mahmoud, \"A Scheme for Generating a Dataset for Anomalous "
                      "Activity Detection in IoT Networks,\" DOI: 10.1007/978-3-030-47358-7_52",
    "raw_rows": dataset_statistics["raw_rows"],
    "raw_columns": dataset_statistics["raw_columns"],
    "cleaned_rows": dataset_statistics["rows_after_dedup"],
    "cleaned_columns": dataset_statistics["raw_columns"],
    "duplicates": dataset_statistics["duplicate_rows"],
    "target_name": "Label",
    "target_classes": ["Normal", "Anomaly"],
    "target_mapping": dataset_statistics["target_mapping"],
    "normal_count": dataset_statistics["label_counts"].get("Normal"),
    "anomaly_count": dataset_statistics["label_counts"].get("Anomaly"),
    "normal_percentage": dataset_statistics["label_percentages"].get("Normal"),
    "anomaly_percentage": dataset_statistics["label_percentages"].get("Anomaly"),
    "numeric_feature_count": dataset_statistics["numeric_column_count_raw"],
    "categorical_feature_count": 0,
    "usable_predictor_count": dataset_statistics["candidate_predictor_count"],
    "excluded_predictors": list(dataset_statistics["excluded_columns_and_reasons"].keys()),
    "missing_value_summary": {
        "ordinary_missing_values": dataset_statistics["ordinary_missing_values"],
        "infinite_values_raw": dataset_statistics["infinite_values_raw"],
        "malformed_value_columns": dataset_statistics["malformed_value_columns"],
    },
}
with open(REPORT_SUPPORT_DIR / "dataset_description.json", "w") as f:
    json.dump(dataset_description, f, indent=2, default=str)
print("Saved: report_support/dataset_description.json")

# ---------------------------------------------------------------- experimental_setup.json
def pkg_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "Not detected")
    except Exception:
        return "Not detected"


y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")
y_val = pd.read_csv(PROCESSED_DIR / "y_validation.csv")
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")

experimental_setup = {
    "os": f"{platform.system()} {platform.release()} ({platform.version()})",
    "python_version": platform.python_version(),
    "cpu": platform.processor() or "Not detected",
    "ram": "Not detected - not queried (no OS-level RAM query performed in this script)",
    "gpu": "Not detected - no GPU-accelerated library invoked; all models trained on CPU",
    "ide_environment": "Jupyter (nbconvert-executed .ipynb, built from cell-marked .py scripts), "
                        "VS Code / Claude Code as the driving terminal",
    "jupyter_version": pkg_version("jupyter") if pkg_version("jupyter") != "Not detected" else "See nbconvert/nbformat versions",
    "nbconvert_version": pkg_version("nbconvert"),
    "nbformat_version": pkg_version("nbformat"),
    "pandas_version": pkg_version("pandas"),
    "numpy_version": pkg_version("numpy"),
    "scikit_learn_version": pkg_version("sklearn"),
    "xgboost_version": pkg_version("xgboost"),
    "shap_version": pkg_version("shap"),
    "matplotlib_version": pkg_version("matplotlib"),
    "seaborn_version": pkg_version("seaborn"),
    "joblib_version": pkg_version("joblib"),
    "random_state": 42,
    "train_rows": int(len(y_train)),
    "validation_rows": int(len(y_val)),
    "test_rows": int(len(y_test)),
}
with open(REPORT_SUPPORT_DIR / "experimental_setup.json", "w") as f:
    json.dump(experimental_setup, f, indent=2)
print("Saved: report_support/experimental_setup.json")

# ---------------------------------------------------------------- leakage_audit.json
leakage_audit = {
    "final_test_isolated_early": {
        "status": "PASS",
        "evidence": "split_final_test() runs immediately after drop_forbidden_predictors(), "
                    "before any preprocessing object is fit.",
        "relevant_script": "03_preprocessing.py",
    },
    "imputer_train_only": {
        "status": "PASS", "evidence": "fit_preprocessing(X_train) fits SimpleImputer on X_train only.",
        "relevant_script": "03_preprocessing.py / common.py",
    },
    "encoder_train_only": {
        "status": "PASS (no-op, documented)",
        "evidence": "No categorical predictors exist after exclusions (verified in 01); "
                    "build_encoder() is a documented no-op rather than silently skipped.",
        "relevant_script": "03_preprocessing.py",
    },
    "scaler_train_only": {
        "status": "PASS", "evidence": "build_scaler()/fit_preprocessing() fits StandardScaler on X_train only.",
        "relevant_script": "03_preprocessing.py / common.py",
    },
    "feature_ranking_train_only": {
        "status": "PASS", "evidence": "load_training_data() in 04 loads X_train_tree.csv exclusively; "
                                        "ANOVA/MI/RF/XGB importance all computed on that frame only.",
        "relevant_script": "04_feature_selection_optimization.py",
    },
    "validation_used_for_tuning": {
        "status": "PASS", "evidence": "run_feature_count_experiments() and tune_*() in 06/08/09 all "
                                        "evaluate on X_val/y_val.",
        "relevant_script": "06_logistic_regression.py, 08_random_forest.py, 09_xgboost.py",
    },
    "test_not_used_for_feature_count_selection": {
        "status": "PASS", "evidence": "best_k is chosen from feature_count_results (validation-only) "
                                        "in each of 06/08/09, before the test set is ever loaded.",
        "relevant_script": "06_logistic_regression.py, 08_random_forest.py, 09_xgboost.py",
    },
    "test_not_used_for_tuning": {
        "status": "PASS", "evidence": "tune_*() functions in 06/08/09 take only X_train/y_train/X_val/y_val "
                                        "as arguments; X_test is not loaded until notebook 11.",
        "relevant_script": "06_logistic_regression.py, 08_random_forest.py, 09_xgboost.py, 11_final_model_comparison_and_predictions.py",
    },
    "label_excluded_from_X": {"status": "PASS", "evidence": "drop_forbidden_predictors() drops Label into y, not X.",
                                "relevant_script": "03_preprocessing.py"},
    "cat_excluded": {"status": "PASS", "evidence": "Cat in ALTERNATE_TARGETS, dropped from X.", "relevant_script": "common.py"},
    "sub_cat_excluded": {"status": "PASS", "evidence": "Sub_Cat in ALTERNATE_TARGETS, dropped from X.", "relevant_script": "common.py"},
    "flow_id_excluded": {"status": "PASS", "evidence": "Flow_ID in IDENTIFIER_COLS, dropped from X.", "relevant_script": "common.py"},
    "src_ip_excluded": {"status": "PASS", "evidence": "Src_IP in IDENTIFIER_COLS, dropped from X.", "relevant_script": "common.py"},
    "dst_ip_excluded": {"status": "PASS", "evidence": "Dst_IP in IDENTIFIER_COLS, dropped from X.", "relevant_script": "common.py"},
    "timestamp_excluded": {"status": "PASS", "evidence": "Timestamp in IDENTIFIER_COLS, dropped from X.", "relevant_script": "common.py"},
}
with open(REPORT_SUPPORT_DIR / "leakage_audit.json", "w") as f:
    json.dump(leakage_audit, f, indent=2)
print("Saved: report_support/leakage_audit.json")

# ---------------------------------------------------------------- notebook_manifest.csv
notebook_manifest = pd.DataFrame([
    {"Notebook": "01_Dataset_Exploration.ipynb", "Purpose": "Raw dataset audit and data-quality statistics",
     "Source_Python_Script": "01_dataset_exploration.py", "Execution_Status": "EXECUTED",
     "Major_Outputs": "metrics/dataset_statistics.json"},
    {"Notebook": "02_EDA.ipynb", "Purpose": "Exploratory data analysis",
     "Source_Python_Script": "02_eda.py", "Execution_Status": "EXECUTED",
     "Major_Outputs": "7 figures in figures/"},
    {"Notebook": "03_Preprocessing.ipynb", "Purpose": "Leakage-safe preprocessing pipeline",
     "Source_Python_Script": "03_preprocessing.py", "Execution_Status": "EXECUTED",
     "Major_Outputs": "processed_data/*.csv, models/preprocessing_pipeline.joblib"},
    {"Notebook": "04_Feature_Selection_Optimization.ipynb", "Purpose": "Feature ranking and subset creation",
     "Source_Python_Script": "04_feature_selection_optimization.py", "Execution_Status": "EXECUTED",
     "Major_Outputs": "feature_selection/*"},
    {"Notebook": "05_Gradient_Descent_Binary_Classification.ipynb", "Purpose": "From-scratch logistic regression (educational)",
     "Source_Python_Script": "05_gradient_descent_binary_classification.py", "Execution_Status": "EXECUTED",
     "Major_Outputs": "metrics/gradient_descent_metrics.csv"},
    {"Notebook": "06_Logistic_Regression.ipynb", "Purpose": "Main linear candidate model",
     "Source_Python_Script": "06_logistic_regression.py", "Execution_Status": "EXECUTED",
     "Major_Outputs": "models/logistic_regression.joblib, metrics/logistic_regression_metrics.csv"},
    {"Notebook": "07_Regularisation_Ridge_Lasso_ElasticNet.ipynb", "Purpose": "L1/L2/ElasticNet sparsity study",
     "Source_Python_Script": "07_regularisation_ridge_lasso_elasticnet.py", "Execution_Status": "EXECUTED",
     "Major_Outputs": "metrics/regularisation_metrics.csv"},
    {"Notebook": "08_Random_Forest.ipynb", "Purpose": "Main ensemble candidate model",
     "Source_Python_Script": "08_random_forest.py", "Execution_Status": "EXECUTED",
     "Major_Outputs": "models/random_forest.joblib, metrics/random_forest_metrics.csv"},
    {"Notebook": "09_XGBoost.ipynb", "Purpose": "Main gradient-boosted candidate model",
     "Source_Python_Script": "09_xgboost.py", "Execution_Status": "EXECUTED",
     "Major_Outputs": "models/xgboost.joblib, metrics/xgboost_metrics.csv"},
    {"Notebook": "10_Explainability_SHAP.ipynb", "Purpose": "SHAP explainability for the strongest candidate",
     "Source_Python_Script": "10_explainability_shap.py", "Execution_Status": "EXECUTED",
     "Major_Outputs": "figures/shap_*.png (if SHAP available), report_support/explainability_summary.json"},
    {"Notebook": "11_Final_Model_Comparison_and_Predictions.ipynb", "Purpose": "Single final test-set evaluation",
     "Source_Python_Script": "11_final_model_comparison_and_predictions.py", "Execution_Status": "EXECUTED",
     "Major_Outputs": "metrics/final_model_comparison.csv, metrics/final_test_metrics.json, predictions/final_predictions.csv"},
])
notebook_manifest.to_csv(REPORT_SUPPORT_DIR / "notebook_manifest.csv", index=False)
print("Saved: report_support/notebook_manifest.csv")

# ---------------------------------------------------------------- artifact_manifest.csv
def dir_manifest_rows(directory, artifact_type, purpose):
    rows = []
    if directory.exists():
        for p in sorted(directory.iterdir()):
            if p.is_file():
                rows.append({"Artifact": str(p.relative_to(ROOT)), "Type": artifact_type,
                             "Generated_By": "pipeline scripts", "Purpose": purpose, "Status": "GENERATED"})
    return rows


artifact_rows = []
artifact_rows += dir_manifest_rows(PROCESSED_DIR, "processed_data", "Leakage-safe train/validation/test matrices")
artifact_rows += dir_manifest_rows(ROOT / "feature_selection", "feature_selection", "Feature ranking and subsets")
artifact_rows += dir_manifest_rows(ROOT / "models", "model", "Frozen fitted models / pipeline")
artifact_rows += dir_manifest_rows(METRICS_DIR, "metrics", "Development and final evaluation metrics")
artifact_rows += dir_manifest_rows(ROOT / "figures", "figure", "Report-quality visualizations")
artifact_rows += dir_manifest_rows(ROOT / "predictions", "predictions", "Final test-set predictions")
for f, purpose in [("README.md", "Project overview and reproduction instructions"),
                    ("requirements.txt", "Python dependency list"),
                    ("PROJECT_PROGRESS.md", "Resumable stage-by-stage progress log")]:
    if (ROOT / f).exists():
        artifact_rows.append({"Artifact": f, "Type": "documentation", "Generated_By": "manual/script",
                               "Purpose": purpose, "Status": "GENERATED"})
artifact_manifest = pd.DataFrame(artifact_rows)
artifact_manifest.to_csv(REPORT_SUPPORT_DIR / "artifact_manifest.csv", index=False)
print(f"Saved: report_support/artifact_manifest.csv ({len(artifact_manifest)} rows)")

# ---------------------------------------------------------------- figure_manifest.csv
figure_defs = [
    ("proposed_system_flow.png", "Proposed System Flow", "generate_flow_diagram.py", "3.1", "End-to-end pipeline diagram"),
    ("class_distribution.png", "Class Distribution", "02_eda.py", "3.5 / 4.1", "Normal vs Anomaly row counts"),
    ("class_percentage.png", "Class Percentage", "02_eda.py", "3.5 / 4.1", "Normal vs Anomaly percentages"),
    ("selected_feature_distributions.png", "Feature Distributions", "02_eda.py", "3.5", "Top correlated feature histograms"),
    ("normal_vs_anomaly_distributions.png", "Normal vs Anomaly", "02_eda.py", "3.5", "Class-conditional distributions"),
    ("selected_boxplots.png", "Selected Boxplots", "02_eda.py", "3.5", "Boxplots by Label"),
    ("correlation_heatmap.png", "Correlation Heatmap", "02_eda.py", "3.5", "Correlation among key features + Label"),
    ("target_feature_relationships.png", "Target-Feature Relationships", "02_eda.py", "3.5", "Scatter + correlation bar"),
    ("feature_ranking.png", "Feature Ranking", "04_feature_selection_optimization.py", "3.6", "Combined consensus ranking"),
    ("feature_count_comparison.png", "Feature-Count Comparison", "11_final_model_comparison_and_predictions.py", "4.2", "Macro-F1 vs feature count, all models"),
    ("gradient_descent_loss_curves.png", "GD Loss Curves", "05_gradient_descent_binary_classification.py", "3.7", "Batch vs Mini-Batch convergence"),
    ("logistic_confusion_matrix.png", "LR Confusion Matrix", "06_logistic_regression.py", "3.9", "Validation confusion matrix"),
    ("logistic_roc_curve.png", "LR ROC Curve", "06_logistic_regression.py", "3.9", "Validation ROC curve"),
    ("logistic_pr_curve.png", "LR PR Curve", "06_logistic_regression.py", "3.9", "Validation PR curve"),
    ("logistic_coefficient_importance.png", "LR Coefficients", "06_logistic_regression.py", "4.3", "Standardized coefficient magnitudes"),
    ("regularisation_metric_vs_c.png", "Regularisation Metric vs C", "07_regularisation_ridge_lasso_elasticnet.py", "4.3", "Macro-F1 vs C by penalty"),
    ("regularisation_nonzero_vs_c.png", "Sparsity vs C", "07_regularisation_ridge_lasso_elasticnet.py", "4.3", "Non-zero coefficients vs C"),
    ("regularisation_coefficient_paths.png", "L1 Coefficient Paths", "07_regularisation_ridge_lasso_elasticnet.py", "4.3", "Coefficient shrinkage paths"),
    ("rf_confusion_matrix.png", "RF Confusion Matrix", "08_random_forest.py", "3.9", "Validation confusion matrix"),
    ("rf_roc_curve.png", "RF ROC Curve", "08_random_forest.py", "3.9", "Validation ROC curve"),
    ("rf_pr_curve.png", "RF PR Curve", "08_random_forest.py", "3.9", "Validation PR curve"),
    ("rf_feature_importance.png", "RF Feature Importance", "08_random_forest.py", "4.3", "Native importance"),
    ("rf_permutation_importance.png", "RF Permutation Importance", "08_random_forest.py", "4.3", "Permutation importance"),
    ("rf_oob_analysis.png", "RF OOB Analysis", "08_random_forest.py", "3.7", "OOB score vs n_estimators"),
    ("xgb_confusion_matrix.png", "XGB Confusion Matrix", "09_xgboost.py", "3.9", "Validation confusion matrix"),
    ("xgb_roc_curve.png", "XGB ROC Curve", "09_xgboost.py", "3.9", "Validation ROC curve"),
    ("xgb_pr_curve.png", "XGB PR Curve", "09_xgboost.py", "3.9", "Validation PR curve"),
    ("xgb_feature_importance.png", "XGB Feature Importance", "09_xgboost.py", "4.3", "Native importance"),
    ("shap_summary.png", "SHAP Summary", "10_explainability_shap.py", "4.3", "SHAP beeswarm (if available)"),
    ("shap_importance.png", "SHAP Importance", "10_explainability_shap.py", "4.3", "SHAP mean |impact| bar"),
    ("final_model_comparison.png", "Final Model Comparison", "11_final_model_comparison_and_predictions.py", "4.2", "Test-set Macro-F1 by model"),
]
figure_manifest_rows = []
for fname, title, script, section, desc in figure_defs:
    exists = (ROOT / "figures" / fname).exists()
    figure_manifest_rows.append({"Figure_File": fname, "Figure_Title": title, "Generated_By": script,
                                  "Report_Section": section, "Description": desc,
                                  "Include_In_Final_Report": exists})
figure_manifest = pd.DataFrame(figure_manifest_rows)
figure_manifest.to_csv(REPORT_SUPPORT_DIR / "figure_manifest.csv", index=False)
print(f"Saved: report_support/figure_manifest.csv "
      f"({figure_manifest['Include_In_Final_Report'].sum()}/{len(figure_manifest)} figures present)")

# ---------------------------------------------------------------- limitations.md / future_work.md
limitations_md = """# Limitations

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
"""
(REPORT_SUPPORT_DIR / "limitations.md").write_text(limitations_md, encoding="utf-8")
print("Saved: report_support/limitations.md")

future_work_md = """# Future Work

- K-fold cross-validation across the full pipeline for a more robust generalization estimate.
- Multi-class modelling of the `Sub_Cat` attack families, not just binary Normal/Anomaly.
- Evaluation on additional or more recent IoT traffic captures to test robustness to concept drift.
- Threshold tuning and cost-sensitive evaluation matched to a specific deployment's false
  positive/false negative cost trade-off.
- Model compression/quantization and benchmarking for deployment on constrained IoT gateway hardware.
- Adversarial robustness testing against evasion attempts targeting the selected features.
- Online/incremental learning to adapt to new attack patterns without full retraining.
"""
(REPORT_SUPPORT_DIR / "future_work.md").write_text(future_work_md, encoding="utf-8")
print("Saved: report_support/future_work.md")
