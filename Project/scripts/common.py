"""Shared utilities for the IoT Intrusion Detection pipeline scripts.

Not a notebook itself - imported by every numbered script so the leakage-safe
preprocessing and evaluation logic is defined exactly once (DRY) and stays
consistent across the whole pipeline.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "Label"
ALTERNATE_TARGETS = ["Cat", "Sub_Cat"]
IDENTIFIER_COLS = ["Flow_ID", "Src_IP", "Dst_IP", "Timestamp"]
TARGET_MAPPING = {"Normal": 0, "Anomaly": 1}


def get_project_root() -> Path:
    """Portable project-root discovery: works from Project/ or Project/scripts/ or
    Project/notebooks/, on any machine/username/drive letter."""
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for c in candidates:
        if (c / "data" / "Final IoT Network Intrusion Dataset.csv").exists():
            return c.resolve()
    raise FileNotFoundError(
        "Could not locate 'data/Final IoT Network Intrusion Dataset.csv' from "
        f"the current working directory ({Path.cwd()})."
    )


ROOT = get_project_root()
DATA_PATH = ROOT / "data" / "Final IoT Network Intrusion Dataset.csv"
EDA_DIR = ROOT / "data" / "outputs"
PROCESSED_DIR = ROOT / "processed_data"
FEATSEL_DIR = ROOT / "feature_selection"
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "metrics"
PREDICTIONS_DIR = ROOT / "predictions"
FIGURES_DIR = ROOT / "figures"
REPORT_SUPPORT_DIR = ROOT / "report_support"
REPORT_DIR = ROOT / "report"

for d in [PROCESSED_DIR, FEATSEL_DIR, MODELS_DIR, METRICS_DIR, PREDICTIONS_DIR,
          FIGURES_DIR, REPORT_SUPPORT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def tic():
    return time.perf_counter()


def toc(t0):
    return time.perf_counter() - t0


# ----------------------------------------------------------------------
# Leakage-safe preprocessing: fit on ONE frame, transform any other frame
# with the SAME fitted objects. Identical logic used everywhere.
# ----------------------------------------------------------------------
def fit_preprocessing(X_fit: pd.DataFrame) -> dict:
    numeric_cols = [c for c in X_fit.columns if pd.api.types.is_numeric_dtype(X_fit[c])]
    X_fit_num = X_fit[numeric_cols].replace([np.inf, -np.inf], np.nan)

    constant_cols = [c for c in numeric_cols if X_fit_num[c].nunique(dropna=False) <= 1]
    numeric_cols = [c for c in numeric_cols if c not in constant_cols]
    X_fit_num = X_fit_num[numeric_cols]

    missing_numeric_cols = [c for c in numeric_cols if X_fit_num[c].isna().any()]

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_fit_num)
    X_fit_imp = pd.DataFrame(imputer.transform(X_fit_num), columns=numeric_cols, index=X_fit.index)

    scaler = StandardScaler()
    scaler.fit(X_fit_imp)

    return {
        "numeric_cols": numeric_cols,
        "constant_cols": constant_cols,
        "missing_numeric_cols": missing_numeric_cols,
        "imputer": imputer,
        "scaler": scaler,
        "final_feature_order": numeric_cols + [f"{c}_missing" for c in missing_numeric_cols],
    }


def transform_preprocessing(X_raw: pd.DataFrame, fitted: dict, scale: bool = True) -> pd.DataFrame:
    """scale=True -> fully processed (imputed + standardized), for scale-sensitive
    models. scale=False -> imputed only (tree-oriented), unscaled."""
    numeric_cols = fitted["numeric_cols"]
    X_num = X_raw[numeric_cols].replace([np.inf, -np.inf], np.nan)

    ind = pd.DataFrame(index=X_raw.index)
    for c in fitted["missing_numeric_cols"]:
        ind[f"{c}_missing"] = X_num[c].isna().astype(int)

    X_imp = pd.DataFrame(fitted["imputer"].transform(X_num), columns=numeric_cols, index=X_raw.index)
    if scale:
        X_imp = pd.DataFrame(fitted["scaler"].transform(X_imp), columns=numeric_cols, index=X_raw.index)

    out = pd.concat([X_imp, ind], axis=1)[fitted["final_feature_order"]]
    return out.astype(np.float32)


def coerce_numeric_predictors(df: pd.DataFrame, expected_numeric_predictors: list) -> dict:
    """Force expected-numeric predictor columns to numeric, coercing malformed
    tokens to NaN. Returns a report of what was coerced (col -> new NaNs)."""
    coercion_report = {}
    for c in expected_numeric_predictors:
        if not pd.api.types.is_numeric_dtype(df[c]):
            before_nan = int(df[c].isna().sum())
            coerced = pd.to_numeric(df[c], errors="coerce")
            new_nan = int(coerced.isna().sum())
            coercion_report[c] = new_nan - before_nan
            df[c] = coerced
    return coercion_report


def eval_binary(y_true, y_pred, y_proba) -> dict:
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_Precision": prec_m, "Macro_Recall": rec_m, "Macro_F1": f1_m, "Weighted_F1": f1_w,
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        "PR_AUC": average_precision_score(y_true, y_proba) if y_proba is not None else np.nan,
        "Normal_Precision": prec_c[0], "Normal_Recall": rec_c[0], "Normal_F1": f1_c[0],
        "Anomaly_Precision": prec_c[1], "Anomaly_Recall": rec_c[1], "Anomaly_F1": f1_c[1],
    }


def fit_time(model, X, y, sample_weight=None):
    t0 = tic()
    if sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return toc(t0)


def predict_full(model, X):
    t0 = tic()
    proba = model.predict_proba(X)[:, 1]
    dt = toc(t0)
    return (proba >= 0.5).astype(int), proba, dt


def model_size_mb(path: Path) -> float:
    return round(Path(path).stat().st_size / (1024 * 1024), 3)
