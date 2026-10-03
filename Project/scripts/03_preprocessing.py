# %% [markdown]
# # IoT Intrusion Detection — Preprocessing
#
# ### Objective
# Build the single authoritative, leakage-safe preprocessing pipeline used by
# every model notebook after this one. The final test set is isolated first
# and never influences any fitted transformation, feature ranking, or model
# decision; every learned statistic (imputation median, scaler mean/std) is
# fit on training data only.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

_candidates = [Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts",
               Path.cwd().parent, Path.cwd().parent.parent / "scripts"]
_scripts_dir = next((c for c in _candidates if (c / "common.py").exists()), None)
if _scripts_dir is None:
    raise FileNotFoundError(f"Could not locate scripts/common.py from cwd={Path.cwd()}")
sys.path.insert(0, str(_scripts_dir))
from common import (ROOT, DATA_PATH, PROCESSED_DIR, MODELS_DIR, REPORT_SUPPORT_DIR,
                     RANDOM_STATE, TARGET, ALTERNATE_TARGETS, IDENTIFIER_COLS, TARGET_MAPPING,
                     coerce_numeric_predictors, fit_preprocessing, transform_preprocessing)

pd.set_option("display.max_columns", 100)


# %% [markdown]
# ## 1. Load Raw Dataset and Validate Schema

# %%
def load_raw_dataset():
    return pd.read_csv(DATA_PATH, low_memory=False)


def validate_schema(df):
    required = ["Label", "Cat", "Sub_Cat"] + IDENTIFIER_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")
    return True


df = load_raw_dataset()
validate_schema(df)
print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns — schema validated.")

# %% [markdown]
# ## 2. Dtype Safety, Duplicate Removal, Target Encoding

# %%
candidate_predictors = [c for c in df.columns if c not in [TARGET] + ALTERNATE_TARGETS + IDENTIFIER_COLS]
coercion_report = coerce_numeric_predictors(df, candidate_predictors)
print("Dtype coercion report (new NaNs introduced):", coercion_report)

rows_before = len(df)


def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


df = remove_duplicates(df)
rows_after = len(df)
print(f"Rows before dedup: {rows_before:,} | after: {rows_after:,} | "
      f"removed: {rows_before - rows_after:,} ({(rows_before - rows_after) / rows_before * 100:.2f}%)")


def encode_target(y_raw):
    unexpected = set(y_raw.unique()) - set(TARGET_MAPPING)
    if unexpected:
        raise ValueError(f"Unexpected Label values: {unexpected}")
    return y_raw.map(TARGET_MAPPING).astype(int)


# %% [markdown]
# ## 3. Drop Forbidden Predictors

# %%
def drop_forbidden_predictors(df):
    drop_cols = [TARGET] + ALTERNATE_TARGETS + IDENTIFIER_COLS
    X = df.drop(columns=drop_cols)
    y = df[TARGET].copy()
    return X, y


X, y = drop_forbidden_predictors(df)
print(f"X shape: {X.shape} (dropped: {[TARGET] + ALTERNATE_TARGETS + IDENTIFIER_COLS})")
del df

# %% [markdown]
# ## 4. Isolate the Final Test Set, Then Split Train / Validation
#
# The final test set is separated **first**, before any preprocessing is fit,
# and is not touched again until notebook 11.

# %%
def split_final_test(X, y):
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def split_train_validation(X_dev, y_dev):
    return train_test_split(X_dev, y_dev, test_size=0.20, random_state=RANDOM_STATE, stratify=y_dev)


X_dev, X_test, y_dev, y_test = split_final_test(X, y)
X_train, X_val, y_train, y_val = split_train_validation(X_dev, y_dev)

y_train_enc = encode_target(y_train)
y_val_enc = encode_target(y_val)
y_test_enc = encode_target(y_test)

print(f"Train:      {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}  (isolated — untouched by every notebook until 11)")

X_test_raw = X_test.copy()  # kept unscaled/unimputed for the notebook-11 raw-input demonstration
del X, y, X_dev, y_dev

# %% [markdown]
# ### Observation
# The 80/20 -> 80/20 split design means the final test set never participates
# in any development decision, while the train/validation split still leaves
# a large, representative validation set for feature selection, model
# comparison, and hyperparameter tuning in notebooks 04-10.

# %% [markdown]
# ## 5. Detect Feature Types, Replace Infinities, Fit Preprocessing on Train Only
#
# `build_numeric_imputer()`, `build_scaler()` fit **exclusively on `X_train`**.
# This dataset has no categorical predictors once identifiers/targets are
# excluded (verified in notebook 01), so `build_categorical_imputer()` /
# `build_encoder()` are written but intentionally no-ops here — documented,
# not silently skipped.

# %%
def detect_feature_types(X):
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def replace_infinities(X, numeric_cols):
    return X[numeric_cols].replace([np.inf, -np.inf], np.nan)


def build_numeric_imputer(X_train_num):
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    imp.fit(X_train_num)
    return imp


def build_categorical_imputer(categorical_cols):
    if not categorical_cols:
        print("No categorical predictors present — categorical imputer is a documented no-op.")
        return None
    from sklearn.impute import SimpleImputer
    return SimpleImputer(strategy="most_frequent")


def build_encoder(categorical_cols):
    if not categorical_cols:
        print("No categorical predictors present — encoder is a documented no-op.")
        return None
    from sklearn.preprocessing import OneHotEncoder
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_scaler(X_train_imputed):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train_imputed)
    return sc


def fit_preprocessing_on_train(X_train):
    return fit_preprocessing(X_train)  # shared, tested implementation in common.py


numeric_cols, categorical_cols = detect_feature_types(X_train)
print(f"Numeric columns: {len(numeric_cols)} | Categorical columns: {len(categorical_cols)}")
_ = build_categorical_imputer(categorical_cols)
_ = build_encoder(categorical_cols)

fitted = fit_preprocessing_on_train(X_train)
print(f"Constant columns removed ({len(fitted['constant_cols'])}): {fitted['constant_cols']}")
print(f"Missing-indicator columns created: {fitted['missing_numeric_cols']}")
print(f"Final feature count (scaled matrix): {len(fitted['final_feature_order'])}")

# %% [markdown]
# ## 6. Transform Train / Validation / Test
#
# Two output flavours are produced from the **same** fitted imputer:
# a fully **scaled** matrix (`_processed`, for scale-sensitive models like
# Logistic Regression / Gradient Descent) and an **imputed-only, unscaled**
# matrix (`_tree`, for tree-based models, which are scale-invariant).

# %%
def transform_validation(X_val, fitted, scale=True):
    return transform_preprocessing(X_val, fitted, scale=scale)


def transform_test(X_test, fitted, scale=True):
    return transform_preprocessing(X_test, fitted, scale=scale)


X_train_processed = transform_preprocessing(X_train, fitted, scale=True)
X_val_processed = transform_validation(X_val, fitted, scale=True)
X_test_processed = transform_test(X_test, fitted, scale=True)

X_train_tree = transform_preprocessing(X_train, fitted, scale=False)
X_val_tree = transform_validation(X_val, fitted, scale=False)
X_test_tree = transform_test(X_test, fitted, scale=False)

print("Scaled shapes:", X_train_processed.shape, X_val_processed.shape, X_test_processed.shape)
print("Tree (unscaled) shapes:", X_train_tree.shape, X_val_tree.shape, X_test_tree.shape)

# %% [markdown]
# ## 7. Validate Processed Matrices
#
# Any failed check raises immediately rather than silently continuing.

# %%
def validate_processed_matrices(frames: dict):
    for name, arr in frames.items():
        assert not arr.isna().any().any(), f"{name} has NaN"
        assert not np.isinf(arr.to_numpy()).any(), f"{name} has inf"
    cols_ref = list(frames["X_train_processed"].columns)
    for name in ["X_val_processed", "X_test_processed"]:
        assert list(frames[name].columns) == cols_ref, f"{name} column mismatch"
    assert not frames["X_train_processed"].columns.duplicated().any(), "duplicate feature names"
    print("All processed-matrix validation checks passed.")


validate_processed_matrices({
    "X_train_processed": X_train_processed, "X_val_processed": X_val_processed,
    "X_test_processed": X_test_processed, "X_train_tree": X_train_tree,
    "X_val_tree": X_val_tree, "X_test_tree": X_test_tree,
})
assert set(y_train_enc.unique()) <= {0, 1} and set(y_test_enc.unique()) <= {0, 1}
print("Target values confirmed restricted to {0, 1}.")

# %% [markdown]
# ## 8. Save Processed Datasets and the Fitted Pipeline

# %%
def save_processed_datasets():
    X_train_processed.to_csv(PROCESSED_DIR / "X_train_processed.csv", index=False)
    X_val_processed.to_csv(PROCESSED_DIR / "X_validation_processed.csv", index=False)
    X_test_processed.to_csv(PROCESSED_DIR / "X_test_processed.csv", index=False)
    X_train_tree.to_csv(PROCESSED_DIR / "X_train_tree.csv", index=False)
    X_val_tree.to_csv(PROCESSED_DIR / "X_validation_tree.csv", index=False)
    X_test_tree.to_csv(PROCESSED_DIR / "X_test_tree.csv", index=False)
    y_train_enc.rename("Label").to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_val_enc.rename("Label").to_csv(PROCESSED_DIR / "y_validation.csv", index=False)
    y_test_enc.rename("Label").to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    print("Saved all 9 processed_data/ files.")


def save_preprocessing_pipeline():
    joblib.dump({
        "numeric_cols": fitted["numeric_cols"],
        "constant_cols": fitted["constant_cols"],
        "missing_numeric_cols": fitted["missing_numeric_cols"],
        "imputer": fitted["imputer"],
        "scaler": fitted["scaler"],
        "final_feature_order": fitted["final_feature_order"],
        "target_mapping": TARGET_MAPPING,
        "identifier_cols_excluded": IDENTIFIER_COLS,
        "alternate_target_cols_excluded": ALTERNATE_TARGETS,
    }, MODELS_DIR / "preprocessing_pipeline.joblib")
    print("Saved models/preprocessing_pipeline.joblib")


save_processed_datasets()
save_preprocessing_pipeline()
X_test_raw.to_csv(PROCESSED_DIR / "X_test_raw_for_demo.csv", index=False)
print("Saved processed_data/X_test_raw_for_demo.csv (raw, unprocessed — used only by notebook 11's demonstration)")

# %% [markdown]
# ## 9. Preprocessing Summary (Report Support)

# %%
missing_before = int(X_train[fitted["numeric_cols"]].replace([np.inf, -np.inf], np.nan).isna().sum().sum())
preprocessing_summary = {
    "raw_rows_before_dedup": rows_before,
    "rows_after_dedup": rows_after,
    "duplicates_removed": rows_before - rows_after,
    "target_mapping": TARGET_MAPPING,
    "excluded_columns": [TARGET] + ALTERNATE_TARGETS + IDENTIFIER_COLS,
    "train_rows": X_train.shape[0], "validation_rows": X_val.shape[0], "test_rows": X_test.shape[0],
    "numeric_feature_count_raw": len(numeric_cols),
    "categorical_feature_count_raw": len(categorical_cols),
    "constant_columns_removed": fitted["constant_cols"],
    "missing_indicator_columns": fitted["missing_numeric_cols"],
    "missing_values_in_train_after_infinity_conversion": missing_before,
    "imputation_method": "SimpleImputer(strategy='median'), fit on X_train only",
    "encoding_method": "None required — every remaining predictor is numeric (verified in notebook 01)",
    "scaling_method": "StandardScaler, fit on X_train only",
    "final_feature_count_scaled": len(fitted["final_feature_order"]),
    "final_feature_count_tree": len(fitted["final_feature_order"]),
    "train_class_distribution": y_train.value_counts().to_dict(),
    "validation_class_distribution": y_val.value_counts().to_dict(),
    "test_class_distribution": y_test.value_counts().to_dict(),
}
with open(REPORT_SUPPORT_DIR / "preprocessing_summary.json", "w") as f:
    json.dump(preprocessing_summary, f, indent=2, default=str)
print("Saved report_support/preprocessing_summary.json")

# %% [markdown]
# ## Notebook Summary

# %%
from IPython.display import display, Markdown

summary_md = f"""
- Loaded the raw dataset, validated its schema, coerced the one malformed
  `Dst_Port` value to numeric, and removed **{rows_before - rows_after:,}**
  exact duplicates ({(rows_before - rows_after) / rows_before * 100:.2f}%).
- Isolated the final test set **first** ({X_test.shape[0]:,} rows) before any
  learned preprocessing was fit, then split the remaining data into
  train ({X_train.shape[0]:,} rows) and validation ({X_val.shape[0]:,} rows).
- Fit imputation and scaling on **training data only**; removed
  {len(fitted['constant_cols'])} constant columns; created
  {len(fitted['missing_numeric_cols'])} missing-indicator columns.
- Produced both a scaled matrix ({len(fitted['final_feature_order'])} features,
  for Logistic Regression / Gradient Descent) and an unscaled, imputed-only
  matrix (for tree models) from the same fitted objects.
- Verified zero remaining NaN/infinity in every processed matrix, identical
  train/validation/test columns, and target values restricted to {{0, 1}}.
- Saved all 9 `processed_data/` files, `models/preprocessing_pipeline.joblib`,
  and `report_support/preprocessing_summary.json`.

**Next notebook (`04_Feature_Selection_Optimization.ipynb`)** uses
`X_train_processed`/`X_train_tree` (training data only) to rank features and
build the 10/15/20/30/All feature subsets used by every model notebook after it.
"""
display(Markdown(summary_md))
