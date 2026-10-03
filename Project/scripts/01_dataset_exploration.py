# %% [markdown]
# # IoT Intrusion Detection — Dataset Exploration
#
# ### Objective
# Load the raw IoT network-flow dataset and establish the data-quality statistics
# (shape, dtypes, missing values, duplicates, target balance) that every later
# stage of this pipeline depends on. No model training happens in this notebook.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_candidates = [Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts",
               Path.cwd().parent, Path.cwd().parent.parent / "scripts"]
_scripts_dir = next((c for c in _candidates if (c / "common.py").exists()), None)
if _scripts_dir is None:
    raise FileNotFoundError(f"Could not locate scripts/common.py from cwd={Path.cwd()}")
sys.path.insert(0, str(_scripts_dir))
from common import ROOT, DATA_PATH, METRICS_DIR, TARGET, ALTERNATE_TARGETS, IDENTIFIER_COLS, coerce_numeric_predictors

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)
print("Project root:", ROOT)
print("Dataset path:", DATA_PATH)
print("Dataset found:", DATA_PATH.exists())

# %% [markdown]
# ## 1. Load the Raw Dataset
#
# We load the CSV exactly as published, then verify the three target-related
# columns (`Label`, `Cat`, `Sub_Cat`) actually exist before doing anything else.

# %%
df = pd.read_csv(DATA_PATH, low_memory=False)
RAW_SHAPE = df.shape
print(f"Raw dataset shape: {RAW_SHAPE[0]:,} rows x {RAW_SHAPE[1]} columns")

required_targets = ["Label", "Cat", "Sub_Cat"]
missing_targets = [c for c in required_targets if c not in df.columns]
if missing_targets:
    raise ValueError(f"Required target columns missing: {missing_targets}")
print("Required target columns present:", required_targets)

# %% [markdown]
# ## 2. Columns and Data Types
#
# Every column name and its inferred pandas dtype, plus a count of numeric vs.
# non-numeric columns. Non-numeric columns are expected only for the three
# target columns and the four high-cardinality identifiers.

# %%
dtype_table = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "n_unique": [df[c].nunique(dropna=False) for c in df.columns],
})
print(dtype_table.to_string())
print("\nDtype value counts:")
print(dtype_table["dtype"].value_counts())

candidate_predictors = [c for c in df.columns if c not in [TARGET] + ALTERNATE_TARGETS + IDENTIFIER_COLS]
print(f"\nCandidate predictor columns (before any cleaning): {len(candidate_predictors)}")

# %% [markdown]
# ### Observation
# Most columns are numeric (`int64`/`float64`) as expected for network-flow
# statistics. A handful are non-numeric by nature (the three targets and four
# identifiers) — Section 3 checks whether every *other* column is genuinely
# numeric, since a single malformed value can silently force a whole column to
# string dtype.

# %% [markdown]
# ## 3. Head / Tail

# %%
print("First 3 rows:")
print(df.head(3).to_string())
print("\nLast 3 rows:")
print(df.tail(3).to_string())

# %% [markdown]
# ## 4. Data-Quality Audit: Missing, Infinite, and Malformed Values
#
# Three *different* data-quality issues are checked separately, because they
# require different fixes: ordinary missing cells (`NaN`), infinite values
# (a rate feature divided by a near-zero duration), and malformed non-numeric
# tokens hiding inside an otherwise-numeric column.

# %%
numeric_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
ordinary_missing = int(df.isna().sum().sum())
raw_inf = int(np.isinf(df[numeric_cols_raw]).sum().sum())
inf_by_col = np.isinf(df[numeric_cols_raw]).sum()

print(f"Ordinary missing values (NaN cells) in raw data: {ordinary_missing}")
print(f"Infinite values in raw data (before dedup):      {raw_inf}")
print("\nInfinite values by column (nonzero only):")
print(inf_by_col[inf_by_col > 0])

expected_numeric_predictors = candidate_predictors
non_numeric_found = {c: str(df[c].dtype) for c in expected_numeric_predictors
                      if not pd.api.types.is_numeric_dtype(df[c])}
print("\nPredictor columns that failed to parse as numeric:", non_numeric_found)

coercion_report = {}
for c, dtype in non_numeric_found.items():
    bad_values = df.loc[pd.to_numeric(df[c], errors="coerce").isna() & df[c].notna(), c].unique()
    coercion_report[c] = {"dtype_before": dtype, "malformed_values": [str(v) for v in bad_values]}
print("Malformed-value report:", coercion_report)

# %% [markdown]
# ### Observation
# The raw file has **zero** ordinary missing values, but that is not the same
# as "fully clean": `Flow_Byts/s` and `Flow_Pkts/s` contain infinite values
# (division by a near-zero flow duration), and — a genuinely new finding,
# verified by code rather than assumed — `Dst_Port` contains one malformed
# token that forces the entire column to text. All three issues are handled
# explicitly (never silently) in the preprocessing notebook (03).

# %% [markdown]
# ## 5. Duplicate Analysis
#
# Exact duplicate network flows carry no additional information and, if left
# in, could end up split across train and test, leaking information. We
# measure them here; actual removal happens in the preprocessing notebook.

# %%
rows_before = len(df)
duplicate_count = int(df.duplicated().sum())
duplicate_pct = duplicate_count / rows_before * 100
rows_after_dedup = rows_before - duplicate_count
print(f"Rows before duplicate removal: {rows_before:,}")
print(f"Exact duplicate rows:          {duplicate_count:,}  ({duplicate_pct:.2f}%)")
print(f"Rows after duplicate removal:  {rows_after_dedup:,}")

# %% [markdown]
# ## 6. Target (`Label`) Inspection
#
# `Label` is the binary classification target for this project. `Cat` and
# `Sub_Cat` are shown for context only — they are excluded from every model
# input because they would leak the answer (see Section 8).

# %%
print("Label unique values:", sorted(df["Label"].unique()))
print("\nLabel counts:")
print(df["Label"].value_counts())
print("\nLabel percentages:")
print((df["Label"].value_counts(normalize=True) * 100).round(2))

print("\nCat value counts (context only, never a model input):")
print(df["Cat"].value_counts())
print("\nSub_Cat value counts (context only, never a model input):")
print(df["Sub_Cat"].value_counts())

TARGET_MAPPING = {"Normal": 0, "Anomaly": 1}
print("\nRequired target mapping:", TARGET_MAPPING)

# %% [markdown]
# ## 7. Descriptive Statistics
#
# Summary statistics for every numeric column, computed on the raw data
# (before dedup/cleaning) purely for descriptive purposes.

# %%
desc = df[numeric_cols_raw].describe().T
print(desc.round(3).to_string())

# %% [markdown]
# ## 8. Unique-Value, Constant, and Near-Constant Feature Inspection
#
# Constant columns (a single distinct value) carry zero predictive
# information and are removed during preprocessing; near-constant columns
# (>99% one value) are flagged for awareness but not automatically dropped,
# since a rare-but-real signal can still be informative for a highly
# imbalanced intrusion-detection target.

# %%
df_dedup_view = df.drop_duplicates()
predictor_cols_for_audit = [c for c in candidate_predictors if pd.api.types.is_numeric_dtype(df_dedup_view[c])
                             or c in non_numeric_found]
# Malformed-but-intended-numeric columns are coerced only for this audit view.
df_audit = df_dedup_view.copy()
coerce_numeric_predictors(df_audit, expected_numeric_predictors)

unique_counts = df_audit[predictor_cols_for_audit].nunique(dropna=False).sort_values()
constant_features = unique_counts[unique_counts <= 1].index.tolist()

top_value_share = {}
for c in predictor_cols_for_audit:
    vc = df_audit[c].value_counts(normalize=True, dropna=False)
    top_value_share[c] = float(vc.iloc[0]) if len(vc) else np.nan
near_constant_features = sorted([c for c, share in top_value_share.items()
                                  if share is not None and 0.99 <= share < 1.0])

print(f"Constant features ({len(constant_features)}): {constant_features}")
print(f"\nNear-constant features (>=99% one value, excluding truly constant) "
      f"({len(near_constant_features)}): {near_constant_features}")

# %% [markdown]
# ## 9. Excluded Predictor Columns (Documented)
#
# These columns are never used as model predictors, for the reasons stated.

# %%
exclusion_reasons = {
    "Label": "Prediction target itself.",
    "Cat": "Alternative/finer target — only knowable once a flow is already known "
           "to be anomalous; using it as a predictor would leak the answer.",
    "Sub_Cat": "Same leakage reason as Cat, at finer granularity.",
    "Flow_ID": "High-cardinality identifier — risks memorization instead of learning "
               "general traffic behaviour.",
    "Src_IP": "High-cardinality identifier — same memorization risk.",
    "Dst_IP": "High-cardinality identifier — same memorization risk.",
    "Timestamp": "Raw timestamp identifier — risks memorizing specific capture windows "
                 "instead of learning generalizable attack behaviour.",
}
for col, reason in exclusion_reasons.items():
    print(f"  {col:10s} -> {reason}")

# %% [markdown]
# ## 10. Save Dataset Statistics
#
# Every number above is written to `metrics/dataset_statistics.json` so later
# notebooks and the final report read the same, single source of truth instead
# of re-typing remembered values.

# %%
dataset_statistics = {
    "raw_rows": RAW_SHAPE[0],
    "raw_columns": RAW_SHAPE[1],
    "ordinary_missing_values": ordinary_missing,
    "infinite_values_raw": raw_inf,
    "infinite_value_columns": {k: int(v) for k, v in inf_by_col[inf_by_col > 0].items()},
    "malformed_value_columns": coercion_report,
    "duplicate_rows": duplicate_count,
    "duplicate_percentage": round(duplicate_pct, 4),
    "rows_after_dedup": rows_after_dedup,
    "target_column": "Label",
    "target_mapping": TARGET_MAPPING,
    "label_counts": df["Label"].value_counts().to_dict(),
    "label_percentages": (df["Label"].value_counts(normalize=True) * 100).round(4).to_dict(),
    "candidate_predictor_count": len(candidate_predictors),
    "constant_features": constant_features,
    "near_constant_features": near_constant_features,
    "excluded_columns_and_reasons": exclusion_reasons,
    "numeric_column_count_raw": len(numeric_cols_raw),
}
with open(METRICS_DIR / "dataset_statistics.json", "w") as f:
    json.dump(dataset_statistics, f, indent=2, default=str)
print("Saved: metrics/dataset_statistics.json")

# %% [markdown]
# ## Notebook Summary

# %%
from IPython.display import display, Markdown

summary_md = f"""
- Loaded the raw IoT dataset: **{RAW_SHAPE[0]:,} rows x {RAW_SHAPE[1]} columns**.
- Verified ordinary missing values = **{ordinary_missing}**; found **{raw_inf}** infinite
  values in two rate features, and one genuinely new malformed-value finding in
  `Dst_Port` that a naive audit would miss.
- Verified **{duplicate_count:,} exact duplicates ({duplicate_pct:.2f}%)**, leaving
  **{rows_after_dedup:,}** rows after removal.
- Confirmed the binary target `Label` and its required mapping (`Normal=0`, `Anomaly=1`);
  class split is **{(df['Label'].value_counts(normalize=True)*100).round(2).to_dict()}**.
- Found **{len(constant_features)}** constant feature(s) and **{len(near_constant_features)}**
  near-constant feature(s) among the candidate predictors.
- Documented all {len(exclusion_reasons)} excluded predictor columns and why.
- Saved all of the above to `metrics/dataset_statistics.json` for reuse by every later
  notebook and the final report — no numbers are retyped by hand.

**Next notebook (`02_EDA.ipynb`)** uses this same raw file to explore class balance,
feature distributions, and correlations in more depth.
"""
display(Markdown(summary_md))
