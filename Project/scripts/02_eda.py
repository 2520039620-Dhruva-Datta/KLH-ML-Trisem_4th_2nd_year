# %% [markdown]
# # IoT Intrusion Detection — Exploratory Data Analysis
#
# ### Objective
# Explore class balance, feature distributions, and correlations in the cleaned
# IoT network-flow data, using reproducible sampling for expensive plots. This
# notebook is descriptive only — no learned transformation is fit here (that
# happens, leakage-safely, in `03_Preprocessing.ipynb`).

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_candidates = [Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts",
               Path.cwd().parent, Path.cwd().parent.parent / "scripts"]
_scripts_dir = next((c for c in _candidates if (c / "common.py").exists()), None)
if _scripts_dir is None:
    raise FileNotFoundError(f"Could not locate scripts/common.py from cwd={Path.cwd()}")
sys.path.insert(0, str(_scripts_dir))
from common import (ROOT, DATA_PATH, FIGURES_DIR, RANDOM_STATE, TARGET, ALTERNATE_TARGETS,
                     IDENTIFIER_COLS, TARGET_MAPPING, coerce_numeric_predictors)

plt.rcParams.update({"figure.dpi": 100, "font.size": 10})
pd.set_option("display.max_columns", 100)


def save_figure(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: figures/{name}")


# %% [markdown]
# ## 1. Load and Clean a Working Copy
#
# A local, lightweight clean copy (dtype fix + duplicate removal + target
# mapping) is built here purely for exploration. The single authoritative,
# leakage-safe preprocessing pipeline lives in notebook 03.

# %%
df = pd.read_csv(DATA_PATH, low_memory=False)
candidate_predictors = [c for c in df.columns if c not in [TARGET] + ALTERNATE_TARGETS + IDENTIFIER_COLS]
coerce_numeric_predictors(df, candidate_predictors)
df = df.drop_duplicates().reset_index(drop=True)
df["Label_encoded"] = df["Label"].map(TARGET_MAPPING)
print(f"Clean working copy: {df.shape[0]:,} rows x {df.shape[1]} columns")

# %% [markdown]
# ## 2. Class Distribution
#
# The target is expected to be strongly imbalanced. This is the single most
# important fact shaping every metric choice later in the project.

# %%
counts = df["Label"].value_counts()
pct = df["Label"].value_counts(normalize=True) * 100
print(counts)
print(pct.round(2))

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(counts.index, counts.values, color=["#4C72B0", "#C44E52"])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
ax.set_title("Class Distribution (Label, after dedup)")
ax.set_ylabel("Row count")
save_figure(fig, "class_distribution.png")
plt.show()

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(pct.index, pct.values, color=["#4C72B0", "#C44E52"])
for i, v in enumerate(pct.values):
    ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
ax.set_ylabel("Percentage of rows")
ax.set_title("Class Percentage (Label, after dedup)")
save_figure(fig, "class_percentage.png")
plt.show()

# %% [markdown]
# ### Observation
# Anomaly traffic dominates the cleaned dataset (~92%), confirming the need
# for macro-averaged and per-class metrics (not raw accuracy) throughout the
# rest of this project.

# %% [markdown]
# ## 3. Informative Numeric Features for Visualization
#
# Rather than plotting all ~70 numeric columns, a small, informative subset is
# selected by absolute correlation with the encoded target — kept small and
# readable per the project's efficiency guidance.

# %%
numeric_cols = [c for c in candidate_predictors if pd.api.types.is_numeric_dtype(df[c])]
corr_with_target_full = df[numeric_cols].corrwith(df["Label_encoded"]).abs().sort_values(ascending=False)
informative_features = corr_with_target_full.head(8).index.tolist()
print("Top correlated features with Label:")
print(corr_with_target_full.head(15))
print("\nSelected for visualization:", informative_features)

# %% [markdown]
# ## 4. Feature Distributions (Sampled)
#
# A reproducible 50,000-row sample keeps histogram rendering fast without
# distorting the underlying distribution shape.

# %%
sample = df.sample(n=min(50000, len(df)), random_state=RANDOM_STATE)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), informative_features):
    ax.hist(sample[col].dropna(), bins=40, color="#4C72B0")
    ax.set_title(col, fontsize=9)
plt.suptitle("Distributions of Top Correlated Features (50,000-row sample)")
plt.tight_layout()
save_figure(fig, "selected_feature_distributions.png")
plt.show()

# %% [markdown]
# ## 5. Normal vs. Anomaly Comparison
#
# For the same informative features, side-by-side distributions by class
# reveal which features visibly separate Normal from Anomaly traffic.

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), informative_features):
    for label, color in [("Normal", "#4C72B0"), ("Anomaly", "#C44E52")]:
        vals = sample.loc[sample["Label"] == label, col].dropna()
        ax.hist(vals, bins=30, alpha=0.5, label=label, color=color, density=True)
    ax.set_title(col, fontsize=9)
axes.ravel()[0].legend(fontsize=8)
plt.suptitle("Normal vs. Anomaly — Top Correlated Features (density, sampled)")
plt.tight_layout()
save_figure(fig, "normal_vs_anomaly_distributions.png")
plt.show()

# %% [markdown]
# ## 6. Boxplots (Selected Features)

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes.ravel(), informative_features[:4]):
    sample.boxplot(column=col, by="Label", ax=ax)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("")
plt.suptitle("Selected Feature Boxplots by Label (sampled)")
plt.tight_layout()
save_figure(fig, "selected_boxplots.png")
plt.show()

# %% [markdown]
# ### Observation
# Several features show visibly different spread/medians between Normal and
# Anomaly traffic (and correspondingly wide outlier ranges) — this is exactly
# the kind of signal the ANOVA test and tree-based importance scores in
# notebook 04 are expected to pick up on quantitatively.

# %% [markdown]
# ## 7. Correlation Analysis and Heatmap

# %%
corr_cols = informative_features + ["Label_encoded"]
corr_matrix = df[corr_cols].corr()
print(corr_matrix.round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=90, fontsize=8)
ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols, fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("Correlation Heatmap — Top Correlated Features + Label")
plt.tight_layout()
save_figure(fig, "correlation_heatmap.png")
plt.show()

# %% [markdown]
# ## 8. Target-Feature Relationships

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(sample[informative_features[0]], sample[informative_features[1]],
                 c=sample["Label_encoded"], cmap="coolwarm", s=4, alpha=0.4)
axes[0].set_xlabel(informative_features[0]); axes[0].set_ylabel(informative_features[1])
axes[0].set_title("Top-2 features, coloured by Label (sampled)")

axes[1].bar(corr_with_target_full.head(10).index, corr_with_target_full.head(10).values, color="#55A868")
axes[1].set_xticklabels(corr_with_target_full.head(10).index, rotation=90, fontsize=8)
axes[1].set_title("|Correlation| with Label (top 10)")
plt.tight_layout()
save_figure(fig, "target_feature_relationships.png")
plt.show()

# %% [markdown]
# ## Notebook Summary

# %%
from IPython.display import display, Markdown

summary_md = f"""
- Built a lightweight, local clean copy ({df.shape[0]:,} rows) for exploration only.
- Confirmed the target imbalance: {pct.round(2).to_dict()}.
- Selected {len(informative_features)} informative features by |correlation| with
  the target: {informative_features}.
- Produced and saved 7 report-quality figures to `figures/`: class distribution,
  class percentage, selected feature distributions, Normal-vs-Anomaly comparison,
  boxplots, correlation heatmap, and target-feature relationships.
- No model, imputer, or scaler was fit here — this notebook is purely descriptive.

**Next notebook (`03_Preprocessing.ipynb`)** builds the single authoritative,
leakage-safe preprocessing pipeline used by every model notebook after it.
"""
display(Markdown(summary_md))
