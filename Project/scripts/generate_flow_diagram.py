"""Generates figures/proposed_system_flow.png, reflecting the ACTUAL implemented
pipeline (not a generic template diagram)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_candidates = [Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts", Path.cwd().parent]
_scripts_dir = next((c for c in _candidates if (c / "common.py").exists()), None)
sys.path.insert(0, str(_scripts_dir))
from common import FIGURES_DIR

STAGES = [
    "Raw IoT Dataset\n(625,783 x 86)",
    "Schema Validation +\nDtype-Safety Coercion",
    "Duplicate Removal\n(164,087 removed, 26.22%)",
    "Target Mapping\nNormal=0, Anomaly=1",
    "Forbidden-Feature Removal\n(Cat, Sub_Cat, Flow_ID, IPs, Timestamp)",
    "Final Test Isolation\n(80/20 stratified split)",
    "Train / Validation Split\n(80/20 of remaining 80%)",
    "Train-Fitted Preprocessing\n(median impute + scale, train-only)",
    "Feature Ranking\n(ANOVA + MI + RF + XGB, train-only)",
    "Top 10 / 15 / 20 / 30 / All\nFeature Subsets",
    "Logistic Regression /\nRandom Forest / XGBoost\n(+ Gradient Descent, Regularisation)",
    "Bounded Hyperparameter\nTuning (validation only)",
    "Feature-Count + Model\nComparison (validation only)",
    "SHAP Explainability\n(+ native/permutation fallback)",
    "Frozen Models\n(fit on train, tuned on validation)",
    "Untouched Final Test Set\n(evaluated exactly once)",
    "Predictions +\nFinal Evaluation",
]

fig, ax = plt.subplots(figsize=(6, 22))
ax.set_xlim(0, 10)
ax.set_ylim(0, len(STAGES) + 1)
ax.axis("off")

box_w, box_h = 8, 0.75
for i, stage in enumerate(STAGES):
    y = len(STAGES) - i
    color = "#4C72B0" if i in (5, 15) else ("#55A868" if i in (13,) else "#EEF2F8")
    text_color = "white" if color != "#EEF2F8" else "black"
    box = FancyBboxPatch((1, y - box_h / 2), box_w, box_h, boxstyle="round,pad=0.05,rounding_size=0.08",
                          facecolor=color, edgecolor="#333333", linewidth=1)
    ax.add_patch(box)
    ax.text(5, y, stage, ha="center", va="center", fontsize=8.5, color=text_color, wrap=True)
    if i < len(STAGES) - 1:
        arrow = FancyArrowPatch((5, y - box_h / 2), (5, y - 1 + box_h / 2),
                                 arrowstyle="-|>", mutation_scale=12, color="#333333")
        ax.add_patch(arrow)

ax.text(0.2, len(STAGES) + 0.6, "Proposed System Flow — Lightweight IoT Intrusion Detection",
        fontsize=11, fontweight="bold")
ax.text(0.2, -0.3, "Blue = split point isolating data; Green = explainability stage", fontsize=7, color="#555555")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "proposed_system_flow.png", bbox_inches="tight", dpi=150)
print(f"Saved: {FIGURES_DIR / 'proposed_system_flow.png'}")
