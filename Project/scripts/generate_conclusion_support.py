"""Generates report_support/lightweight_comparison.csv and conclusion_evidence.json
from the final, saved test-set results (never retyped/remembered values)."""
import json
import sys
from pathlib import Path

import pandas as pd

_candidates = [Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts", Path.cwd().parent]
_scripts_dir = next((c for c in _candidates if (c / "common.py").exists()), None)
sys.path.insert(0, str(_scripts_dir))
from common import METRICS_DIR, REPORT_SUPPORT_DIR, FEATSEL_DIR

final_comparison = pd.read_csv(METRICS_DIR / "final_model_comparison.csv")
final_test_metrics = json.loads((METRICS_DIR / "final_test_metrics.json").read_text())
explainability = json.loads((REPORT_SUPPORT_DIR / "explainability_summary.json").read_text())

all_feature_count = len((FEATSEL_DIR / "all_selected_features.txt").read_text().splitlines())

# ---------------------------------------------------------------- lightweight_comparison.csv
rows = []
for _, r in final_comparison.iterrows():
    rows.append({
        "Model": r["Model"],
        "Selected_Feature_Count": int(r["Feature_Count"]),
        "Total_Available_Features": all_feature_count,
        "Feature_Reduction_Percentage": round(100 * (1 - r["Feature_Count"] / all_feature_count), 1),
        "F1": r["Macro_F1"],
        "ROC_AUC": r["ROC_AUC"],
        "PR_AUC": r["PR_AUC"],
        "Inference_Time": r["Inference_Time"],
        "Model_Size_MB": r["Model_Size_MB"],
    })
lightweight_comparison = pd.DataFrame(rows)
lightweight_comparison.to_csv(REPORT_SUPPORT_DIR / "lightweight_comparison.csv", index=False)
print("Saved: report_support/lightweight_comparison.csv")
print(lightweight_comparison.to_string(index=False))

# ---------------------------------------------------------------- conclusion_evidence.json
winner = final_test_metrics["Model"]
winner_row = final_comparison[final_comparison["Model"] == winner].iloc[0]
runner_up = final_comparison[final_comparison["Model"] != winner].sort_values("Macro_F1", ascending=False).iloc[0]

conclusion_evidence = {
    "winning_model": winner,
    "final_test_macro_f1": final_test_metrics["Macro_F1"],
    "final_test_roc_auc": final_test_metrics["ROC_AUC"],
    "final_test_pr_auc": final_test_metrics["PR_AUC"],
    "winning_model_feature_count": int(final_test_metrics["Feature_Count"]),
    "total_available_features": all_feature_count,
    "feature_reduction_percentage": round(100 * (1 - final_test_metrics["Feature_Count"] / all_feature_count), 1),
    "winning_model_size_mb": final_test_metrics["Model_Size_MB"],
    "runner_up_model": runner_up["Model"],
    "runner_up_macro_f1": runner_up["Macro_F1"],
    "runner_up_model_size_mb": runner_up["Model_Size_MB"],
    "shap_status": explainability["shap_status"],
    "top_explainability_features": [f["feature"] for f in explainability["top_shap_features"][:5]] \
        if explainability["shap_status"] == "SUCCEEDED" else "See native/permutation importance in notebooks 08/09",
    "limitations_file": "report_support/limitations.md",
}
with open(REPORT_SUPPORT_DIR / "conclusion_evidence.json", "w") as f:
    json.dump(conclusion_evidence, f, indent=2, default=str)
print("\nSaved: report_support/conclusion_evidence.json")
print(json.dumps(conclusion_evidence, indent=2, default=str))
