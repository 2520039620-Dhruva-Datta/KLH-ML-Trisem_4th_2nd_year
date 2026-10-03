"""Converts the canonical `# %%` / `# %% [markdown]` cell-marked .py scripts in
scripts/ into real .ipynb notebooks under notebooks/.

jupytext is not installed in this environment; per the project's explicit
fallback rule, this uses nbformat directly instead (does not require jupytext).
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
NOTEBOOKS_DIR = ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# (script filename, notebook filename)
SCRIPT_TO_NOTEBOOK = [
    ("01_dataset_exploration.py", "01_Dataset_Exploration.ipynb"),
    ("02_eda.py", "02_EDA.ipynb"),
    ("03_preprocessing.py", "03_Preprocessing.ipynb"),
    ("04_feature_selection_optimization.py", "04_Feature_Selection_Optimization.ipynb"),
    ("05_gradient_descent_binary_classification.py", "05_Gradient_Descent_Binary_Classification.ipynb"),
    ("06_logistic_regression.py", "06_Logistic_Regression.ipynb"),
    ("07_regularisation_ridge_lasso_elasticnet.py", "07_Regularisation_Ridge_Lasso_ElasticNet.ipynb"),
    ("08_random_forest.py", "08_Random_Forest.ipynb"),
    ("09_xgboost.py", "09_XGBoost.ipynb"),
    ("10_explainability_shap.py", "10_Explainability_SHAP.ipynb"),
    ("11_final_model_comparison_and_predictions.py", "11_Final_Model_Comparison_and_Predictions.ipynb"),
]

CELL_MARK = re.compile(r"^# %%(\s*\[markdown\])?\s*$")


def parse_py_to_cells(source: str):
    lines = source.splitlines()
    cells = []
    current_kind = None
    current_lines = []

    def flush():
        if current_kind is None:
            return
        text = "\n".join(current_lines).strip("\n")
        if current_kind == "markdown":
            # Strip the leading "# " (or bare "#") jupytext-style comment prefix.
            md_lines = []
            for ln in current_lines:
                if ln.startswith("# "):
                    md_lines.append(ln[2:])
                elif ln.strip() == "#":
                    md_lines.append("")
                else:
                    md_lines.append(ln)
            text = "\n".join(md_lines).strip("\n")
        if text.strip():
            cells.append((current_kind, text))

    for line in lines:
        m = CELL_MARK.match(line)
        if m:
            flush()
            current_kind = "markdown" if m.group(1) else "code"
            current_lines = []
        else:
            if current_kind is None:
                continue  # ignore anything before the first cell marker (e.g. module docstring)
            current_lines.append(line)
    flush()
    return cells


def build_notebook(py_path: Path, nb_path: Path):
    source = py_path.read_text(encoding="utf-8")
    cells = parse_py_to_cells(source)
    if not cells:
        raise ValueError(f"No cells parsed from {py_path} - check '# %%' markers.")

    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    for kind, text in cells:
        if kind == "markdown":
            nb["cells"].append(nbf.v4.new_markdown_cell(text))
        else:
            nb["cells"].append(nbf.v4.new_code_cell(text))
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3 (.MLvenv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    nbf.write(nb, nb_path)
    print(f"  {py_path.name:50s} -> {nb_path.name}  ({len(cells)} cells)")


def main(only=None):
    print("Building notebooks from canonical scripts (nbformat, jupytext unavailable)...")
    for script_name, nb_name in SCRIPT_TO_NOTEBOOK:
        if only and script_name not in only:
            continue
        py_path = SCRIPTS_DIR / script_name
        if not py_path.exists():
            print(f"  [SKIP] {script_name} not found yet.")
            continue
        build_notebook(py_path, NOTEBOOKS_DIR / nb_name)
    print("Done.")


if __name__ == "__main__":
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    main(only)
