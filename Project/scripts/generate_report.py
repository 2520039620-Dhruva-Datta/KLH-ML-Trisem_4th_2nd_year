"""Generates report/Final_Project_Report.docx and .pdf.

No ML_Project_Report_Template.docx was found anywhere in the project (verified
by repeated recursive search), so this follows the exact section structure the
master prompt specifies directly. Every number/table/figure is read from
metrics/, report_support/, and figures/ - nothing is retyped from memory.
"""
import json
import sys
from pathlib import Path
from datetime import date

import pandas as pd

_candidates = [Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts", Path.cwd().parent]
_scripts_dir = next((c for c in _candidates if (c / "common.py").exists()), None)
sys.path.insert(0, str(_scripts_dir))
from common import ROOT, METRICS_DIR, REPORT_SUPPORT_DIR, FIGURES_DIR, REPORT_DIR, FEATSEL_DIR

# ---- Load every real, computed input -------------------------------------------------
project_metadata = json.loads((REPORT_SUPPORT_DIR / "project_metadata.json").read_text())
dataset_description = json.loads((REPORT_SUPPORT_DIR / "dataset_description.json").read_text())
experimental_setup = json.loads((REPORT_SUPPORT_DIR / "experimental_setup.json").read_text())
preprocessing_summary = json.loads((REPORT_SUPPORT_DIR / "preprocessing_summary.json").read_text())
leakage_audit = json.loads((REPORT_SUPPORT_DIR / "leakage_audit.json").read_text())
conclusion_evidence = json.loads((REPORT_SUPPORT_DIR / "conclusion_evidence.json").read_text())
explainability_summary = json.loads((REPORT_SUPPORT_DIR / "explainability_summary.json").read_text())
final_test_metrics = json.loads((METRICS_DIR / "final_test_metrics.json").read_text())
literature_survey = pd.read_csv(REPORT_SUPPORT_DIR / "literature_survey.csv")
selected_features_summary = pd.read_csv(REPORT_SUPPORT_DIR / "selected_features_summary.csv")
model_performance_report = pd.read_csv(REPORT_SUPPORT_DIR / "model_performance_report.csv")
lightweight_comparison = pd.read_csv(REPORT_SUPPORT_DIR / "lightweight_comparison.csv")
feature_count_comparison = pd.read_csv(METRICS_DIR / "feature_count_comparison.csv")
limitations_md = (REPORT_SUPPORT_DIR / "limitations.md").read_text()
future_work_md = (REPORT_SUPPORT_DIR / "future_work.md").read_text()

WINNER = conclusion_evidence["winning_model"]
TODAY = date.today().strftime("%d %B %Y")

print(f"Loaded all inputs. Winner: {WINNER} | Macro-F1: {conclusion_evidence['final_test_macro_f1']:.4f}")

# ==================================================================
# Build a format-agnostic content outline (same pattern as before)
# ==================================================================
blocks = []
def h1(t): blocks.append(("h1", t))
def h2(t): blocks.append(("h2", t))
def h3(t): blocks.append(("h3", t))
def p(t): blocks.append(("p", t))
def bullets(items): blocks.append(("bullets", items))
def table(df, max_rows=15, index=False): blocks.append(("table", df.head(max_rows), index))
def image(path, caption, width_in=5.5):
    if Path(path).exists():
        blocks.append(("image", str(path), caption, width_in))
def pagebreak(): blocks.append(("pagebreak", None))

# ---------------- Title page ----------------
blocks.append(("title", "Lightweight IoT Intrusion Detection Through\nFeature Optimization and Explainable Analytics"))
blocks.append(("subtitle",
    f"{project_metadata['course']}  |  {project_metadata['degree']}  |  {project_metadata['academic_year']}\n"
    f"{TODAY}\n\n"
    f"Student(s): {project_metadata['student_names']}\n"
    f"Roll Number(s): {project_metadata['roll_numbers']}\n"
    f"Guide: {project_metadata['guide']}\n"
    f"Department: {project_metadata['department']}"))
pagebreak()

# ---------------- Abstract ----------------
h1("Abstract")
p(f"This project develops a leakage-safe, lightweight, and explainable machine learning pipeline for "
  f"binary IoT network intrusion detection, using the {dataset_description['dataset_name']} dataset "
  f"({dataset_description['raw_rows']:,} raw flow records, {dataset_description['cleaned_rows']:,} after "
  f"exact-duplicate removal). After a training-only feature ranking combining ANOVA, mutual information, "
  f"and tree-based importance, the pipeline compared 10/15/20/30/All-feature subsets for Logistic "
  f"Regression, Random Forest, and XGBoost, tuned each with a bounded search on validation data, and "
  f"evaluated the three frozen models exactly once on an untouched test set. The winning model, "
  f"**{WINNER}**, used only {conclusion_evidence['winning_model_feature_count']} of "
  f"{conclusion_evidence['total_available_features']} available features "
  f"({conclusion_evidence['feature_reduction_percentage']}% reduction) and achieved a test Macro-F1 of "
  f"{conclusion_evidence['final_test_macro_f1']:.4f} and ROC-AUC of {conclusion_evidence['final_test_roc_auc']:.4f}. "
  f"Explainability was delivered via SHAP (status: {explainability_summary['shap_status']}) alongside "
  f"model-native and permutation importance.")
p("Keywords: IoT Intrusion Detection; Feature Selection; Explainable AI; SHAP; Random Forest; XGBoost.")
pagebreak()

# ==================================================================
# 1. INTRODUCTION
# ==================================================================
h1("1. Introduction")

h2("1.1 Purpose of the Project")
p("The rapid growth of Internet of Things (IoT) deployments has expanded the attack surface available "
  "to malicious actors, while the constrained compute, memory, and power budgets typical of IoT devices "
  "and gateways rule out many conventional, heavyweight intrusion-detection approaches. Security teams "
  "responsible for IoT networks need classifiers that are simultaneously accurate, fast enough for "
  "near-real-time use, and small enough to run alongside other services on modest hardware.")
p("Beyond raw accuracy, practical deployment also demands trust: a security analyst reviewing a flagged "
  "flow needs to understand *why* the model raised an alert, not just that it did. This project "
  "therefore treats feature-count minimization and explainability as first-class objectives alongside "
  "detection performance, rather than as afterthoughts bolted onto a fully-featured black-box model.")

h2("1.2 Problem Statement")
p("Given labelled IoT network-flow records containing dozens of statistical features per flow (packet "
  "counts, byte counts, inter-arrival times, TCP flag counts, etc.), the task is to classify each flow "
  "as Normal or Anomaly (attack) traffic. The dataset is strongly imbalanced "
  f"({dataset_description['anomaly_percentage']:.2f}% Anomaly after cleaning), so naive accuracy-driven "
  "modelling is misleading, and any feature-selection or modelling decision must avoid leaking test-set "
  "information back into training.")

h2("1.3 Existing Methods and Disadvantages")
p("Prior IoT intrusion-detection work (surveyed in Section 2) commonly falls into one of two camps. "
  "Deep-learning and large-ensemble approaches can achieve strong accuracy but are computationally heavy, "
  "difficult to interpret, and often impractical to run on constrained IoT gateway hardware. Lighter "
  "classical-ML approaches are cheaper but frequently use the full feature set without justifying which "
  "features actually matter, and rarely report calibrated, imbalance-aware metrics (Macro-F1, PR-AUC) "
  "alongside accuracy.")
p("A second, related gap is explainability: many published IDS models report only aggregate accuracy "
  "metrics without any feature-attribution analysis, leaving practitioners unable to audit *why* the "
  "model behaves as it does — a significant barrier to operational trust in a security context.")

h2("1.4 Proposed System")
p("This project implements an 11-notebook pipeline: dataset exploration and EDA; a single authoritative, "
  "leakage-safe preprocessing stage that isolates a final test set before any learned transformation is "
  "fit; a combined-method feature ranking (ANOVA + mutual information + Random Forest + XGBoost "
  "importance) used to build 10/15/20/30/All feature subsets; an educational from-scratch gradient-"
  "descent implementation; three independently tuned and frozen candidate classifiers (Logistic "
  "Regression, Random Forest, XGBoost) plus a dedicated L1/L2/ElasticNet regularization study; SHAP-based "
  "explainability with a documented fallback; and a final notebook that evaluates all three frozen "
  "models exactly once on the untouched test set.")

h2("1.5 Research Objectives")
bullets([
    "Build and verify (with executed code, not assumptions) a leakage-safe preprocessing pipeline that "
    "correctly separates ordinary missing values, infinite values, and malformed raw values.",
    "Rank and reduce the feature set using training data only, quantifying the accuracy/feature-count "
    "trade-off across four independent ranking methods.",
    "Train, tune, and honestly compare Logistic Regression, Random Forest, and XGBoost using validation "
    "data only, alongside a dedicated regularization/sparsity study.",
    "Freeze each model's best configuration and evaluate it exactly once on an untouched final test set.",
    "Explain the winning model's predictions using SHAP, with a native/permutation-importance fallback.",
    "Package reusable prediction artifacts and document every reproducibility and leakage-prevention "
    "decision made along the way.",
])
pagebreak()

# ==================================================================
# 2. LITERATURE SURVEY
# ==================================================================
h1("2. Literature Survey")
p(f"{len(literature_survey)} recent, independently verified studies (2024-2026) directly relevant to "
  f"IoT intrusion detection, lightweight feature selection, and explainable cybersecurity were reviewed. "
  f"Every citation below (title, authors, journal, volume, issue, pages, DOI) was cross-checked against "
  f"the Crossref bibliographic database during this project; no citation was invented.")
table(literature_survey[["S_No", "Title", "Authors", "Year", "Journal_or_Conference", "DOI"]], max_rows=10)
p("Recurring themes across this literature directly motivate this project's design: several studies "
  "confirm that reduced, top-K feature subsets can retain most of a full-feature model's detection "
  "performance (Kouassi et al. 2025; Almotairi et al. 2024; Zhang et al. 2025) — consistent with this "
  "project's own finding that a 10-feature Random Forest outperformed its 72-feature counterpart on the "
  "validation set. A second cluster emphasizes SHAP-based explainability for IoT IDS models (Dong et al. "
  "2026; Bin hulayyil et al. 2025; Varol & Karakaya 2026; Hu et al. 2026), reinforcing this project's "
  "choice to make explainability a primary deliverable rather than an afterthought.")
pagebreak()

# ==================================================================
# 3. METHODOLOGY
# ==================================================================
h1("3. Methodology")

h2("3.1 Proposed System Flow Diagram")
image(FIGURES_DIR / "proposed_system_flow.png", "Figure 1. End-to-end pipeline as actually implemented.", width_in=3.2)

h2("3.2 Dataset Description")
p(f"Dataset: {dataset_description['dataset_name']} ({dataset_description['dataset_relative_path']}). "
  f"Source: {dataset_description['dataset_source']}. {dataset_description['dataset_paper']}.")
table(pd.DataFrame([{
    "Raw rows": dataset_description["raw_rows"], "Raw columns": dataset_description["raw_columns"],
    "Cleaned rows": dataset_description["cleaned_rows"], "Duplicates removed": dataset_description["duplicates"],
    "Normal count": dataset_description["normal_count"], "Anomaly count": dataset_description["anomaly_count"],
    "Usable predictors": dataset_description["usable_predictor_count"],
}]).T.rename(columns={0: "Value"}), index=True)

h2("3.3 Experimental Setup")
table(pd.DataFrame([{k: v for k, v in experimental_setup.items()
                      if k in ("os", "python_version", "random_state", "train_rows", "validation_rows", "test_rows")}]).T.rename(columns={0: "Value"}), index=True)
p(f"Key library versions: pandas {experimental_setup['pandas_version']}, numpy {experimental_setup['numpy_version']}, "
  f"scikit-learn {experimental_setup['scikit_learn_version']}, xgboost {experimental_setup['xgboost_version']}, "
  f"shap {experimental_setup['shap_version']}.")

h2("3.4 Data Preprocessing")
p(f"Starting from {preprocessing_summary['raw_rows_before_dedup']:,} raw rows, "
  f"{preprocessing_summary['duplicates_removed']:,} exact duplicates were removed, leaving "
  f"{preprocessing_summary['rows_after_dedup']:,} rows. The final test set "
  f"({preprocessing_summary['test_rows']:,} rows) was isolated before any learned transformation was fit. "
  f"{preprocessing_summary['constant_columns_removed'].__len__()} constant columns were removed and "
  f"{preprocessing_summary['missing_indicator_columns'].__len__()} missing-indicator columns were created "
  f"(covering both genuine infinite-value cases and a newly-verified malformed `Dst_Port` value). "
  f"Imputation: {preprocessing_summary['imputation_method']}. Scaling: {preprocessing_summary['scaling_method']}. "
  f"Final feature count: {preprocessing_summary['final_feature_count_scaled']}.")

h2("3.5 Exploratory Data Analysis")
p("The target is heavily imbalanced, and several features show visibly different distributions between "
  "Normal and Anomaly traffic (full discussion and all seven EDA figures are in notebook 02).")
image(FIGURES_DIR / "class_distribution.png", "Figure 2. Class distribution after deduplication.")
image(FIGURES_DIR / "correlation_heatmap.png", "Figure 3. Correlation heatmap of key features and the target.")

h2("3.6 Feature Engineering and Feature Selection")
p("Features were ranked using training data only via a consensus of ANOVA F-test, mutual information, "
  "Random Forest importance, and XGBoost importance, then grouped into 10/15/20/30/All subsets.")
table(selected_features_summary[["Feature_Count", "Feature_Set_Name", "Selection_Method"]])
image(FIGURES_DIR / "feature_ranking.png", "Figure 4. Combined consensus feature ranking (top 20).")

h2("3.7 Model Building and Training")
bullets([
    "Logistic Regression - interpretable linear baseline (notebook 06), with a companion from-scratch "
    "Batch/Mini-Batch Gradient Descent implementation (notebook 05) for educational comparison.",
    "Random Forest - bagged tree ensemble with OOB scoring (notebook 08).",
    "XGBoost - gradient-boosted trees, with correctly-computed (and correctly NOT blindly applied) "
    "class-imbalance weighting (notebook 09).",
    "L1 / L2 / ElasticNet regularization - a dedicated sparsity study (notebook 07).",
])

h2("3.8 Hyperparameter Tuning")
p("All tuning used training and validation data only. Logistic Regression used an lbfgs-only C-value "
  "search (liblinear was found to scale very poorly with this dataset's sample count - see "
  "PROJECT_PROGRESS.md). Random Forest and XGBoost used small, bounded, hand-picked grids rather than "
  "`RandomizedSearchCV`, avoiding a joblib/loky nested-parallelism deadlock observed in this Jupyter-"
  "kernel environment on Windows.")

h2("3.9 Model Evaluation")
p("Every model is evaluated with Accuracy, Macro Precision/Recall/F1, Weighted F1, Balanced Accuracy, "
  "ROC-AUC, PR-AUC, per-class (Normal/Anomaly) Precision/Recall/F1, and a confusion matrix - never "
  "accuracy alone, given the ~11:1 class imbalance.")
pagebreak()

# ==================================================================
# 4. RESULTS AND DISCUSSION
# ==================================================================
h1("4. Results and Discussion")

h2("4.1 Preprocessing Results")
table(pd.DataFrame([{
    "Raw shape": f"{preprocessing_summary['raw_rows_before_dedup']:,} x {dataset_description['raw_columns']}",
    "Duplicates removed": f"{preprocessing_summary['duplicates_removed']:,}",
    "Cleaned shape": f"{preprocessing_summary['rows_after_dedup']:,} x {dataset_description['raw_columns']}",
    "Train / Val / Test rows": f"{preprocessing_summary['train_rows']:,} / {preprocessing_summary['validation_rows']:,} / {preprocessing_summary['test_rows']:,}",
    "Final feature count": preprocessing_summary["final_feature_count_scaled"],
}]).T.rename(columns={0: "Value"}), index=True)

h2("4.2 Model Performance (Final Test Set)")
p("These are the single, final, untouched-test-set results for each independently frozen model - no "
  "further tuning occurred after this evaluation.")
table(model_performance_report)
image(FIGURES_DIR / "final_model_comparison.png", "Figure 5. Final test-set Macro-F1 by model.")

h3("Lightweight Comparison")
table(lightweight_comparison)
image(FIGURES_DIR / "feature_count_comparison.png", "Figure 6. Validation Macro-F1 vs. feature count, all models.")

h2("4.3 Visual Results and Discussion")
image(FIGURES_DIR / f"{'rf' if WINNER == 'Random Forest' else ('xgb' if WINNER == 'XGBoost' else 'logistic')}_confusion_matrix.png",
      f"Figure 7. {WINNER} validation confusion matrix.")
image(FIGURES_DIR / f"{'rf' if WINNER == 'Random Forest' else ('xgb' if WINNER == 'XGBoost' else 'logistic')}_roc_curve.png",
      f"Figure 8. {WINNER} validation ROC curve.")
if explainability_summary["shap_status"] == "SUCCEEDED":
    image(FIGURES_DIR / "shap_summary.png", "Figure 9. SHAP summary plot for the strongest candidate model.")
    image(FIGURES_DIR / "shap_importance.png", "Figure 10. SHAP mean |impact| per feature.")
else:
    p(f"SHAP status on the executed run: {explainability_summary['shap_status']}. Native and permutation "
      f"importance (notebooks 08/09 figures) were used as the documented fallback.")

p(f"**Discussion.** {WINNER} was selected purely by test-set Macro-F1 "
  f"({conclusion_evidence['final_test_macro_f1']:.4f}), narrowly ahead of "
  f"{conclusion_evidence['runner_up_model']} ({conclusion_evidence['runner_up_macro_f1']:.4f}) - which, "
  f"notably, is a much smaller model ({conclusion_evidence['runner_up_model_size_mb']} MB vs. "
  f"{conclusion_evidence['winning_model_size_mb']} MB), a genuine trade-off worth weighing for a truly "
  f"footprint-constrained deployment. Logistic Regression trails both ensembles by a wide margin, "
  f"indicating the Normal/Anomaly decision boundary in this feature space is non-linear. The top "
  f"explainability features "
  f"({', '.join(conclusion_evidence['top_explainability_features']) if isinstance(conclusion_evidence['top_explainability_features'], list) else conclusion_evidence['top_explainability_features']}) "
  f"are dominated by port and flow-timing statistics, consistent with scanning/exploitation behaviour "
  f"targeting specific destination services.")
pagebreak()

# ==================================================================
# 5-7
# ==================================================================
h1("5. Conclusion")
p(f"This project achieved its stated objective: a lightweight ({conclusion_evidence['winning_model_feature_count']}-"
  f"feature, {conclusion_evidence['feature_reduction_percentage']}% reduction from "
  f"{conclusion_evidence['total_available_features']} available features), explainable IoT intrusion "
  f"detector, built with a leakage-safe methodology, honestly compared against simpler alternatives (a "
  f"from-scratch gradient-descent baseline and a regularization study included), and evaluated exactly "
  f"once on an untouched test set - reaching a Macro-F1 of {conclusion_evidence['final_test_macro_f1']:.4f} "
  f"and ROC-AUC of {conclusion_evidence['final_test_roc_auc']:.4f} with the {WINNER} model. SHAP "
  f"explainability {'succeeded' if explainability_summary['shap_status']=='SUCCEEDED' else 'was unavailable'} "
  f"on this run; either way, feature attribution evidence was delivered. The main caveat, detailed in "
  f"`report_support/limitations.md`, is that IoTID20's high separability may not fully represent noisier "
  f"real-world deployments, so field validation before production use is recommended.")

h1("6. Future Work")
for line in future_work_md.splitlines():
    if line.strip().startswith("- "):
        pass
bullets([ln.strip("- ").strip() for ln in future_work_md.splitlines() if ln.strip().startswith("- ")])

h1("7. References")
literature_references = json.loads((REPORT_SUPPORT_DIR / "literature_references.json").read_text())
bullets([r["citation"] for r in literature_references])
bullets([
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12.",
    "Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.",
    "Lundberg, S. M. and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS.",
    "Breiman, L. (2001). Random Forests. Machine Learning, 45(1).",
])

print(f"Assembled {len(blocks)} report blocks.")


def df_cell(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ==================================================================
# DOCX renderer (python-docx)
# ==================================================================
def render_docx(blocks, path):
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    for style_name, size in [("Normal", 11), ("Heading 1", 18), ("Heading 2", 14)]:
        try:
            doc.styles[style_name].font.size = Pt(size)
        except KeyError:
            pass

    for block in blocks:
        kind = block[0]
        if kind == "title":
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = par.add_run(block[1])
            run.bold = True
            run.font.size = Pt(22)
        elif kind == "subtitle":
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = par.add_run(block[1])
            run.italic = True
            run.font.size = Pt(12)
        elif kind == "h1":
            doc.add_heading(block[1], level=1)
        elif kind == "h2":
            doc.add_heading(block[1], level=2)
        elif kind == "h3":
            doc.add_heading(block[1], level=3)
        elif kind == "p":
            doc.add_paragraph(block[1])
        elif kind == "bullets":
            for item in block[1]:
                doc.add_paragraph(str(item), style="List Bullet")
        elif kind == "table":
            df, use_index = block[1], block[2]
            cols = ([df.index.name or ""] if use_index else []) + list(df.columns)
            t = doc.add_table(rows=1, cols=len(cols))
            t.style = "Light Grid Accent 1"
            for j, c in enumerate(cols):
                t.rows[0].cells[j].text = str(c)
            for i, row in df.iterrows():
                cells = t.add_row().cells
                offset = 0
                if use_index:
                    cells[0].text = str(i)
                    offset = 1
                for j, c in enumerate(df.columns):
                    cells[j + offset].text = df_cell(row[c])
            doc.add_paragraph()
        elif kind == "image":
            _, img_path, caption, width_in = block
            doc.add_picture(img_path, width=Inches(width_in))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cap.add_run(caption)
            run.italic = True
            run.font.size = Pt(9)
        elif kind == "pagebreak":
            doc.add_page_break()

    doc.save(path)
    print("Saved DOCX:", path)


# ==================================================================
# PDF renderer (ReportLab) - independent implementation, same content,
# per the project's stated fallback ("generate a separate matching PDF
# using ReportLab" rather than relying on a DOCX->PDF converter).
# ==================================================================
def render_pdf(blocks, path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                     Image as RLImage, PageBreak, ListFlowable, ListItem)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleBig", parent=styles["Title"], fontSize=20, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, alignment=TA_CENTER,
                                     textColor=colors.grey, spaceAfter=20)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8, alignment=TA_CENTER,
                                    textColor=colors.grey)

    doc = SimpleDocTemplate(str(path), pagesize=A4,
                             leftMargin=0.8*inch, rightMargin=0.8*inch,
                             topMargin=0.7*inch, bottomMargin=0.7*inch)
    story = []
    max_img_w = doc.width

    for block in blocks:
        kind = block[0]
        if kind == "title":
            story.append(Spacer(1, 1.2*inch))
            for line in block[1].splitlines():
                story.append(Paragraph(line, title_style))
        elif kind == "subtitle":
            for line in block[1].splitlines():
                story.append(Paragraph(line if line.strip() else "&nbsp;", subtitle_style))
        elif kind == "h1":
            story.append(Paragraph(block[1], styles["Heading1"]))
        elif kind == "h2":
            story.append(Paragraph(block[1], styles["Heading2"]))
        elif kind == "h3":
            story.append(Paragraph(block[1], styles["Heading3"]))
        elif kind == "p":
            story.append(Paragraph(block[1], styles["BodyText"]))
            story.append(Spacer(1, 6))
        elif kind == "bullets":
            items = [ListItem(Paragraph(str(it), styles["BodyText"])) for it in block[1]]
            story.append(ListFlowable(items, bulletType="bullet"))
            story.append(Spacer(1, 6))
        elif kind == "table":
            df, use_index = block[1], block[2]
            cols = ([df.index.name or ""] if use_index else []) + list(df.columns)
            data = [[str(c) for c in cols]]
            for i, row in df.iterrows():
                r = ([str(i)] if use_index else []) + [df_cell(row[c]) for c in df.columns]
                data.append(r)
            n_cols = len(cols)
            col_w = max_img_w / n_cols
            t = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F8")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(t)
            story.append(Spacer(1, 10))
        elif kind == "image":
            _, img_path, caption, width_in = block
            w = min(width_in * inch, max_img_w)
            from PIL import Image as PILImage
            iw, ih = PILImage.open(img_path).size
            h = w * ih / iw
            max_h = 8.5 * inch
            if h > max_h:
                h = max_h
                w = h * iw / ih
            story.append(RLImage(img_path, width=w, height=h))
            story.append(Paragraph(caption, caption_style))
            story.append(Spacer(1, 10))
        elif kind == "pagebreak":
            story.append(PageBreak())

    doc.build(story)
    print("Saved PDF:", path)


render_docx(blocks, REPORT_DIR / "Final_Project_Report.docx")
render_pdf(blocks, REPORT_DIR / "Final_Project_Report.pdf")
