"""Create the assessment report from the verified model outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report"
TEAL = colors.HexColor("#064A56")
PALE = colors.HexColor("#EEF5F5")
LINE = colors.HexColor("#D5E1E3")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    summary = json.loads((ROOT / "results" / "validation_summary.json").read_text(encoding="utf-8"))
    final = summary["final_evaluation"]
    tuning = pd.read_csv(ROOT / "results" / "tuning_windows.csv")
    december = pd.read_csv(ROOT / "data" / "december_chart_inputs.csv")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=TEAL, spaceAfter=4)
    subtitle = ParagraphStyle("SubX", parent=styles["Normal"], fontSize=9.5, leading=13, textColor=colors.HexColor("#52666C"), spaceAfter=10)
    h = ParagraphStyle("HX", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=TEAL, spaceBefore=10, spaceAfter=5)
    b = ParagraphStyle("BX", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=colors.HexColor("#1D2D32"), spaceAfter=6)
    s = ParagraphStyle("SX", parent=b, fontSize=8, leading=10, textColor=colors.HexColor("#52666C"))
    cell = ParagraphStyle("CX", parent=b, fontSize=8.2, leading=10, spaceAfter=0)

    def p(text, style=b): return Paragraph(text, style)
    def table(rows, widths):
        result = Table(rows, colWidths=widths, repeatRows=1)
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return result
    def heads(labels): return [p(f"<b>{x}</b>", ParagraphStyle("head", parent=cell, textColor=colors.white)) for x in labels]

    mean = tuning.groupby("model")[["all_MAE", "all_RMSE", "all_MAPE", "all_MedAPE"]].mean().reset_index().sort_values("all_MAE")
    model_rows = [heads(["Candidate", "Mean MAE", "Mean RMSE", "Mean MAPE", "Median APE"])]
    pretty = {"two_stage_l1": "Two-stage gradient boosting, L1 loss", "two_stage_squared": "Two-stage gradient boosting, squared loss", "linear_only": "Linear stage only"}
    for _, row in mean.iterrows():
        model_rows.append([p(pretty[row.model], cell), p(f"${row.all_MAE:,.2f}", cell), p(f"${row.all_RMSE:,.2f}", cell), p(f"{row.all_MAPE:.2f}%", cell), p(f"{row.all_MedAPE:.2f}%", cell)])
    final_rows = [heads(["Metric", "Selected model", "Rate-per-mile baseline", "Improvement"])]
    for label, key, fmt in [("MAE", "MAE", "${:,.2f}"), ("RMSE", "RMSE", "${:,.2f}"), ("MAPE", "MAPE", "{:.2f}%"), ("Median APE", "MedAPE", "{:.2f}%"), ("R-squared", "R2", "{:.3f}")]:
        m, base = final[f"all_{key}"], final[f"baseline_all_{key}"]
        improvement = "higher" if key == "R2" else f"{(1-m/base)*100:.1f}% lower"
        final_rows.append([p(label, cell), p(fmt.format(m), cell), p(fmt.format(base), cell), p(improvement, cell)])

    story = [p("Freight rate prediction", title), p("Validation approach and December forecast | Machine Learning Engineer assessment", subtitle)]
    story += [p("Summary", h), p(
        "The model predicts posted freight rates for 12,000 unlabeled November-December loads using 48,000 labeled January-October loads. "
        "The selected approach uses a smooth linear log-rate model for extrapolating distance, time trend, and quarter-end effects, then a gradient-boosted residual model for nonlinear geographic and load interactions."
    )]
    story += [p("Validation protocol", h), p(
        "The prediction task is future-dated, so validation is chronological rather than random. The first four rolling two-month windows (May-Jun, Jun-Jul, Jul-Aug, and Aug-Sep) were used to choose among three pre-specified candidates by mean all-row MAE. "
        "The final Sep-Oct window was held back from that choice and evaluated once. All reported headline metrics include every row in the window."
    )]
    story += [table(model_rows, [2.5*inch, .85*inch, .9*inch, .8*inch, .85*inch]), Spacer(1, 7)]
    story += [p("Final locked Sep-Oct evaluation", h), p(
        f"The selected squared-loss two-stage model trained on {final['train_rows']:,} Jan-Aug rows and predicted {final['test_rows']:,} Sep-Oct rows. "
        "It materially improves typical error and MAE over the training-only rate-per-mile baseline."
    ), table(final_rows, [1.25*inch, 1.35*inch, 1.45*inch, 1.15*inch])]
    story += [p("Data preparation and model", h), p(
        "Negative weights are treated as sign errors and converted to absolute values with a flag. Missing weight uses equipment-level training medians; missing market index uses a same-date feature-only batch average and a missingness flag. "
        "The model does not use city IDs, so unseen cities are handled through coordinates, route geometry, distance, and equipment. Target-based extreme-rate flags are applied only to training labels for robust fitting. Diagnostic clean-label metrics are not used for selection or presented as hidden-test performance."
    )]
    story += [PageBreak(), p("Final prediction files", h), p(
        "The final model is refit on all labeled rows. It writes one positive prediction for every template load ID and fills all 31 December rows. The supplied scorer accepted both output files and created the chart below. "
        "December market-index values use the mean from the unlabeled batch for each date; this is a prediction-time batch-feature assumption."
    )]
    story += [Image(str(ROOT / "scorer_results" / "candidate_december.png"), width=6.9*inch, height=2.25*inch), Spacer(1, 3), p(
        f"The fixed Lexington-to-Fort-Wayne December forecast ranges from ${december.predicted_rate.min():,.2f} to ${december.predicted_rate.max():,.2f}. "
        "The assessment provider calculates the final hidden-label score after submission.", s
    )]
    story += [p("Key limitations", h), p(
        "The final score is unavailable because validation labels were not supplied. The model assumes the trend and quarter-end pattern observed through Q3 continue through Q4. "
        "Same-date market averages are available only when all batch features are present; a real-time single-load deployment would need a different market-index forecast."
    ), p("Reproducibility", h), p(
        "All inputs, code, dependencies, model-selection outputs, predictions, scorer, and the report generator are included. Running the four commands in the README recreates the evaluation, predictions, chart, and report.", s
    )]
    doc = SimpleDocTemplate(str(OUT / "freight_rate_report.pdf"), pagesize=letter, leftMargin=.75*inch, rightMargin=.75*inch, topMargin=.6*inch, bottomMargin=.55*inch, title="Freight Rate Prediction Report")
    doc.build(story)
    print(OUT / "freight_rate_report.pdf")


if __name__ == "__main__":
    main()
