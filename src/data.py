from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def load_all():
    tr = pd.read_csv(DATA / "train_test.csv")
    va = pd.read_csv(DATA / "validation.csv")
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")
    tmpl = pd.read_csv(DATA / "validation_predictions_template.csv")
    return tr, va, dec, tmpl
