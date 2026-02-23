import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef


def _read_gt(solution_dir):
    gt = pd.read_csv(str(solution_dir) + "/y_test.csv")
    if "id" not in gt.columns:
        gt.insert(0, "id", range(len(gt)))
    if "y_true" not in gt.columns:
        last = [c for c in gt.columns if c != "id"][-1]
        gt = gt.rename(columns={last: "y_true"})
    return gt[["id", "y_true"]]


def _read_preds(pred_dir):
    p = str(pred_dir) + "/predictions.csv"
    if not Path(p).exists():
        raise FileNotFoundError("predictions.csv not found")
    pred = pd.read_csv(p)
    if "id" not in pred.columns:
        pred.insert(0, "id", range(len(pred)))
    return pred


def _score(solution_dir, prediction_dir, score_dir):
    score_dir.mkdir(parents=True, exist_ok=True)
    gt = _read_gt(solution_dir)
    pr = _read_preds(prediction_dir)
    df = gt.merge(pr, on="id", how="inner")
    if len(df) == 0:
        raise ValueError("No overlapping ids between y_test and predictions.")

    y_true = df["y_true"].astype(int).to_numpy()
    if "y_prob" in df.columns:
        y_prob = df["y_prob"].astype(float).to_numpy()
    elif "label" in df.columns:
        y_prob = df["label"].astype(int).to_numpy()
    else:
        y_prob = df.iloc[:, -1].astype(float).to_numpy()

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "auc": auc,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n": int(len(y_true))}

    (score_dir / "scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution_dir", required=True)
    ap.add_argument("--prediction_dir", required=True)
    ap.add_argument("--score_dir", required=True)
    args = ap.parse_args()

    _score(Path(args.solution_dir), Path(args.prediction_dir), Path(args.score_dir))
