import argparse
import os
import sys
import subprocess
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

try:
    from scipy.io import loadmat
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---- config ----
TRAINING_STATIONS = [
    'Annapolis', 'Atlantic_City', 'Charleston', 'Washington', 'Wilmington',
    'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook']
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']
HIST_DAYS = 7
FUTURE_DAYS = 14


def _matlab2datetime(matlab_datenum):
    return (datetime.fromordinal(int(matlab_datenum))
            + timedelta(days=float(matlab_datenum) % 1)
            - timedelta(days=366))


def _build_from_mat(input_dir, work_dir):
    mat_path = input_dir / "NEUSTG_19502020_12stations.mat"
    if not mat_path.exists():
        raise FileNotFoundError("Expected MAT file not found and CSVs missing.")
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is not available to read MAT files.")

    d = loadmat(mat_path)
    lat = d['lattg'].flatten()
    lon = d['lontg'].flatten()
    sea_level = d['sltg']  # (T, S)
    station_names = [s[0] for s in d['sname'].flatten()]
    time = d['t'].flatten()
    time_dt = np.array([_matlab2datetime(t) for t in time], dtype='datetime64[ns]')

    T, S = sea_level.shape
    df_hourly = pd.DataFrame({
        "time": np.tile(pd.to_datetime(time_dt), S),
        "station_name": np.repeat(station_names, T),
        "latitude": np.repeat(lat, T),
        "longitude": np.repeat(lon, T),
        "sea_level": sea_level.reshape(-1, order="F")})

    train = df_hourly[df_hourly["station_name"].isin(TRAINING_STATIONS)].copy()
    test = df_hourly[df_hourly["station_name"].isin(TESTING_STATIONS)].copy()

    # ---- daily aggregation for test index ----
    daily = test.copy()
    daily["date"] = pd.to_datetime(daily["time"]).dt.floor("D")
    daily = (daily.groupby(["station_name", "date"])
             .agg(sea_level=("sea_level", "mean"),
                  sea_level_max=("sea_level", "max"))
             .reset_index())

    rows = []
    for stn, g in daily.groupby("station_name"):
        g = g.sort_values("date").reset_index(drop=True)
        for anchor in g["date"].unique():
            hist_start = pd.to_datetime(anchor) - pd.Timedelta(days=HIST_DAYS)
            hist_end = pd.to_datetime(anchor) - pd.Timedelta(days=1)
            future_end = pd.to_datetime(anchor) + pd.Timedelta(days=FUTURE_DAYS - 1)
            if (g["date"].min() <= hist_start) and (g["date"].max() >= future_end):
                rows.append({
                    "station_name": stn,
                    "hist_start": hist_start.date().isoformat(),
                    "hist_end": hist_end.date().isoformat(),
                    "future_start": pd.to_datetime(anchor).date().isoformat(),
                    "future_end": future_end.date().isoformat()})
    test_index = pd.DataFrame(rows).reset_index().rename(columns={"index": "id"})

    work_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(work_dir / "train_hourly.csv", index=False)
    test.to_csv(work_dir / "test_hourly.csv", index=False)
    test_index.to_csv(work_dir / "test_index.csv", index=False)
    return work_dir / "train_hourly.csv", work_dir / "test_hourly.csv", work_dir / "test_index.csv"


def _run_ingestion(input_dir, output_dir, submission_dir, work_dir=Path("/tmp/ingestion_work")):
    output_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    train_csv, test_csv, index_csv = _build_from_mat(input_dir, work_dir)

    # ---- run participant model ----
    model_py = submission_dir / "model.py"
    if not model_py.exists():
        raise FileNotFoundError("Participant submission must contain model.py at submission root.")

    pred_out = work_dir / "predictions.csv"
    cmd = [sys.executable, "model.py",
           "--train_hourly", str(train_csv),
           "--test_hourly", str(test_csv),
           "--test_index", str(index_csv),
           "--predictions_out", str(pred_out)]
    res = subprocess.run(cmd, cwd=submission_dir,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

    if not pred_out.exists():
        raise FileNotFoundError("Participant did not create predictions.csv")
    preds = pd.read_csv(pred_out)
    if "id" not in preds.columns or (("y_prob" not in preds.columns) and ("label" not in preds.columns)):
        raise ValueError("predictions.csv must have columns: id,y_prob (or id,label).")

    shutil.copy(pred_out, output_dir / "predictions.csv")
    print(f"Predictions copied to {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--submission_dir", required=True)
    ap.add_argument("--work_dir", default="/tmp/ingestion_work")
    args = ap.parse_args()

    _run_ingestion(
        Path(args.input_dir), Path(args.output_dir),
        Path(args.submission_dir), Path(args.work_dir))
