import argparse
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import rankdata
from scipy.special import logit, expit


# ---- config ----
HIST_DAYS = 7
FUTURE_DAYS = 14
TOP_K = 30
CALIB_POWER = 1.10
F_TARGET = 0.85
USE_PREVALENCE = True
THR_K = 1.5


def _station_daily(hourly):
    # hourly sea-level -> per-station daily aggregates + percentiles
    h = hourly.copy()
    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.floor("D")
    sdata = {}
    for stn, grp in h.groupby("station_name"):
        v = grp["sea_level"].values
        ok = ~np.isnan(v)
        pct = np.full(len(v), np.nan)
        pct[ok] = rankdata(v[ok]) / ok.sum()
        grp = grp.copy()
        grp["pct"] = pct
        d = grp.groupby("date").agg(
            sl_mean=("sea_level", "mean"), sl_max=("sea_level", "max"),
            sl_min=("sea_level", "min"), sl_std=("sea_level", "std"),
            pct_mean=("pct", "mean"), pct_max=("pct", "max"),
            pct_min=("pct", "min"), pct_std=("pct", "std"),
            count=("sea_level", "count")).reset_index()
        d["sl_std"] = d["sl_std"].fillna(0)
        d["pct_std"] = d["pct_std"].fillna(0)
        sv = v[ok]
        mu = float(sv.mean()) if len(sv) > 0 else 0.0
        sd = float(sv.std()) if len(sv) > 1 else 1.0
        sdata[stn] = {
            "daily": d.sort_values("date").reset_index(drop=True),
            "mu": mu, "sd": sd, "thr": mu + THR_K * sd}
    return sdata


def _feats(hist, mu, sd, anchor):
    # 69-dim feature vector from a 7-day history window
    f = []
    vals = hist["sl_mean"].values
    mx = hist["sl_max"].values
    sdv = hist["sl_std"].values
    pv = hist["pct_mean"].values
    pmx = hist["pct_max"].values
    pmn = hist["pct_min"].values

    # ---- percentile space (10) ----
    f.extend([pv.mean(), pv.std(), pmx.max(), pmn.min(),
              pv[-1] - pv[0], pv[-1], pmx[-1], pmx.max() - pmn.min(),
              pv[-3:].mean(), pv[-1] - pv[-3]])

    # ---- exceedance counts (4) ----
    f.extend([(pmx > 0.90).sum(), (pmx > 0.95).sum(),
              (pmx > 0.99).sum(), (pv > 0.85).sum()])

    # ---- daily percentiles (14) ----
    for i in range(7):
        f.extend([pv[i], pmx[i]])

    # ---- diffs and autocorr (5) ----
    if len(pv) > 1:
        pd_ = np.diff(pv)
        f.extend([pd_.mean(), pd_.std(), pd_.max()])
    else:
        f.extend([0, 0, 0])
    if len(pv) > 2:
        f.extend([np.corrcoef(pv[:-1], pv[1:])[0, 1],
                  np.corrcoef(pv[:-2], pv[2:])[0, 1]])
    else:
        f.extend([0, 0])

    # ---- z-score space (7) ----
    z = (vals - mu) / (sd + 1e-8)
    zmx = (mx - mu) / (sd + 1e-8)
    f.extend([z.mean(), z.std(), z.max(), z.min(),
              z[-1] - z[0], z[-1]])
    f.append(zmx.max() - z.min())

    # ---- daily z-scores (14) ----
    for i in range(7):
        f.extend([z[i], zmx[i]])

    # ---- misc stats (4) ----
    f.extend([sdv.mean() / (sd + 1e-8), sdv.max() / (sd + 1e-8),
              z[-3:].mean(), hist["count"].sum() / (7 * 24)])

    # ---- seasonality + lunar (8) ----
    if anchor is not None:
        doy = anchor.timetuple().tm_yday
        f.extend([np.sin(2 * np.pi * doy / 365.25),
                  np.cos(2 * np.pi * doy / 365.25),
                  np.sin(2 * np.pi * anchor.month / 12),
                  np.cos(2 * np.pi * anchor.month / 12)])
        phase = (((anchor - datetime(2000, 1, 6, 18, 14)).total_seconds()
                  / 86400) % 29.53) / 29.53
        f.extend([np.sin(2 * np.pi * phase),
                  np.cos(2 * np.pi * phase),
                  float(abs(np.sin(2 * np.pi * phase)) < 0.3)])
        f.append(anchor.year + doy / 365.25)

    # ---- higher moments (3) ----
    f.extend([
        float(pd.Series(pv).skew()) if len(pv) > 2 else 0,
        float(pd.Series(pv).kurtosis()) if len(pv) > 3 else 0,
        np.percentile(pv, 75) - np.percentile(pv, 25) if len(pv) > 3 else 0])

    return np.array(f, dtype=np.float32)


def _test_feats(sdata, index):
    # build feature matrix for test windows defined in index
    Xs, ids, stns = [], [], []
    for _, row in index.iterrows():
        stn = row["station_name"]
        ids.append(row["id"])
        stns.append(stn)
        if stn not in sdata:
            Xs.append(np.zeros(69, dtype=np.float32))
            continue
        d = sdata[stn]
        g = d["daily"]
        mu, sd = d["mu"], d["sd"]
        hs = pd.Timestamp(row["hist_start"])
        he = pd.Timestamp(row["hist_end"])
        fs = pd.Timestamp(row["future_start"])
        mask = (g["date"] >= hs) & (g["date"] <= he)
        h = g[mask].sort_values("date")
        if len(h) == HIST_DAYS:
            Xs.append(_feats(h, mu, sd, fs.to_pydatetime()))
        else:
            Xs.append(np.zeros(69, dtype=np.float32))
    X = np.array(Xs, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0), ids, stns


def _prevalence_shift(probs, f_target):
    # shift probabilities in logit space so predicted flood rate matches f_target
    if f_target >= 0.999 or f_target <= 0.001:
        return probs
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    lg = logit(p)
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (expit(lg + mid) >= 0.5).mean()
        if rate > f_target:
            hi = mid
        else:
            lo = mid
    return expit(lg + (lo + hi) / 2)


class Model:
    def __init__(self):
        self.models = {}
        self.ind_auc = {}

    def load(self):
        here = os.path.dirname(__file__)
        pkl = os.path.join(here, "model.pkl")
        if not os.path.exists(pkl):
            pkl = os.path.join(here, "xgb_d10_n300_132_models.pkl")
        with open(pkl, "rb") as f:
            payload = pickle.load(f)
        self.models = payload["models"]
        self.ind_auc = payload.get("ind_auc", {})

    def _select_models(self):
        # global top-K selection by ind_auc
        ml = []
        for k, clf in self.models.items():
            if isinstance(k, tuple) and len(k) in (2, 3):
                ml.append((k, clf, self.ind_auc.get(k, 0.5)))
        ml.sort(key=lambda x: -x[2])
        return ml[:TOP_K]

    def predict(self, X_te, ids):
        sel = self._select_models()

        # ---- soft vote across selected models ----
        plist = []
        for mk, clf, _ in sel:
            plist.append(clf.predict_proba(X_te)[:, 1])
        avg = np.mean(plist, axis=0)

        # ---- calibration ----
        if CALIB_POWER != 1.0:
            avg = np.clip(avg, 1e-8, 1 - 1e-8) ** CALIB_POWER

        # ---- prevalence matching ----
        if USE_PREVALENCE:
            avg = _prevalence_shift(avg, F_TARGET)

        out = pd.DataFrame({"id": ids, "y_prob": avg})
        return out.sort_values("id").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hourly", required=True)
    ap.add_argument("--test_hourly", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--predictions_out", required=True)
    args = ap.parse_args()

    m = Model()
    m.load()

    train = pd.read_csv(args.train_hourly)
    test = pd.read_csv(args.test_hourly)
    idx = pd.read_csv(args.test_index)

    all_h = pd.concat([train, test], ignore_index=True)
    sdata = _station_daily(all_h)
    X_te, ids, stns = _test_feats(sdata, idx)

    out = m.predict(X_te, ids)
    out.to_csv(args.predictions_out, index=False)