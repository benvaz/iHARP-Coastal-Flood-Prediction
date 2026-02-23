# HDR ML Challenge Y2 — Coastal Flood Prediction

Cross-station ensemble for 14-day coastal flood risk prediction from 7-day sea level history.
Submitted to the [iHARP Coastal Flooding Challenge](https://www.codabench.org/competitions/9855/) as part of the 2025 HDR Scientific Mood (Modeling Out of Distribution) Challenge.

## Approach

An XGBoost ensemble trained on all pairwise combinations of 9 training stations (132 models total), with global top-K model selection by out-of-distribution AUC. Features are station-agnostic: percentile-space statistics, exceedance counts, z-scores, seasonal harmonics, and lunar phase — designed to transfer across stations without encoding station identity.

Post-processing applies power calibration and prevalence matching in logit space to align predicted flood rates with the scoring distribution.

## Results

| Metric | Score |
|--------|-------|
| MCC    | 0.346 |
| AUC    | 0.832 |
| F1     | 0.913 |
| Acc    | 0.849 |

## Repository Structure

```
├── submission/
│   ├── model.py            # inference code (called by ingestion)
│   ├── model.pkl            # 132 XGBoost models + OOD AUC scores
│   └── requirements.txt     # scipy, scikit-learn
├── competition/
│   ├── ingestion.py         # competition ingestion program
│   └── scoring.py           # competition scoring program
├── .gitignore
├── LICENSE
└── README.md
```

## Submission

The submission zip contains `model.py`, `model.pkl`, and `requirements.txt`. The ingestion program passes hourly sea level CSVs and a test index; the model builds 69 features per window and returns flood probabilities.

```
python model.py \
    --train_hourly train_hourly.csv \
    --test_hourly test_hourly.csv \
    --test_index test_index.csv \
    --predictions_out predictions.csv
```

## Team

- Ben Vaziritabar — Cardiff University

## License

MIT
