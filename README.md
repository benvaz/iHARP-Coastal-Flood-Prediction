# Coastal Flood Prediction — HDR SMood Challenge Y2

Cross-station XGBoost ensemble predicting 14-day coastal flood risk from 7-day sea level history for the [2025 HDR SMood Challenge](https://www.codabench.org/competitions/9855/).

## Structure

```
submission/
  model.py              # inference (CodaBench submission)
  model.pkl             # 132 XGBoost models + OOD AUC scores
  requirements.txt
competition/
  ingestion.py          # competition ingestion program
  scoring.py            # competition scoring program
```

## Approach

132 XGBoost models trained on all pairwise combinations of 9 training stations, with global top-K selection by out-of-distribution AUC. 69 station-agnostic features: percentile-space statistics, exceedance counts, z-scores, seasonal harmonics, and lunar phase. Post-processing applies power calibration and prevalence matching in logit space.

## References

- Competition: [iHARP Coastal Flooding Challenge](https://www.codabench.org/competitions/9855/)
- [Challenge sample repo](https://github.com/Imageomics/HDR-SMood-Challenge-sample)

## License

MIT
