# Philadelphia Property Assessments

Compare OLS home-price predictions with Philadelphia's assessed market values using single-family property sales from 2022-2024. A simulation examines how grouping by actual sale price can create misleading valuation-ratio patterns.

## Data and method

The public City data in `csvs/` contains property characteristics, recorded transfers, and annual assessment history. `provenance.json` records the source queries and extraction date.

The pipeline joins records by parcel number and screens for likely market sales. It fits OLS to log sale price using 2022-2023 sales, with five-fold cross-validation grouped by property. Preprocessing is learned within each training fold. Predictions are evaluated against 2024 sale prices, and City assessments are compared on the same sales.

Accuracy is measured with dollar MAE and RMSE. Confidence intervals use 2,000 paired parcel-bootstrap resamples. The random seed is 5420.

## Run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_analysis.py
```

Run these commands from the project directory.

## Files

- `results.txt`: cohort counts, matching rates, CV scores, test metrics, confidence intervals, and simulation checks.
- `outputs/baseline.png`: OLS-versus-City test RMSE with 95% confidence intervals.
- `outputs/baseline.txt`: manually maintained analysis summary.
- `cohort.py`: record matching, transaction screening, and feature preparation.
- `simulation.py`: the known-value-plus-noise simulation.

## Checks

```sh
.venv/bin/python -B -m unittest test_analysis
```

## Limitations

Property features and classifications come from a 2026 snapshot. Historical assessments may include later revisions. Transaction screening identifies likely market sales but cannot independently verify arm's-length status, so the analysis is retrospective.
