# Philadelphia baseline checkpoint

Submit only `outputs/baseline.txt` and `outputs/baseline.png`. The text follows the required template and stays under 500 words. The PNG is one OLS-versus-City RMSE plot with error bars. The completed simulation check is reported in the text.

Replace the group/member placeholders and add a shareable code URL before submitting. Use first names only. The member whose last name comes first alphabetically uploads for the group.

The runner implements parcel matching, screened single-family sales, log-price OLS, five-fold parcel-grouped validation, a common 2024 City benchmark, and 2,000 paired parcel-bootstrap resamples. Ridge and random forest are outside this checkpoint.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_analysis.py
```

`run_analysis.py` computes the results and saves all numbers/checks in one readable `results.txt` log. It also saves the simple `outputs/baseline.png`. The write-up `outputs/baseline.txt` is a normal, manually edited file; the script does not generate or overwrite it.

Input CSVs and source metadata are in `csvs/`. The script writes no PDF, CSV output, or JSON results cache.

Current features are a 2026 snapshot; retained sales are screened likely market transactions. Both limitations appear in the submission. Synthetic validity checks: `.venv/bin/python -m unittest test_analysis`.
