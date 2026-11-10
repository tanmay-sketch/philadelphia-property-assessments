"""Small scientific-validity checks using synthetic inputs only."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from cohort import load_cohort
from run_analysis import (
    FEATURES, MODELS, bootstrap_summary, dollar_predictions, feature_frame,
    preprocessor,
)


class AnalysisChecks(unittest.TestCase):
    def test_cohort_preserves_repeats_and_missing_city_but_excludes_invalid_sales(self):
        def transfer(record, parcel, year=2024, **changes):
            row = dict(record_id=record, document_id=record, document_type="DEED",
                       property_count="1", opa_account_num=parcel, zip_code="19104",
                       total_consideration="100000", discrepancy="False",
                       other_consideration="0", document_date=f"{year}-06-01")
            return dict(row, **changes)

        first, missing_city = "000000001", "000000002"
        transfers = [
            transfer("a", first, 2022), transfer("b", first),
            transfer("c", missing_city),
            transfer("d", "000000003", discrepancy="True"),
            transfer("e", "000000004", document_type="MORTGAGE"),
            transfer("f", "000000005"),
            transfer("g", "000000006", total_consideration="1"),
            transfer("h", "000000007", property_count="2"),
            transfer("i", ""),
        ]
        properties = []
        for index in range(1, 8):
            properties.append(dict(
                parcel_number=f"{index:09d}", category_code="1",
                year_built="2025" if index == 5 else "1920",
                assessment_date="2026-01-01", total_livable_area="1200",
                total_area="1500", number_of_bedrooms="3", number_of_bathrooms="1",
                number_stories="2", interior_condition="4", exterior_condition="4",
                quality_grade="C", zip_code="19104", sale_price="99999999",
            ))
        history = [dict(parcel_number=first, year="2022", market_value="90000"),
                   dict(parcel_number=first, year="2024", market_value="110000")]
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            pd.DataFrame(transfers).to_csv(raw / "transfers_2022_2024_candidates.csv", index=False)
            pd.DataFrame(properties).to_csv(raw / "properties_selected_columns.csv", index=False)
            pd.DataFrame(history).to_csv(raw / "assessments_2022_2024.csv", index=False)
            original_files = set(raw.iterdir())
            sales, audit = load_cohort(raw)
            self.assertEqual(set(raw.iterdir()), original_files, "No property-level outputs should be written")
        self.assertEqual(len(sales), 3)
        self.assertEqual(sales.parcel_id.value_counts().to_dict(), {first: 2, missing_city: 1})
        self.assertTrue(sales.loc[sales.parcel_id.eq(missing_city), "city_value"].isna().all())
        self.assertTrue(sales.age.ge(0).all())
        self.assertEqual(audit["single_parcel_deed_match"]["with_nonblank_key"], 6)
        self.assertAlmostEqual(audit["single_parcel_deed_match"]["rate_all"], 6 / 7)
        self.assertEqual(audit["single_parcel_deed_match"]["rate_nonblank_keys"], 1)

    def test_paired_parcel_bootstrap_gives_zero_difference_for_identical_models(self):
        actual = np.array([100., 120., 200., 210., 300.])
        predictions = np.repeat((actual * .9 + 10)[:, None], len(MODELS), axis=1)
        parcel = np.array(["a", "a", "b", "c", "c"])
        metrics, differences = bootstrap_summary(
            actual, predictions, parcel,
            replicates=200, seed=123,
        )
        self.assertFalse(differences.empty)
        np.testing.assert_array_equal(differences[["estimate", "ci_low", "ci_high"]], 0)
        self.assertTrue(metrics["n"].eq(len(actual)).all())
        self.assertTrue(np.isfinite(metrics[["estimate", "ci_low", "ci_high"]]).all().all())
        expected_rmse = np.sqrt(np.mean((predictions[:, 0] - actual) ** 2))
        np.testing.assert_allclose(metrics.loc[metrics.metric.eq("RMSE"), "estimate"], expected_rmse)

    def test_predictors_exclude_outcomes_and_handle_unseen_categories(self):
        n = 20
        frame = pd.DataFrame({
            "total_livable_area": np.linspace(800, 1800, n),
            "total_area": np.linspace(1000, 2200, n), "age": np.arange(n) + 20.,
            "number_of_bedrooms": np.full(n, 3.), "number_of_bathrooms": np.ones(n),
            "number_stories": np.full(n, 2.), "interior_condition": ["4"] * n,
            "exterior_condition": ["4"] * n, "quality_grade": ["C"] * n,
            "zip_code": ["19104"] * n, "price": np.full(n, 100000.),
            "city_value": np.full(n, 200000.), "parcel_id": [f"p{i}" for i in range(n)],
        })
        train = feature_frame(frame)
        self.assertEqual(set(train), set(FEATURES))
        self.assertFalse({"price", "city_value", "parcel_id", "sale_price"} & set(train))
        # Changing outcome/benchmark/identifier fields must not change predictors.
        changed = frame.assign(price=-1, city_value=-2, parcel_id="different")
        pd.testing.assert_frame_equal(train, feature_frame(changed))
        processor = preprocessor()
        x_train = processor.fit_transform(train)
        heldout = frame.iloc[:2].copy()
        heldout["zip_code"], heldout["quality_grade"] = "99999", "UNSEEN"
        x_test = processor.transform(feature_frame(heldout))
        self.assertEqual(x_test.shape[1], x_train.shape[1])
        self.assertTrue(np.isfinite(x_test.toarray() if hasattr(x_test, "toarray") else x_test).all())
        # A correctly specified noiseless log-price relationship needs no smearing.
        log_price = 10 + .5 * train.total_livable_area.to_numpy()
        prediction, smear = dollar_predictions(
            LinearRegression(tol=1e-10), x_train, log_price, x_train,
        )
        self.assertAlmostEqual(smear, 1., places=8)
        np.testing.assert_allclose(prediction, np.exp(log_price), rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
