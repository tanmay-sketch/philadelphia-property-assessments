"""Prepare a conservative retrospective Philadelphia sales cohort in memory.

The filters identify likely market transactions; the public fields do not verify
arm's-length status. No cleaned property-level file is written by this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "total_livable_area", "total_area", "age", "number_of_bedrooms",
    "number_of_bathrooms", "number_stories",
]
CATEGORICAL_FEATURES = [
    "interior_condition", "exterior_condition", "quality_grade", "zip_code",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _read(path: Path) -> pd.DataFrame:
    # Preserve leading zeros and distinguish identifiers from measurements.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame.apply(lambda col: col.str.strip())


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def load_cohort(raw_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Return eligible sales, optional same-year City values, and aggregate audit.

    Returned identifiers exist only for parcel-grouped folds and bootstrap.
    Training uses sale_year 2022/2023 and testing uses 2024. All repeated sales
    remain in the data; duplicate source records and ambiguous join keys do not.
    """
    raw_dir = Path(raw_dir)
    audit: dict = {"attrition": [], "feature_columns": FEATURES.copy()}
    transfers = _read(raw_dir / "transfers_2022_2024_candidates.csv")
    properties = _read(raw_dir / "properties_selected_columns.csv")
    history = _read(raw_dir / "assessments_2022_2024.csv")
    audit["source_rows"] = {
        "transfers_union_date_window": len(transfers),
        "properties": len(properties), "assessment_history": len(history),
    }
    counts_path = raw_dir / "server_counts.json"
    if counts_path.exists():
        expected = {r["source"]: int(r["n"]) for r in json.loads(counts_path.read_text())["rows"]}
        observed = {"transfers_union_window": len(transfers), "properties": len(properties),
                    "assessment_rows_2022_2024": len(history)}
        if expected != observed:
            raise ValueError("CSV row counts differ from saved server counts")
        audit["csv_completeness_verified_against_server_counts"] = observed

    # Exact duplicates may be safely collapsed. Conflicting records are excluded
    # rather than selecting arbitrary property features or assessment values.
    prop_exact = int(properties.duplicated().sum())
    properties = properties.drop_duplicates()
    conflicting_prop = properties["parcel_number"].duplicated(keep=False)
    audit["property_key_audit"] = {
        "exact_duplicate_rows_removed": prop_exact,
        "conflicting_parcel_keys_excluded": int(properties.loc[conflicting_prop, "parcel_number"].nunique()),
        "conflicting_rows_excluded": int(conflicting_prop.sum()),
        "blank_key_rows_excluded": int(properties["parcel_number"].eq("").sum()),
    }
    properties = properties.loc[~conflicting_prop & properties["parcel_number"].ne("")].copy()
    properties = properties.rename(columns={"parcel_number": "parcel_id"})
    property_keys = set(properties["parcel_id"])

    # Denominators include all single-parcel regular deeds in the downloaded
    # union date window, before price, feature, and benchmark exclusions.
    single = transfers.loc[transfers["document_type"].str.upper().eq("DEED") &
                           _numeric(transfers["property_count"]).eq(1)]
    nonblank = single["opa_account_num"].ne("")
    matched = single["opa_account_num"].isin(property_keys)
    audit["single_parcel_deed_match"] = {
        "denominator_scope": "Downloaded union of document-date or display-date 2022-2024 windows; before other screens",
        "all_single_parcel_deeds": len(single),
        "with_nonblank_key": int(nonblank.sum()), "matched": int(matched.sum()),
        "rate_all": _rate(int(matched.sum()), len(single)),
        "rate_nonblank_keys": _rate(int(matched.sum()), int(nonblank.sum())),
    }

    def retain(frame: pd.DataFrame, mask: pd.Series, rule: str) -> pd.DataFrame:
        kept = frame.loc[mask].copy()
        audit["attrition"].append({"rule": rule, "before": len(frame),
                                    "removed": len(frame) - len(kept), "remaining": len(kept)})
        return kept

    sales = transfers.drop_duplicates().copy()
    audit["transfer_duplicates"] = {"exact_rows_removed": len(transfers) - len(sales)}
    bad_record = sales["record_id"].duplicated(keep=False) | sales["record_id"].eq("")
    audit["transfer_duplicates"]["ambiguous_or_blank_record_id_rows_removed"] = int(bad_record.sum())
    sales = sales.loc[~bad_record].copy()
    sales = retain(sales, sales["document_type"].str.upper().eq("DEED"), "regular DEED")
    sales = retain(sales, _numeric(sales["property_count"]).eq(1), "property_count = 1")
    sales = retain(sales, sales["opa_account_num"].ne(""), "nonblank OPA account key")
    sales = retain(sales, sales["opa_account_num"].isin(property_keys), "unique current property key match")

    # Keep a compact explicit allowlist from properties. Current market values,
    # sale details and owner/address fields never enter the modeling frame.
    property_cols = ["parcel_id", "category_code", "year_built", "assessment_date"] + [
        f for f in FEATURES if f != "age"
    ]
    sales = sales.rename(columns={"opa_account_num": "parcel_id", "zip_code": "transfer_zip_code"})
    sales = sales.merge(properties[property_cols], on="parcel_id", how="left", validate="many_to_one")
    sales = retain(sales, sales["category_code"].eq("1"), "current category_code = 1 (single family)")
    sales["price"] = _numeric(sales["total_consideration"])
    sales = retain(sales, sales["price"].gt(1000), "total_consideration > $1,000")
    audit["screened_discrepancy_values_before_flag_exclusion"] = _counts(sales["discrepancy"])
    sales = retain(sales, sales["discrepancy"].str.lower().eq("false"), "discrepancy explicitly False")
    other = _numeric(sales["other_consideration"])
    # Non-numeric nonblank values are not accepted as equivalent to missing.
    sales = retain(sales, other.eq(0) | sales["other_consideration"].eq(""),
                   "other_consideration is zero or blank")
    date = pd.to_datetime(sales["document_date"].str[:10], errors="coerce", format="%Y-%m-%d")
    sales = retain(sales, date.notna(), "nonmissing valid document_date; no fallback")
    sales["document_date"] = date.loc[sales.index]
    sales["sale_year"] = sales["document_date"].dt.year.astype(int)
    sales = retain(sales, sales["sale_year"].isin([2022, 2023, 2024]), "document_date in 2022-2024")

    built = _numeric(sales["year_built"])
    built = built.where(built.gt(0) & built.eq(np.floor(built)))
    sales = retain(sales, built.isna() | built.le(sales["sale_year"]), "known year_built not after sale year")
    sales["age"] = sales["sale_year"] - built.loc[sales.index]
    audit["snapshot_assessment_date_years"] = _counts(sales["assessment_date"].str[:4])
    audit["pre_benchmark_rows_by_year"] = _counts(sales["sale_year"])

    history["sale_year"] = _numeric(history["year"])
    history["city_value"] = _numeric(history["market_value"])
    history = history.rename(columns={"parcel_number": "parcel_id"})
    history = history[["parcel_id", "sale_year", "city_value"]]
    exact_history = int(history.duplicated().sum())
    history = history.drop_duplicates()
    ambiguous_history = history.duplicated(["parcel_id", "sale_year"], keep=False)
    audit["assessment_key_audit"] = {
        "exact_duplicate_rows_removed": exact_history,
        "conflicting_parcel_year_pairs_excluded": int(history.loc[ambiguous_history, ["parcel_id", "sale_year"]].drop_duplicates().shape[0]),
        "conflicting_rows_excluded": int(ambiguous_history.sum()),
        "rows_by_year_before_conflict_exclusion": _counts(history["sale_year"]),
    }
    history = history.loc[~ambiguous_history & history["parcel_id"].ne("") &
                          history["sale_year"].isin([2022, 2023, 2024])].copy()
    history["sale_year"] = history["sale_year"].astype(int)
    sales = sales.merge(history, on=["parcel_id", "sale_year"], how="left", validate="many_to_one")
    sales["city_value"] = sales["city_value"].where(sales["city_value"].gt(0))
    available_city = sales["city_value"].notna()
    audit["historical_city_benchmark_coverage"] = {
        "eligible_sales": len(sales),
        "with_positive_unique_same_year_value": int(available_city.sum()),
        "missing_nonpositive_or_conflicting_value": int((~available_city).sum()),
        "available_rows_by_sale_year": _counts(sales.loc[available_city, "sale_year"]),
        "missing_rows_by_sale_year": _counts(sales.loc[~available_city, "sale_year"]),
        "policy": "City availability does not restrict model training or all-sale evaluation; use the same nonmissing subset for City comparisons.",
    }

    invalid: dict[str, int] = {}
    for feature in NUMERIC_FEATURES:
        values = _numeric(sales[feature])
        bad = values.lt(0) if feature == "age" else values.le(0)
        invalid[feature] = int(bad.sum())
        sales[feature] = values.mask(bad)
    for feature in CATEGORICAL_FEATURES:
        values = sales[feature].str.upper()
        if feature == "zip_code":
            values = values.str[:5].where(values.str[:5].str.fullmatch(r"\d{5}"), "")
        sales[feature] = values.replace("", np.nan)
    audit["invalid_nonpositive_numeric_values_set_missing"] = invalid
    audit["feature_missingness"] = {
        feature: {"n": int(sales[feature].isna().sum()),
                  "fraction": float(sales[feature].isna().mean())}
        for feature in FEATURES
    }
    train = sales.loc[sales["sale_year"].isin([2022, 2023])]
    test = sales.loc[sales["sale_year"].eq(2024)]
    train_parcels = set(train["parcel_id"])
    test_parcels = set(test["parcel_id"])
    audit["final_cohort"] = {
        "sales": len(sales), "unique_parcels": int(sales["parcel_id"].nunique()),
        "rows_by_year": _counts(sales["sale_year"]),
        "train_sales": len(train), "test_sales": len(test),
        "train_unique_parcels": len(train_parcels), "test_unique_parcels": len(test_parcels),
        "parcels_in_both_train_and_test": len(train_parcels & test_parcels),
        "test_sales_on_previously_seen_parcels": int(test["parcel_id"].isin(train_parcels).sum()),
        "test_sales_on_unseen_parcels": int((~test["parcel_id"].isin(train_parcels)).sum()),
        "parcels_with_multiple_sales": int(sales["parcel_id"].value_counts().gt(1).sum()),
        "repeated_sale_rows_beyond_one_per_parcel": len(sales) - int(sales["parcel_id"].nunique()),
    }
    higher = sales.loc[sales["price"].gt(10000)]
    audit["price_threshold_sensitivity_over_10000"] = {
        "sales": len(higher), "rows_by_year": _counts(higher["sale_year"]),
        "removed_from_main_cohort": len(sales) - len(higher),
    }
    audit["notes"] = [
        "Retrospective comparison: property features are a 2026-11-09 current snapshot, not verified sale-date features.",
        "Screened likely-market sales are not independently validated arm's-length transactions.",
        "Main price is total_consideration; source field definitions and discrepancy semantics remain unverified.",
        "Main date is document_date with no fallback; sale-year City market_value comes from historical assessments downloaded in 2026.",
        "Eligible sales lacking a positive unique historical City value remain in the cohort with city_value missing; City comparisons must use a common nonmissing subset.",
        "Historical assessment values may reflect later corrections; original assessment publication vintage is unavailable.",
        "A >$1,000 threshold retains low-price transactions; >$10,000 sensitivity counts are reported without outcome-based ratio filtering.",
        "Missing feature values remain missing for training-fold imputation; zero bedrooms, bathrooms and stories are treated as suspect missing data.",
        "All repeated sales are retained; parcel IDs are for internal grouping only and excluded from predictor features and published output.",
    ]
    result = sales[["parcel_id", "sale_year", "price", "city_value", "document_date"] + FEATURES].reset_index(drop=True)
    if result.empty or not np.isfinite(result["price"].to_numpy()).all() or not result["price"].gt(1000).all():
        raise ValueError("Cohort must contain finite sale prices greater than $1,000")
    city = result["city_value"].dropna()
    if not np.isfinite(city.to_numpy()).all() or not city.gt(0).all():
        raise ValueError("Observed City benchmark values must be finite and positive")
    return result, audit
