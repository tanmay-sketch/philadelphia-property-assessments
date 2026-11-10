"""Oracle simulation illustrating artifacts in valuation-ratio charts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(seed: int = 5420) -> tuple[pd.DataFrame, dict]:
    """Return aggregate oracle ratios and checks without writing or plotting."""
    rng = np.random.default_rng(seed)
    n = 50_000
    x = rng.normal(size=n)
    oracle = np.exp(12 + 0.5 * x)
    rows = []
    checks = []
    noise_levels = (0.0, 0.1, 0.3, 0.5)
    for sigma in noise_levels:
        epsilon = rng.normal(0, sigma, size=n)
        actual = oracle * np.exp(epsilon - sigma**2 / 2)
        individual_ratio = oracle / actual
        expected_ratio = float(np.exp(sigma**2))
        observed_ratio = float(individual_ratio.mean())
        # Exact variance of the oracle/actual ratio under the specified DGP.
        standard_error = np.sqrt(
            (np.exp(3 * sigma**2) - np.exp(2 * sigma**2)) / n
        )
        tolerance = max(0.002, 5 * float(standard_error))
        if abs(observed_ratio - expected_ratio) > tolerance:
            raise AssertionError("Simulated mean ratio differs from its oracle expectation.")
        checks.append(
            {
                "sigma": sigma,
                "n": n,
                "expected_mean_individual_ratio": expected_ratio,
                "observed_mean_individual_ratio": observed_ratio,
                "validation_tolerance": tolerance,
                "ratio_of_sums": float(oracle.sum() / actual.sum()),
            }
        )
        for grouping, values in (("actual_price", actual), ("oracle_value", oracle)):
            # Quintiles are recalculated for each simulated actual-price distribution.
            # Oracle-value quintiles remain identical across the noise levels.
            bins = pd.qcut(values, q=5, labels=False)
            for bin_id in range(5):
                selected = np.asarray(bins == bin_id)
                rows.append(
                    {
                        "sigma": sigma,
                        "grouping": grouping,
                        "quintile": bin_id + 1,
                        "n": int(selected.sum()),
                        "group_value_min": float(values[selected].min()),
                        "group_value_max": float(values[selected].max()),
                        "mean_actual_price": float(actual[selected].mean()),
                        "mean_oracle_prediction": float(oracle[selected].mean()),
                        "mean_individual_ratio": float(individual_ratio[selected].mean()),
                        "ratio_of_sums": float(oracle[selected].sum() / actual[selected].sum()),
                        "expected_mean_ratio_given_x": expected_ratio,
                    }
                )
    summary = pd.DataFrame(rows)
    zero_noise = summary.loc[summary["sigma"].eq(0)]
    if not np.allclose(zero_noise[["mean_individual_ratio", "ratio_of_sums"]], 1):
        raise AssertionError("Every zero-noise bin must have a valuation ratio of one.")

    validation = {
        "seed": seed,
        "observations_per_noise_level": n,
        "dgp": "X ~ Normal(0,1); m(X)=exp(12+0.5X); Y=m(X)*exp(epsilon-sigma^2/2)",
        "zero_noise_ratios_equal_one": True,
        "mean_ratio_checks": checks,
    }
    return summary, validation
