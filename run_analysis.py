"""Run the fixed OLS checkpoint; save results.txt and baseline.png."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/philly-baseline-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from cohort import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_cohort
from simulation import simulate
SEED = 5420
MODELS = ["OLS", "City"]
FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)

def feature_frame(frame):
    result = frame[FEATURES].copy()
    # Prespecified hedonic size transformation; never chosen using test errors.
    for col in ["total_livable_area", "total_area"]:
        result[col] = np.log1p(result[col])
    return result

def preprocessor():
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), list(NUMERIC_FEATURES)),
        ("categorical", make_pipeline(SimpleImputer(strategy="constant", fill_value="Missing"), OneHotEncoder(handle_unknown="ignore", sparse_output=True)), list(CATEGORICAL_FEATURES)),
    ])

def dollar_predictions(estimator, x_fit, log_y_fit, x_predict):
    estimator.fit(x_fit, log_y_fit)
    smear = float(np.mean(np.exp(log_y_fit - estimator.predict(x_fit))))
    prediction = np.exp(estimator.predict(x_predict)) * smear
    if not np.isfinite(prediction).all() or not (prediction > 0).all():
        raise ValueError("Nonfinite/nonpositive dollar predictions; do not silently clip predictions.")
    return prediction, smear

def bootstrap_summary(y, prediction, parcel, replicates=2000, seed=SEED):
    """Compare OLS and City using the same resampled test parcels."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    cluster, labels = pd.factorize(parcel)
    n_clusters = len(labels)
    rng = np.random.default_rng(seed)
    error = prediction - y[:, None]
    absolute, squared = abs(error), error ** 2
    metric_draws = np.empty((replicates, prediction.shape[1], 2))
    for b in range(replicates):
        weights = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)[cluster]
        total = weights.sum()
        metric_draws[b, :, 0] = weights @ absolute / total
        metric_draws[b, :, 1] = np.sqrt(weights @ squared / total)
    overall = []
    for model, name in enumerate(MODELS):
        for metric, estimate, index in [("MAE", absolute[:, model].mean(), 0), ("RMSE", np.sqrt(squared[:, model].mean()), 1)]:
            low, high = np.quantile(metric_draws[:, model, index], [.025, .975])
            overall.append({"model": name, "metric": metric, "n": len(y), "estimate": estimate, "ci_low": low, "ci_high": high})
    draws = metric_draws[:, 0, 1] - metric_draws[:, 1, 1]
    low, high = np.quantile(draws, [.025, .975])
    difference = {"model": "OLS", "reference": "City", "metric": "RMSE difference",
                  "estimate": float(np.sqrt(squared[:, 0].mean()) - np.sqrt(squared[:, 1].mean())),
                  "ci_low": low, "ci_high": high}
    return pd.DataFrame(overall), pd.DataFrame([difference])

def compute_checkpoint(raw, replicates, seed):
    cohort, audit = load_cohort(raw)
    train = cohort[cohort.sale_year.isin([2022, 2023])].reset_index(drop=True)
    test = cohort[cohort.sale_year.eq(2024)].reset_index(drop=True)
    x = feature_frame(train)
    y = train.price.to_numpy()
    log_y = np.log(y)
    parcels = train.parcel_id.to_numpy()
    oof = np.full(len(train), np.nan)
    folds = []
    for fold, (fit, valid) in enumerate(GroupKFold(5).split(x, groups=parcels), 1):
        assert not set(parcels[fit]) & set(parcels[valid])
        processor = preprocessor()
        x_fit = processor.fit_transform(x.iloc[fit])
        oof[valid], _ = dollar_predictions(LinearRegression(tol=1e-8), x_fit, log_y[fit], processor.transform(x.iloc[valid]))
        folds.append({"fold": fold, "fit_n": len(fit), "validation_n": len(valid), "overlapping_parcels": 0})
    assert np.isfinite(oof).all()
    cv = [{"model": "OLS", "rmse": float(np.sqrt(np.mean((oof-y)**2))), "mae": float(np.mean(abs(oof-y)))}]
    # This fixed OLS specification has no hyperparameter grid. All transforms
    # and the price correction are learned from training data only.
    processor = preprocessor()
    x_fit = processor.fit_transform(x)
    prediction, smear = dollar_predictions(LinearRegression(tol=1e-8), x_fit, log_y, processor.transform(feature_frame(test)))
    common = test.city_value.notna() & test.city_value.gt(0)
    held = test.loc[common].reset_index(drop=True)
    pred = np.column_stack([prediction[common], held.city_value])
    metrics, differences = bootstrap_summary(held.price.to_numpy(), pred, held.parcel_id, replicates, seed)
    simulation, checks = simulate(seed)
    return {"audit": audit, "cv": cv, "metrics": metrics.to_dict("records"), "paired_rmse_differences": differences.to_dict("records"),
            "simulation": simulation.to_dict("records"), "simulation_checks": checks, "bootstrap_replicates": replicates,
            "manifest": {"seed":seed, "cv_folds":folds, "test_used_for_tuning":False,
                         "feature_allowlist":FEATURES, "smearing_factor":smear, "sklearn_version":sklearn.__version__}}

def baseline_figure(result, path):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    metrics = pd.DataFrame(result["metrics"])
    fig, ax = plt.subplots(figsize=(7.2,3.8))
    for y, (model,color) in enumerate([("City","#BC4939"),("OLS","#1769A6")]):
        row = metrics[(metrics.model==model)&(metrics.metric=="RMSE")].iloc[0]
        ax.errorbar(row.estimate,y,xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],fmt="o",color=color,capsize=5,markersize=7)
        ax.annotate(f"${row.estimate:,.0f}",(row.estimate,y),xytext=(0,12),textcoords="offset points",ha="center",fontsize=10)
    ax.set(yticks=[0,1],yticklabels=["City-assessed values","OLS-predicted prices"],ylim=(-.5,1.5),xlim=(0,230000),xlabel="RMSE against actual sale price ($)")
    ax.set_title(f"2024 sales (n = {int(metrics.iloc[0]['n']):,})",fontsize=11,pad=12)
    ax.legend(handles=[Line2D([],[],marker="o",linestyle="none",color="0.3",label="RMSE"),
                       Line2D([],[],marker="|",linewidth=1.5,color="0.3",label="95% bootstrap CI")],
              loc="upper left",frameon=False,fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"${x/1000:,.0f}k"))
    ax.grid(axis="x",alpha=.15)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path,dpi=200,facecolor="white")
    plt.close(fig)


def write_results_log(result, path):
    """Save numerical outputs and checks; the submission text is edited separately."""
    audit = result["audit"]
    with path.open("w") as log:
        def section(title, content):
            print(f"\n{title}\n{'-' * len(title)}\n{content}", file=log)

        print("Philadelphia OLS baseline: results log", file=log)
        section("Source record counts", "\n".join(f"{k}: {v:,}" for k,v in audit["source_rows"].items()))
        section("Sale filters", pd.DataFrame(audit["attrition"]).to_string(index=False))
        section("Parcel matching", "\n".join(f"{k}: {v}" for k,v in audit["single_parcel_deed_match"].items()))
        section("Final cohort", "\n".join(f"{k}: {v}" for k,v in audit["final_cohort"].items()))
        section("City benchmark coverage", "\n".join(f"{k}: {v}" for k,v in audit["historical_city_benchmark_coverage"].items()))
        for key in ["property_key_audit", "transfer_duplicates", "assessment_key_audit", "feature_missingness", "price_threshold_sensitivity_over_10000"]:
            section(key.replace("_", " "), "\n".join(f"{k}: {v}" for k,v in audit[key].items()))
        method = result["manifest"]
        section("Method", f"Fixed log-price OLS; no tuning grid. Five parcel-grouped folds.\n"
                f"Seed: {method['seed']}\nFeatures: {', '.join(method['feature_allowlist'])}\n"
                "Area transform: log1p. Numeric imputation/scaling and category encoding within each fold.\n"
                f"Training-only smearing factor: {method['smearing_factor']:.10f}\n"
                f"scikit-learn: {method['sklearn_version']}\n2024 used for tuning: False")
        section("CV fold sizes", pd.DataFrame(method["cv_folds"]).to_string(index=False))
        section("Pooled out-of-fold scores: 2022-2023, dollars", pd.DataFrame(result["cv"]).to_string(index=False))
        section("Held-out scores: same 2024 sales, dollars", pd.DataFrame(result["metrics"]).to_string(index=False, float_format=lambda x:f"{x:.6f}"))
        section("Paired OLS minus City RMSE, dollars", pd.DataFrame(result["paired_rmse_differences"]).to_string(index=False, float_format=lambda x:f"{x:.6f}"))
        section("Uncertainty", f"{result['bootstrap_replicates']} paired parcel-cluster bootstrap replicates; 2.5th and 97.5th percentiles.\n"
                "Repeat sales stay together; predictions remain fixed. Conditional on fitted models.")
        checks = result["simulation_checks"]
        section("Simulation", f"{checks['dgp']}\nSeed: {checks['seed']}; observations per noise level: {checks['observations_per_noise_level']}\n"
                f"Zero-noise ratios equal one: {checks['zero_noise_ratios_equal_one']}")
        section("Simulation mean-ratio checks", pd.DataFrame(checks["mean_ratio_checks"]).to_string(index=False, float_format=lambda x:f"{x:.6f}"))
        simulation = pd.DataFrame(result["simulation"])
        endpoints = simulation[simulation.quintile.isin([1,5])].pivot(index=["sigma","grouping"],columns="quintile",values="mean_individual_ratio")
        section("Simulation ratios: lowest and highest quintiles", endpoints.rename(columns={1:"lowest",5:"highest"}).to_string(float_format=lambda x:f"{x:.6f}"))
        section("RMSE definition", "sqrt(mean((predicted price - sale price)^2)).\n"
                f"CV uses {audit['final_cohort']['train_sales']:,} held-out-fold predictions from 2022-2023.\n"
                "It is a root-mean-square error in dollars, not a total dollar loss across the dataset.")
        section("Limitations", "\n".join(audit["notes"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("csvs"))
    parser.add_argument("--output",type=Path,default=Path("outputs"))
    parser.add_argument("--bootstrap",type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap<100:
        parser.error("Use at least 100 bootstrap replicates")
    result=compute_checkpoint(args.data,args.bootstrap,SEED)
    write_results_log(result, Path("results.txt"))
    args.output.mkdir(parents=True,exist_ok=True)
    baseline_figure(result,args.output/"baseline.png")
    print(pd.DataFrame(result["metrics"]).to_string(index=False), flush=True)
    print(f"Saved results.txt and {args.output/'baseline.png'}. baseline.txt is edited separately.")


if __name__=="__main__":
    main()
