import numpy as np
import pandas as pd
import statsmodels.api as sm



# =============================================================================
# 1. Building a Log binomial model
# =============================================================================

def make_binary_outcome(df, source_col, new_col=None, threshold="median", higher_is_one=True):
    d = df.copy()

    if new_col is None:
        new_col = f"{source_col}_bin"

    s = d[source_col]

    if threshold == "median":
        cut = s.median()
    elif threshold == "mean":
        cut = s.mean()
    elif isinstance(threshold, (int, float)):
        cut = float(threshold)
    else:
        raise ValueError("threshold must be 'median', 'mean', or a numeric cutoff")

    if higher_is_one:
        d[new_col] = (s >= cut).astype(int)
    else:
        d[new_col] = (s <= cut).astype(int)

    return d, cut


def prepare_poisson_mediation_data(df, sample, x, m, y_bin, covariates=None, standardize_xm=True):
    if covariates is None:
        covariates = []

    needed = [x, m, y_bin] + covariates

    if sample == "Combined":
        d = df[needed].dropna().copy()
    else:
        d = df.loc[df["course"] == sample, needed].dropna().copy()

    if standardize_xm:
        d[x] = (d[x] - d[x].mean()) / d[x].std(ddof=0)
        d[m] = (d[m] - d[m].mean()) / d[m].std(ddof=0)

    return d


def fit_poisson_mediation(df, sample, x, m, y_bin, covariates=None):
    if covariates is None:
        covariates = []

    d = prepare_poisson_mediation_data(df, sample, x, m, y_bin, covariates=covariates)

    # a-path: M ~ X + C
    Xa = sm.add_constant(d[[x] + covariates])
    model_a = sm.OLS(d[m], Xa).fit(cov_type="HC3")

    # total effect: Y ~ X + C
    Xt = sm.add_constant(d[[x] + covariates])
    model_total = sm.GLM(
        d[y_bin],
        Xt,
        family=sm.families.Poisson(link=sm.families.links.Log())
    ).fit(cov_type="HC3")

    # direct + mediator effect: Y ~ X + M + C
    Xb = sm.add_constant(d[[x, m] + covariates])
    model_b = sm.GLM(
        d[y_bin],
        Xb,
        family=sm.families.Poisson(link=sm.families.links.Log())
    ).fit(cov_type="HC3")

    a = model_a.params[x]
    theta2 = model_b.params[m]
    theta1 = model_b.params[x]
    total_logrr = model_total.params[x]

    indirect_logrr = a * theta2
    direct_rr = np.exp(theta1)
    indirect_rr = np.exp(indirect_logrr)
    total_rr = np.exp(total_logrr)

    result = pd.DataFrame({
        "sample": [sample],
        "mediator": [m],
        "outcome": [y_bin],
        "n": [len(d)],
        "a_path": [a],
        "a_p": [model_a.pvalues[x]],
        "theta2_mediator_logrr": [theta2],
        "theta2_p": [model_b.pvalues[m]],
        "direct_logrr": [theta1],
        "direct_rr": [direct_rr],
        "direct_p": [model_b.pvalues[x]],
        "total_logrr": [total_logrr],
        "total_rr": [total_rr],
        "total_p": [model_total.pvalues[x]],
        "indirect_logrr": [indirect_logrr],
        "indirect_rr": [indirect_rr],
        "r2_mediator_model": [model_a.rsquared],
    })

    return result, model_a, model_total, model_b, d


def bootstrap_poisson_indirect(df, sample, x, m, y_bin, covariates=None, n_boot=3000, seed=2026):
    if covariates is None:
        covariates = []

    rng = np.random.default_rng(seed)
    d = prepare_poisson_mediation_data(df, sample, x, m, y_bin, covariates=covariates)

    n = len(d)
    ab_vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = d.iloc[idx].copy()

        try:
            Xa = sm.add_constant(boot[[x] + covariates])
            ma = sm.OLS(boot[m], Xa).fit()

            Xb = sm.add_constant(boot[[x, m] + covariates])
            mb = sm.GLM(
                boot[y_bin],
                Xb,
                family=sm.families.Poisson(link=sm.families.links.Log())
            ).fit()

            ab_vals.append(ma.params[x] * mb.params[m])
        except Exception:
            continue

    ab_vals = np.array(ab_vals)

    out = pd.DataFrame({
        "sample": [sample],
        "mediator": [m],
        "outcome": [y_bin],
        "indirect_logrr_boot_mean": [ab_vals.mean()],
        "indirect_logrr_ci_low_95": [np.quantile(ab_vals, 0.025)],
        "indirect_logrr_ci_high_95": [np.quantile(ab_vals, 0.975)],
        "indirect_rr_boot_mean": [np.exp(ab_vals.mean())],
        "indirect_rr_ci_low_95": [np.exp(np.quantile(ab_vals, 0.025))],
        "indirect_rr_ci_high_95": [np.exp(np.quantile(ab_vals, 0.975))],
        "prop_logrr_positive": [(ab_vals > 0).mean()],
    })

    return out, ab_vals


def run_poisson_mediations(df, x, mediators, y_bin, sample="Combined", covariates=None, n_boot=3000, seed=2026):
    if covariates is None:
        covariates = []

    rows = []
    for i, m in enumerate(mediators):
        res, _, _, _, _ = fit_poisson_mediation(
            df=df,
            sample=sample,
            x=x,
            m=m,
            y_bin=y_bin,
            covariates=covariates
        )

        boot_res, _ = bootstrap_poisson_indirect(
            df=df,
            sample=sample,
            x=x,
            m=m,
            y_bin=y_bin,
            covariates=covariates,
            n_boot=n_boot,
            seed=seed + i
        )

        rows.append(
            res.merge(boot_res, on=["sample", "mediator", "outcome"], how="left")
        )

    return pd.concat(rows, ignore_index=True)

