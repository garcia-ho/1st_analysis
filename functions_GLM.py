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




# =============================================================================
# 2. Multiple mediators with Poisson outcome model
# =============================================================================

def prepare_multiple_poisson_mediation_data(
    df, sample, x, mediators, y_bin, covariates=None, standardize=True
):
    if covariates is None:
        covariates = []

    needed = [x] + mediators + [y_bin] + covariates

    if sample == "Combined":
        d = df[needed].dropna().copy()
    else:
        d = df.loc[df["course"] == sample, needed].dropna().copy()

    if standardize:
        d[x] = (d[x] - d[x].mean()) / d[x].std(ddof=0)
        for m in mediators:
            d[m] = (d[m] - d[m].mean()) / d[m].std(ddof=0)

    return d


def fit_multiple_poisson_mediation(df, sample, x, mediators, y_bin, covariates=None):
    if covariates is None:
        covariates = []

    d = prepare_multiple_poisson_mediation_data(
        df=df,
        sample=sample,
        x=x,
        mediators=mediators,
        y_bin=y_bin,
        covariates=covariates
    )

    # total effect model: Y ~ X + C
    Xt = sm.add_constant(d[[x] + covariates])
    model_total = sm.GLM(
        d[y_bin],
        Xt,
        family=sm.families.Poisson(link=sm.families.links.Log())
    ).fit(cov_type="HC3")

    # mediator models: M_j ~ X + C
    mediator_models = {}
    a_paths = {}
    a_ps = {}

    for m in mediators:
        Xm = sm.add_constant(d[[x] + covariates])
        mm = sm.OLS(d[m], Xm).fit(cov_type="HC3")
        mediator_models[m] = mm
        a_paths[m] = mm.params[x]
        a_ps[m] = mm.pvalues[x]

    # outcome model: Y ~ X + all mediators + C
    Xy = sm.add_constant(d[[x] + mediators + covariates])
    model_y = sm.GLM(
        d[y_bin],
        Xy,
        family=sm.families.Poisson(link=sm.families.links.Log())
    ).fit(cov_type="HC3")

    theta1 = model_y.params[x]
    theta1_p = model_y.pvalues[x]
    total_logrr = model_total.params[x]
    total_p = model_total.pvalues[x]

    b_paths = {m: model_y.params[m] for m in mediators}
    b_ps = {m: model_y.pvalues[m] for m in mediators}

    indirect_components_logrr = {m: a_paths[m] * b_paths[m] for m in mediators}
    joint_indirect_logrr = sum(indirect_components_logrr.values())

    summary = pd.DataFrame({
        "sample": [sample],
        "outcome": [y_bin],
        "n": [len(d)],
        "total_logrr": [total_logrr],
        "total_rr": [np.exp(total_logrr)],
        "total_p": [total_p],
        "direct_logrr": [theta1],
        "direct_rr": [np.exp(theta1)],
        "direct_p": [theta1_p],
        "joint_indirect_logrr": [joint_indirect_logrr],
        "joint_indirect_rr": [np.exp(joint_indirect_logrr)],
    })

    component_table = pd.DataFrame({
        "mediator": mediators,
        "a_path": [a_paths[m] for m in mediators],
        "a_p": [a_ps[m] for m in mediators],
        "theta2_mediator_logrr": [b_paths[m] for m in mediators],
        "theta2_p": [b_ps[m] for m in mediators],
        "indirect_component_logrr": [indirect_components_logrr[m] for m in mediators],
        "indirect_component_rr": [np.exp(indirect_components_logrr[m]) for m in mediators],
    })

    return summary, component_table, mediator_models, model_total, model_y, d



def bootstrap_multiple_poisson_mediation(
    df, sample, x, mediators, y_bin, covariates=None, n_boot=3000, seed=2026
):
    if covariates is None:
        covariates = []

    rng = np.random.default_rng(seed)

    d = prepare_multiple_poisson_mediation_data(
        df=df,
        sample=sample,
        x=x,
        mediators=mediators,
        y_bin=y_bin,
        covariates=covariates
    )

    n = len(d)
    joint_vals = []
    comp_rows = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = d.iloc[idx].copy()

        try:
            # mediator models
            a_vals = {}
            for m in mediators:
                Xm = sm.add_constant(boot[[x] + covariates])
                mm = sm.OLS(boot[m], Xm).fit()
                a_vals[m] = mm.params[x]

            # outcome model
            Xy = sm.add_constant(boot[[x] + mediators + covariates])
            my = sm.GLM(
                boot[y_bin],
                Xy,
                family=sm.families.Poisson(link=sm.families.links.Log())
            ).fit()

            b_vals = {m: my.params[m] for m in mediators}

            comp = {m: a_vals[m] * b_vals[m] for m in mediators}
            joint = sum(comp.values())

            joint_vals.append(joint)
            comp_rows.append(comp)

        except Exception:
            continue

    joint_vals = np.array(joint_vals)
    comp_df = pd.DataFrame(comp_rows)

    joint_summary = pd.DataFrame({
        "sample": [sample],
        "outcome": [y_bin],
        "joint_indirect_logrr_boot_mean": [joint_vals.mean()],
        "joint_indirect_logrr_ci_low_95": [np.quantile(joint_vals, 0.025)],
        "joint_indirect_logrr_ci_high_95": [np.quantile(joint_vals, 0.975)],
        "joint_indirect_rr_boot_mean": [np.exp(joint_vals.mean())],
        "joint_indirect_rr_ci_low_95": [np.exp(np.quantile(joint_vals, 0.025))],
        "joint_indirect_rr_ci_high_95": [np.exp(np.quantile(joint_vals, 0.975))],
        "prop_joint_logrr_positive": [(joint_vals > 0).mean()],
    })

    comp_summary = pd.DataFrame({
        "mediator": mediators,
        "indirect_component_logrr_boot_mean": [comp_df[m].mean() for m in mediators],
        "indirect_component_logrr_ci_low_95": [comp_df[m].quantile(0.025) for m in mediators],
        "indirect_component_logrr_ci_high_95": [comp_df[m].quantile(0.975) for m in mediators],
        "indirect_component_rr_boot_mean": [np.exp(comp_df[m].mean()) for m in mediators],
        "indirect_component_rr_ci_low_95": [np.exp(comp_df[m].quantile(0.025)) for m in mediators],
        "indirect_component_rr_ci_high_95": [np.exp(comp_df[m].quantile(0.975)) for m in mediators],
        "prop_component_logrr_positive": [(comp_df[m] > 0).mean() for m in mediators],
    })

    return joint_summary, comp_summary, joint_vals, comp_df


def run_multiple_poisson_mediator_model(
    df, x, mediators, y_bin, sample="Combined", covariates=None, n_boot=3000, seed=2026
):
    if covariates is None:
        covariates = []

    summary, component_table, mediator_models, model_total, model_y, d = fit_multiple_poisson_mediation(
        df=df,
        sample=sample,
        x=x,
        mediators=mediators,
        y_bin=y_bin,
        covariates=covariates
    )

    boot_joint, boot_components, joint_vals, comp_df = bootstrap_multiple_poisson_mediation(
        df=df,
        sample=sample,
        x=x,
        mediators=mediators,
        y_bin=y_bin,
        covariates=covariates,
        n_boot=n_boot,
        seed=seed
    )

    full_summary = summary.merge(
        boot_joint,
        on=["sample", "outcome"],
        how="left"
    )

    full_components = component_table.merge(
        boot_components,
        on="mediator",
        how="left"
    )

    return full_summary, full_components, mediator_models, model_total, model_y, d





# =============================================================================
# 3. Add AI course background variable
# =============================================================================



def add_ai_course_background(df, source_col="ai_course_taken_num", new_col="ai_course_yes"):
    d = df.copy()
    d[new_col] = np.where(
        d[source_col] == 1, 1,
        np.where(d[source_col] == 0, 0, np.nan)
    )
    return d


def vanderweele_bias_factor(gamma, lam):
    """
    Bias factor for unmeasured mediator-outcome confounding:
        B = gamma * lambda / (gamma + lambda - 1)
    gamma: max RR relating U to Y within strata of A, M, C
    lam:   max RR relating U to A within strata of M, C
    """
    gamma = np.asarray(gamma, dtype=float)
    lam = np.asarray(lam, dtype=float)
    return (gamma * lam) / (gamma + lam - 1)


def correct_indirect_rr(observed_rr, ci_low, ci_high, gamma, lam):
    """
    VanderWeele correction for an observed indirect RR.
    For binary outcomes, corrected indirect effect = observed * B
    """
    B = vanderweele_bias_factor(gamma, lam)
    return {
        "bias_factor": B,
        "corrected_rr": observed_rr * B,
        "corrected_ci_low": ci_low * B,
        "corrected_ci_high": ci_high * B,
    }


def sensitivity_grid_indirect_rr(
    observed_rr,
    ci_low,
    ci_high,
    gamma_values,
    lambda_values
):
    rows = []
    for lam in lambda_values:
        for gamma in gamma_values:
            out = correct_indirect_rr(
                observed_rr=observed_rr,
                ci_low=ci_low,
                ci_high=ci_high,
                gamma=gamma,
                lam=lam
            )
            rows.append({
                "gamma": gamma,
                "lambda": lam,
                "bias_factor": out["bias_factor"],
                "corrected_rr": out["corrected_rr"],
                "corrected_ci_low": out["corrected_ci_low"],
                "corrected_ci_high": out["corrected_ci_high"],
            })
    return pd.DataFrame(rows)




def sensitivity_grid_direct_rr(
    observed_rr,
    ci_low,
    ci_high,
    gamma_values,
    lambda_values
):
    rows = []
    for lam in lambda_values:
        for gamma in gamma_values:
            out = correct_direct_rr(
                observed_rr=observed_rr,
                ci_low=ci_low,
                ci_high=ci_high,
                gamma=gamma,
                lam=lam
            )
            rows.append({
                "gamma": gamma,
                "lambda": lam,
                "bias_factor": out["bias_factor"],
                "corrected_rr": out["corrected_rr"],
                "corrected_ci_low": out["corrected_ci_low"],
                "corrected_ci_high": out["corrected_ci_high"],
            })
    return pd.DataFrame(rows)




def summarize_ai_course_background(df, course_col="ai_course_yes"):
    s = df[course_col].dropna()
    return pd.DataFrame({
        "n": [len(s)],
        "n_yes": [(s == 1).sum()],
        "n_no": [(s == 0).sum()],
        "prop_yes": [(s == 1).mean()],
        "prop_no": [(s == 0).mean()],
    })





