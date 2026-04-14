import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import networkx as nx

from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor



# =============================================================================
# 1. Correlation Matrix
# =============================================================================

label_map = {
    "ses_factor1_score": "SES1",
    "ses_factor2_score": "SES2",
    "ses_index": "Overall SES",
    "conceptual_exposure_score": "Conceptual exposure",
    "practical_ai_use_score": "Practical AI use",
    "learning_ecology_score": "Learning ecology",
    "language_load_score": "Language load",
    "epistemic_stance_score": "Epistemic stance",
    "ai_factor1_score": "AI understanding",
    "ai_factor2_score": "AI confidence",
        "ai_lit_score": "AI literacy",
}


def partial_corr_one(df, x, y, controls):
    sub = df[[x, y] + controls].dropna().copy()

    if len(controls) == 0:
        r, p = stats.pearsonr(sub[x], sub[y])
        return r, p, len(sub)

    Xc = sm.add_constant(sub[controls])

    model_x = sm.OLS(sub[x], Xc).fit()
    model_y = sm.OLS(sub[y], Xc).fit()

    resid_x = model_x.resid
    resid_y = model_y.resid

    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(sub)

def partial_corr_matrix(df, vars):
    dat = df[vars].dropna().copy()

    p = len(vars)
    est = pd.DataFrame(np.eye(p), index=vars, columns=vars, dtype=float)
    pval = pd.DataFrame(np.zeros((p, p)), index=vars, columns=vars, dtype=float)
    nmat = pd.DataFrame(np.full((p, p), len(dat)), index=vars, columns=vars, dtype=float)

    for i, x in enumerate(vars):
        for j, y in enumerate(vars):
            if i >= j:
                continue

            controls = [v for v in vars if v not in [x, y]]
            r, p, n = partial_corr_one(dat, x, y, controls)

            est.loc[x, y] = r
            est.loc[y, x] = r
            pval.loc[x, y] = p
            pval.loc[y, x] = p
            nmat.loc[x, y] = n
            nmat.loc[y, x] = n

    return {
        "estimate": est,
        "p_value": pval,
        "n": nmat,
        "data": dat
    }

def partial_corr_table(pc_obj, upper_only=True):

    est = pc_obj["estimate"]
    pval = pc_obj["p_value"]
    nmat = pc_obj["n"]

    rows = []
    vars = list(est.index)

    for i, v1 in enumerate(vars):
        for j, v2 in enumerate(vars):
            if upper_only and i >= j:
                continue
            if not upper_only and i == j:
                continue

            rows.append({
                "var1": v1,
                "var2": v2,
                "partial_r": est.loc[v1, v2],
                "p_value": pval.loc[v1, v2],
                "n": nmat.loc[v1, v2],
            })

    return pd.DataFrame(rows)

# multiple testing correction
def adjust_partial_corr_p(pc_table, method="fdr_bh", alpha=0.05):
    out = pc_table.copy()

    reject, p_adj, _, _ = multipletests(out["p_value"], alpha=alpha, method=method)

    out["p_adj"] = p_adj
    out["sig"] = reject
    return out


# =============================================================================
# 2. Network Analysis
# =============================================================================
# mapping for network
node_groups = {
    "SES1": "SES",
    "SES2": "SES",
    "Conceptual exposure": "Mediator",
    "Practical AI use": "Mediator",
    "Learning ecology": "Mediator",
    "Language load": "Mediator",
    "Epistemic stance": "Mediator",
    "AI understanding": "AI",
    "AI confidence": "AI",
}

def build_network_edges(pc_obj, adjust_method="fdr_bh", alpha=0.05, min_abs_r=0.0):
    tbl = partial_corr_table(pc_obj, upper_only=True).copy()

    reject, p_adj, _, _ = multipletests(tbl["p_value"], alpha=alpha, method=adjust_method)
    tbl["p_adj"] = p_adj
    tbl["sig"] = reject

    edges = tbl[
        (tbl["sig"]) & (tbl["partial_r"].abs() >= min_abs_r)
    ].copy()

    edges["weight"] = edges["partial_r"].abs()
    edges["sign"] = np.where(edges["partial_r"] >= 0, "positive", "negative")

    return edges.sort_values("weight", ascending=False).reset_index(drop=True)

def build_network_graph(edges, node_groups=None):
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(
            row["var1"],
            row["var2"],
            partial_r=row["partial_r"],
            weight=row["weight"],
            sign=row["sign"],
            p_adj=row["p_adj"]
        )

    if node_groups is not None:
        for node, group in node_groups.items():
            if node in G.nodes:
                G.nodes[node]["group"] = group

    return G


def network_centrality_table(G):
    if len(G.nodes) == 0:
        return pd.DataFrame(columns=["node", "degree", "strength", "betweenness"])

    degree = dict(G.degree())
    strength = {
        n: sum(abs(G[n][nbr]["partial_r"]) for nbr in G.neighbors(n))
        for n in G.nodes()
    }
    betweenness = nx.betweenness_centrality(G, weight="weight")

    out = pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [degree[n] for n in G.nodes()],
        "strength": [strength[n] for n in G.nodes()],
        "betweenness": [betweenness[n] for n in G.nodes()],
    })

    return out.sort_values(["strength", "degree"], ascending=False).reset_index(drop=True)


def network_adjacency_matrix(G):
    nodes = list(G.nodes())
    mat = pd.DataFrame(0.0, index=nodes, columns=nodes)

    for u, v, data in G.edges(data=True):
        mat.loc[u, v] = data["partial_r"]
        mat.loc[v, u] = data["partial_r"]

    return mat


# =============================================================================
# 3. Hierarchical Regression
# =============================================================================


blocks_with_interactions = {
    "M1_SES": [
        "ses_factor1_score",
        "ses_factor2_score",
    ],
    "M2_SES_plus_mediators": [
        "ses_factor1_score",
        "ses_factor2_score",
        "conceptual_exposure_score",
        "practical_ai_use_score",
        "learning_ecology_score",
        "language_load_score",
        "epistemic_stance_score",
    ],
    "M3_add_interactions": [
        "ses_factor1_score",
        "ses_factor2_score",
        "conceptual_exposure_score",
        "practical_ai_use_score",
        "learning_ecology_score",
        "language_load_score",
        "epistemic_stance_score",
        "learning_x_epistemic",
        "language_x_epistemic",
    ],
}


def fit_hierarchical_models(
    df,
    outcome,
    blocks,
    standardize=True,
    cov_type="HC3"
):
    
    all_vars = [outcome]
    for cols in blocks.values():
        all_vars.extend(cols)
    all_vars = list(dict.fromkeys(all_vars))

    d = df[all_vars].dropna().copy()

    if standardize:
        for col in all_vars:
            d[col] = (d[col] - d[col].mean()) / d[col].std(ddof=0)

    fitted = {}
    model_rows = []
    coef_rows = []

    y = d[outcome]

    for model_name, xcols in blocks.items():
        X = sm.add_constant(d[xcols])
        model = sm.OLS(y, X).fit(cov_type=cov_type)
        fitted[model_name] = model

        model_rows.append({
            "outcome": outcome,
            "model": model_name,
            "n": int(model.nobs),
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "aic": model.aic,
            "bic": model.bic,
            "f_pvalue": model.f_pvalue,
        })

        ci = model.conf_int()
        for term in model.params.index:
            if term == "const":
                continue
            coef_rows.append({
                "outcome": outcome,
                "model": model_name,
                "term": term,
                "beta": model.params[term],
                "p_value": model.pvalues[term],
                "ci_low_95": ci.loc[term, 0],
                "ci_high_95": ci.loc[term, 1],
            })

    model_table = pd.DataFrame(model_rows)
    coef_table = pd.DataFrame(coef_rows)

    return {
        "data": d,
        "models": fitted,
        "model_table": model_table,
        "coef_table": coef_table,
    }




def hierarchical_model_comparison(fit_obj):
    model_names = list(fit_obj["models"].keys())
    rows = []

    for i in range(len(model_names) - 1):
        m_small_name = model_names[i]
        m_big_name = model_names[i + 1]

        m_small = fit_obj["models"][m_small_name]
        m_big = fit_obj["models"][m_big_name]

        comp = anova_lm(m_small, m_big)
        p_change = comp["Pr(>F)"].iloc[1]
        df_diff = comp["df_diff"].iloc[1]
        ss_diff = comp["ss_diff"].iloc[1]

        rows.append({
            "outcome": fit_obj["model_table"]["outcome"].iloc[0],
            "from_model": m_small_name,
            "to_model": m_big_name,
            "r2_from": m_small.rsquared,
            "r2_to": m_big.rsquared,
            "delta_r2": m_big.rsquared - m_small.rsquared,
            "df_diff": df_diff,
            "ss_diff": ss_diff,
            "p_change": p_change,
        })

    return pd.DataFrame(rows)


def adjust_hierarchical_results(model_comp_table, coef_table,
                                block_method="holm",
                                coef_method="fdr_bh",
                                alpha=0.05):
    comp = model_comp_table.copy()
    coef = coef_table.copy()

    if not comp.empty:
        reject_block, p_block_adj, _, _ = multipletests(
            comp["p_change"], alpha=alpha, method=block_method
        )
        comp["p_change_adj"] = p_block_adj
        comp["sig_block_adj"] = reject_block

    if not coef.empty:
        coef["p_value_adj"] = np.nan
        coef["sig_adj"] = False

        for outcome in coef["outcome"].unique():
            for model in coef.loc[coef["outcome"] == outcome, "model"].unique():
                idx = coef.index[(coef["outcome"] == outcome) & (coef["model"] == model)]
                reject_coef, p_coef_adj, _, _ = multipletests(
                    coef.loc[idx, "p_value"], alpha=alpha, method=coef_method
                )
                coef.loc[idx, "p_value_adj"] = p_coef_adj
                coef.loc[idx, "sig_adj"] = reject_coef

    return comp, coef


def run_hierarchical_regression(
    df,
    outcomes,
    blocks,
    standardize=True,
    cov_type="HC3",
    block_method="holm",
    coef_method="fdr_bh",
    alpha=0.05
):
    all_model_tables = []
    all_comp_tables = []
    all_coef_tables = []
    fitted_by_outcome = {}

    for outcome in outcomes:
        fit_obj = fit_hierarchical_models(
            df=df,
            outcome=outcome,
            blocks=blocks,
            standardize=standardize,
            cov_type=cov_type
        )
        comp_table = hierarchical_model_comparison(fit_obj)
        comp_table, coef_table = adjust_hierarchical_results(
            comp_table,
            fit_obj["coef_table"],
            block_method=block_method,
            coef_method=coef_method,
            alpha=alpha
        )

        all_model_tables.append(fit_obj["model_table"])
        all_comp_tables.append(comp_table)
        all_coef_tables.append(coef_table)
        fitted_by_outcome[outcome] = fit_obj["models"]

    return {
        "model_table": pd.concat(all_model_tables, ignore_index=True),
        "comparison_table": pd.concat(all_comp_tables, ignore_index=True),
        "coef_table": pd.concat(all_coef_tables, ignore_index=True),
        "models": fitted_by_outcome,
    }



# =============================================================================
# 4. VIF Colinearity Check
# =============================================================================


def get_locked_blocks(outcome):
    base_predictors = [
        "ses_factor1_score",
        "ses_factor2_score",
    ]

    mediator_predictors = base_predictors + [
        "conceptual_exposure_score",
        "practical_ai_use_score",
        "learning_ecology_score",
        "language_load_score",
        "epistemic_stance_score",
    ]

    interaction_predictors = mediator_predictors + [
        "learning_x_epistemic",
        "language_x_epistemic",
    ]

    if outcome == "ai_factor1_score":
        return {
            "M1_SES": base_predictors,
            "M2_SES_plus_mediators": mediator_predictors,
        }

    elif outcome == "ai_factor2_score":
        return {
            "M1_SES": base_predictors,
            "M2_SES_plus_mediators": mediator_predictors,
            "M3_add_interactions": interaction_predictors,
        }

    else:
        raise ValueError("outcome must be 'ai_factor1_score' or 'ai_factor2_score'")
    
def add_final_interaction_terms(df):
    d = df.copy()
    d["learning_x_epistemic"] = d["learning_ecology_score"] * d["epistemic_stance_score"]
    d["language_x_epistemic"] = d["language_load_score"] * d["epistemic_stance_score"]
    return d

def run_locked_final_models(df, alpha=0.05):
    d = add_final_interaction_terms(df).copy()

    out_understanding = run_hierarchical_regression(
        df=d,
        outcomes=["ai_factor1_score"],
        blocks=get_locked_blocks("ai_factor1_score"),
        standardize=True,
        cov_type="HC3",
        block_method="holm",
        coef_method="fdr_bh",
        alpha=alpha
    )

    out_confidence = run_hierarchical_regression(
        df=d,
        outcomes=["ai_factor2_score"],
        blocks=get_locked_blocks("ai_factor2_score"),
        standardize=True,
        cov_type="HC3",
        block_method="holm",
        coef_method="fdr_bh",
        alpha=alpha
    )

    return {
        "ai_understanding": out_understanding,
        "ai_confidence": out_confidence,
    }

def get_final_model_name(outcome):
    return "M2_SES_plus_mediators" if outcome == "ai_factor1_score" else "M3_add_interactions"


def vif_table_from_locked_model(locked_obj, outcome, model_name):
    model = locked_obj["models"][outcome][model_name]
    exog = pd.DataFrame(model.model.exog, columns=model.model.exog_names)

    if "const" in exog.columns:
        exog = exog.drop(columns=["const"])

    rows = []
    for i, col in enumerate(exog.columns):
        rows.append({
            "term": col,
            "vif": variance_inflation_factor(exog.values, i)
        })

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)

def get_final_model_name(outcome):
    return "M2_SES_plus_mediators" if outcome == "ai_factor1_score" else "M3_add_interactions"






# =============================================================================
#  Visualization
# =============================================================================

def plot_partial_corr_heatmap(pc_obj, adjust_method="fdr_bh", alpha=0.05, sig_only=False, figsize=(10, 8)):
    est = pc_obj["estimate"].copy()
    tbl = partial_corr_table(pc_obj, upper_only=False)
    tbl = adjust_partial_corr_p(tbl, method=adjust_method, alpha=alpha)

    if sig_only:
        keep = tbl[tbl["sig"]][["var1", "var2"]].copy()
        keep["flag"] = 1

        flag_mat = pd.DataFrame(0, index=est.index, columns=est.columns)
        for _, row in keep.iterrows():
            flag_mat.loc[row["var1"], row["var2"]] = 1
            flag_mat.loc[row["var2"], row["var1"]] = 1

        for i in est.index:
            flag_mat.loc[i, i] = 1

        est = est.where(flag_mat == 1)

    plt.figure(figsize=figsize)
    sns.heatmap(
        est,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )
    plt.title(
        f"Partial correlation matrix"
        + (f" ({adjust_method}, significant only)" if sig_only else "")
    )
    plt.tight_layout()
    plt.show()


def plot_network_graph(G, layout="spring", figsize=(10, 8), seed=2026):
    if len(G.nodes) == 0:
        print("No edges in graph.")
        return

    plt.figure(figsize=figsize)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=1.0)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed, k=1.0)

    edge_colors = [
        "#1f77b4" if G[u][v]["sign"] == "positive" else "#d62728"
        for u, v in G.edges()
    ]
    edge_widths = [
        1.5 + 6 * G[u][v]["weight"]
        for u, v in G.edges()
    ]

    # node colors by group if available
    default_color = "#cccccc"
    group_color_map = {
        "SES": "#f4a261",
        "Mediator": "#2a9d8f",
        "AI": "#e9c46a",
    }
    node_colors = []
    for n in G.nodes():
        grp = G.nodes[n].get("group", None)
        node_colors.append(group_color_map.get(grp, default_color))

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=3000,
        edgecolors="black",
        linewidths=1.5
    )
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=20
    )

    plt.title("Partial-correlation network")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_interaction_effect( df, outcome, focal, moderator, controls=None, 
                            use_observed_levels=True, n_points=100, title=None):

    if controls is None:
        controls = []

    needed = [outcome, focal, moderator] + controls
    d = df[needed].dropna().copy()

    # standardize 
    for col in needed:
        d[col] = (d[col] - d[col].mean()) / d[col].std(ddof=0)

    d["interaction_term"] = d[focal] * d[moderator]

    X_cols = [focal, moderator, "interaction_term"] + controls
    X = sm.add_constant(d[X_cols])
    y = d[outcome]

    model = sm.OLS(y, X).fit(cov_type="HC3")

    focal_grid = np.linspace(
        d[focal].quantile(0.05),
        d[focal].quantile(0.95),
        n_points
    )

    # choose moderator levels from actual observed standardized values
    uniq = np.sort(d[moderator].unique())

    if use_observed_levels:
        if len(uniq) < 3:
            raise ValueError(
                f"{moderator} has fewer than 3 distinct observed values after standardization."
            )

        low_idx = max(0, int(np.floor(0.10 * (len(uniq) - 1))))
        mid_idx = int(np.floor(0.50 * (len(uniq) - 1)))
        high_idx = min(len(uniq) - 1, int(np.floor(0.90 * (len(uniq) - 1))))

        # make sure the three levels are distinct
        chosen = [uniq[low_idx], uniq[mid_idx], uniq[high_idx]]
        chosen = sorted(pd.unique(chosen))

        if len(chosen) < 3:
            chosen = [uniq[0], uniq[len(uniq) // 2], uniq[-1]]

        mod_low, mod_mid, mod_high = chosen[0], chosen[1], chosen[2]
    else:
        mod_low, mod_mid, mod_high = -1.0, 0.0, 1.0

    print(f"{moderator} observed levels used:")
    print("Low  =", round(float(mod_low), 3))
    print("Mid  =", round(float(mod_mid), 3))
    print("High =", round(float(mod_high), 3))

    pred_rows = []
    for label, mod_val in [("Low", mod_low), ("Mid", mod_mid), ("High", mod_high)]:
        tmp = pd.DataFrame({
            focal: focal_grid,
            moderator: mod_val,
            "interaction_term": focal_grid * mod_val,
        })

        for c in controls:
            tmp[c] = 0.0

        tmp = sm.add_constant(tmp, has_constant="add")
        pred = model.predict(tmp[X.columns])

        pred_rows.append(pd.DataFrame({
            focal: focal_grid,
            "predicted": pred,
            "moderator_level": label
        }))

    pred_df = pd.concat(pred_rows, ignore_index=True)

    plt.figure(figsize=(12, 7))

    style_map = {
        "Low": "--",
        "Mid": "-",
        "High": ":"
    }

    for label in ["Low", "Mid", "High"]:
        sub = pred_df[pred_df["moderator_level"] == label]
        plt.plot(
            sub[focal],
            sub["predicted"],
            linestyle=style_map[label],
            linewidth=2.5,
            label=label
        )

    plt.xlabel(focal)
    plt.ylabel(outcome)
    plt.title(title if title is not None else f"{moderator} × {focal} on {outcome}")
    plt.legend(title=moderator)
    plt.tight_layout()
    plt.show()

    return model, pred_df


label_map = {
    "ai_factor2_score": "AI confidence",
    "learning_ecology_score": "Learning ecology",
    "language_load_score": "Language load",
    "epistemic_stance_score": "Epistemic stance",
}


def plot_locked_ai_confidence_interactions(df):
    # Plot 1: learning ecology × epistemic stance
    model1, pred1 = plot_interaction_effect(
        df=df,
        outcome="ai_factor2_score",
        focal="learning_ecology_score",
        moderator="epistemic_stance_score",
        controls=[
            "ses_factor1_score",
            "ses_factor2_score",
            "conceptual_exposure_score",
            "practical_ai_use_score",
            "language_load_score",
        ],
        title="Learning ecology × Epistemic stance on AI confidence"
    )

    # Plot 2: language load × epistemic stance
    model2, pred2 = plot_interaction_effect(
        df=df,
        outcome="ai_factor2_score",
        focal="language_load_score",
        moderator="epistemic_stance_score",
        controls=[
            "ses_factor1_score",
            "ses_factor2_score",
            "conceptual_exposure_score",
            "practical_ai_use_score",
            "learning_ecology_score",
        ],
        title="Language load × Epistemic stance on AI confidence"
    )

    return {
        "learning_x_epistemic_model": model1,
        "learning_x_epistemic_pred": pred1,
        "language_x_epistemic_model": model2,
        "language_x_epistemic_pred": pred2,
    }