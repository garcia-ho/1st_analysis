import importlib.util
from pathlib import Path
from functions_Network import build_network_graph
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import networkx as nx
from statsmodels.stats.anova import anova_lm
from functions_Network import *

import itertools


def summarize_columns(df, cols, round_digits=3):

    out = (
        df[cols]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
        .round(round_digits)
    )
    return out


def merge_and_score_followup_ai(
    baseline_df,
    followup_file_1,
    followup_file_2,
    ai_fa,
    ai_efa_items,
    baseline_id_col="id",
    followup_id_col="Q00_Identification",
    followup_prefix="post_",
    functions_file="functions.py",
    item_cols=None,
    how="inner",
    drop_temp=True
):

   # Merge follow-up data to baseline, preprocess the 10 post AI items,and add ai_factor1_score_post / ai_factor2_score_post.

    # load old helper functions
    spec = importlib.util.spec_from_file_location("old_functions", functions_file)
    old_functions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_functions)

    f1 = pd.read_csv(followup_file_1)
    f2 = pd.read_csv(followup_file_2)
    followup_df = pd.concat([f1, f2], ignore_index=True)

    baseline = baseline_df.copy()
    followup = followup_df.copy()

    baseline[baseline_id_col] = baseline[baseline_id_col].astype(str).str.strip()
    followup[followup_id_col] = followup[followup_id_col].astype(str).str.strip()

    # rename follow-up ID to match baseline
    followup = followup.rename(columns={followup_id_col: baseline_id_col})

    # infer raw follow-up AI item columns before prefixing, if not supplied
    if item_cols is None:
        raw_item_cols = [c for c in followup.columns if c != baseline_id_col][:10]
        if len(raw_item_cols) != 10:
            raise ValueError(
                f"Could not infer 10 follow-up AI item columns; found {len(raw_item_cols)}"
            )
        item_cols = [f"{followup_prefix}{c}" for c in raw_item_cols]

    # prefix all non-ID follow-up columns
    rename_map = {
        c: f"{followup_prefix}{c}"
        for c in followup.columns
        if c != baseline_id_col
    }
    followup = followup.rename(columns=rename_map)

    # merge
    d = baseline.merge(followup, on=baseline_id_col, how=how)


    if len(item_cols) != 10:
        raise ValueError(f"item_cols must contain exactly 10 columns, got {len(item_cols)}")

    parsed_cols = []
    for i, col in enumerate(item_cols, start=1):
        if col not in d.columns:
            raise KeyError(f"Missing follow-up item column: {col}")

        parsed_col = f"followup_ai_item_{i}_num"
        d[col] = d[col].map(old_functions.clean_text)
        d[parsed_col] = d[col].apply(old_functions.extract_leading_code)
        parsed_cols.append(parsed_col)

  
    item_map_post = {
        1: "ai_concept_input_variation_scored_num_post",
        2: "ai_ability_input_sensitivity_scored_num_post",
        3: "ai_concept_blackbox_scored_num_post",
        4: "ai_ability_explainability_scored_num_post",
        5: "ai_concept_data_bias_scored_num_post",
        6: "ai_ability_training_data_scored_num_post",
        7: "ai_concept_prompt_wording_scored_num_post",
        8: "ai_ability_prompting_scored_num_post",
        9: "ai_concept_social_ethics_scored_num_post",
        10: "ai_ability_social_ethics_scored_num_post",
    }

    # reverse-keyed post items from the follow-up questionnaire
    reverse_items = {1, 3, 5, 6, 7, 9}

    for i, parsed_col in enumerate(parsed_cols, start=1):
        vals = d[parsed_col]
        target_col = item_map_post[i]
        d[target_col] = 6 - vals if i in reverse_items else vals


    # add original composite mean scores for pre and post

    conceptual_cols_pre = [
        "ai_concept_data_bias_scored_num",
        "ai_concept_blackbox_scored_num",
        "ai_concept_input_variation_scored_num",
        "ai_concept_prompt_wording_scored_num",
        "ai_concept_social_ethics_scored_num",
    ]

    confidence_cols_pre = [
        "ai_ability_training_data_scored_num",
        "ai_ability_explainability_scored_num",
        "ai_ability_input_sensitivity_scored_num",
        "ai_ability_prompting_scored_num",
        "ai_ability_social_ethics_scored_num",
    ]

    conceptual_cols_post = [f"{c}_post" for c in conceptual_cols_pre]
    confidence_cols_post = [f"{c}_post" for c in confidence_cols_pre]
    missing_pre = [c for c in conceptual_cols_pre + confidence_cols_pre if c not in d.columns]

    if missing_pre:
        raise KeyError(f"Missing baseline scored AI columns needed for pre composites: {missing_pre}")

    missing_post = [c for c in conceptual_cols_post + confidence_cols_post if c not in d.columns]

    if missing_post:
        raise KeyError(f"Missing post scored AI columns needed for post composites: {missing_post}")

    d["ai_conceptual_score_pre"] = d[conceptual_cols_pre].mean(axis=1)
    d["ai_confidence_score_pre"] = d[confidence_cols_pre].mean(axis=1)
    d["ai_lit_score_pre"] = d[conceptual_cols_pre + confidence_cols_pre].mean(axis=1)
    d["ai_conceptual_score_post"] = d[conceptual_cols_post].mean(axis=1)
    d["ai_confidence_score_post"] = d[confidence_cols_post].mean(axis=1)
    d["ai_lit_score_post"] = d[conceptual_cols_post + confidence_cols_post].mean(axis=1)

# AI-factors 1 and 2 for post scores. (Not orignial factor space)
    ai_efa_items_post = [f"{c}_post" for c in ai_efa_items]
    missing = [c for c in ai_efa_items_post if c not in d.columns]
    if missing:
        raise KeyError(f"Missing post scored AI columns: {missing}")

    post_input = d[ai_efa_items_post].dropna().copy()
    post_input_for_transform = post_input.copy()
    post_input_for_transform.columns = ai_efa_items

    scores_post = ai_fa.transform(post_input_for_transform)

    df_scores_post = pd.DataFrame(
        scores_post,
        index=post_input.index,
        columns=["ai_factor1_score_post", "ai_factor2_score_post"]
    )

    d.loc[df_scores_post.index, "ai_factor1_score_post"] = df_scores_post["ai_factor1_score_post"]
    d.loc[df_scores_post.index, "ai_factor2_score_post"] = df_scores_post["ai_factor2_score_post"]

    if drop_temp:
        d = d.drop(columns=parsed_cols, errors="ignore")

    return d


#____________________
# statistical test
#____________________


def paired_pre_post_test(df, pre_col, post_col):
    d = df[[pre_col, post_col]].dropna().copy()
    diff = d[post_col] - d[pre_col]

    # Wilcoxon signed-rank test
    try:
        w_stat, w_p = stats.wilcoxon(d[post_col]-d[pre_col], alternative="greater", method="approx")
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    # effect size for paired data: dz
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan

    out = pd.DataFrame({
        "mean_change_post_minus_pre": [diff.mean()],
        "wilcoxon_p": [w_p]
    })

    return out


def run_ai_pre_post_tests(df):

    res1 = paired_pre_post_test(df, "ai_conceptual_score_pre", "ai_conceptual_score_post")
    res2 = paired_pre_post_test(df, "ai_confidence_score_pre", "ai_confidence_score_post")
    out = pd.concat([res1, res2], ignore_index=True)

    # multiple testing correction (Benjamini-Hochberg)
    reject_w, p_w_adj, _, _ = multipletests(out["wilcoxon_p"], method="holm", alpha=0.05)
    out["wilcoxon_p_adj"] = p_w_adj
    out["wilcoxon_sig_adj"] = reject_w
    out.index = ["Ai_understanding", "Ai_confidence"]

    return out




#____________________
# Rebuild Network analysis
#____________________

# new work between SES1,2, mediators and 4 AI literacy (Pre and post)

node_groups_two_wave = {
    "ses_factor1_score": "SES",
    "ses_factor2_score": "SES",
    "conceptual_exposure_score": "Mediator",
    "practical_ai_use_score": "Mediator",
    "learning_ecology_score": "Mediator",
    "language_load_score": "Mediator",
    "epistemic_stance_score": "Mediator",
    "ai_factor1_score": "AI_pre",
    "ai_factor2_score": "AI_pre",
    "ai_factor1_score_post": "AI_post",
    "ai_factor2_score_post": "AI_post",
}

node_groups_change = {
    "ses_factor1_score": "SES",
    "ses_factor2_score": "SES",
    "conceptual_exposure_score": "Mediator",
    "practical_ai_use_score": "Mediator",
    "learning_ecology_score": "Mediator",
    "language_load_score": "Mediator",
    "epistemic_stance_score": "Mediator",
    "delta_ai_factor1": "AI_change",
    "delta_ai_factor2": "AI_change",

}


def run_two_wave_ai_network(
    df,
    alpha=0.05,
    adjust_method="fdr_bh",
    min_abs_r=0.0,
    node_groups=None
):
    vars_two_wave = [
        "ses_factor1_score",
        "ses_factor2_score",
        "conceptual_exposure_score",
        "practical_ai_use_score",
        "learning_ecology_score",
        "language_load_score",
        "epistemic_stance_score",
        "ai_factor1_score",
        "ai_factor2_score",
        "ai_factor1_score_post",
        "ai_factor2_score_post",
        # "ai_conceptual_score_pre",
        # "ai_conceptual_score_post",
        # "ai_confidence_score_pre",
        # "ai_confidence_score_post"
    ]

    d = df[vars_two_wave].dropna().copy()

    pc_obj = partial_corr_matrix(d, vars_two_wave)
    edges = build_network_edges(
        pc_obj,
        adjust_method=adjust_method,
        alpha=alpha,
        min_abs_r=min_abs_r
    )
    G = build_network_graph(edges, node_groups=node_groups)

    return {
        "data": d,
        "vars": vars_two_wave,
        "pc_obj": pc_obj,
        "edges": edges,
        "graph": G,
    }



def run_change_score_ai_network(
    df,
    alpha=0.05,
    adjust_method="fdr_bh",
    min_abs_r=0.0,
    node_groups=None
):
    d = df.copy()

    d["delta_ai_factor1"] = d["ai_factor1_score_post"] - d["ai_factor1_score"]
    d["delta_ai_factor2"] = d["ai_factor2_score_post"] - d["ai_factor2_score"]

    # d["delta_ai_factor1"] = d["ai_conceptual_score_post"] - d["ai_conceptual_score_pre"]
    # d["delta_ai_factor2"] = d["ai_confidence_score_post"] - d["ai_confidence_score_pre"]
    vars_change = [
        "ses_factor1_score",
        "ses_factor2_score",
        "conceptual_exposure_score",
        "practical_ai_use_score",
        "learning_ecology_score",
        "language_load_score",
        "epistemic_stance_score",
        "delta_ai_factor1",
        "delta_ai_factor2",
    ]

    d_net = d[vars_change].dropna().copy()

    pc_obj = partial_corr_matrix(d_net, vars_change)
    edges = build_network_edges(
        pc_obj,
        adjust_method=adjust_method,
        alpha=alpha,
        min_abs_r=min_abs_r
    )
    G = build_network_graph(edges, node_groups=node_groups)

    return {
        "data": d_net,
        "vars": vars_change,
        "pc_obj": pc_obj,
        "edges": edges,
        "graph": G,
    }




#____________________
# Test AI literacy Gain in different SES groups
#____________________

def compare_gain_by_all_ses_binary(
    df,
    ses_cols,
    pre_post_pairs,
    alpha=0.05,
    low_label="Low SES",
    high_label="High SES"
):

    d = df.copy()

    # 1) gain scores
    for gain_col, (pre_col, post_col) in pre_post_pairs.items():
        d[gain_col] = d[post_col] - d[pre_col]

    gain_cols = list(pre_post_pairs.keys())
    result_rows = []
    desc_rows = []

    for ses_col in ses_cols:
        cutoff = d[ses_col].median()
        group_col = f"{ses_col}_group"
        d[group_col] = np.where(d[ses_col] <= cutoff, low_label, high_label)

        for gain_col in gain_cols:
            sub = d[[ses_col, group_col, gain_col]].dropna().copy()
            low_vals = sub.loc[sub[group_col] == low_label, gain_col]
            high_vals = sub.loc[sub[group_col] == high_label, gain_col]
            # descriptives

            desc = (
                sub.groupby(group_col)[gain_col]
                .agg(["count", "mean", "std"])
                .reset_index()
            )

            desc["ses_col"] = ses_col
            desc["gain"] = gain_col
            desc_rows.append(desc)

            # Welch t-test
            t_stat, p_val = stats.ttest_ind(
                low_vals,
                high_vals,
                equal_var=False,
                nan_policy="omit"
            )

            result_rows.append({
                "ses_col": ses_col,
                "gain": gain_col,
                "test": "Welch t-test",
                "mean_diff": low_vals.mean() - high_vals.mean(),
                "p_value": p_val,

            })

    results = pd.DataFrame(result_rows)

    # 2) Holm correction across all SES × gain tests
    reject, p_adj, _, _ = multipletests(results["p_value"], alpha=alpha, method="holm")
    results["p_value_adj"] = p_adj
    results["sig_adj"] = reject
    descriptives = pd.concat(desc_rows, ignore_index=True)

    return {
        "data": d,
        "results": results,
        "descriptives": descriptives,

    }




#_--------------------
# Test AI literacy Gain in different Mediator groups
#_--------------------

def compare_gain_by_key_mediator_binary(
    df,
    mediator_cols=("learning_ecology_score", "language_load_score"),
    pre_post_pairs=None,
    alpha=0.05,
    low_label="Low",
    high_label="High"
):
  
    if pre_post_pairs is None:
        raise ValueError("pre_post_pairs must be provided.")

    d = df.copy()

    # 1) gain scores
    for gain_col, (pre_col, post_col) in pre_post_pairs.items():
        d[gain_col] = d[post_col] - d[pre_col]

    gain_cols = list(pre_post_pairs.keys())

    result_rows = []
    desc_rows = []

    for med_col in mediator_cols:
        cutoff = d[med_col].median()
        group_col = f"{med_col}_group"

        d[group_col] = np.where(d[med_col] <= cutoff, low_label, high_label)

        for gain_col in gain_cols:
            sub = d[[med_col, group_col, gain_col]].dropna().copy()

            low_vals = sub.loc[sub[group_col] == low_label, gain_col]
            high_vals = sub.loc[sub[group_col] == high_label, gain_col]

            # descriptives
            desc = (
                sub.groupby(group_col)[gain_col]
                .agg(["count", "mean", "std"])
                .reset_index()
            )
            desc["mediator"] = med_col
            desc["gain"] = gain_col
            desc_rows.append(desc)

            # Welch t-test
            t_stat, p_val = stats.ttest_ind(
                low_vals,
                high_vals,
                equal_var=False,
                nan_policy="omit"
            )

            result_rows.append({
                "mediator": med_col,
                "gain": gain_col,
                "test": "Welch t-test",
                "mean_diff": low_vals.mean() - high_vals.mean(),
                "p_value": p_val,
            })

    results = pd.DataFrame(result_rows)

    # 2) Holm correction across all mediator × gain tests
    reject, p_adj, _, _ = multipletests(results["p_value"], alpha=alpha, method="holm")
    results["p_value_adj"] = p_adj
    results["sig_adj"] = reject

    descriptives = pd.concat(desc_rows, ignore_index=True)

    return {
        "data": d,
        "results": results,
        "descriptives": descriptives,
    }




#_--------------------
# Test do 2 Mediators still predict AI understanding in Phase II
#_--------------------


def test_phase1_ai_understanding_mediators_in_phase2(
    df,
    mediators=("language_load_score", "epistemic_stance_score"),
    pre_col="ai_factor1_score",
    post_col="ai_factor1_score_post",
    standardize=True,
    cov_type="HC3",
    alpha=0.05,
    p_adjust_method="holm"

):

    model_rows = []
    coef_rows = []
    models = {}

    for med in mediators:
        needed = [pre_col, post_col, med]
        d = df[needed].dropna().copy()

        if standardize:
            for col in needed:
                d[col] = (d[col] - d[col].mean()) / d[col].std(ddof=0)

        y = d[post_col]
        X = sm.add_constant(d[[pre_col, med]])
        model = sm.OLS(y, X).fit(cov_type=cov_type)
        models[med] = model
        ci = model.conf_int()

        model_rows.append({
            "mediator": med,
            "n": int(model.nobs),
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_pvalue": model.f_pvalue,
        })

        for term in [pre_col, med]:
            coef_rows.append({
                "mediator": med,
                "term": term,
                "beta": model.params[term],
                "p_value": model.pvalues[term]
            })

    model_table = pd.DataFrame(model_rows)
    coef_table = pd.DataFrame(coef_rows)

    # adjust only the mediator terms, not the pre-score term
    mask = coef_table["term"].isin(mediators)

    reject, p_adj, _, _ = multipletests(
        coef_table.loc[mask, "p_value"],
        alpha=alpha,
        method=p_adjust_method

    )

    coef_table["p_value_adj"] = np.nan
    coef_table["sig_adj"] = False
    coef_table.loc[mask, "p_value_adj"] = p_adj
    coef_table.loc[mask, "sig_adj"] = reject

    return {
        "model_table": model_table,
        "coef_table": coef_table,
        "models": models,
    }



#_--------------------
# Visualization
#_--------------------

ses_cols = [
    "ses_parent1_edu_num",
    "ses_parent2_edu_num",
    "ses_household_income_num",
    "ses_financial_constraint_scored_num",
    "ses_home_area_num",
    "ses_housing_type_num",
    "ses_school_type_num",
    "ses_device_access_num",
    "ses_internet_quality_num",
]

reverse_cols = [
    "ses_financial_constraint_scored_num",
    "ses_device_access_num",
    "ses_internet_quality_num",
]


def add_ses_index_mean(df, ses_cols, reverse_cols=None, new_col="ses_index"):

    d = df.copy()
    if reverse_cols is None:
        reverse_cols = []

    for col in reverse_cols:
        if col not in ses_cols:
            continue
        col_min = d[col].min()
        col_max = d[col].max()
        d[col] = col_max + col_min - d[col]
    d[new_col] = d[ses_cols].mean(axis=1)
    return d


def plot_boxplots(df, cols, label_map=None, figsize=(16, 10), rotation=10):
    """
    Colorful box plots for SES / mediator variables.
    """
    d = df[cols].dropna().copy()

    plot_df = d.copy()
    if label_map is not None:
        plot_df = plot_df.rename(columns=label_map)

    data = [plot_df[c].dropna() for c in plot_df.columns]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=plot_df.columns,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )

    colors = plt.cm.Set3(np.linspace(0, 1, len(plot_df.columns)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    ax.set_xticklabels(plot_df.columns, rotation=rotation, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Distribution of variables")

    plt.tight_layout()
    plt.show()




def plot_pre_post_boxplots(df, pairs, label_map=None, figsize=(8, 5)):

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n / 2, figsize[1]))
    if n == 1:
        axes = [axes]
    for ax, (pre_col, post_col) in zip(axes, pairs):
        sub = df[[pre_col, post_col]].dropna()
        cols = [pre_col, post_col]
        labels = [label_map.get(c, c) if label_map else c for c in cols]
        ax.boxplot([sub[pre_col], sub[post_col]], labels=labels)
        ax.set_title(f"{labels[0]} vs {labels[1]}")
        ax.set_ylabel("Score")

    plt.tight_layout()
    plt.show()


ai_label_map = {
    "ai_factor1_score": "AI understanding (pre)",
    "ai_factor1_score_post": "AI understanding (post)",
    "ai_factor2_score": "AI confidence (pre)",
    "ai_factor2_score_post": "AI confidence (post)",
    "ai_lit_score_pre": "AI literacy (pre)",
    "ai_lit_score_post": "AI literacy (post)",
}