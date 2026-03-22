"""
functions.py — Utility functions for Phase 1 AI Literacy analysis.

Sections
--------
1. Constants / Config       – rename_map, item_labels
2. Text Cleaning            – clean_text, extract_leading_code
3. Data Loading & Scoring   – add_combined_panel, prepare_dataset
4. Statistical Utilities    – cronbach_alpha, cliffs_delta
5. Reporting Tables         – construct_summary_table, reliability_table, effect_size_table
"""

import re

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")
from scipy.stats import skew, mannwhitneyu, spearmanr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo




# =============================================================================
# 1. Constants / Config
# =============================================================================

rename_map = {
    "Q01_Identification": "id",

    # Section A: SES
    "Q02_Socioeconomic Background 1": "ses_parent1_edu",
    "Q03_Socioeconomic Background 2": "ses_parent2_edu",
    "Q04_Socioeconomic Background 3": "ses_household_income",
    "Q05_Socioeconomic Background 4": "ses_school_type",
    "Q06_Socioeconomic Background 5": "ses_housing_type",
    "Q07_Socioeconomic Background 6": "ses_household_size",
    "Q08_Socioeconomic Background 7": "ses_home_area",
    "Q09_Socioeconomic Background 8": "ses_device_access",
    "Q10_Socioeconomic Background 9": "ses_internet_quality",
    "Q11_Socioeconomic Background 10->Cantonese": "ses_lang_cantonese",
    "Q11_Socioeconomic Background 10->Mandarin": "ses_lang_mandarin",
    "Q11_Socioeconomic Background 10->English": "ses_lang_english",
    "Q11_Socioeconomic Background 10->Language(s) not mentioned above": "ses_lang_other",
    "Q12_Socioeconomic Background 11": "ses_financial_constraint",

    # Section B: AI literacy
    "Q13_AI Literacy 1": "ai_course_taken",
    "Q14_AI Literacy 2": "ai_concept_data_bias_r",
    "Q15_AI Literacy 3": "ai_ability_training_data",
    "Q16_AI Literacy 4": "ai_concept_blackbox_r",
    "Q17_AI Literacy 5": "ai_ability_explainability",
    "Q18_AI Literacy 6": "ai_concept_input_variation_r",
    "Q19_AI Literacy 7": "ai_ability_input_sensitivity",
    "Q20_AI Literacy 8": "ai_concept_prompt_wording_r",
    "Q21_AI Literacy 9": "ai_ability_prompting",
    "Q22_AI Literacy 10": "ai_concept_social_ethics_r",
    "Q23_AI Literacy 11": "ai_ability_social_ethics",

    # Section C: Mediators
    "Q24_Mediators 1": "med_conceptual_exposure_1",
    "Q25_Mediators 2": "med_conceptual_exposure_2",
    "Q26_Mediators 3": "med_practical_ai_use_1",
    "Q27_Mediators 4": "med_practical_ai_use_2",
    "Q28_Mediators 5": "med_learning_ecology_1",
    "Q29_Mediator 6": "med_learning_ecology_2",
    "Q30_Mediators 7": "med_language_load_1",
    "Q31_Mediators 8": "med_language_load_2",
    "Q32_Mediators 9": "med_epistemic_stance_1",
    "Q33_Mediators 10": "med_epistemic_stance_2",
}

item_labels = {
    "ses_parent1_edu_num": "Parent 1 edu",
    "ses_parent2_edu_num": "Parent 2 edu",
    "ses_household_income_num": "Household income",
    "ses_home_area_num": "Home area",
    "ses_device_access_scored_num": "Device access",
    "ses_internet_quality_scored_num": "Internet quality",
    "ses_financial_constraint_scored_num": "Low financial strain",

    "ai_concept_data_bias_scored_num": "Bias in data",
    "ai_concept_blackbox_scored_num": "Black-box nature",
    "ai_concept_input_variation_scored_num": "Input sensitivity",
    "ai_concept_prompt_wording_scored_num": "Prompt wording matters",
    "ai_concept_social_ethics_scored_num": "Social/ethics limits",
    "ai_ability_training_data_scored_num": "Explain training data",
    "ai_ability_explainability_scored_num": "Explainability",
    "ai_ability_input_sensitivity_scored_num": "Input sensitivity ability",
    "ai_ability_prompting_scored_num": "Prompting ability",
    "ai_ability_social_ethics_scored_num": "Social/ethics ability",
}

ses_predictors = [
    "ses_parent1_edu_num",
    "ses_parent2_edu_num",
    "ses_household_income_num",
    "ses_home_area_num",
    "ses_device_access_scored_num",
    "ses_internet_quality_scored_num",
    "ses_financial_constraint_scored_num",
]

def get_item_labels():
    return {
        "ses_parent1_edu_num": "Parent 1 edu",
        "ses_parent2_edu_num": "Parent 2 edu",
        "ses_household_income_num": "Household income",
        "ses_home_area_num": "Home area",
        "ses_device_access_scored_num": "Device access",
        "ses_internet_quality_scored_num": "Internet quality",
        "ses_financial_constraint_scored_num": "Low financial strain",

        "ai_concept_data_bias_scored_num": "Bias in data",
        "ai_concept_blackbox_scored_num": "Black-box nature",
        "ai_concept_input_variation_scored_num": "Input sensitivity",
        "ai_concept_prompt_wording_scored_num": "Prompt wording matters",
        "ai_concept_social_ethics_scored_num": "Social/ethics limits",
        "ai_ability_training_data_scored_num": "Explain training data",
        "ai_ability_explainability_scored_num": "Explainability",
        "ai_ability_input_sensitivity_scored_num": "Input sensitivity ability",
        "ai_ability_prompting_scored_num": "Prompting ability",
        "ai_ability_social_ethics_scored_num": "Social/ethics ability",
    }

mediator_map = {
    "conceptual_exposure": [
        "med_conceptual_exposure_1_num",
        "med_conceptual_exposure_2_num",
    ],
    "practical_ai_use": [
        "med_practical_ai_use_1_num",
        "med_practical_ai_use_2_num",
    ],
    "learning_ecology": [
        "med_learning_ecology_1_num",
        "med_learning_ecology_2_num",
    ],
    "language_load": [
        "med_language_load_1_num",
        "med_language_load_2_num",
    ],
    "epistemic_stance": [
        "med_epistemic_stance_1_num",
        "med_epistemic_stance_2_num",
    ],
}


# =============================================================================
# 2. Text Cleaning
# =============================================================================

def clean_text(x):
    """Normalise a raw survey response string."""
    if pd.isna(x):
        return np.nan
    x = str(x)
    x = x.replace("\xa0", " ")
    x = re.sub(r"\s+", " ", x).strip()
    x = re.sub(r"\s*::\s*", " : ", x)
    x = re.sub(r"\s*:\s*", " : ", x)
    x = x.replace("--", "–")
    return x


def extract_leading_code(x):
    """
    Pull the leading integer from a Likert-coded string such as '3 : Agree'.
    Returns np.nan if no code can be found.
    """
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    m = re.match(r"^(\d+)\s*:", x)
    if m:
        return int(m.group(1))
    if x in ["0", "1"]:
        return int(x)
    return np.nan


# =============================================================================
# 3. Data Loading & Scoring
# =============================================================================

def add_combined_panel(df, group_col="course", combined_label="Combined"):
    """
    Append a copy of *df* whose *group_col* is relabelled *combined_label*,
    so that grouped operations automatically include an aggregated panel.
    """
    comb = df.copy()
    comb[group_col] = combined_label
    out = pd.concat([df, comb], ignore_index=True)
    out[group_col] = pd.Categorical(
        out[group_col],
        categories=["1111", "1204", combined_label],
        ordered=True,
    )
    return out


def zscore_series(s):
    s = pd.Series(s, dtype="float")
    return (s - s.mean()) / s.std(ddof=0)


def prepare_dataset(path_1111, path_1204):
    """
    Load the two CSV files, rename columns, extract numeric codes,
    compute reverse-scored AI literacy items, and build the SES index.

    Returns
    -------
    df   : pd.DataFrame  – fully scored dataset
    meta : dict          – lists of column names for each construct
    """
    d1 = pd.read_csv(path_1111).assign(course="1111")
    d2 = pd.read_csv(path_1204).assign(course="1204")

    df = pd.concat([d1, d2], ignore_index=True).rename(columns=rename_map)

    # ── clean text ──────────────────────────────────────────────────────────
    for col in df.columns:
        df[col] = df[col].map(clean_text)

    # ── extract numeric codes ────────────────────────────────────────────────
    non_extract = {
        "id", "course",
        "ses_lang_cantonese", "ses_lang_mandarin",
        "ses_lang_english", "ses_lang_other",
    }
    for col in df.columns:
        if col not in non_extract:
            df[col + "_num"] = df[col].apply(extract_leading_code)

    for col in ["ses_lang_cantonese", "ses_lang_mandarin",
                "ses_lang_english", "ses_lang_other"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── AI literacy scoring ──────────────────────────────────────────────────
    ai_reverse = [
        "ai_concept_data_bias_r_num",
        "ai_concept_blackbox_r_num",
        "ai_concept_input_variation_r_num",
        "ai_concept_prompt_wording_r_num",
        "ai_concept_social_ethics_r_num",
    ]
    ai_forward = [
        "ai_ability_training_data_num",
        "ai_ability_explainability_num",
        "ai_ability_input_sensitivity_num",
        "ai_ability_prompting_num",
        "ai_ability_social_ethics_num",
    ]

    for col in ai_reverse:
        df[col.replace("_r_num", "_scored_num")] = 6 - df[col]
    for col in ai_forward:
        df[col.replace("_num", "_scored_num")] = df[col]

    ai_scored_cols = [
        "ai_concept_data_bias_scored_num",
        "ai_concept_blackbox_scored_num",
        "ai_concept_input_variation_scored_num",
        "ai_concept_prompt_wording_scored_num",
        "ai_concept_social_ethics_scored_num",
        "ai_ability_training_data_scored_num",
        "ai_ability_explainability_scored_num",
        "ai_ability_input_sensitivity_scored_num",
        "ai_ability_prompting_scored_num",
        "ai_ability_social_ethics_scored_num",
    ]
    df["ai_lit_score"] = df[ai_scored_cols].mean(axis=1)

    # ── SES scoring ──────────────────────────────────────────────────────────
    df["ses_device_access_scored_num"] = 4 - df["ses_device_access_num"]
    df["ses_internet_quality_scored_num"] = 4 - df["ses_internet_quality_num"]
    df["ses_financial_constraint_scored_num"] = 5 - df["ses_financial_constraint_num"]

    ses_scored_cols = [
        "ses_parent1_edu_num",
        "ses_parent2_edu_num",
        "ses_household_income_num",
        "ses_home_area_num",
        "ses_device_access_scored_num",
        "ses_internet_quality_scored_num",
        "ses_financial_constraint_scored_num",
    ]
    z_ses = df[ses_scored_cols].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
    df["ses_index"] = z_ses.mean(axis=1)

    meta = {
        "ai_scored_cols": ai_scored_cols,
        "ses_scored_cols": ses_scored_cols,
    }
    return df, meta


# =============================================================================
# 4. Statistical Utilities
# =============================================================================

def cronbach_alpha(df_items):
    """Return Cronbach's alpha for a DataFrame of item columns."""
    x = df_items.dropna()
    k = x.shape[1]
    if k < 2 or len(x) < 3:
        return np.nan
    item_vars = x.var(axis=0, ddof=1)
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var == 0 or np.isnan(total_var):
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def cliffs_delta(x, y):
    """
    Compute Cliff's delta (non-parametric effect size) for two samples.
    Positive values indicate y > x.
    """
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) == 0 or len(y) == 0:
        return np.nan
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2 * u / (len(x) * len(y)) - 1


# =============================================================================
# 5. Reporting Tables
# =============================================================================

def construct_summary_table(df):
    """
    Descriptive statistics (n, mean, SD, median, IQR, skew) for SES index
    and AI literacy score, broken out by course and combined.
    """
    rows = []
    constructs = {
        "SES index": "ses_index",
        "AI literacy score": "ai_lit_score",
    }
    for sample, d in add_combined_panel(df).groupby("course", observed=False):
        for construct_name, col in constructs.items():
            s = d[col].dropna()
            rows.append({
                "sample": sample,
                "construct": construct_name,
                "n": s.size,
                "mean": s.mean(),
                "sd": s.std(ddof=1),
                "median": s.median(),
                "iqr": s.quantile(0.75) - s.quantile(0.25),
                "skew": skew(s, bias=False) if s.size >= 8 else np.nan,
            })
    return pd.DataFrame(rows)


def reliability_table(df, meta):
    """Cronbach's alpha for each construct by course and combined."""
    rows = []
    for sample, d in add_combined_panel(df).groupby("course", observed=False):
        rows.append({
            "sample": sample,
            "construct": "SES index ingredients",
            "alpha": cronbach_alpha(d[meta["ses_scored_cols"]]),
            "n_complete": d[meta["ses_scored_cols"]].dropna().shape[0],
        })
        rows.append({
            "sample": sample,
            "construct": "AI literacy score",
            "alpha": cronbach_alpha(d[meta["ai_scored_cols"]]),
            "n_complete": d[meta["ai_scored_cols"]].dropna().shape[0],
        })
    return pd.DataFrame(rows)


def effect_size_table(df, meta):
    """
    Mann-Whitney U p-values and Cliff's delta (1204 vs 1111) for every
    scored item plus the composite indices, sorted by effect size.
    """
    rows = []
    g1111 = df.query("course == '1111'")
    g1204 = df.query("course == '1204'")

    cols = meta["ses_scored_cols"] + meta["ai_scored_cols"] + ["ses_index", "ai_lit_score"]
    for col in cols:
        x = g1111[col]
        y = g1204[col]
        pval = mannwhitneyu(y.dropna(), x.dropna(), alternative="two-sided").pvalue
        rows.append({
            "variable": col,
            "label": item_labels.get(col, col),
            "cliffs_delta_1204_vs_1111": cliffs_delta(y, x),
            "mw_pvalue": pval,
        })
    return pd.DataFrame(rows).sort_values("cliffs_delta_1204_vs_1111", ascending=False)


# =============================================================================
# 6. Visualizations
# =============================================================================

def plot_item_profile(df, meta, construct="SES", figsize=(14, 8)):

    item_labels = get_item_labels()

    if construct.upper() == "SES":
        cols = meta["ses_scored_cols"]
        title = "SES item profile: class 1111, class 1204, and combined"
        ylabel = "Mean scored response"
    elif construct.upper() == "AI":
        cols = meta["ai_scored_cols"]
        title = "AI literacy item profile: class 1111, class 1204, and combined"
        ylabel = "Mean scored response"
    else:
        raise ValueError("construct must be 'SES' or 'AI'")

    plot_df = (
        add_combined_panel(df)[["course"] + cols]
        .melt(id_vars="course", value_vars=cols, var_name="item", value_name="score")
        .dropna()
    )

    summary = (
        plot_df.groupby(["course", "item"], observed=False)["score"]
        .agg(mean="mean", sd="std", n="size")
        .reset_index()
    )

    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    summary["lower"] = summary["mean"] - 1.96 * summary["se"]
    summary["upper"] = summary["mean"] + 1.96 * summary["se"]
    summary["item_label"] = summary["item"].map(item_labels)

    order = [item_labels[c] for c in cols]

    plt.figure(figsize=figsize)

    for course in ["1111", "1204", "Combined"]:
        sub = summary[summary["course"] == course].copy()
        sub["item_label"] = pd.Categorical(sub["item_label"], categories=order, ordered=True)
        sub = sub.sort_values("item_label")

        x = np.arange(len(sub))
        plt.plot(x, sub["mean"], marker="o", label=course)
        plt.errorbar(
            x, sub["mean"],
            yerr=1.96 * sub["se"],
            fmt="none",
            capsize=4,
            alpha=0.7
        )

    plt.xticks(np.arange(len(order)), order, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.title(title)
    plt.legend(title="Sample")
    plt.tight_layout()
    plt.show()

def plot_ses_ai_correlation_heatmaps(df, meta, figsize=(18, 6), annotate=False):
    item_labels = get_item_labels()
    ses_cols = meta["ses_scored_cols"]
    ai_cols = meta["ai_scored_cols"]

    plot_df = add_combined_panel(df)

    # fixed labels and order across all panels
    row_labels = [item_labels[c] for c in ses_cols]
    col_labels = [item_labels[c] for c in ai_cols]

    fig, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    courses = ["1111", "1204", "Combined"]
    last_mappable = None

    for i, (ax, course) in enumerate(zip(axes, courses)):
        sub = plot_df[plot_df["course"] == course]
        corr = sub[ses_cols + ai_cols].corr().loc[ses_cols, ai_cols]
        corr.index = row_labels
        corr.columns = col_labels

        hm = sns.heatmap(
            corr,
            ax=ax,
            annot=annotate,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            cbar=False,
            linewidths=0.5,
            linecolor="white"
        )
        last_mappable = hm.collections[0]

        ax.set_title(course, fontsize=13, pad=10)

        # only left panel keeps y tick labels
        if i == 0:
            ax.set_ylabel("SES items")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        # all panels keep x labels, but compactly
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

    # one shared colorbar for all panels
    cbar = fig.colorbar(
        last_mappable,
        ax=axes,
        shrink=0.85,
        location="right",
        pad=0.02
    )
    cbar.set_label("Pearson r", rotation=90)

    fig.suptitle(
        "SES-by-AI item correlations: class 1111, class 1204, and combined",
        y=1.03,
        fontsize=15
    )
    plt.show()

def plot_ranked_effect_sizes(effect_df, top_n=12, figsize=(10, 7)):
    plot_df = effect_df.copy()
    plot_df = plot_df.sort_values("cliffs_delta_1204_vs_1111", ascending=False).head(top_n)
    plot_df = plot_df.iloc[::-1]  # reverse for horizontal plot

    plt.figure(figsize=figsize)

    colors = ["#4C72B0" if x > 0 else "#DD8452" for x in plot_df["cliffs_delta_1204_vs_1111"]]

    plt.barh(
        plot_df["label"],
        plot_df["cliffs_delta_1204_vs_1111"],
        color=colors
    )

    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Cliff's delta (1204 vs 1111)")
    plt.ylabel("")
    plt.title(f"Top {top_n} ranked class differences")

    for i, (_, row) in enumerate(plot_df.iterrows()):
        plt.text(
            row["cliffs_delta_1204_vs_1111"],
            i,
            f"  p={row['mw_pvalue']:.3f}",
            va="center"
        )

    plt.tight_layout()
    plt.show()

def plot_mediator_summary(summary_df, figsize=(18, 8), errorbar="sd"):
    plot_df = summary_df.copy()

    label_map = {
        "conceptual_exposure_score": "Conceptual exposure",
        "practical_ai_use_score": "Practical AI use",
        "learning_ecology_score": "Learning ecology",
        "language_load_score": "Language load",
        "epistemic_stance_score": "Epistemic stance",
    }

    plot_df["mediator_label"] = plot_df["mediator"].map(label_map)
    mediator_order = [
        "Conceptual exposure",
        "Practical AI use",
        "Learning ecology",
        "Language load",
        "Epistemic stance",
    ]
    plot_df["mediator_label"] = pd.Categorical(
        plot_df["mediator_label"],
        categories=mediator_order,
        ordered=True
    )

    plot_df["sample"] = pd.Categorical(
        plot_df["sample"],
        categories=["1111", "1204", "Combined"],
        ordered=True
    )

    if errorbar == "sd":
        plot_df["ymin"] = plot_df["mean"] - plot_df["sd"]
        plot_df["ymax"] = plot_df["mean"] + plot_df["sd"]
        err_label = "Mean ± 1 SD"
    elif errorbar == "se":
        plot_df["se"] = plot_df["sd"] / np.sqrt(plot_df["n"])
        plot_df["ymin"] = plot_df["mean"] - 1.96 * plot_df["se"]
        plot_df["ymax"] = plot_df["mean"] + 1.96 * plot_df["se"]
        err_label = "Mean ± 95% CI"
    else:
        raise ValueError("errorbar must be 'sd' or 'se'")

    fig, ax = plt.subplots(figsize=figsize)

    offsets = {"1111": -0.22, "1204": 0.0, "Combined": 0.22}
    x_base = np.arange(len(mediator_order))

    for sample in ["1111", "1204", "Combined"]:
        sub = plot_df[plot_df["sample"] == sample].sort_values("mediator_label")
        x = x_base + offsets[sample]

        ax.errorbar(
            x=x,
            y=sub["mean"],
            yerr=[sub["mean"] - sub["ymin"], sub["ymax"] - sub["mean"]],
            fmt="o",
            capsize=4,
            linewidth=2,
            label=sample
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(mediator_order, ha="right")
    ax.set_ylabel(err_label)
    ax.set_xlabel("")
    ax.set_title("Mediator summary by class and combined")
    ax.legend(title="Sample")
    plt.tight_layout()
    plt.show()



# Change of total indirect effect using 4 left-out SES variables
def plot_direct_effect_verification(direct_df, figsize=(12, 9)):
    plot_df = direct_df.copy()

    order = ["none", "home_language", "school", "housing", "household_size"]
    label_map = {
        "none": "No extra check",
        "home_language": "Control home language",
        "school": "Control school type",
        "housing": "Control housing type",
        "household_size": "Control household size",
    }

    plot_df = plot_df[plot_df["check_group"].isin(order)].copy()
    plot_df["check_group"] = pd.Categorical(plot_df["check_group"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("check_group")
    plot_df["check_label"] = plot_df["check_group"].map(label_map)

    x = np.arange(len(plot_df))

    plt.figure(figsize=figsize)
    plt.errorbar(
        x=x,
        y=plot_df["beta_ses_std"],
        yerr=[
            plot_df["beta_ses_std"] - plot_df["ci_low_95"],
            plot_df["ci_high_95"] - plot_df["beta_ses_std"]
        ],
        fmt="o",
        capsize=5,
        linewidth=2
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, plot_df["check_label"], rotation=20, ha="right")
    plt.ylabel("Standardized SES effect on AI literacy")
    plt.xlabel("")
    plt.title("Direct SES effect across verification checks")

    for i, row in plot_df.reset_index(drop=True).iterrows():
        plt.text(
            i,
            row["beta_ses_std"] + 0.01,
            f"p={row['p_hc3']:.3f}",
            ha="center",
            va="bottom",
            fontsize=14
        )

    plt.tight_layout()
    plt.show()


# mediation analysis using 4 left-out SES variables
def plot_mediation_verification(med_df, figsize=(12, 9)):
    plot_df = med_df.copy()

    order = ["none", "home_language", "school", "housing", "household_size"]
    label_map = {
        "none": "No extra check",
        "home_language": "Control home language",
        "school": "Control school type",
        "housing": "Control housing type",
        "household_size": "Control household size",
    }

    plot_df = plot_df[plot_df["check_group"].isin(order)].copy()
    plot_df["check_group"] = pd.Categorical(plot_df["check_group"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("check_group")
    plot_df["check_label"] = plot_df["check_group"].map(label_map)

    y = np.arange(len(plot_df))

    colors = [
        "#4C72B0" if (lo > 0 or hi < 0) else "#999999"
        for lo, hi in zip(plot_df["indirect_ci_low_95"], plot_df["indirect_ci_high_95"])
    ]

    plt.figure(figsize=figsize)

    for i, row in plot_df.reset_index(drop=True).iterrows():
        plt.errorbar(
            x=row["indirect_ab"],
            y=i,
            xerr=[[row["indirect_ab"] - row["indirect_ci_low_95"]],
                  [row["indirect_ci_high_95"] - row["indirect_ab"]]],
            fmt="o",
            capsize=5,
            color=colors[i],
            linewidth=2
        )

    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.yticks(y, plot_df["check_label"])
    plt.xlabel("Indirect effect (a × b)")
    plt.ylabel("")
    plt.title("Mediation verification across categorical checks")

    for i, row in plot_df.reset_index(drop=True).iterrows():
        plt.text(
            row["indirect_ci_high_95"] + 0.003,
            i,
            f"Pr(>0)={row['prop_ab_positive']:.2f}",
            va="center",
            fontsize=12
        )

    plt.tight_layout()
    plt.show()



# Mediation analysis part
def prepare_mediation_plot_df(mediation_results):
    plot_df = mediation_results.copy()

    mediator_map = {
        "Conceptual exposure": "Conceptual exposure",
        "Practical AI use": "Practical AI use",
        "Learning ecology": "Learning ecology",
        "Language load": "Language load",
        "Epistemic stance": "Epistemic stance",
        "conceptual_exposure_score": "Conceptual exposure",
        "practical_ai_use_score": "Practical AI use",
        "learning_ecology_score": "Learning ecology",
        "language_load_score": "Language load",
        "epistemic_stance_score": "Epistemic stance",
    }

    outcome_map = {
        "AI conceptual understanding": "AI conceptual understanding",
        "AI ability/confidence": "AI ability/confidence",
        "ai_factor1_score": "AI conceptual understanding",
        "ai_factor2_score": "AI ability/confidence",
    }

    plot_df["mediator"] = plot_df["mediator"].map(mediator_map).fillna(plot_df["mediator"])
    plot_df["outcome"] = plot_df["outcome"].map(outcome_map).fillna(plot_df["outcome"])

    mediator_order = [
        "Conceptual exposure",
        "Practical AI use",
        "Learning ecology",
        "Language load",
        "Epistemic stance",
    ]
    plot_df["mediator"] = pd.Categorical(
        plot_df["mediator"],
        categories=mediator_order,
        ordered=True
    )

    plot_df["sample"] = pd.Categorical(
        plot_df["sample"],
        categories=["1111", "1204", "Combined"],
        ordered=True
    )

    return plot_df.sort_values(["outcome", "sample", "mediator"])


def plot_indirect_effect_forest(mediation_results, figsize=(16, 10)):

    plot_df = prepare_mediation_plot_df(mediation_results)

    outcomes = ["AI conceptual understanding", "AI ability/confidence"]
    sample = "Combined"

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    mediator_order = list(plot_df["mediator"].cat.categories)

    for j, outcome in enumerate(outcomes):
        ax = axes[j]
        sub = plot_df[
            (plot_df["outcome"] == outcome) &
            (plot_df["sample"] == sample)
        ].copy().sort_values("mediator")

        y = np.arange(len(mediator_order))
        x = sub["indirect_boot_mean"].values
        xerr_left = x - sub["indirect_ci_low_95"].values
        xerr_right = sub["indirect_ci_high_95"].values - x

        colors = ["#4C72B0" if lo > 0 or hi < 0 else "#999999"
                  for lo, hi in zip(sub["indirect_ci_low_95"], sub["indirect_ci_high_95"])]

        for k in range(len(sub)):
            ax.errorbar(
                x[k], y[k],
                xerr=[[xerr_left[k]], [xerr_right[k]]],
                fmt="o",
                capsize=4,
                color=colors[k]
            )
            # Add p-value text
            ax.text(
                sub["indirect_ci_high_95"].values[k] + 0.005,
                y[k],
                f"Pr(>0)={sub['prop_ab_positive'].values[k]:.2f}",
                va="center",
                fontsize=12
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{outcome}")
        ax.set_yticks(y)
        ax.set_yticklabels(mediator_order)
        ax.set_xlabel("Bootstrap indirect effect (a × b)")

    fig.suptitle("Simple mediation indirect effects (Combined)", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_a_b_paths(mediation_results, figsize=(16, 10)):
    plot_df = prepare_mediation_plot_df(mediation_results)

    outcomes = ["AI conceptual understanding", "AI ability/confidence"]
    sample = "Combined"

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    mediator_order = list(plot_df["mediator"].cat.categories)

    for j, outcome in enumerate(outcomes):
        ax = axes[j]
        sub = plot_df[
            (plot_df["outcome"] == outcome) &
            (plot_df["sample"] == sample)
        ].copy().sort_values("mediator")

        y = np.arange(len(mediator_order))

        ax.scatter(sub["a_path"], y, s=70, label="a path: SES → M")
        ax.scatter(sub["b_path"], y, s=70, marker="s", label="b path: M → AI | SES")

        # Add p-values for a and b paths
        for k in range(len(sub)):
            ax.text(
                sub["a_path"].values[k] + 0.01,
                y[k] - 0.15,
                f"p={sub['a_p'].values[k]:.3f}",
                fontsize=12,
                ha="center"
            )
            ax.text(
                sub["b_path"].values[k] + 0.01,
                y[k] + 0.15,
                f"p={sub['b_p'].values[k]:.3f}",
                fontsize=12,
                ha="center"
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{outcome}")
        ax.set_yticks(y)
        ax.set_yticklabels(mediator_order)
        ax.set_xlabel("Standardized path coefficient")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("a- and b-path patterns in simple mediation models (Combined)", y=0.99)
    plt.tight_layout()
    plt.show()


# Plot the weak mediators
def plot_ab_diagnostic(mediation_results, sample="Combined", figsize=(16, 10)):
    d = mediation_results.copy()
    d = d[d["sample"] == sample].copy()

    plt.figure(figsize=figsize)

    outcome_markers = {
        "AI conceptual understanding": "o",
        "AI ability/confidence": "s",
        "ai_factor1_score": "o",
        "ai_factor2_score": "s",
    }

    for _, row in d.iterrows():
        marker = outcome_markers.get(row["outcome"], "o")
        plt.scatter(row["a_path"], row["b_path"], s=180, marker=marker)

        plt.text(
            row["a_path"] + 0.01,
            row["b_path"] + 0.01,
            f"{row['mediator']}\n{row['outcome']}",
            fontsize=15
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("a-path: SES → mediator")
    plt.ylabel("b-path: mediator → AI | SES")
    plt.title(f"a–b path diagnostic plot ({sample})")
    plt.tight_layout()
    plt.show()



# =============================================================================
# 7. Linear modeling
# =============================================================================

# Total effect of SES - AI
def fit_total_effect_model(df, sample="Combined"):


    if sample == "Combined":
        d = df[["ses_index", "ai_lit_score"]].dropna().copy()
    else:
        d = df.loc[df["course"] == sample, ["ses_index", "ai_lit_score"]].dropna().copy()

    d["ses_z"] = zscore_series(d["ses_index"])
    d["ai_z"] = zscore_series(d["ai_lit_score"])

    X = sm.add_constant(d["ses_z"])
    y = d["ai_z"]

    model = sm.OLS(y, X).fit(cov_type="HC3")

    rho, rho_p = spearmanr(d["ses_index"], d["ai_lit_score"], nan_policy="omit")

    out = pd.DataFrame({
        "sample": [sample],
        "n": [len(d)],
        "beta_ses_std": [model.params["ses_z"]],
        "se_hc3": [model.bse["ses_z"]],
        "t_hc3": [model.tvalues["ses_z"]],
        "p_hc3": [model.pvalues["ses_z"]],
        "ci_low_95": [model.conf_int().loc["ses_z", 0]],
        "ci_high_95": [model.conf_int().loc["ses_z", 1]],
        "r_squared": [model.rsquared],
        "spearman_rho": [rho],
        "spearman_p": [rho_p],
    })

    return out, model, d



# =============================================================================
# 8. Exploratory Factor Analysis 
# =============================================================================

def prepare_efa_data(df, sample, items):
    if sample == "Combined":
        d = df[items].dropna().copy()
    else:
        d = df.loc[df["course"] == sample, items].dropna().copy()
    return d

#  EFA diagnostics (Bartlett's test)
def efa_diagnostics(df, sample, items):
    d = prepare_efa_data(df, sample, items)

    bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(d)
    kmo_per_item, kmo_overall = calculate_kmo(d)

    summary = pd.DataFrame({
        "sample": [sample],
        "n_complete": [len(d)],
        "bartlett_chi2": [bartlett_chi2],
        "bartlett_p": [bartlett_p],
        "kmo_overall": [kmo_overall]
    })

    item_kmo = pd.DataFrame({
        "item": items,
        "kmo_item": kmo_per_item
    }).sort_values("kmo_item")

    return summary, item_kmo


def run_scree_analysis(df, sample, items, figsize=(14, 8)):
    d = prepare_efa_data(df, sample, items)

    fa = FactorAnalyzer(rotation=None)
    fa.fit(d)

    eigenvalues, _ = fa.get_eigenvalues()

    eig_table = pd.DataFrame({
        "factor_number": np.arange(1, len(eigenvalues) + 1),
        "eigenvalue": eigenvalues
    })

    plt.figure(figsize=figsize)
    plt.plot(eig_table["factor_number"], eig_table["eigenvalue"], marker="o")
    plt.axhline(1, linestyle="--")
    plt.xticks(eig_table["factor_number"])
    plt.xlabel("Factor number")
    plt.ylabel("Eigenvalue")
    plt.title(f"Scree plot: {sample}")
    plt.tight_layout()
    plt.show()

    return eig_table


# After diagnostics, fit an EFA model
def fit_efa(df, sample, items, n_factors=2, rotation="oblimin", method="minres"):
    d = prepare_efa_data(df, sample, items)

    fa = FactorAnalyzer(
        n_factors=n_factors,
        rotation=rotation,
        method=method
    )
    fa.fit(d)

    loadings = pd.DataFrame(
        fa.loadings_,
        index=items,
        columns=[f"Factor{j+1}" for j in range(n_factors)]
    )

    variance = pd.DataFrame({
        "SS Loadings": fa.get_factor_variance()[0],
        "Proportion Var": fa.get_factor_variance()[1],
        "Cumulative Var": fa.get_factor_variance()[2],
    }, index=[f"Factor{j+1}" for j in range(n_factors)])

    return fa, d, loadings, variance


# Clean loadings for presentation
def clean_loadings(loadings, cutoff=0.30):
    out = loadings.copy()
    out = out.round(3)
    out = out.where(out.abs() >= cutoff, "")
    return out



def plot_loading_heatmap(loadings, title="Factor loadings", figsize=(20, 8)):
    plt.figure(figsize=figsize)
    sns.heatmap(
        loadings,
        annot=True,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white"
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()




# =============================================================================
# 9. Mediators Factors
# =============================================================================

# calculate Cronbach’s alpha
def alpha_two_items(df_pair):
    d = df_pair.dropna()
    if d.shape[0] < 3 or d.shape[1] != 2:
        return np.nan

    corr = d.iloc[:, 0].corr(d.iloc[:, 1])
    if pd.isna(corr):
        return np.nan

    return (2 * corr) / (1 + corr)



def mediator_reliability_table(df, mediator_map):
    rows = []

    for sample in ["1111", "1204", "Combined"]:
        for mediator_name, items in mediator_map.items():
            if sample == "Combined":
                d = df[items].dropna().copy()
            else:
                d = df.loc[df["course"] == sample, items].dropna().copy()

            rho, pval = spearmanr(d[items[0]], d[items[1]], nan_policy="omit")
            alpha = alpha_two_items(d)

            rows.append({
                "sample": sample,
                "mediator": mediator_name,
                "n_complete": len(d),
                "spearman_rho": rho,
                "alpha_2item": alpha,
            })

    return pd.DataFrame(rows)


# Add composite scores for each mediator by averaging the two items.
def add_mediator_composites(df, mediator_map):
    d = df.copy()

    for mediator_name, items in mediator_map.items():
        d[f"{mediator_name}_score"] = d[items].mean(axis=1)

    return d

mediator_direction = pd.DataFrame({
    "mediator": [
        "conceptual_exposure",
        "practical_ai_use",
        "learning_ecology",
        "language_load",
        "epistemic_stance",
    ],
    "higher_score_means": [
        "more conceptual exposure",
        "more practical AI use",
        "more supportive learning ecology",
        "more language load / burden",
        "stronger epistemic stance (check exact wording)",
    ]
})


def mediator_summary_table(df):
    mediator_scores = [
        "conceptual_exposure_score",
        "practical_ai_use_score",
        "learning_ecology_score",
        "language_load_score",
        "epistemic_stance_score",
    ]

    rows = []
    for sample in ["1111", "1204", "Combined"]:
        if sample == "Combined":
            d = df[mediator_scores]
        else:
            d = df.loc[df["course"] == sample, mediator_scores]

        for col in mediator_scores:
            s = d[col].dropna()
            rows.append({
                "sample": sample,
                "mediator": col,
                "n": len(s),
                "mean": s.mean(),
                "sd": s.std(ddof=1),
                "median": s.median(),
            })

    return pd.DataFrame(rows)




# =============================================================================
# 10. Mediation Analysis
# =============================================================================

mediator_vars = [
    "conceptual_exposure_score",
    "practical_ai_use_score",
    "learning_ecology_score",
    "language_load_score",     
    "epistemic_stance_score",
]

outcome_vars = [
    "ai_factor1_score",   # conceptual understanding
    "ai_factor2_score",   # ability/confidence
]


label_map = {
    "conceptual_exposure_score": "Conceptual exposure",
    "practical_ai_use_score": "Practical AI use",
    "learning_ecology_score": "Learning ecology",
    "language_load_score": "Language load",   # or Language ease if reversed
    "epistemic_stance_score": "Epistemic stance",
    "ai_factor1_score": "AI conceptual understanding",
    "ai_factor2_score": "AI ability/confidence",
}


def prepare_simple_mediation_data(df, sample, x, m, y):
    cols = [x, m, y]

    if sample == "Combined":
        d = df[cols].dropna().copy()
    else:
        d = df.loc[df["course"] == sample, cols].dropna().copy()

    d[x] = zscore_series(d[x])
    d[m] = zscore_series(d[m])
    d[y] = zscore_series(d[y])

    return d


# SLR
def fit_simple_mediation(df, sample, x, m, y):
    d = prepare_simple_mediation_data(df, sample, x, m, y)

    # a path: M ~ X
    Xa = sm.add_constant(d[[x]])
    model_a = sm.OLS(d[m], Xa).fit(cov_type="HC3")

    # total effect: Y ~ X
    Xt = sm.add_constant(d[[x]])
    model_total = sm.OLS(d[y], Xt).fit(cov_type="HC3")

    # b path and direct effect: Y ~ X + M
    Xb = sm.add_constant(d[[x, m]])
    model_b = sm.OLS(d[y], Xb).fit(cov_type="HC3")

    a = model_a.params[x]
    b = model_b.params[m]
    c_total = model_total.params[x]
    c_prime = model_b.params[x]
    indirect = a * b

    result = pd.DataFrame({
        "sample": [sample],
        "mediator": [m],
        "outcome": [y],
        "n": [len(d)],
        "a_path": [a],
        "a_p": [model_a.pvalues[x]],
        "b_path": [b],
        "b_p": [model_b.pvalues[m]],
        "c_total": [c_total],
        "c_total_p": [model_total.pvalues[x]],
        "c_prime": [c_prime],
        "c_prime_p": [model_b.pvalues[x]],
        "indirect_ab": [indirect],
        "direct_plus_indirect_check": [c_prime + indirect],
        "r2_mediator_model": [model_a.rsquared],
        "r2_outcome_model": [model_b.rsquared],
    })

    return result, model_a, model_total, model_b, d



def bootstrap_indirect_effect(df, sample, x, m, y, n_boot=3000, seed=2026):
    rng = np.random.default_rng(seed)
    d = prepare_simple_mediation_data(df, sample, x, m, y)

    n = len(d)
    ab_vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = d.iloc[idx].copy()

        try:
            Xa = sm.add_constant(boot[[x]])
            model_a = sm.OLS(boot[m], Xa).fit()

            Xb = sm.add_constant(boot[[x, m]])
            model_b = sm.OLS(boot[y], Xb).fit()

            a = model_a.params[x]
            b = model_b.params[m]
            ab_vals.append(a * b)
        except Exception:
            continue

    ab_vals = np.array(ab_vals)

    out = pd.DataFrame({
        "sample": [sample],
        "mediator": [m],
        "outcome": [y],
        "indirect_boot_mean": [ab_vals.mean()],
        "indirect_ci_low_95": [np.quantile(ab_vals, 0.025)],
        "indirect_ci_high_95": [np.quantile(ab_vals, 0.975)],
        "prop_ab_positive": [(ab_vals > 0).mean()],
    })

    return out, ab_vals



# Run all simple mediations for every combination of sample, mediator, and outcome,
def run_all_simple_mediations(df, x="ses_index", mediators=None, outcomes=None, n_boot=3000, seed=2026):
    if mediators is None:
        mediators = mediator_vars
    if outcomes is None:
        outcomes = outcome_vars

    rows = []
    boot_rows = []

    counter = 0
    for sample in ["1111", "1204", "Combined"]:
        for m in mediators:
            for y in outcomes:
                res, _, _, _, _ = fit_simple_mediation(df, sample, x, m, y)
                boot_res, _ = bootstrap_indirect_effect(
                    df, sample, x, m, y,
                    n_boot=n_boot,
                    seed=seed + counter
                )
                counter += 1

                merged = res.merge(
                    boot_res,
                    on=["sample", "mediator", "outcome"],
                    how="left"
                )

                rows.append(merged)

    result_table = pd.concat(rows, ignore_index=True)
    return result_table



# =============================================================================
# 11. Post-Mediation Analysis using 4 left-out SES catogories
#.    Only use combined data
# =============================================================================

def add_leftout_ses_categories(df):
    d = df.copy()

    if "ses_school_type_num" in d.columns:
        d["school_type_cat"] = d["ses_school_type_num"].map({
            1: "gov_aided",
            2: "dss",
            3: "international",
            4: "other",
        })

    if "ses_housing_type_num" in d.columns:
        d["housing_type_cat"] = d["ses_housing_type_num"].map({
            1: "prh",
            2: "subsidized_ownership",
            3: "private_owned",
            4: "private_rented",
        })

    if "ses_household_size_num" in d.columns:
        d["household_size_cat"] = d["ses_household_size_num"].map({
            1: "1_2",
            2: "3_4",
            3: "5_6",
            4: "7plus",
        })

    lang_cols = [
        "ses_lang_cantonese",
        "ses_lang_mandarin",
        "ses_lang_english",
        "ses_lang_other",
    ]

    if all(col in d.columns for col in lang_cols):
        def get_lang_group(row):
            langs = []
            if row["ses_lang_cantonese"] == 1:
                langs.append("cantonese")
            if row["ses_lang_mandarin"] == 1:
                langs.append("mandarin")
            if row["ses_lang_english"] == 1:
                langs.append("english")
            if row["ses_lang_other"] == 1:
                langs.append("other")
            if len(langs) == 0:
                return "unknown"
            if len(langs) == 1:
                return langs[0]
            return "multilingual"

        d["home_language_cat"] = d[lang_cols].apply(get_lang_group, axis=1)

    return d

# create dummy variables
def add_selected_leftout_dummies(df, group_name=None):
    d = add_leftout_ses_categories(df)

    group_map = {
        "school": ["school_type_cat"],
        "housing": ["housing_type_cat"],
        "household_size": ["household_size_cat"],
        "home_language": ["home_language_cat"],
    }

    if group_name is None:
        return d, []

    cat_cols = group_map[group_name]
    dummy_df = pd.get_dummies(d[cat_cols], drop_first=True, dtype=float)
    d = pd.concat([d, dummy_df], axis=1)

    return d, list(dummy_df.columns)


# similar to fit_total_effect_model
def verify_direct_effect(df, check_group=None, sample="Combined"):
    d, dummy_cols = add_selected_leftout_dummies(df, group_name=check_group)

    if sample == "Combined":
        sub = d.copy()
    else:
        sub = d[d["course"] == sample].copy()

    needed = ["ses_index", "ai_lit_score"] + dummy_cols
    sub = sub[needed].dropna().copy()

    sub["ses_z"] = zscore_series(sub["ses_index"])
    sub["ai_z"] = zscore_series(sub["ai_lit_score"])

    X = sm.add_constant(sub[["ses_z"] + dummy_cols])
    y = sub["ai_z"]

    model = sm.OLS(y, X).fit(cov_type="HC3")

    return pd.DataFrame({
        "sample": [sample],
        "check_group": [check_group if check_group else "none"],
        "n": [len(sub)],
        "beta_ses_std": [model.params["ses_z"]],
        "p_hc3": [model.pvalues["ses_z"]],
        "ci_low_95": [model.conf_int().loc["ses_z", 0]],
        "ci_high_95": [model.conf_int().loc["ses_z", 1]],
        "r_squared": [model.rsquared],
    })



def verify_one_mediation(df, mediator, outcome, check_group=None, sample="Combined", n_boot=3000, seed=2026):
    d, dummy_cols = add_selected_leftout_dummies(df, group_name=check_group)

    if sample == "Combined":
        sub = d.copy()
    else:
        sub = d[d["course"] == sample].copy()

    needed = ["ses_index", mediator, outcome] + dummy_cols
    sub = sub[needed].dropna().copy()

    sub["ses_index"] = zscore_series(sub["ses_index"])
    sub[mediator] = zscore_series(sub[mediator])
    sub[outcome] = zscore_series(sub[outcome])

    Xa = sm.add_constant(sub[["ses_index"] + dummy_cols])
    model_a = sm.OLS(sub[mediator], Xa).fit(cov_type="HC3")

    Xt = sm.add_constant(sub[["ses_index"] + dummy_cols])
    model_total = sm.OLS(sub[outcome], Xt).fit(cov_type="HC3")

    Xb = sm.add_constant(sub[["ses_index", mediator] + dummy_cols])
    model_b = sm.OLS(sub[outcome], Xb).fit(cov_type="HC3")

    a = model_a.params["ses_index"]
    b = model_b.params[mediator]
    indirect = a * b

    rng = np.random.default_rng(seed)
    ab_vals = []
    n = len(sub)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = sub.iloc[idx].copy()

        try:
            Xa_boot = sm.add_constant(boot[["ses_index"] + dummy_cols])
            ma = sm.OLS(boot[mediator], Xa_boot).fit()

            Xb_boot = sm.add_constant(boot[["ses_index", mediator] + dummy_cols])
            mb = sm.OLS(boot[outcome], Xb_boot).fit()

            ab_vals.append(ma.params["ses_index"] * mb.params[mediator])
        except:
            continue

    ab_vals = np.array(ab_vals)

    return pd.DataFrame({
        "sample": [sample],
        "check_group": [check_group if check_group else "none"],
        "mediator": [mediator],
        "outcome": [outcome],
        "n": [len(sub)],
        "a_path": [a],
        "b_path": [b],
        "indirect_ab": [indirect],
        "indirect_ci_low_95": [np.quantile(ab_vals, 0.025)],
        "indirect_ci_high_95": [np.quantile(ab_vals, 0.975)],
        "prop_ab_positive": [(ab_vals > 0).mean()],
    })




# =============================================================================
# 12. Check Weak Mediators
# =============================================================================

def diagnose_mediation_paths(mediation_results, a_alpha=0.05, b_alpha=0.05, ab_alpha=True):

    d = mediation_results.copy()

    # whether the bootstrap CI excludes zero
    d["indirect_sig"] = ~(
        (d["indirect_ci_low_95"] <= 0) & (d["indirect_ci_high_95"] >= 0)
    )

    def classify(row):
        a_sig = row["a_p"] < a_alpha
        b_sig = row["b_p"] < b_alpha
        ab_sig = row["indirect_sig"]

        if ab_sig:
            return "supported mediation"
        if a_sig and not b_sig:
            return "fails at b-path"
        if not a_sig and b_sig:
            return "fails at a-path"
        if not a_sig and not b_sig:
            return "both paths weak"
        return "inconsistent / partial"

    d["diagnosis"] = d.apply(classify, axis=1)
    return d



# For a given mediator, create a table comparing the a and b paths across the two outcomes.
def compare_one_mediator_across_outcomes(mediation_results, mediator_name, sample="Combined"):
    d = mediation_results.copy()
    d = d[(d["sample"] == sample) & (d["mediator"] == mediator_name)].copy()

    return d[[
        "sample", "mediator", "outcome",
        "a_path", "a_p", "b_path", "b_p",
        "indirect_ab", "indirect_ci_low_95", "indirect_ci_high_95",
        "prop_ab_positive"
    ]].sort_values("outcome")


# 2-item within each mediator
def run_item_level_mediation(df, sample, x, mediator_items, y, n_boot=3000, seed=2026):
    rows = []

    for i, m in enumerate(mediator_items):
        res, _, _, _, _ = fit_simple_mediation(df, sample, x, m, y)
        boot_res, _ = bootstrap_indirect_effect(
            df, sample, x, m, y,
            n_boot=n_boot,
            seed=seed + i
        )

        merged = res.merge(
            boot_res,
            on=["sample", "mediator", "outcome"],
            how="left"
        )
        rows.append(merged)

    return pd.concat(rows, ignore_index=True)
