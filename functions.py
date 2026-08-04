"""
functions.py — Utility functions for Phase 1 AI Literacy analysis.

"""

import re

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")
from scipy.stats import skew, mannwhitneyu, spearmanr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import itertools



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


# One source of truth for item identity, order, subscale, and scoring at both
# waves. `post_position` documents the required follow-up column order.
AI_ITEM_CODEBOOK = (
    {
        "item": "data_bias", "subscale": "conceptual", "reverse": True,
        "pre_source": "Q14_AI Literacy 2", "pre_numeric": "ai_concept_data_bias_r_num",
        "pre_scored": "ai_concept_data_bias_scored_num",
        "post_source": "Q05_AI Conceptual", "post_position": 5,
        "post_scored": "ai_concept_data_bias_scored_num_post",
    },
    {
        "item": "training_data", "subscale": "confidence", "reverse": False,
        "pre_source": "Q15_AI Literacy 3", "pre_numeric": "ai_ability_training_data_num",
        "pre_scored": "ai_ability_training_data_scored_num",
        "post_source": "Q06_AI Conceptual 2", "post_position": 6,
        "post_scored": "ai_ability_training_data_scored_num_post",
    },
    {
        "item": "blackbox", "subscale": "conceptual", "reverse": True,
        "pre_source": "Q16_AI Literacy 4", "pre_numeric": "ai_concept_blackbox_r_num",
        "pre_scored": "ai_concept_blackbox_scored_num",
        "post_source": "Q03_AI Representation 1", "post_position": 3,
        "post_scored": "ai_concept_blackbox_scored_num_post",
    },
    {
        "item": "explainability", "subscale": "confidence", "reverse": False,
        "pre_source": "Q17_AI Literacy 5", "pre_numeric": "ai_ability_explainability_num",
        "pre_scored": "ai_ability_explainability_scored_num",
        "post_source": "Q04_AI Representation 2", "post_position": 4,
        "post_scored": "ai_ability_explainability_scored_num_post",
    },
    {
        "item": "input_variation", "subscale": "conceptual", "reverse": True,
        "pre_source": "Q18_AI Literacy 6", "pre_numeric": "ai_concept_input_variation_r_num",
        "pre_scored": "ai_concept_input_variation_scored_num",
        "post_source": "Q01_AI Perception 1", "post_position": 1,
        "post_scored": "ai_concept_input_variation_scored_num_post",
    },
    {
        "item": "input_sensitivity", "subscale": "confidence", "reverse": False,
        "pre_source": "Q19_AI Literacy 7", "pre_numeric": "ai_ability_input_sensitivity_num",
        "pre_scored": "ai_ability_input_sensitivity_scored_num",
        "post_source": "Q02_AI Perception 2", "post_position": 2,
        "post_scored": "ai_ability_input_sensitivity_scored_num_post",
    },
    {
        "item": "prompt_wording", "subscale": "conceptual", "reverse": True,
        "pre_source": "Q20_AI Literacy 8", "pre_numeric": "ai_concept_prompt_wording_r_num",
        "pre_scored": "ai_concept_prompt_wording_scored_num",
        "post_source": "Q07_AI Interaction 1", "post_position": 7,
        "post_scored": "ai_concept_prompt_wording_scored_num_post",
    },
    {
        "item": "prompting", "subscale": "confidence", "reverse": False,
        "pre_source": "Q21_AI Literacy 9", "pre_numeric": "ai_ability_prompting_num",
        "pre_scored": "ai_ability_prompting_scored_num",
        "post_source": "Q08_AI Interaction 2", "post_position": 8,
        "post_scored": "ai_ability_prompting_scored_num_post",
    },
    {
        "item": "social_ethics_limits", "subscale": "conceptual", "reverse": True,
        "pre_source": "Q22_AI Literacy 10", "pre_numeric": "ai_concept_social_ethics_r_num",
        "pre_scored": "ai_concept_social_ethics_scored_num",
        "post_source": "Q09_AI Societal Impact 1", "post_position": 9,
        "post_scored": "ai_concept_social_ethics_scored_num_post",
    },
    {
        "item": "social_ethics_ability", "subscale": "confidence", "reverse": False,
        "pre_source": "Q23_AI Literacy 11", "pre_numeric": "ai_ability_social_ethics_num",
        "pre_scored": "ai_ability_social_ethics_scored_num",
        "post_source": "Q10_AI Societal Impact 2", "post_position": 10,
        "post_scored": "ai_ability_social_ethics_scored_num_post",
    },
)

AI_EFA_ITEM_ORDER = (
    "data_bias", "blackbox", "input_variation", "prompt_wording",
    "social_ethics_limits", "training_data", "explainability",
    "input_sensitivity", "prompting", "social_ethics_ability",
)
_AI_ITEM_BY_NAME = {row["item"]: row for row in AI_ITEM_CODEBOOK}
AI_EFA_ITEMS = tuple(_AI_ITEM_BY_NAME[name]["pre_scored"] for name in AI_EFA_ITEM_ORDER)
AI_EFA_ITEMS_POST = tuple(_AI_ITEM_BY_NAME[name]["post_scored"] for name in AI_EFA_ITEM_ORDER)
AI_CONCEPTUAL_ITEMS = tuple(
    _AI_ITEM_BY_NAME[name]["pre_scored"]
    for name in AI_EFA_ITEM_ORDER
    if _AI_ITEM_BY_NAME[name]["subscale"] == "conceptual"
)
AI_CONFIDENCE_ITEMS = tuple(
    _AI_ITEM_BY_NAME[name]["pre_scored"]
    for name in AI_EFA_ITEM_ORDER
    if _AI_ITEM_BY_NAME[name]["subscale"] == "confidence"
)


# Canonical SES index used in every phase. Higher values always mean higher
# SES/resources. The index is the mean of Phase-I-pooled z-scored items.
SES_INDEX_CODEBOOK = (
    {"item": "parent1_education", "numeric": "ses_parent1_edu_num", "scored": "ses_parent1_edu_num", "minimum": 1, "maximum": 6, "reverse": False},
    {"item": "parent2_education", "numeric": "ses_parent2_edu_num", "scored": "ses_parent2_edu_num", "minimum": 1, "maximum": 6, "reverse": False},
    {"item": "household_income", "numeric": "ses_household_income_num", "scored": "ses_household_income_num", "minimum": 1, "maximum": 5, "reverse": False},
    {"item": "home_area", "numeric": "ses_home_area_num", "scored": "ses_home_area_num", "minimum": 1, "maximum": 5, "reverse": False},
    {"item": "device_access", "numeric": "ses_device_access_num", "scored": "ses_device_access_scored_num", "minimum": 1, "maximum": 3, "reverse": True},
    {"item": "internet_quality", "numeric": "ses_internet_quality_num", "scored": "ses_internet_quality_scored_num", "minimum": 1, "maximum": 3, "reverse": True},
    {"item": "financial_constraint", "numeric": "ses_financial_constraint_num", "scored": "ses_financial_constraint_scored_num", "minimum": 1, "maximum": 4, "reverse": True},
)
SES_INDEX_ITEMS = tuple(row["scored"] for row in SES_INDEX_CODEBOOK)

# Shared by both Phase-I notebooks so the fitted SES factors are identical.
SES_EFA_ITEMS_2DIM = (
    "ses_parent1_edu_num",
    "ses_parent2_edu_num",
    "ses_household_income_num",
    "ses_financial_constraint_scored_num",
    "ses_space_per_person",
    "ses_housing_type_ord",
    "ses_school_type_ord",
)

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


def validate_item_order(columns, expected_columns, context="questionnaire"):
    """Validate that required item columns exist and retain their defined order."""
    columns = list(columns)
    expected_columns = list(expected_columns)
    missing = [col for col in expected_columns if col not in columns]
    if missing:
        raise KeyError(f"{context}: missing required item columns: {missing}")

    positions = [columns.index(col) for col in expected_columns]
    if positions != sorted(positions):
        raise ValueError(
            f"{context}: item columns are out of order. Expected: {expected_columns}"
        )


def validate_score_range(values, minimum, maximum, context="score"):
    """Reject parsed responses outside an item's documented scale."""
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    invalid = s[~s.between(minimum, maximum)]
    if not invalid.empty:
        bad_values = sorted(pd.unique(invalid).tolist())
        raise ValueError(
            f"{context}: values outside [{minimum}, {maximum}]: {bad_values}"
        )


def validate_unique_ids(df, id_col, context="dataset", allow_missing=False):
    """Validate identifiers without printing or exposing their values."""
    if id_col not in df.columns:
        raise KeyError(f"{context}: missing ID column {id_col!r}")

    ids = df[id_col].astype("string").str.strip().replace("", pd.NA)
    if not allow_missing and ids.isna().any():
        raise ValueError(f"{context}: found {int(ids.isna().sum())} missing IDs")

    duplicate_count = int(ids.dropna().duplicated().sum())
    if duplicate_count:
        raise ValueError(f"{context}: found {duplicate_count} duplicate IDs")
    return ids


def score_pre_ai_items(df):
    """Score Phase-I AI items from the shared pre/post item codebook."""
    d = df.copy()
    # Preserve the historical derived-column order while sourcing all scoring
    # rules from the codebook.
    for item_name in AI_EFA_ITEM_ORDER:
        item = _AI_ITEM_BY_NAME[item_name]
        source = item["pre_numeric"]
        target = item["pre_scored"]
        if source not in d.columns:
            raise KeyError(f"Phase I AI scoring: missing parsed column {source!r}")
        validate_score_range(d[source], 1, 5, context=source)
        d[target] = 6 - d[source] if item["reverse"] else d[source]
        validate_score_range(d[target], 1, 5, context=target)
    return d


def score_ses_index(df, new_col="ses_index"):
    """Apply the canonical SES directions and pooled-z-score index definition."""
    d = df.copy()
    for item in SES_INDEX_CODEBOOK:
        source = item["numeric"]
        target = item["scored"]
        if source not in d.columns:
            raise KeyError(f"SES scoring: missing parsed column {source!r}")
        validate_score_range(
            d[source], item["minimum"], item["maximum"], context=source
        )
        if item["reverse"]:
            d[target] = item["maximum"] + item["minimum"] - d[source]
        elif target != source:
            d[target] = d[source]

    ses_scored_cols = list(SES_INDEX_ITEMS)
    means = d[ses_scored_cols].mean()
    sds = d[ses_scored_cols].std(ddof=0)
    if (sds == 0).any():
        constant_cols = sds.index[sds == 0].tolist()
        raise ValueError(f"SES scoring: constant item columns: {constant_cols}")

    # Keep the original per-column arithmetic so regenerated CSV values remain
    # byte-stable apart from intentional scoring changes.
    z_ses = d[ses_scored_cols].apply(
        lambda series: (series - series.mean()) / series.std(ddof=0)
    )
    d[new_col] = z_ses.mean(axis=1, skipna=False)
    return d, means.to_dict(), sds.to_dict()


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
    d1 = pd.read_csv(path_1111)
    d2 = pd.read_csv(path_1204)

    expected_ai_columns = [row["pre_source"] for row in AI_ITEM_CODEBOOK]
    validate_item_order(d1.columns, expected_ai_columns, context="Phase I course 1111")
    validate_item_order(d2.columns, expected_ai_columns, context="Phase I course 1204")

    d1 = d1.assign(course="1111")
    d2 = d2.assign(course="1204")

    df = pd.concat([d1, d2], ignore_index=True).rename(columns=rename_map)
    df["id"] = validate_unique_ids(df, "id", context="Combined Phase I")

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
    df = score_pre_ai_items(df)
    ai_scored_cols = list(AI_EFA_ITEMS)
    df["ai_lit_score"] = df[ai_scored_cols].mean(axis=1)

    # ── SES scoring ──────────────────────────────────────────────────────────
    df, ses_reference_means, ses_reference_sds = score_ses_index(df)
    ses_scored_cols = list(SES_INDEX_ITEMS)

    meta = {
        "ai_scored_cols": ai_scored_cols,
        "ses_scored_cols": ses_scored_cols,
        "ses_reference_means": ses_reference_means,
        "ses_reference_sds": ses_reference_sds,
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


# Special: T-test for 'other' school vs non-other school
def school_others_ses_ttest(
    df,
    school_col="ses_school_type_num",
    others_code=4,
    metrics=None
):
    if metrics is None:
        metrics = [
            "ses_parent1_edu_num",
            "ses_parent2_edu_num",
            "ses_household_income_num",
            "ses_home_area_num",
            "ses_device_access_scored_num",
            "ses_internet_quality_scored_num",
            "ses_financial_constraint_scored_num",
            "ses_index",
        ]

    d = df.copy()
    d["school_group"] = np.where(d[school_col] == others_code, "Others", "Non-others")

    label_map = {
        "ses_parent1_edu_num": "Parent 1 education",
        "ses_parent2_edu_num": "Parent 2 education",
        "ses_household_income_num": "Household income",
        "ses_home_area_num": "Home area",
        "ses_device_access_scored_num": "Device access",
        "ses_internet_quality_scored_num": "Internet quality",
        "ses_financial_constraint_scored_num": "Low financial strain",
        "ses_index": "SES index",
    }

    def cohens_d(x, y):
        x = pd.Series(x).dropna()
        y = pd.Series(y).dropna()
        nx, ny = len(x), len(y)
        sx, sy = x.std(ddof=1), y.std(ddof=1)
        pooled_sd = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
        if pooled_sd == 0:
            return np.nan
        return (x.mean() - y.mean()) / pooled_sd

    rows = []

    for metric in metrics:
        sub = d[["school_group", metric]].dropna().copy()
        others = sub.loc[sub["school_group"] == "Others", metric]
        non_others = sub.loc[sub["school_group"] == "Non-others", metric]

        test = ttest_ind(
            others,
            non_others,
            equal_var=False,
            alternative="two-sided"
        )

        rows.append({
            "metric": label_map.get(metric, metric),
            "n_others": len(others),
            "mean_diff_others_minus_nonothers": others.mean() - non_others.mean(),
            "t_stat": test.statistic,
            "p-value": test.pvalue,
            "cohens_d": cohens_d(others, non_others),
        })

    return pd.DataFrame(rows).sort_values("p-value").reset_index(drop=True)





# =============================================================================
# * Special treatment for some SES variables
# =============================================================================

# Spliting SES into 2 dimentions 
# (2 versions, one with 'other school' as a separate category, one with 'other school' assigned to the nearest known group)
def add_assigned_school_type_ord(
    df,
    school_col="ses_school_type_num",
    metrics=None,
    new_col="ses_school_type_ord"
):
    if metrics is None:
        metrics = [
            "ses_parent1_edu_num",
            "ses_parent2_edu_num",
            "ses_household_income_num",
            "ses_home_area_num",
            "ses_device_access_scored_num",
            "ses_internet_quality_scored_num",
            "ses_financial_constraint_scored_num",
        ]

    d = df.copy()

    keep = [school_col] + metrics
    work = d[keep].dropna().copy()

# assign 'other school' group to the known group with the smallest Euclidean distance
    zcols = []
    for col in metrics:
        zcol = f"{col}_z"
        work[zcol] = (work[col] - work[col].mean()) / work[col].std(ddof=0)
        zcols.append(zcol)

    # centroids for known school groups only
    known = work[work[school_col].isin([1, 2, 3])].copy()
    centroids = known.groupby(school_col)[zcols].mean()

    d[new_col] = d[school_col]

    # assign 'Others school' student by student
    others_idx = work.index[work[school_col] == 4]

    for idx in others_idx:
        row = work.loc[idx, zcols]
        dists = ((centroids - row.values) ** 2).sum(axis=1) ** 0.5
        d.loc[idx, new_col] = dists.idxmin()

    return d


def add_ses_space_house_variables(
    df,
    home_area_col="ses_home_area_num",
    household_size_col="ses_household_size_num",
    school_type_col="ses_school_type_num",
    housing_type_col="ses_housing_type_num"
):
    d = df.copy()

    # space per person
    d["ses_space_per_person"] = d[home_area_col] / d[household_size_col].replace(0, np.nan)

    # assign Others school type by nearest centroid, then keep ordered coding
    d = add_assigned_school_type_ord(
        d,
        school_col=school_type_col,
        metrics=[
            "ses_parent1_edu_num",
            "ses_parent2_edu_num",
            "ses_household_income_num",
            "ses_home_area_num",
            "ses_internet_quality_scored_num",
            "ses_financial_constraint_scored_num",
            "ses_device_access_scored_num"
        ],
        new_col="ses_school_type_ord"
    )

    # housing type as ordered SES resource
    housing_map = {
        1: 1,  # Public Rental Housing
        2: 2,  # Subsidised Home Ownership
        3: 3,  # Private Owned
        4: 2,  # Private Rented
    }
    d["ses_housing_type_ord"] = d[housing_type_col].map(housing_map)

    return d



# =============================================================================
# 5. Reporting Tables
# =============================================================================

def prepare_partial_corr_data(df, vars):
    return df[vars].dropna().copy()

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


ses_map = {
        "ses_factor1_score": "SES Factor 1",
        "ses_factor2_score": "SES Factor 2",
    "ses_index": "SES index",
    }

outcome_map = {
        "ai_factor1_score": "AI conceptual understanding",
        "ai_factor2_score": "AI ability/confidence",
        "ai_lit_score": "AI literacy overall",
    }

ses_order = ["SES Factor 1", "SES Factor 2", "SES index"]
    
outcome_order = [
        "AI conceptual understanding",
        "AI ability/confidence",
        "AI literacy overall",
    ]

mediator_order = [
        "Conceptual exposure",
        "Practical AI use",
        "Learning ecology",
        "Language load",
        "Epistemic stance",
    ]




# Mediation analysis part
def plot_indirect_effect_forest(results, sample="Combined", include_ai_overall=False, figsize=(18, 14)):
    plot_df = results.copy()
    sample_col = next((c for c in ["sample", "group", "subset"] if c in plot_df.columns), None)
    ses_col = next((c for c in ["ses_dimension", "x", "ses", "predictor"] if c in plot_df.columns), None)
    outcome_col = next((c for c in ["outcome", "y", "ai_outcome", "target"] if c in plot_df.columns), None)
    mediator_col = next((c for c in ["mediator", "m", "mediator_var"] if c in plot_df.columns), None)

    if ses_col is None or outcome_col is None or mediator_col is None:
        raise ValueError(
            "Could not find required grouping columns for SES, mediator, and outcome."
        )

    if sample_col is not None:
        # Try exact match first, then case-insensitive fallback.
        keep = plot_df[sample_col] == sample
        if not keep.any():
            keep = plot_df[sample_col].astype(str).str.lower() == str(sample).lower()
        plot_df = plot_df[keep].copy()

    plot_df = plot_df.rename(
        columns={ses_col: "ses_dimension", outcome_col: "outcome", mediator_col: "mediator"}
    )

    # Accept alternate column names if upstream output schema changed.
    effect_col = next(
        (c for c in ["indirect_boot_mean", "indirect_ab", "ab", "indirect_effect"] if c in plot_df.columns),
        None,
    )
    ci_low_col = next(
        (c for c in ["indirect_ci_low_95", "ci_low_95", "ab_ci_low", "indirect_ci_low"] if c in plot_df.columns),
        None,
    )
    ci_high_col = next(
        (c for c in ["indirect_ci_high_95", "ci_high_95", "ab_ci_high", "indirect_ci_high"] if c in plot_df.columns),
        None,
    )

    if effect_col is None or ci_low_col is None or ci_high_col is None:
        raise ValueError(
            "Could not find indirect-effect columns. Expected one of "
            "[indirect_boot_mean, indirect_ab, ab, indirect_effect] and matching CI columns."
        )

    plot_df["ses_dimension"] = plot_df["ses_dimension"].replace(ses_map)
    plot_df["outcome"] = plot_df["outcome"].replace(outcome_map)
    plot_df["mediator"] = plot_df["mediator"].replace(mediator_label_map)

    observed_ses = set(plot_df["ses_dimension"].dropna())
    observed_outcomes = set(plot_df["outcome"].dropna())

    ses_levels = [s for s in ses_order if s in observed_ses]
    if not ses_levels:
        ses_levels = sorted(observed_ses)

    base_outcomes = outcome_order if include_ai_overall else [o for o in outcome_order if o != "AI literacy overall"]
    outcome_levels = [o for o in base_outcomes if o in observed_outcomes]
    if not outcome_levels:
        outcome_levels = sorted(observed_outcomes)

    if not ses_levels or not outcome_levels:
        raise ValueError("No SES/outcome combinations found for the selected sample.")

    nrows, ncols = len(ses_levels), len(outcome_levels)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.array(axes, dtype=object).reshape(nrows, ncols)

    sig_color = "#D55E00"
    nonsig_color = "#9A9A9A"

    for i, ses_dim in enumerate(ses_levels):
        for j, outcome in enumerate(outcome_levels):
            ax = axes[i, j]
            sub = plot_df[
                (plot_df["ses_dimension"] == ses_dim) &
                (plot_df["outcome"] == outcome)
            ].copy()

            sub = sub.set_index("mediator").reindex(mediator_order).reset_index()
            y = np.arange(len(mediator_order))
            n_plotted = 0

            for k, (_, row) in enumerate(sub.iterrows()):
                eff = row.get(effect_col, np.nan)
                lo = row.get(ci_low_col, np.nan)
                hi = row.get(ci_high_col, np.nan)
                if pd.isna(eff) or pd.isna(lo) or pd.isna(hi):
                    continue
                if lo > hi:
                    continue

                sig = (lo > 0) or (hi < 0)
                color = sig_color if sig else nonsig_color

                ax.errorbar(
                    x=eff,
                    y=k,
                    xerr=[[eff - lo], [hi - eff]],
                    fmt="o",
                    capsize=5,
                    color=color,
                    ecolor=color,
                    elinewidth=3 if sig else 2,
                    markersize=8
                )
                n_plotted += 1

            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{ses_dim} → {outcome}")
            ax.set_yticks(y)
            ax.set_yticklabels(mediator_order)
            if n_plotted == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No CI estimates",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#666666",
                    fontsize=11,
                )

    for ax in axes[-1, :]:
        ax.set_xlabel("Indirect effect (a × b)")

    fig.suptitle(f"Indirect effects by SES dimension and AI outcome ({sample})", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_a_b_paths(results, sample="Combined", alpha=0.05, figsize=(18, 14)):
    plot_df = results.copy()
    plot_df = plot_df[plot_df["sample"] == sample].copy()

    plot_df["ses_dimension"] = plot_df["ses_dimension"].replace(ses_map)
    plot_df["outcome"] = plot_df["outcome"].replace(outcome_map)
    plot_df["mediator"] = plot_df["mediator"].replace(mediator_label_map)

    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)
    axes = np.array(axes)

    for i, ses_dim in enumerate(ses_order):
        for j, outcome in enumerate(outcome_order):
            ax = axes[i, j]

            sub = plot_df[
                (plot_df["ses_dimension"] == ses_dim) &
                (plot_df["outcome"] == outcome)
            ].copy()

            for _, row in sub.iterrows():
                if pd.isna(row.get("a_path", np.nan)) or pd.isna(row.get("b_path", np.nan)):
                    continue

                sig_a = row["a_p"] < alpha
                sig_b = row["b_p"] < alpha

                if sig_a and sig_b:
                    color = "#D55E00"   # orange
                elif sig_a:
                    color = "#0072B2"   # blue
                elif sig_b:
                    color = "#009E73"   # green
                else:
                    color = "#9A9A9A"   # gray

                ax.scatter(
                    row["a_path"],
                    row["b_path"],
                    s=90,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5
                )

                # larger font for significant labels
                is_sig = sig_a or sig_b
                ax.text(
                    row["a_path"] + 0.01,
                    row["b_path"] + 0.01,
                    row["mediator"],
                    fontsize=15 if is_sig else 8,
                    fontweight="bold" if is_sig else "normal",
                    color=color if is_sig else "black"
                )

            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{ses_dim} → {outcome}", fontsize=12)

            if i == 2:
                ax.set_xlabel("a path: SES → mediator")
            if j == 0:
                ax.set_ylabel("b path: mediator → AI | SES")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="a and b significant",
               markerfacecolor="#D55E00", markeredgecolor="black", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="a significant only",
               markerfacecolor="#0072B2", markeredgecolor="black", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="b significant only",
               markerfacecolor="#009E73", markeredgecolor="black", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="neither significant",
               markerfacecolor="#9A9A9A", markeredgecolor="black", markersize=9),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01)
    )

    fig.suptitle(f"a–b path diagnostic plot ({sample})", y=0.98, fontsize=16)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
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
        "beta_ses_std": [model.params["ses_z"]],
        "p_hc3": [model.pvalues["ses_z"]],
        "ci_low_95": [model.conf_int().loc["ses_z", 0]],
        "ci_high_95": [model.conf_int().loc["ses_z", 1]],
        "r_squared": [model.rsquared],
        "spearman_rho": [rho],
        "spearman_p": [rho_p],
    })

    return out, model, d


def run_total_effect_models(df, ses_cols, ai_cols):
    if isinstance(ses_cols, str):
        ses_cols = [ses_cols]
    if isinstance(ai_cols, str):
        ai_cols = [ai_cols]

    results = []

    for ses_col in ses_cols:
        for ai_col in ai_cols:
            d = df.copy()
            d["ses_index"] = d[ses_col]
            d["ai_lit_score"] = d[ai_col]

            res, _, _ = fit_total_effect_model(d, sample="Combined")
            res["ses_dimension"] = ses_col
            res["ai_outcome"] = ai_col
            results.append(res)

    out = pd.concat(results, ignore_index=True)

    first_cols = ["sample", "ses_dimension", "ai_outcome"]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + other_cols]

    return out

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



def plot_loading_heatmap(loadings, cutoff=0.30, figsize=(16, 10), title="Factor loadings heatmap"):
    plot_df = loadings.copy()

    factor_cols = [c for c in plot_df.columns if c.lower().startswith("factor")]
    plot_df = plot_df[factor_cols]

    abs_df = plot_df.abs()
    dominant_factor = abs_df.idxmax(axis=1)
    dominant_value = abs_df.max(axis=1)

    dim_number = dominant_factor.str.extract(r"(\d+)")[0]
    dim_number = dim_number.where(dominant_value >= cutoff, "")

    order_df = pd.DataFrame({
        "dominant_factor": dominant_factor,
        "dominant_value": dominant_value,
        "dim_number": dim_number
    }, index=plot_df.index)

    order_df["_factor_num"] = order_df["dominant_factor"].str.extract(r"(\d+)")[0].astype(float)
    order_df["_factor_num"] = order_df["_factor_num"].fillna(999)

    row_order = order_df.sort_values(
        ["_factor_num", "dominant_value"],
        ascending=[True, False]
    ).index

    plot_df = plot_df.loc[row_order]
    dim_number = dim_number.loc[row_order]

    # annotation table: blank out values below cutoff
    annot_df = plot_df.copy()
    annot_df = annot_df.applymap(lambda x: f"{x:.2f}" if abs(x) >= cutoff else "")

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        plot_df,
        annot=annot_df,
        fmt="",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Loading"},
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Factor")
    ax.set_ylabel("Item")

    ncols = plot_df.shape[1]
    for i, lab in enumerate(dim_number):
        ax.text(
            ncols + 0.15,
            i + 0.5,
            lab,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold"
        )

    ax.text(
        ncols + 0.15,
        -0.35,
        "Dim",
        va="bottom",
        ha="left",
        fontsize=11,
        fontweight="bold"
    )

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


# Add 2 AI factor scores and 2 SES factor scores into one dataframe.
def add_ses_ai_factor_scores(
    df_base, ai_df, ai_items, ai_fa, ses_df, ses_items, ses_fa
):

    df_analysis = df_base.copy()

    # AI factor scores
    ai_complete = ai_df[ai_items].dropna().copy()
    ai_scores = pd.DataFrame(
        ai_fa.transform(ai_complete),
        index=ai_complete.index,
        columns=["ai_factor1_score", "ai_factor2_score"]
    )
    df_analysis.loc[ai_scores.index, "ai_factor1_score"] = ai_scores["ai_factor1_score"]
    df_analysis.loc[ai_scores.index, "ai_factor2_score"] = ai_scores["ai_factor2_score"]

    # SES factor scores
    ses_complete = ses_df[ses_items].dropna().copy()
    ses_scores = pd.DataFrame(
        ses_fa.transform(ses_complete),
        index=ses_complete.index,
        columns=["ses_factor1_score", "ses_factor2_score"]
    )
    df_analysis.loc[ses_scores.index, "ses_factor1_score"] = ses_scores["ses_factor1_score"]
    df_analysis.loc[ses_scores.index, "ses_factor2_score"] = ses_scores["ses_factor2_score"]

    return df_analysis



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

ses_label_map = {
    "ses_factor1_score": "SES Factor 1",
    "ses_factor2_score": "SES Factor 2",
    "ses_index": "SES index",
}

outcome_label_map = {
    "ai_factor1_score": "AI conceptual understanding",
    "ai_factor2_score": "AI ability/confidence",
}

mediator_label_map = {
    "conceptual_exposure_score": "Conceptual exposure",
    "practical_ai_use_score": "Practical AI use",
    "learning_ecology_score": "Learning ecology",
    "language_load_score": "Language load",
    "epistemic_stance_score": "Epistemic stance",
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



# Run all simple mediations for every combination of 2*SES, 5*mediator, and 2*AI literacy,
def run_all_simple_mediations(
    df, x="ses_index", mediators=None, outcomes=None, sample_list=None, n_boot=3000, seed=2026
):

    if mediators is None:
        mediators = mediator_vars
    if outcomes is None:
        outcomes = outcome_vars
    if sample_list is None:
        sample_list = ["1111", "1204", "Combined"]

    # allow one SES variable or many
    if isinstance(x, str):
        x_list = [x]
    else:
        x_list = list(x)

    rows = []
    counter = 0

    for sample in sample_list:
        for x_var in x_list:
            for m in mediators:
                for y in outcomes:
                    res, _, _, _, _ = fit_simple_mediation(
                        df=df,
                        sample=sample,
                        x=x_var,
                        m=m,
                        y=y
                    )

                    boot_res, _ = bootstrap_indirect_effect(
                        df=df,
                        sample=sample,
                        x=x_var,
                        m=m,
                        y=y,
                        n_boot=n_boot,
                        seed=seed + counter
                    )
                    counter += 1

                    merged = res.merge(
                        boot_res,
                        on=["sample", "mediator", "outcome"],
                        how="left"
                    )

                    merged["ses_dimension"] = x_var
                    # Bootstrap p-value
                    merged["p_indirect_boot"] = 2 * np.minimum(
                        merged["prop_ab_positive"],
                        1 - merged["prop_ab_positive"]
                    )
                    rows.append(merged)

    out = pd.concat(rows, ignore_index=True)

    # put ses_dimension near the front
    first_cols = ["sample", "ses_dimension", "mediator", "outcome"]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + other_cols]

    return out



# =============================================================================
# 11.  Model with Interaction Effects
# =============================================================================

def fit_interaction_mediation_model(
    df,
    sample,
    x,
    mediator1,
    y,
    mediator2=None,
    interaction_type="ses_mediator"
):
    """
    interaction_type:
        - "ses_mediator"      : Y ~ X + M1 + X*M1
        - "mediator_mediator" : Y ~ X + M1 + M2 + M1*M2
        - "both"              : Y ~ X + M1 + M2 + X*M1 + X*M2 + M1*M2
    """

    needed = [x, mediator1, y]
    if mediator2 is not None:
        needed.append(mediator2)

    if sample == "Combined":
        d = df[needed].dropna().copy()
    else:
        d = df.loc[df["course"] == sample, needed].dropna().copy()

    # standardize within analysis sample
    for col in needed:
        d[col] = (d[col] - d[col].mean()) / d[col].std(ddof=0)

    # a-path for mediator1
    Xa1 = sm.add_constant(d[[x]])
    model_a1 = sm.OLS(d[mediator1], Xa1).fit(cov_type="HC3")

    model_a2 = None
    if mediator2 is not None:
        Xa2 = sm.add_constant(d[[x]])
        model_a2 = sm.OLS(d[mediator2], Xa2).fit(cov_type="HC3")

    # total effect
    Xt = sm.add_constant(d[[x]])
    model_total = sm.OLS(d[y], Xt).fit(cov_type="HC3")

    # build outcome model
    X_cols = [x, mediator1]

    if interaction_type == "ses_mediator":
        d["x_m1"] = d[x] * d[mediator1]
        X_cols += ["x_m1"]

    elif interaction_type == "mediator_mediator":
        if mediator2 is None:
            raise ValueError("mediator2 is required for interaction_type='mediator_mediator'")
        d["m1_m2"] = d[mediator1] * d[mediator2]
        X_cols += [mediator2, "m1_m2"]

    elif interaction_type == "both":
        if mediator2 is None:
            raise ValueError("mediator2 is required for interaction_type='both'")
        d["x_m1"] = d[x] * d[mediator1]
        d["x_m2"] = d[x] * d[mediator2]
        d["m1_m2"] = d[mediator1] * d[mediator2]
        X_cols += [mediator2, "x_m1", "x_m2", "m1_m2"]

    else:
        raise ValueError("interaction_type must be 'ses_mediator', 'mediator_mediator', or 'both'")

    Xy = sm.add_constant(d[X_cols])
    model_y = sm.OLS(d[y], Xy).fit(cov_type="HC3")

    out = {
        "sample": sample,
        "ses_dimension": x,
        "mediator1": mediator1,
        "mediator2": mediator2,
        "outcome": y,
        "interaction_type": interaction_type,
        "n": len(d),

        "a_m1": model_a1.params[x],
        "a_m1_p": model_a1.pvalues[x],

        "c_total": model_total.params[x],
        "c_total_p": model_total.pvalues[x],

        "b_m1": model_y.params.get(mediator1, np.nan),
        "b_m1_p": model_y.pvalues.get(mediator1, np.nan),

        "b_x": model_y.params.get(x, np.nan),
        "b_x_p": model_y.pvalues.get(x, np.nan),

        "r2_total_model": model_total.rsquared,
        "r2_outcome_model": model_y.rsquared,
    }

    if model_a2 is not None:
        out["a_m2"] = model_a2.params[x]
        out["a_m2_p"] = model_a2.pvalues[x]
    else:
        out["a_m2"] = np.nan
        out["a_m2_p"] = np.nan

    if interaction_type == "ses_mediator":
        out["b_m2"] = np.nan
        out["b_m2_p"] = np.nan
        out["b_x_m1"] = model_y.params.get("x_m1", np.nan)
        out["b_x_m1_p"] = model_y.pvalues.get("x_m1", np.nan)
        out["b_x_m2"] = np.nan
        out["b_x_m2_p"] = np.nan
        out["b_m1_m2"] = np.nan
        out["b_m1_m2_p"] = np.nan

    elif interaction_type == "mediator_mediator":
        out["b_m2"] = model_y.params.get(mediator2, np.nan)
        out["b_m2_p"] = model_y.pvalues.get(mediator2, np.nan)
        out["b_x_m1"] = np.nan
        out["b_x_m1_p"] = np.nan
        out["b_x_m2"] = np.nan
        out["b_x_m2_p"] = np.nan
        out["b_m1_m2"] = model_y.params.get("m1_m2", np.nan)
        out["b_m1_m2_p"] = model_y.pvalues.get("m1_m2", np.nan)

    elif interaction_type == "both":
        out["b_m2"] = model_y.params.get(mediator2, np.nan)
        out["b_m2_p"] = model_y.pvalues.get(mediator2, np.nan)
        out["b_x_m1"] = model_y.params.get("x_m1", np.nan)
        out["b_x_m1_p"] = model_y.pvalues.get("x_m1", np.nan)
        out["b_x_m2"] = model_y.params.get("x_m2", np.nan)
        out["b_x_m2_p"] = model_y.pvalues.get("x_m2", np.nan)
        out["b_m1_m2"] = model_y.params.get("m1_m2", np.nan)
        out["b_m1_m2_p"] = model_y.pvalues.get("m1_m2", np.nan)

    return pd.DataFrame([out]), model_a1, model_a2, model_total, model_y, d



def run_interaction_mediation_models(
    df,
    ses_cols=None,
    mediators=None,
    outcomes=None,
    sample="Combined",
    interaction_type="ses_mediator"
):
    if ses_cols is None:
        ses_cols = ["ses_index", "ses_factor1_score", "ses_factor2_score"]

    if mediators is None:
        mediators = [
            "practical_ai_use_score",
            "learning_ecology_score",
            "language_load_score",
            "epistemic_stance_score",
        ]

    if outcomes is None:
        outcomes = ["ai_factor1_score", "ai_factor2_score", "ai_lit_score"]

    rows = []

    for x in ses_cols:
        for y in outcomes:
            if interaction_type == "ses_mediator":
                for m1 in mediators:
                    res, _, _, _, _, _ = fit_interaction_mediation_model(
                        df=df,
                        sample=sample,
                        x=x,
                        mediator1=m1,
                        mediator2=None,
                        y=y,
                        interaction_type=interaction_type
                    )
                    rows.append(res)

            elif interaction_type in ["mediator_mediator", "both"]:
                for m1, m2 in itertools.combinations(mediators, 2):
                    res, _, _, _, _, _ = fit_interaction_mediation_model(
                        df=df,
                        sample=sample,
                        x=x,
                        mediator1=m1,
                        mediator2=m2,
                        y=y,
                        interaction_type=interaction_type
                    )
                    rows.append(res)

            else:
                raise ValueError("interaction_type must be 'ses_mediator', 'mediator_mediator', or 'both'")

    return pd.concat(rows, ignore_index=True)



def summarize_interaction_mediation_results(results, alpha=0.05, interaction_only=False):
    d = results.copy()

    d["sig_a_m1"] = d["a_m1_p"] < alpha
    d["sig_a_m2"] = d["a_m2_p"] < alpha
    d["sig_b_m1"] = d["b_m1_p"] < alpha
    d["sig_b_m2"] = d["b_m2_p"] < alpha
    d["sig_b_x"] = d["b_x_p"] < alpha

    d["sig_x_m1"] = d["b_x_m1_p"] < alpha if "b_x_m1_p" in d.columns else False
    d["sig_x_m2"] = d["b_x_m2_p"] < alpha if "b_x_m2_p" in d.columns else False
    d["sig_m1_m2"] = d["b_m1_m2_p"] < alpha if "b_m1_m2_p" in d.columns else False

    sig_cols = [
        "sig_a_m1", "sig_a_m2", "sig_b_m1", "sig_b_m2",
        "sig_b_x", "sig_x_m1", "sig_x_m2", "sig_m1_m2"
    ]
    d["n_sig_terms"] = d[sig_cols].sum(axis=1)

    if interaction_only:
        d = d[(d["sig_x_m1"]) | (d["sig_x_m2"]) | (d["sig_m1_m2"])].copy()
    else:
        d = d[d["n_sig_terms"] > 0].copy()

    keep_cols = [
        "ses_dimension",
        "mediator1",
        "mediator2",
        "outcome",
        "interaction_type",
        "a_m1", "a_m1_p",
        "a_m2", "a_m2_p",
        "b_m1", "b_m1_p",
        "b_m2", "b_m2_p",
        "b_x", "b_x_p",
        "b_x_m1", "b_x_m1_p",
        "b_x_m2", "b_x_m2_p",
        "b_m1_m2", "b_m1_m2_p",
        "n_sig_terms",
        "r2_outcome_model",
    ]
    keep_cols = [c for c in keep_cols if c in d.columns]

    d = d[keep_cols].sort_values(
        by=["n_sig_terms", "r2_outcome_model"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return d



def plot_significant_mediator_interactions(results, alpha=0.05, sample="Combined", figsize=(16, 6)):
    d = results.copy()
    d = d[d["sample"] == sample].copy()

    # collect significant interaction terms into one long table
    rows = []

    for _, row in d.iterrows():
        interaction_type = row.get("interaction_type", "")

        if ("b_x_m1_p" in d.columns) and pd.notna(row.get("b_x_m1_p")) and row["b_x_m1_p"] < alpha:
            rows.append({
                "ses_dimension": row["ses_dimension"],
                "mediator1": row["mediator1"],
                "mediator2": row.get("mediator2", np.nan),
                "outcome": row["outcome"],
                "interaction_type": interaction_type,
                "term": "SES × mediator1",
                "coef": row["b_x_m1"],
                "p_value": row["b_x_m1_p"],
            })

        if ("b_x_m2_p" in d.columns) and pd.notna(row.get("b_x_m2_p")) and row["b_x_m2_p"] < alpha:
            rows.append({
                "ses_dimension": row["ses_dimension"],
                "mediator1": row["mediator1"],
                "mediator2": row.get("mediator2", np.nan),
                "outcome": row["outcome"],
                "interaction_type": interaction_type,
                "term": "SES × mediator2",
                "coef": row["b_x_m2"],
                "p_value": row["b_x_m2_p"],
            })

        if ("b_m1_m2_p" in d.columns) and pd.notna(row.get("b_m1_m2_p")) and row["b_m1_m2_p"] < alpha:
            rows.append({
                "ses_dimension": row["ses_dimension"],
                "mediator1": row["mediator1"],
                "mediator2": row.get("mediator2", np.nan),
                "outcome": row["outcome"],
                "interaction_type": interaction_type,
                "term": "mediator1 × mediator2",
                "coef": row["b_m1_m2"],
                "p_value": row["b_m1_m2_p"],
            })

    sig_df = pd.DataFrame(rows)

    if sig_df.empty:
        print("No significant interaction terms found.")
        return

    sig_df["ses_dimension"] = sig_df["ses_dimension"].replace(ses_map)
    sig_df["mediator1"] = sig_df["mediator1"].replace(mediator_label_map)
    sig_df["mediator2"] = sig_df["mediator2"].replace(mediator_label_map)
    sig_df["outcome"] = sig_df["outcome"].replace(outcome_map)

    def make_label(row):
        if row["term"] == "SES × mediator1":
            return f'{row["ses_dimension"]} × {row["mediator1"]} → {row["outcome"]}'
        elif row["term"] == "SES × mediator2":
            return f'{row["ses_dimension"]} × {row["mediator2"]} → {row["outcome"]}'
        else:
            return f'{row["mediator1"]} × {row["mediator2"]} → {row["outcome"]}'

    sig_df["label"] = sig_df.apply(make_label, axis=1)
    sig_df = sig_df.sort_values("coef").reset_index(drop=True)

    plt.figure(figsize=figsize)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.scatter(sig_df["coef"], np.arange(len(sig_df)), s=90)
    plt.yticks(np.arange(len(sig_df)), sig_df["label"])
    plt.xlabel("Interaction coefficient")
    plt.title(f"Significant interaction terms ({sample})")
    plt.tight_layout()
    plt.show()
