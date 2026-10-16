"""Small-sample utilities for the Phase I high-school report.

The module reuses the validated questionnaire codebook and scoring rules in
``functions.py``.  The high-school sample is intentionally used for
description and exploratory comparisons only.  The two-factor EFA models are
estimated from the earlier 232-student Phase I reference sample and then
applied to the high-school responses.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from statsmodels.stats.multitest import multipletests

from functions import (
    AI_EFA_ITEMS,
    SES_EFA_ITEMS_2DIM,
    add_mediator_composites,
    add_ses_space_house_variables,
    clean_text,
    cronbach_alpha,
    extract_leading_code,
    fit_efa,
    mediator_map,
    phase_i_schema,
    prepare_dataset,
    score_pre_ai_items,
    score_ses_index,
    validate_unique_ids,
)


PALETTE = {
    "navy": "#172554",
    "blue": "#2563EB",
    "cyan": "#06B6D4",
    "orange": "#F97316",
    "magenta": "#DB2777",
    "green": "#10B981",
    "slate": "#475569",
    "light": "#F1F5F9",
}

AI_FACTOR_LABELS = {
    "ai_understanding": "AI understanding",
    "ai_confidence": "AI confidence",
}

SES_FACTOR_LABELS = {
    "ses_education": "Parental educational capital",
    "ses_resources": "Household material resources",
}

MEDIATOR_LABELS = {
    "practical_ai_use_score": "Practical AI use",
    "language_load_score": "Language difficulty",
    "epistemic_stance_score": "Evaluative stance",
}

OUTCOME_LABELS = {
    **AI_FACTOR_LABELS,
    **MEDIATOR_LABELS,
}

AI_ITEM_LABELS = {
    "ai_concept_data_bias_scored_num": "Recognises data bias",
    "ai_concept_blackbox_scored_num": "Recognises the black-box problem",
    "ai_concept_input_variation_scored_num": "Recognises input sensitivity",
    "ai_concept_prompt_wording_scored_num": "Recognises prompt wording effects",
    "ai_concept_social_ethics_scored_num": "Recognises social and ethical limits",
    "ai_ability_training_data_scored_num": "Can explain training data",
    "ai_ability_explainability_scored_num": "Can explain AI decisions",
    "ai_ability_input_sensitivity_scored_num": "Can manage input sensitivity",
    "ai_ability_prompting_scored_num": "Can write effective prompts",
    "ai_ability_social_ethics_scored_num": "Can address social and ethical issues",
}

AI_IDEA_PAIRS = {
    "Data and training": (
        "ai_concept_data_bias_scored_num",
        "ai_ability_training_data_scored_num",
    ),
    "Black box and explainability": (
        "ai_concept_blackbox_scored_num",
        "ai_ability_explainability_scored_num",
    ),
    "Input sensitivity": (
        "ai_concept_input_variation_scored_num",
        "ai_ability_input_sensitivity_scored_num",
    ),
    "Prompt wording": (
        "ai_concept_prompt_wording_scored_num",
        "ai_ability_prompting_scored_num",
    ),
    "Social and ethical impact": (
        "ai_concept_social_ethics_scored_num",
        "ai_ability_social_ethics_scored_num",
    ),
}


def set_report_style():
    """Apply a restrained, high-contrast style used across the report."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _parse_phase_i_files(paths, school_labels):
    """Load and parse Phase I files while preserving non-substantive missing codes."""
    if len(paths) != len(school_labels):
        raise ValueError("paths and school_labels must have the same length")

    parts = []
    for path, school in zip(paths, school_labels):
        raw = pd.read_csv(path)
        _, rename = phase_i_schema(raw.columns, context=f"High school {school}")
        part = raw.rename(columns=rename).copy()
        part["school"] = str(school)
        part["source_file"] = Path(path).name
        parts.append(part)

    df = pd.concat(parts, ignore_index=True, sort=False)
    df["id"] = validate_unique_ids(df, "id", context="Combined high-school Phase I")

    for col in df.columns:
        df[col] = df[col].map(clean_text)

    non_extract = {
        "id",
        "school",
        "source_file",
        "ses_lang_cantonese",
        "ses_lang_mandarin",
        "ses_lang_english",
        "ses_lang_other",
    }
    for col in df.columns:
        if col not in non_extract:
            df[f"{col}_num"] = df[col].apply(extract_leading_code)

    for col in [
        "ses_lang_cantonese",
        "ses_lang_mandarin",
        "ses_lang_english",
        "ses_lang_other",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Code 7 means "Not applicable" for parent education.  It is missing
    # information, not a higher education level.
    for col in ["ses_parent1_edu_num", "ses_parent2_edu_num"]:
        df[col] = df[col].where(df[col].between(1, 6))

    df = score_pre_ai_items(df)
    df["ai_lit_score"] = df[list(AI_EFA_ITEMS)].mean(axis=1)

    # Apply the canonical directions without estimating a high-school-only
    # z-score index.  Higher values consistently mean more resources.
    df["ses_device_access_scored_num"] = 4 - df["ses_device_access_num"]
    df["ses_internet_quality_scored_num"] = 5 - df["ses_internet_quality_num"]
    df["ses_financial_constraint_scored_num"] = 5 - df["ses_financial_constraint_num"]

    df = add_mediator_composites(df, mediator_map)
    df = add_ses_space_house_variables(df)
    df["prior_ai_course"] = df["ai_course_taken_num"].map({0: "No", 1: "Yes"})
    return df


def fit_phase_i_reference_models(base_dir="datasets"):
    """Fit the established two-factor EFA models on the 232-student reference sample."""
    base = Path(base_dir)
    legacy, _ = prepare_dataset(
        base / "AI_Lit_Que_1111_25.csv",
        base / "AI_Lit_Que_1204_25.csv",
        course_labels=["1111_25", "1204_25"],
    )
    current, _ = prepare_dataset(
        base / "AI_Lit_Que_1111_26.csv",
        base / "AI_Lit_Que_1201_26.csv",
        course_labels=["1111_26", "1201_26"],
    )
    reference = pd.concat([legacy, current], ignore_index=True, sort=False)
    reference["id"] = validate_unique_ids(reference, "id", context="Phase I reference")
    reference, _, _ = score_ses_index(reference)
    reference["course"] = "Combined"
    reference_ses = add_ses_space_house_variables(reference)

    ai_fa, _, ai_loadings, ai_variance = fit_efa(
        reference,
        "Combined",
        list(AI_EFA_ITEMS),
        n_factors=2,
        rotation="oblimin",
        method="minres",
    )
    ses_fa, _, ses_loadings, ses_variance = fit_efa(
        reference_ses,
        "Combined",
        list(SES_EFA_ITEMS_2DIM),
        n_factors=2,
        rotation="oblimin",
        method="minres",
    )

    ai_signs = np.array(
        [
            1 if ai_loadings.loc[list(AI_EFA_ITEMS)[:5], "Factor1"].mean() >= 0 else -1,
            1 if ai_loadings.loc[list(AI_EFA_ITEMS)[5:], "Factor2"].mean() >= 0 else -1,
        ]
    )
    ses_signs = np.array(
        [
            1
            if ses_loadings.loc[
                ["ses_parent1_edu_num", "ses_parent2_edu_num"], "Factor1"
            ].mean()
            >= 0
            else -1,
            1
            if ses_loadings.loc[
                [
                    "ses_household_income_num",
                    "ses_financial_constraint_scored_num",
                    "ses_space_per_person",
                    "ses_housing_type_ord",
                ],
                "Factor2",
            ].mean()
            >= 0
            else -1,
        ]
    )

    return {
        "reference_n": len(reference),
        "ai_fa": ai_fa,
        "ses_fa": ses_fa,
        "ai_signs": ai_signs,
        "ses_signs": ses_signs,
        "ai_loadings": ai_loadings,
        "ses_loadings": ses_loadings,
        "ai_variance": ai_variance,
        "ses_variance": ses_variance,
    }


def prepare_hs_analysis(
    paths=("datasets/HS/Invictus_PhaseI.csv", "datasets/HS/STTSS_PhaseI.csv"),
    school_labels=("Invictus", "STTSS"),
    reference_dir="datasets",
):
    """Return the scored high-school data and the fixed reference EFA model."""
    df = _parse_phase_i_files(paths, school_labels)
    reference = fit_phase_i_reference_models(reference_dir)

    ai_complete = df[list(AI_EFA_ITEMS)].dropna()
    ai_scores = reference["ai_fa"].transform(ai_complete) * reference["ai_signs"]
    df.loc[ai_complete.index, ["ai_understanding", "ai_confidence"]] = ai_scores

    ses_complete = df[list(SES_EFA_ITEMS_2DIM)].dropna()
    ses_scores = reference["ses_fa"].transform(ses_complete) * reference["ses_signs"]
    df.loc[ses_complete.index, ["ses_education", "ses_resources"]] = ses_scores

    for factor, label in SES_FACTOR_LABELS.items():
        valid = df[factor].dropna().sort_values()
        lower_n = len(valid) // 2
        group = pd.Series(pd.NA, index=df.index, dtype="object")
        group.loc[valid.index[:lower_n]] = "Lower"
        group.loc[valid.index[lower_n:]] = "Higher"
        df[f"{factor}_group"] = pd.Categorical(
            group, categories=["Lower", "Higher"], ordered=True
        )

    return df, reference


def sample_overview(df):
    """Create a compact sample overview table."""
    rows = [
        ("All responses", int(len(df))),
        ("Complete SES profiles", int(df["ses_education"].notna().sum())),
        ("Prior AI course: yes", int(df["ai_course_taken_num"].eq(1).sum())),
        ("Prior AI course: no", int(df["ai_course_taken_num"].eq(0).sum())),
    ]
    for school, n in df["school"].value_counts().items():
        rows.append((f"{school} responses", int(n)))
    return pd.DataFrame(rows, columns=["Measure", "Count"])


def construct_descriptives(df):
    """Descriptive statistics for the two AI factors and retained mediators."""
    columns = list(AI_FACTOR_LABELS) + list(MEDIATOR_LABELS)
    out = (
        df[columns]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .T.reset_index()
        .rename(columns={"index": "measure", "count": "n", "std": "sd"})
    )
    out["Measure"] = out["measure"].map(OUTCOME_LABELS)
    return out[["measure", "Measure", "n", "mean", "sd", "median", "min", "max"]]


def ai_item_profile(df):
    """Rank scored AI-literacy items from lowest to highest mean."""
    rows = []
    conceptual = set(list(AI_EFA_ITEMS)[:5])
    for col in AI_EFA_ITEMS:
        values = df[col].dropna()
        sem = values.std(ddof=1) / np.sqrt(len(values))
        ci = stats.t.ppf(0.975, max(len(values) - 1, 1)) * sem
        rows.append(
            {
                "item": col,
                "Item": AI_ITEM_LABELS[col],
                "Dimension": "AI understanding" if col in conceptual else "AI confidence",
                "n": len(values),
                "mean": values.mean(),
                "ci_low": values.mean() - ci,
                "ci_high": values.mean() + ci,
            }
        )
    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)


def ai_idea_consistency(df):
    """Check whether the two scored questions within each AI idea move together."""
    rows = []
    for idea, pair in AI_IDEA_PAIRS.items():
        rows.append(
            {
                "Big Idea": idea,
                "Cronbach alpha": cronbach_alpha(df[list(pair)]),
                "Complete pairs": int(df[list(pair)].dropna().shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _hedges_g(higher, lower):
    higher = np.asarray(pd.Series(higher).dropna(), dtype=float)
    lower = np.asarray(pd.Series(lower).dropna(), dtype=float)
    n1, n0 = len(higher), len(lower)
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan, np.nan
    pooled = np.sqrt(
        ((n1 - 1) * np.var(higher, ddof=1) + (n0 - 1) * np.var(lower, ddof=1))
        / (n1 + n0 - 2)
    )
    if pooled == 0:
        return np.nan, np.nan, np.nan
    d = (np.mean(higher) - np.mean(lower)) / pooled
    correction = 1 - 3 / (4 * (n1 + n0) - 9)
    g = correction * d
    se = np.sqrt((n1 + n0) / (n1 * n0) + (g**2) / (2 * (n1 + n0 - 2)))
    return g, g - 1.96 * se, g + 1.96 * se


def compare_ses_groups(df):
    """Welch comparisons for five outcomes across each rank-based SES split."""
    outcomes = list(AI_FACTOR_LABELS) + list(MEDIATOR_LABELS)
    rows = []
    for factor, factor_label in SES_FACTOR_LABELS.items():
        group_col = f"{factor}_group"
        for outcome in outcomes:
            higher = df.loc[df[group_col].eq("Higher"), outcome].dropna()
            lower = df.loc[df[group_col].eq("Lower"), outcome].dropna()
            test = stats.ttest_ind(higher, lower, equal_var=False, nan_policy="omit")
            diff = higher.mean() - lower.mean()
            se = np.sqrt(higher.var(ddof=1) / len(higher) + lower.var(ddof=1) / len(lower))
            dof_num = (higher.var(ddof=1) / len(higher) + lower.var(ddof=1) / len(lower)) ** 2
            dof_den = (
                (higher.var(ddof=1) / len(higher)) ** 2 / (len(higher) - 1)
                + (lower.var(ddof=1) / len(lower)) ** 2 / (len(lower) - 1)
            )
            dof = dof_num / dof_den if dof_den > 0 else np.nan
            critical = stats.t.ppf(0.975, dof) if np.isfinite(dof) else np.nan
            g, g_low, g_high = _hedges_g(higher, lower)
            rows.append(
                {
                    "ses_factor": factor,
                    "SES dimension": factor_label,
                    "outcome": outcome,
                    "Outcome": OUTCOME_LABELS[outcome],
                    "n_lower": len(lower),
                    "n_higher": len(higher),
                    "mean_lower": lower.mean(),
                    "mean_higher": higher.mean(),
                    "difference": diff,
                    "difference_ci_low": diff - critical * se,
                    "difference_ci_high": diff + critical * se,
                    "welch_t": test.statistic,
                    "p_value": test.pvalue,
                    "hedges_g": g,
                    "g_ci_low": g_low,
                    "g_ci_high": g_high,
                }
            )
    out = pd.DataFrame(rows)
    out["p_bh"] = out.groupby("ses_factor")["p_value"].transform(
        lambda values: multipletests(values, method="fdr_bh")[1]
    )
    return out


def continuous_associations(df):
    """Spearman associations using the continuous EFA factor scores."""
    outcomes = list(AI_FACTOR_LABELS) + list(MEDIATOR_LABELS)
    rows = []
    for factor, factor_label in SES_FACTOR_LABELS.items():
        for outcome in outcomes:
            d = df[[factor, outcome]].dropna()
            rho, p_value = stats.spearmanr(d[factor], d[outcome])
            rows.append(
                {
                    "ses_factor": factor,
                    "SES dimension": factor_label,
                    "outcome": outcome,
                    "Outcome": OUTCOME_LABELS[outcome],
                    "n": len(d),
                    "rho": rho,
                    "p_value": p_value,
                }
            )
    out = pd.DataFrame(rows)
    out["p_bh"] = out.groupby("ses_factor")["p_value"].transform(
        lambda values: multipletests(values, method="fdr_bh")[1]
    )
    return out


def prior_course_fisher_tests(df):
    """Fisher exact tests for prior AI-course prevalence across SES groups."""
    rows = []
    for factor, factor_label in SES_FACTOR_LABELS.items():
        group_col = f"{factor}_group"
        table = pd.crosstab(df[group_col], df["ai_course_taken_num"]).reindex(
            index=["Lower", "Higher"], columns=[0, 1], fill_value=0
        )
        odds_ratio, p_value = stats.fisher_exact(table.values)
        rows.append(
            {
                "SES dimension": factor_label,
                "lower_prior_course": int(table.loc["Lower", 1]),
                "lower_total": int(table.loc["Lower"].sum()),
                "higher_prior_course": int(table.loc["Higher", 1]),
                "higher_total": int(table.loc["Higher"].sum()),
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def plot_score_construction_tree(consistency=None):
    """Illustrate how paired AI questions and SES items become four EFA scores."""
    set_report_style()
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, width, height, label, face, edge, size=9, weight="bold"):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.3,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            color=PALETTE["navy"],
            linespacing=1.08,
        )
        return (x + width / 2, y, x + width / 2, y + height)

    def arrow(start, end, color=PALETTE["slate"], width=1.15):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": width, "shrinkA": 1, "shrinkB": 1},
        )

    ax.text(0.01, 0.965, "AI literacy: 10 questions across five Big Ideas", fontsize=13, fontweight="bold", color=PALETTE["navy"], va="top")
    ai_root = box(0.385, 0.855, 0.23, 0.065, "AI-literacy questions (10)", "#E0F2FE", PALETTE["blue"], size=10)

    consistency_values = {}
    if consistency is not None:
        consistency_values = consistency.set_index("Big Idea")["Cronbach alpha"].to_dict()
    short_idea_labels = {
        "Data and training": "Data and training",
        "Black box and explainability": "Black box and explainability",
        "Input sensitivity": "Input sensitivity",
        "Prompt wording": "Prompt wording",
        "Social and ethical impact": "Social and ethical impact",
    }
    idea_labels = []
    for idea in AI_IDEA_PAIRS:
        label = short_idea_labels[idea]
        if idea in consistency_values:
            label = f"{label}\nPair consistency = {consistency_values[idea]:.2f}"
        idea_labels.append(label)
    idea_x = [0.012, 0.210, 0.408, 0.606, 0.804]
    idea_centres = [x + 0.092 for x in idea_x]
    ax.plot([0.104, 0.896], [0.822, 0.822], color=PALETTE["slate"], linewidth=1.1)
    ax.plot([0.5, 0.5], [0.855, 0.822], color=PALETTE["slate"], linewidth=1.1)

    understanding_centres = []
    confidence_centres = []
    for x, centre, label in zip(idea_x, idea_centres, idea_labels):
        ax.plot([centre, centre], [0.822, 0.805], color=PALETTE["slate"], linewidth=1.1)
        box(x, 0.715, 0.184, 0.090, label, "#F8FAFC", "#94A3B8", size=9.0)
        u = box(x, 0.615, 0.086, 0.068, "Understanding\nquestion", "#DBEAFE", PALETTE["blue"], size=7.4)
        c = box(x + 0.098, 0.615, 0.086, 0.068, "Confidence\nquestion", "#FFEDD5", PALETTE["orange"], size=7.4)
        arrow((centre, 0.715), (u[0], 0.683), PALETTE["blue"], 1.0)
        arrow((centre, 0.715), (c[0], 0.683), PALETTE["orange"], 1.0)
        understanding_centres.append(u[0])
        confidence_centres.append(c[0])

    u_bus_y, c_bus_y = 0.575, 0.535
    ax.plot([min(understanding_centres), max(understanding_centres)], [u_bus_y, u_bus_y], color=PALETTE["blue"], linewidth=1.6)
    ax.plot([min(confidence_centres), max(confidence_centres)], [c_bus_y, c_bus_y], color=PALETTE["orange"], linewidth=1.6)
    for centre in understanding_centres:
        ax.plot([centre, centre], [0.615, u_bus_y], color=PALETTE["blue"], linewidth=1.25)
    for centre in confidence_centres:
        ax.plot([centre, centre], [0.615, c_bus_y], color=PALETTE["orange"], linewidth=1.25)

    box(0.245, 0.425, 0.235, 0.075, "AI understanding\n5 understanding questions", "#DBEAFE", PALETTE["blue"], size=9.3)
    box(0.520, 0.425, 0.235, 0.075, "AI confidence\n5 confidence questions", "#FFEDD5", PALETTE["orange"], size=9.3)
    arrow((0.363, u_bus_y), (0.363, 0.500), PALETTE["blue"], 1.6)
    arrow((0.638, c_bus_y), (0.638, 0.500), PALETTE["orange"], 1.6)
    ax.text(0.5, 0.517, "EFA groups similar response patterns", ha="center", va="center", fontsize=8.4, color=PALETTE["slate"], fontweight="bold")

    box(
        0.145,
        0.335,
        0.710,
        0.047,
        "Consistency check: within each Big Idea, do its two paired questions tell a similar story?",
        "#FEF3C7",
        "#F59E0B",
        size=10.0,
    )

    ax.plot([0.01, 0.99], [0.300, 0.300], color="#CBD5E1", linewidth=1.0)
    ax.text(0.01, 0.280, "Socioeconomic background: six measures", fontsize=13, fontweight="bold", color=PALETTE["navy"], va="top")

    ses_root = box(0.020, 0.120, 0.155, 0.075, "SES measures\n(6)", "#DCFCE7", PALETTE["green"], size=9.5)
    edu_group = box(0.255, 0.185, 0.300, 0.062, "Parent 1 + Parent 2 education", "#F0FDF4", PALETTE["green"], size=8.7)
    resource_group = box(0.255, 0.065, 0.300, 0.082, "Income + financial strain +\nliving space + housing", "#F0FDF4", PALETTE["green"], size=8.7)
    edu_factor = box(0.690, 0.185, 0.285, 0.062, "Parental educational capital", "#DCFCE7", PALETTE["green"], size=9.0)
    resource_factor = box(0.690, 0.065, 0.285, 0.082, "Household material resources", "#DCFCE7", PALETTE["green"], size=9.0)
    arrow((ses_root[2], 0.158), (0.255, 0.216), PALETTE["green"], 1.4)
    arrow((ses_root[2], 0.158), (0.255, 0.106), PALETTE["green"], 1.4)
    arrow((edu_group[2], 0.216), (0.690, 0.216), PALETTE["green"], 1.4)
    arrow((resource_group[2], 0.106), (0.690, 0.106), PALETTE["green"], 1.4)
    ax.text(0.622, 0.160, "EFA", ha="center", va="center", fontsize=9.5, fontweight="bold", color=PALETTE["slate"])

    fig.tight_layout()
    return fig


def plot_preprocessing_flowchart(consistency=None):
    """Portrait A4 workflow for preparing AI, SES, and mediator variables."""
    set_report_style()
    fig, ax = plt.subplots(figsize=(7.2, 10.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def panel(y, height, title, edge):
        patch = FancyBboxPatch(
            (0.02, y),
            0.96,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="#FFFFFF",
            edgecolor=edge,
            linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(0.05, y + height - 0.026, title, ha="left", va="center", fontsize=13, fontweight="bold", color=PALETTE["navy"])

    def box(x, y, width, height, label, face, edge, size=9.5, weight="bold"):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.010",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.3,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            color=PALETTE["navy"],
            linespacing=1.14,
        )

    def arrow(start, end, color=PALETTE["slate"], width=1.4):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": width, "shrinkA": 1, "shrinkB": 1},
        )

    consistency_values = {}
    if consistency is not None:
        consistency_values = consistency.set_index("Big Idea")["Cronbach alpha"].to_dict()

    panel(0.515, 0.465, "1. AI literacy", PALETTE["blue"])
    box(0.20, 0.865, 0.60, 0.050, "10 scored questions", "#E0F2FE", PALETTE["blue"], size=11)
    arrow((0.50, 0.865), (0.50, 0.828), PALETTE["blue"])

    idea_lines = ["Five Big Ideas - each has one understanding and one confidence question"]
    for idea in AI_IDEA_PAIRS:
        value = consistency_values.get(idea, np.nan)
        suffix = f"pair consistency {value:.2f}" if np.isfinite(value) else "two-question pair"
        idea_lines.append(f"{idea}: {suffix}")
    box(0.09, 0.660, 0.82, 0.165, "\n".join(idea_lines), "#F8FAFC", "#94A3B8", size=9.0, weight="normal")
    arrow((0.50, 0.660), (0.50, 0.648), PALETTE["slate"])
    box(0.32, 0.605, 0.36, 0.040, "EFA groups similar response patterns", "#EEF2FF", PALETTE["navy"], size=9.2)
    ax.plot([0.50, 0.50], [0.605, 0.593], color=PALETTE["slate"], linewidth=1.3)
    ax.plot([0.27, 0.73], [0.593, 0.593], color=PALETTE["slate"], linewidth=1.3)
    arrow((0.27, 0.593), (0.27, 0.581), PALETTE["blue"])
    arrow((0.73, 0.593), (0.73, 0.581), PALETTE["orange"])
    box(0.07, 0.530, 0.40, 0.050, "AI understanding\n5 understanding questions", "#DBEAFE", PALETTE["blue"], size=9.5)
    box(0.53, 0.530, 0.40, 0.050, "AI confidence\n5 confidence questions", "#FFEDD5", PALETTE["orange"], size=9.5)

    panel(0.270, 0.225, "2. Socioeconomic background", PALETTE["green"])
    box(0.20, 0.410, 0.60, 0.040, "6 SES measures", "#DCFCE7", PALETTE["green"], size=10.5)
    ax.plot([0.50, 0.50], [0.410, 0.399], color=PALETTE["green"], linewidth=1.3)
    ax.plot([0.26, 0.74], [0.399, 0.399], color=PALETTE["green"], linewidth=1.3)
    arrow((0.26, 0.399), (0.26, 0.387), PALETTE["green"])
    arrow((0.74, 0.399), (0.74, 0.387), PALETTE["green"])
    box(0.05, 0.337, 0.42, 0.048, "Parent 1 + Parent 2 education", "#F0FDF4", PALETTE["green"], size=9.2)
    box(0.53, 0.337, 0.42, 0.048, "Income + financial strain +\nliving space + housing", "#F0FDF4", PALETTE["green"], size=8.8)
    arrow((0.26, 0.337), (0.26, 0.324), PALETTE["green"])
    arrow((0.74, 0.337), (0.74, 0.324), PALETTE["green"])
    ax.text(0.50, 0.327, "EFA", ha="center", va="center", fontsize=9, fontweight="bold", color=PALETTE["slate"])
    box(0.05, 0.282, 0.42, 0.040, "Parental educational capital", "#DCFCE7", PALETTE["green"], size=9.3)
    box(0.53, 0.282, 0.42, 0.040, "Household material resources", "#DCFCE7", PALETTE["green"], size=9.3)

    panel(0.010, 0.240, "3. Mediators", PALETTE["magenta"])
    box(0.20, 0.172, 0.60, 0.038, "10 questions: 5 mediator pairs", "#FCE7F3", PALETTE["magenta"], size=10.0)
    arrow((0.50, 0.172), (0.50, 0.157), PALETTE["magenta"])
    box(0.25, 0.120, 0.50, 0.034, "Average the two questions in each pair", "#FDF2F8", PALETTE["magenta"], size=9.2)
    ax.plot([0.50, 0.50], [0.120, 0.109], color=PALETTE["magenta"], linewidth=1.3)
    ax.plot([0.31, 0.80], [0.109, 0.109], color=PALETTE["magenta"], linewidth=1.3)
    arrow((0.31, 0.109), (0.31, 0.098), PALETTE["magenta"])
    arrow((0.80, 0.109), (0.80, 0.098), "#94A3B8")
    box(0.04, 0.026, 0.54, 0.070, "Retained for this report (3)\nPractical AI use | Language difficulty | Evaluative stance", "#FCE7F3", PALETTE["magenta"], size=9.0)
    box(0.62, 0.026, 0.34, 0.070, "Not retained (2)\nConceptual exposure\nLearning ecology", "#F1F5F9", "#94A3B8", size=8.8)

    fig.tight_layout(pad=0.4)
    return fig


def plot_analysis_workflow():
    """Portrait workflow showing how the prepared variables enter each test."""
    set_report_style()
    fig, ax = plt.subplots(figsize=(7.2, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, width, height, label, face, edge, size=9.5, weight="bold"):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.010",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.3,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=size, fontweight=weight, color=PALETTE["navy"], linespacing=1.15)

    def arrow(start, end, color=PALETTE["slate"]):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4, "shrinkA": 1, "shrinkB": 1})

    ax.text(0.03, 0.975, "Prepared variable families", fontsize=14, fontweight="bold", color=PALETTE["navy"], va="top")
    box(0.08, 0.865, 0.84, 0.075, "SES factors (2)\nParental educational capital | Household material resources", "#DCFCE7", PALETTE["green"], size=10)
    box(0.08, 0.770, 0.84, 0.075, "AI-literacy outcomes (2)\nAI understanding | AI confidence", "#DBEAFE", PALETTE["blue"], size=10)
    box(0.08, 0.675, 0.84, 0.075, "Retained mediators (3)\nPractical AI use | Language difficulty | Evaluative stance", "#FCE7F3", PALETTE["magenta"], size=10)

    ax.plot([0.03, 0.97], [0.640, 0.640], color="#CBD5E1", linewidth=1.0)
    ax.text(0.03, 0.615, "Analysis paths and named statistical tests", fontsize=14, fontweight="bold", color=PALETTE["navy"], va="top")

    box(0.10, 0.525, 0.80, 0.060, "Do lower and higher SES halves differ across the five outcomes?", "#F8FAFC", "#94A3B8", size=9.8)
    arrow((0.50, 0.525), (0.50, 0.500))
    box(0.18, 0.442, 0.64, 0.055, "Welch independent-samples t-test", "#EEF2FF", PALETTE["blue"], size=10)
    arrow((0.50, 0.442), (0.50, 0.418))
    box(0.18, 0.360, 0.64, 0.055, "Hedges g effect size + 95% interval + BH correction", "#EEF2FF", PALETTE["blue"], size=9.5)

    box(0.10, 0.275, 0.80, 0.060, "Do continuous SES scores move with the five outcomes?", "#F8FAFC", "#94A3B8", size=9.8)
    arrow((0.50, 0.275), (0.50, 0.250))
    box(0.18, 0.192, 0.64, 0.055, "Spearman rank correlation + BH correction", "#ECFEFF", PALETTE["cyan"], size=9.8)

    box(0.10, 0.105, 0.80, 0.060, "Is prior AI-course exposure distributed differently across SES halves?", "#F8FAFC", "#94A3B8", size=9.4)
    arrow((0.50, 0.105), (0.50, 0.080))
    box(0.25, 0.020, 0.50, 0.055, "Fisher exact test", "#FFF7ED", PALETTE["orange"], size=10.2)

    fig.tight_layout(pad=0.4)
    return fig


def plot_overview(df):
    """Show sample composition and prior AI-course exposure."""
    set_report_style()
    fig = plt.figure(figsize=(10.8, 5.2))
    grid = fig.add_gridspec(2, 4, height_ratios=[0.8, 1.2], hspace=0.35, wspace=0.5)
    cards = [
        (str(len(df)), "students analysed", PALETTE["navy"]),
        (str(df["ses_education"].notna().sum()), "complete SES profiles", PALETTE["blue"]),
        (str(df["ai_course_taken_num"].eq(1).sum()), "with prior AI course", PALETTE["magenta"]),
        (str(df["ai_course_taken_num"].eq(0).sum()), "without prior AI course", PALETTE["cyan"]),
    ]
    for i, (value, label, color) in enumerate(cards):
        ax = fig.add_subplot(grid[0, i])
        ax.set_facecolor(color)
        ax.text(0.5, 0.60, value, ha="center", va="center", fontsize=25, fontweight="bold", color="white")
        ax.text(0.5, 0.23, label, ha="center", va="center", fontsize=9, color="white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_school = fig.add_subplot(grid[1, :2])
    school_counts = df["school"].value_counts().sort_values()
    ax_school.barh(school_counts.index, school_counts.values, color=[PALETTE["orange"], PALETTE["blue"]])
    for y, value in enumerate(school_counts.values):
        ax_school.text(value + 0.25, y, str(value), va="center", fontweight="bold")
    ax_school.set_title("Responses by school", loc="left")
    ax_school.set_xlabel("Students")
    ax_school.set_xlim(0, max(school_counts.values) + 3)
    sns.despine(ax=ax_school, left=True)

    ax_course = fig.add_subplot(grid[1, 2:])
    course_counts = df["prior_ai_course"].value_counts().reindex(["No", "Yes"])
    ax_course.bar(course_counts.index, course_counts.values, color=[PALETTE["cyan"], PALETTE["magenta"]], width=0.55)
    for x, value in enumerate(course_counts.values):
        ax_course.text(x, value + 0.25, str(value), ha="center", fontweight="bold")
    ax_course.set_title("Prior AI-course experience", loc="left")
    ax_course.set_ylabel("Students")
    ax_course.set_ylim(0, max(course_counts.values) + 3)
    sns.despine(ax=ax_course)
    fig.suptitle("Who took part?", x=0.02, ha="left", fontsize=17, fontweight="bold", color=PALETTE["navy"])
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def _raincloud(ax, data, x, y, order, colors):
    sns.violinplot(data=data, x=x, y=y, order=order, palette=colors, inner=None, cut=0, linewidth=0, alpha=0.35, ax=ax)
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        order=order,
        width=0.22,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": PALETTE["navy"], "linewidth": 1.2},
        medianprops={"color": PALETTE["navy"], "linewidth": 1.6},
        whiskerprops={"color": PALETTE["navy"]},
        capprops={"color": PALETTE["navy"]},
        ax=ax,
    )
    sns.stripplot(data=data, x=x, y=y, order=order, color=PALETTE["navy"], alpha=0.72, jitter=0.13, size=4, ax=ax)


def plot_overall_distributions(df):
    """Show the two AI factors and three mediator score distributions."""
    set_report_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), gridspec_kw={"width_ratios": [1, 1.45]})

    ai_long = df[list(AI_FACTOR_LABELS)].melt(var_name="measure", value_name="score").dropna()
    ai_long["Measure"] = ai_long["measure"].map(AI_FACTOR_LABELS)
    _raincloud(
        axes[0],
        ai_long,
        "Measure",
        "score",
        list(AI_FACTOR_LABELS.values()),
        [PALETTE["blue"], PALETTE["orange"]],
    )
    axes[0].axhline(0, color=PALETTE["slate"], linestyle="--", linewidth=1)
    axes[0].set_title("AI factors", loc="left")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("EFA score (sample average = 0)")
    axes[0].tick_params(axis="x", rotation=12)

    med_long = df[list(MEDIATOR_LABELS)].melt(var_name="measure", value_name="score").dropna()
    med_long["Measure"] = med_long["measure"].map(MEDIATOR_LABELS)
    _raincloud(
        axes[1],
        med_long,
        "Measure",
        "score",
        list(MEDIATOR_LABELS.values()),
        [PALETTE["cyan"], PALETTE["magenta"], PALETTE["green"]],
    )
    axes[1].axhline(3, color=PALETTE["slate"], linestyle="--", linewidth=1)
    axes[1].set_title("Learning experiences", loc="left")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean item score (1-5)")
    axes[1].set_ylim(0.8, 5.2)
    axes[1].tick_params(axis="x", rotation=12)
    fig.suptitle("Overall score distributions", x=0.02, ha="left", fontsize=17, fontweight="bold", color=PALETTE["navy"])
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


def plot_ai_item_profile(profile):
    """Plot ranked AI-literacy item means with 95% confidence intervals."""
    set_report_style()
    ordered = profile.sort_values("mean")
    colors = ordered["Dimension"].map(
        {"AI understanding": PALETTE["blue"], "AI confidence": PALETTE["orange"]}
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    y = np.arange(len(ordered))
    ax.hlines(y, 1, ordered["mean"], color=colors, linewidth=5, alpha=0.35)
    ax.errorbar(
        ordered["mean"],
        y,
        xerr=[ordered["mean"] - ordered["ci_low"], ordered["ci_high"] - ordered["mean"]],
        fmt="none",
        ecolor=PALETTE["slate"],
        elinewidth=1.2,
        capsize=3,
        zorder=2,
    )
    ax.scatter(ordered["mean"], y, c=colors, s=85, edgecolor="white", linewidth=1.2, zorder=3)
    ax.axvline(3, linestyle="--", color=PALETTE["slate"], linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["Item"])
    ax.set_xlim(1, 5)
    ax.set_xlabel("Mean scored response (1-5; higher is better)")
    ax.set_title("Where students feel strongest and weakest", loc="left", color=PALETTE["navy"])
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["blue"], markersize=8, label="AI understanding"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["orange"], markersize=8, label="AI confidence"),
    ]
    ax.legend(handles=handles, loc="lower right")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig


def plot_ses_group_comparisons(df):
    """Compare all five outcomes across the two SES groupings."""
    set_report_style()
    outcomes = list(AI_FACTOR_LABELS) + list(MEDIATOR_LABELS)
    zdata = df.copy()
    for outcome in outcomes:
        sd = zdata[outcome].std(ddof=0)
        zdata[outcome] = (zdata[outcome] - zdata[outcome].mean()) / sd

    fig, axes = plt.subplots(2, 5, figsize=(11.2, 6.0), sharey=True)
    for row, (factor, factor_label) in enumerate(SES_FACTOR_LABELS.items()):
        group_col = f"{factor}_group"
        for col, outcome in enumerate(outcomes):
            ax = axes[row, col]
            plot_data = zdata[[group_col, outcome]].dropna()
            _raincloud(
                ax,
                plot_data,
                group_col,
                outcome,
                ["Lower", "Higher"],
                [PALETTE["cyan"], PALETTE["magenta"]],
            )
            ax.axhline(0, color=PALETTE["slate"], linestyle="--", linewidth=0.8)
            ax.set_title(OUTCOME_LABELS[outcome], fontsize=10.5)
            ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel(f"{factor_label}\nStandardised score")
            else:
                ax.set_ylabel("")
            ax.tick_params(axis="x", labelrotation=25, labelsize=10)
    fig.suptitle("Outcome distributions in lower and higher SES groups", x=0.02, ha="left", fontsize=16, fontweight="bold", color=PALETTE["navy"])
    fig.text(0.02, 0.01, "Groups are rank-based halves of each EFA score; each dot is one student.", fontsize=9, color=PALETTE["slate"])
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    return fig


def plot_effect_sizes(results):
    """Forest plot of Hedges g for higher versus lower SES groups."""
    set_report_style()
    order = list(OUTCOME_LABELS.values())
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    for ax, (factor, factor_label) in zip(axes, SES_FACTOR_LABELS.items()):
        part = results.loc[results["ses_factor"].eq(factor)].set_index("Outcome").loc[order]
        y = np.arange(len(part))[::-1]
        colors = [PALETTE["blue"] if name in AI_FACTOR_LABELS.values() else PALETTE["green"] for name in part.index]
        ax.axvline(0, color=PALETTE["slate"], linewidth=1)
        ax.errorbar(
            part["hedges_g"],
            y,
            xerr=[part["hedges_g"] - part["g_ci_low"], part["g_ci_high"] - part["hedges_g"]],
            fmt="none",
            ecolor=PALETTE["slate"],
            capsize=3,
            linewidth=1.3,
        )
        ax.scatter(part["hedges_g"], y, c=colors, s=80, edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(part.index)
        ax.set_xlim(-1.8, 1.8)
        ax.set_xlabel("Hedges g: higher SES minus lower SES")
        ax.set_title(factor_label, loc="left")
        ax.axvspan(-0.2, 0.2, color=PALETTE["light"], zorder=0)
    fig.suptitle("Estimated SES group differences", x=0.02, ha="left", fontsize=16, fontweight="bold", color=PALETTE["navy"])
    fig.text(0.02, 0.01, "Positive values favour the higher-SES group. Wide intervals reflect the small sample.", fontsize=9, color=PALETTE["slate"])
    fig.tight_layout(rect=[0, 0.05, 1, 0.91])
    return fig


def plot_association_matrix(associations):
    """Bubble heatmap for continuous SES-outcome Spearman associations."""
    set_report_style()
    matrix = associations.pivot(index="Outcome", columns="SES dimension", values="rho")
    matrix = matrix.loc[list(OUTCOME_LABELS.values()), list(SES_FACTOR_LABELS.values())]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for y, outcome in enumerate(matrix.index):
        for x, factor in enumerate(matrix.columns):
            value = matrix.loc[outcome, factor]
            color = PALETTE["magenta"] if value >= 0 else PALETTE["cyan"]
            ax.scatter(x, y, s=420 + 1150 * abs(value), color=color, alpha=0.82, edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=10, ha="center")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, len(matrix.columns) - 0.4)
    ax.set_ylim(len(matrix.index) - 0.4, -0.6)
    ax.set_title("Continuous SES-outcome patterns", loc="left", color=PALETTE["navy"])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def compact_test_table(results):
    """Return a reader-facing table with one row per SES dimension and outcome."""
    out = results.copy()
    out["Higher - lower"] = out["difference"].map(lambda value: f"{value:+.2f}")
    out["Effect size"] = out["hedges_g"].map(lambda value: f"{value:+.2f}")
    out["p"] = out["p_value"].map(lambda value: f"{value:.3f}")
    out["BH p"] = out["p_bh"].map(lambda value: f"{value:.3f}")
    return out[["SES dimension", "Outcome", "Higher - lower", "Effect size", "p", "BH p"]]

