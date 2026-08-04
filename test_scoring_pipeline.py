from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from functions import (
    AI_ITEM_CODEBOOK,
    AI_EFA_ITEMS,
    SES_INDEX_ITEMS,
    fit_efa,
    prepare_dataset,
    score_pre_ai_items,
    validate_item_order,
    validate_score_range,
    validate_unique_ids,
)
from phase_2_functions import merge_and_score_followup_ai


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"
BASELINE_FILES = (DATA / "AI_Lit_Que_1111.csv", DATA / "AI_Lit_Que_1204.csv")
FOLLOWUP_FILES = (DATA / "AI_Lit_follow_1111.csv", DATA / "AI_Lit_follow_1204.csv")


def test_raw_item_order_matches_codebook():
    expected_pre = [row["pre_source"] for row in AI_ITEM_CODEBOOK]
    expected_post = [
        row["post_source"]
        for row in sorted(AI_ITEM_CODEBOOK, key=lambda row: row["post_position"])
    ]

    for path in BASELINE_FILES:
        columns = pd.read_csv(path, nrows=0).columns
        validate_item_order(columns, expected_pre, context=path.name)

    for path in FOLLOWUP_FILES:
        columns = pd.read_csv(path, nrows=0).columns
        validate_item_order(columns, expected_post, context=path.name)
        assert list(columns[1:]) == expected_post


def test_reverse_key_and_score_ranges():
    reverse_positions = {
        row["post_position"] for row in AI_ITEM_CODEBOOK if row["reverse"]
    }
    assert reverse_positions == {1, 3, 5, 7, 9}

    synthetic = pd.DataFrame(
        {row["pre_numeric"]: [1.0, 5.0] for row in AI_ITEM_CODEBOOK}
    )
    scored = score_pre_ai_items(synthetic)
    for row in AI_ITEM_CODEBOOK:
        expected = [5.0, 1.0] if row["reverse"] else [1.0, 5.0]
        assert scored[row["pre_scored"]].tolist() == expected

    with pytest.raises(ValueError, match="outside"):
        validate_score_range(pd.Series([0, 3, 6]), 1, 5, context="test item")


def test_phase_i_ids_ranges_and_canonical_ses():
    baseline, meta = prepare_dataset(*BASELINE_FILES)
    assert len(baseline) == 141
    validate_unique_ids(baseline, "id", context="Phase I")
    assert baseline[list(AI_EFA_ITEMS)].stack().between(1, 5).all()
    assert meta["ses_scored_cols"] == list(SES_INDEX_ITEMS)
    assert np.isclose(baseline["ses_index"].mean(), 0.0, atol=1e-12)

    duplicate = pd.DataFrame({"id": ["same", "same"]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_unique_ids(duplicate, "id", context="synthetic")


def test_phase_ii_expected_matches_scoring_and_ses_preservation():
    baseline, _ = prepare_dataset(*BASELINE_FILES)
    ai_fa, _, _, _ = fit_efa(
        baseline, "Combined", list(AI_EFA_ITEMS), n_factors=2, rotation="oblimin"
    )
    merged = merge_and_score_followup_ai(
        baseline_df=baseline,
        followup_file_1=FOLLOWUP_FILES[0],
        followup_file_2=FOLLOWUP_FILES[1],
        ai_fa=ai_fa,
        ai_efa_items=list(AI_EFA_ITEMS),
        expected_matches=41,
    )

    assert len(merged) == 41
    validate_unique_ids(merged, "id", context="Phase II")
    assert merged[list(row["post_scored"] for row in AI_ITEM_CODEBOOK)].stack().between(1, 5).all()

    baseline_ses = baseline.set_index("id")["ses_index"]
    expected_ses = merged["id"].map(baseline_ses)
    assert np.allclose(merged["ses_index"], expected_ses)

    training_item = next(row for row in AI_ITEM_CODEBOOK if row["item"] == "training_data")
    raw_training = merged[f'post_{training_item["post_source"]}'].str.extract(r"^(\d+)")[0].astype(float)
    assert np.allclose(merged[training_item["post_scored"]], raw_training)

