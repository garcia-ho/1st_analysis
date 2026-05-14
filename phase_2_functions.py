
import pandas as pd

def merge_followup_to_baseline(
    baseline_df,
    followup_file_1,
    followup_file_2,
    baseline_id_col="id",
    followup_id_col="Q00_Identification",
    how="inner"
):
    # read and combine follow-up files
    f1 = pd.read_csv(followup_file_1)
    f2 = pd.read_csv(followup_file_2)
    followup_df = pd.concat([f1, f2], ignore_index=True)

    # clean id columns as strings
    baseline = baseline_df.copy()
    followup = followup_df.copy()

    baseline[baseline_id_col] = baseline[baseline_id_col].astype(str).str.strip()
    followup[followup_id_col] = followup[followup_id_col].astype(str).str.strip()

    # rename follow-up id column to match baseline
    followup = followup.rename(columns={followup_id_col: baseline_id_col})

    # merge
    merged_df = baseline.merge(
        followup,
        on=baseline_id_col,
        how=how,
        suffixes=("", "_follow")
    )

    return merged_df

