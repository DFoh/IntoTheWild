import pandas as pd

from analysis.util import load_result_dataframe, load_demographics_raw_data, save_cleaned_demographics_data, \
    save_merged_dataframe


def df_remove_dnfers(df: pd.DataFrame) -> pd.DataFrame:
    # Remove rows where "finish_time_s" is NaN or where "avg_speed_m_s" is NaN
    df_cleaned = df.dropna(subset=["finish_time_s", "avg_speed_m_s"])
    num_removed = len(df) - len(df_cleaned)
    print(f"Removed {num_removed} DNFers from the dataset")
    return df_cleaned


if __name__ == '__main__':
    # Whether or not to remove DNFers from the merged dataset. This will remove all rows where "finish_time_s" is NaN or where "avg_speed_m_s" is NaN.
    remove_dnfers = True
    df_kinematics = load_result_dataframe("kinematic_params.xlsx")
    df_demographics = load_demographics_raw_data()
    bibs_demo = set(df_demographics["Bib"])
    bibs_kinematics = set(df_kinematics["Bib"])
    print(f"Found {len(bibs_demo)} bibs in demographics and {len(bibs_kinematics)} bibs in kinematics")
    bibs_not_in_kinematics = bibs_demo - bibs_kinematics
    print(f"Bibs in demographics but not in kinematics: {bibs_not_in_kinematics}")
    bibs_not_in_demo = bibs_kinematics - bibs_demo
    print(f"Bibs in kinematics but not in demographics: {bibs_not_in_demo}")
    bibs_resulting = bibs_demo.intersection(bibs_kinematics)
    print(f"Finally including {len(bibs_resulting)} bibs in the merged dataset")
    df_merged = pd.merge(df_kinematics, df_demographics, on="Bib", how="inner")
    foo = 1
    df_demo_cleaned = df_demographics[df_demographics["Bib"].isin(bibs_resulting)].copy()
    df_demo_cleaned.sort_values("Bib", inplace=True)
    df_demo_cleaned.reset_index(drop=True, inplace=True)
    df_merged_bibs = df_merged["Bib"].unique().tolist()
    df_merged_bibs.sort()
    assert df_demo_cleaned["Bib"].tolist() == df_merged_bibs, "Bibs do not match after merge"
    if remove_dnfers:
        df_demo_cleaned = df_remove_dnfers(df_demo_cleaned)
        bibs_cleaned = set(df_demo_cleaned["Bib"])
        df_merged = df_merged[df_merged["Bib"].isin(bibs_cleaned)].copy()

    save_merged_dataframe(df_merged)
    save_cleaned_demographics_data(df_demo_cleaned)
