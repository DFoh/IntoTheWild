import warnings
from pathlib import Path

import pandas as pd

from analysis.util import load_result_dataframe, PATH_ROOT


def get_demographics_data():
    path_demo = Path(PATH_ROOT) / "demographics.xlsx"
    if not path_demo.exists():
        raise FileNotFoundError(f"File {path_demo} not found")
    df_demo = pd.read_excel(path_demo)
    columns_of_interest = ["Bib", "participant_id", "age", "sex", "body_mass_kg", "height_cm", "finish_time_s", "avg_speed_m_s"]
    df_out = df_demo[columns_of_interest].copy()
    df_out.sort_values("Bib", inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


if __name__ == '__main__':
    df_kinematics = load_result_dataframe("kinematic_params.xlsx")
    df_demographics = get_demographics_data()
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
    df_demo_reduced = df_demographics[df_demographics["Bib"].isin(bibs_resulting)].copy()
    df_demo_reduced.sort_values("Bib", inplace=True)
    df_demo_reduced.reset_index(drop=True, inplace=True)
    df_merged_bibs = df_merged["Bib"].unique().tolist()
    df_merged_bibs.sort()
    assert df_demo_reduced["Bib"].tolist() == df_merged_bibs, "Bibs do not match after merge"
    df_merged.to_excel(Path(PATH_ROOT) / "merged_data.xlsx", index=False)
    df_demo_reduced.to_excel(Path(PATH_ROOT) / "demographics_reduced.xlsx", index=False)

