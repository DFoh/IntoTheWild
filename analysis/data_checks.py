from pathlib import Path

from labtools.batch_processor import BatchProcessor
from pandas.core.computation.ops import isnumeric

from analysis.util import load_cleaned_demographics_data, load_path_root, load_result_dataframe, safe_result_dataframe
import pandas as pd


def check_mat_files():
    path_root = load_path_root()
    path_mat_root = Path(path_root) / "TrackGrandPrix" / "kinematics" / "mat"
    bp = BatchProcessor(path_mat_root,
                        level_names=["Bib", "Lap"],
                        file_pattern="_filt.mat")

    excpected_lap_count = 13
    expected_participant_count = 53
    assert len(bp.index["Bib"].unique()) == expected_participant_count
    missing_laps = bp.index.groupby("Bib").count().Lap[bp.index.groupby("Bib").count().Lap != excpected_lap_count]
    if not missing_laps.empty:
        print(f"{len(missing_laps)} participants are missing laps:")
        for bib, count in missing_laps.items():
            print(f"{bib} is missing {excpected_lap_count - count}")

    df_wide = bp.index.copy()
    df_wide["Bib"] = df_wide["Bib"].astype(int)
    df_wide["path"] = df_wide["path"].apply(lambda x: 1 if x is not None else 0).astype(int)
    df_wide["Lap"] = df_wide.apply(lambda x: x["Lap"].replace(f"{x['Bib']}_lap_", "").replace("_filt", ""), axis=1).astype(int)

    df_wide = df_wide.pivot(
        index=["Bib"],
        columns="Lap",
        values="path",
    ).fillna(0).astype(int)

    path_mat_check_out = path_root / "mat_file_check.xlsx"
    df_wide.to_excel(path_mat_check_out)


def check_kinematics_output():
    df_kinematic_params = load_result_dataframe("kinematic_params.xlsx")
    df_demo = load_cleaned_demographics_data()

    df_kinematic_params = pd.merge(df_kinematic_params, df_demo[["Bib", "participant_id"]], how="inner", on="Bib")

    df_check = df_kinematic_params[["Bib", "participant_id", "Lap", "running_speed_ms.Left"]]

    df_check["running_speed_ms.Left"] = (
        pd.to_numeric(df_check["running_speed_ms.Left"], errors="coerce").notna().astype(int)
    )

    df_wide = df_check.pivot(
        index=["Bib", "participant_id"],
        columns="Lap",
        values="running_speed_ms.Left",
    ).fillna(0).astype(int)

    df_wide.columns = [f"lap_{c}" for c in df_wide.columns]
    df_wide = df_wide.reset_index()
    df_wide.sort_values("participant_id", inplace=True)
    df_wide.reset_index(inplace=True, drop=True)

    safe_result_dataframe(df_wide, filename="data_checks_auto.xlsx")


if __name__ == '__main__':
    check_mat_files()
