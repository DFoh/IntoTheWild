import sys
import warnings
from pathlib import Path

import pandas as pd

# for Windows:
PATH_ROOT_WIN = r"C:\Users\dominik.fohrmann\OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University\Dokumente\Projects\IntoTheWild\data\TrackGrandPrix"
# for macOS:
PATH_ROOT_MAC = r"/Users/dominikfohrmann/OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University/Dokumente/Projects/IntoTheWild/data/TrackGrandPrix"

if sys.platform.startswith('win'):
    PATH_ROOT = PATH_ROOT_WIN
elif sys.platform.startswith('darwin'):
    PATH_ROOT = PATH_ROOT_MAC
else:
    raise OSError("Unsupported operating system. Please use Windows or macOS.")


def get_result_frame_path(filename: str) -> Path:
    path_safe = Path(PATH_ROOT) / "kinematics" / filename
    return path_safe


def safe_result_dataframe(df_result: pd.DataFrame, filename: str):
    path_safe = get_result_frame_path(filename)
    try:
        df_result.to_excel(path_safe, index=False)
    except Exception as e:
        warnings.warn(f"Could not save to Excel file: {e}")


def get_event_frame_path() -> Path:
    path_safe = get_result_frame_path("events.xlsx")
    return path_safe


def safe_event_dataframe(df_events: pd.DataFrame):
    safe_result_dataframe(df_events, "events.xlsx")


def safe_running_speed_dataframe(df_running_speed: pd.DataFrame):
    safe_result_dataframe(df_running_speed, "running_speed.xlsx")


def load_result_dataframe(filename: str) -> pd.DataFrame:
    path_safe = get_result_frame_path(filename)
    if not path_safe.exists():
        raise FileNotFoundError(f"Result file {path_safe} does not exist.")
    try:
        df_result = pd.read_excel(path_safe)
        print(f"Result loaded from {path_safe}")
        return df_result
    except Exception as e:
        warnings.warn(f"Could not load result from Excel file: {e}")
        return pd.DataFrame()


def load_events_from_excel() -> pd.DataFrame:
    path_safe = get_event_frame_path()
    try:
        df_events = pd.read_excel(path_safe)
        # parse the "Events" column from string back to dict
        df_events["Events"] = df_events["Events"].apply(lambda x: eval(x) if isinstance(x, str) else x)
        print(f"Events loaded from {path_safe}")
        return df_events
    except Exception as e:
        warnings.warn(f"Could not load events from Excel file: {e}")
        return pd.DataFrame(columns=["Heat", "Bib", "Lap", "Events"])


def load_demographics_raw_data():
    path_demo = Path(PATH_ROOT) / "demographics.xlsx"
    if not path_demo.exists():
        raise FileNotFoundError(f"File {path_demo} not found")
    df_demo = pd.read_excel(path_demo)
    columns_of_interest = ["Bib", "participant_id", "age", "sex", "body_mass_kg", "height_cm", "finish_time_s",
                           "avg_speed_m_s"]
    df_out = df_demo[columns_of_interest].copy()
    df_out.sort_values("Bib", inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


def save_merged_dataframe(df_merged: pd.DataFrame):
    path_merged = Path(PATH_ROOT) / "merged_data.xlsx"
    try:
        df_merged.to_excel(path_merged, index=False)
        print(f"Merged data saved to {path_merged}")
    except Exception as e:
        warnings.warn(f"Could not save merged data to Excel file: {e}")


def load_merged_dataframe() -> pd.DataFrame:
    path_merged = Path(PATH_ROOT) / "merged_data.xlsx"
    if not path_merged.exists():
        raise FileNotFoundError(f"File {path_merged} not found")
    df_merged = pd.read_excel(path_merged)
    print(f"Merged data loaded from {path_merged}")
    return df_merged


def save_cleaned_demographics_data(df_demo: pd.DataFrame):
    path_demo_cleaned = Path(PATH_ROOT) / "demographics_cleaned.xlsx"
    try:
        df_demo.to_excel(path_demo_cleaned, index=False)
        print(f"Cleaned demographics data saved to {path_demo_cleaned}")
    except Exception as e:
        warnings.warn(f"Could not save cleaned demographics data to Excel file: {e}")


def load_cleaned_demographics_data() -> pd.DataFrame:
    path_demo_cleaned = Path(PATH_ROOT) / "demographics_cleaned.xlsx"
    if not path_demo_cleaned.exists():
        raise FileNotFoundError(f"File {path_demo_cleaned} not found")
    df_demo = pd.read_excel(path_demo_cleaned)
    print(f"Cleaned demographics data loaded from {path_demo_cleaned}")
    return df_demo


def make_file_path(path_mat_root: Path, heat: str, bib_number: int, lap_no: int) -> Path:
    """
    create the file path for each heat, bib number and lap number based on the root path
    heat: str, e.g. "heat_1"
    bib_number: int, e.g. 100
    lap_no: int, e.g. 1
    Heat_1/100/100_lap_2_filt.mat
    """

    return path_mat_root / heat / str(bib_number) / f"{bib_number}_lap_{lap_no}_filt.mat"
