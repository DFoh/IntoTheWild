import sys
import pandas as pd
import warnings
from pathlib import Path

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


def make_file_path(path_mat_root: Path, heat: str, bib_number: int, lap_no: int) -> Path:
    """
    create the file path for each heat, bib number and lap number based on the root path
    heat: str, e.g. "heat_1"
    bib_number: int, e.g. 100
    lap_no: int, e.g. 1
    Heat_1/100/100_lap_2_filt.mat
    """

    return path_mat_root / heat / str(bib_number) / f"{bib_number}_lap_{lap_no}_filt.mat"
