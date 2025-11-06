from pathlib import Path

import numpy as np
import pandas as pd


def make_empty_raw_number_dataframe() -> pd.DataFrame:
    column_dtypes = {
        'frame_number': np.int64,
        'cam_id': np.int64,
        'text': 'object',
        'conf': np.float64,
        'x1': np.int64,
        'y1': np.int64,
        'x2': np.int64,
        'y2': np.int64,
        'height_number_px': np.int64,
        'width_number_px': np.int64
    }
    return pd.DataFrame(columns=column_dtypes.keys()).astype(column_dtypes)


def make_empty_final_frame_dataframe():
    column_dtypes = {
        'heat': np.int64,
        'number': np.int64,
        'trial_name': 'object',
        'final_frame': np.int64
    }
    return pd.DataFrame(columns=column_dtypes.keys()).astype(column_dtypes)


def get_path_numbers_dataframe(heat: int, trial_name: str):
    return Path(f"detection_results_raw/start_numbers_raw_heat_{heat}_{trial_name}.xlsx")


def get_path_final_frames_dataframe(heat: int, trial_name: str):
    return Path(f"detection_results_final/start_numbers_final_heat_{heat}_{trial_name}.xlsx")


def load_numbers_dataframe_from_excel(heat: int, trial_name) -> pd.DataFrame:
    path = get_path_numbers_dataframe(heat, trial_name)
    if not path.exists():
        return make_empty_raw_number_dataframe()
    return pd.read_excel(path)


def load_final_frames_dataframe_from_excel(heat: int, trial_name) -> pd.DataFrame:
    path = get_path_final_frames_dataframe(heat, trial_name)
    if not path.exists():
        return make_empty_final_frame_dataframe()
    return pd.read_excel(path)


def save_numbers_dataframe_to_excel(df: pd.DataFrame, heat: int, trial_name: str):
    path = get_path_numbers_dataframe(heat, trial_name)
    df.to_excel(path, index=False)


def save_final_frames_dataframe_to_excel(df: pd.DataFrame, heat: int, trial_name: str):
    path = get_path_final_frames_dataframe(heat, trial_name)
    df.to_excel(path, index=False)
