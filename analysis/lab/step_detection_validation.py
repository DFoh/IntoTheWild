import warnings
from pathlib import Path

import pandas as pd
import re
import math
import numpy as np
import json

from analysis.util import load_path_lab_data


def parse_speed(name):
    m = re.search(r"(?:Laufanalyse|lab)_(.+?)kmh", str(name), flags=re.IGNORECASE)
    return float(m.group(1).strip("_").replace("_", ".")) if m else math.nan


def harmonize_event_arrays(ev_1: np.ndarray, ev_2: np.ndarray):
    same_length = len(ev_1) == len(ev_2)
    thresh = 200  # ms, threshold for stride-off detection
    if same_length:
        avg_diff = np.mean(ev_1 - ev_2)
        max_diff = np.max(ev_1 - ev_2)
        min_diff = np.min(ev_1 - ev_2)
        if abs(avg_diff) < thresh:
            return ev_1, ev_2
        else:
            foo = 1
    else:
        max_len = max(len(ev_1), len(ev_2))
        min_len = min(len(ev_1), len(ev_2))
        len_diff = len(ev_1) - len(ev_2)
        if abs(ev_1[0] - ev_2[0]) < thresh:
            return ev_1[:min_len], ev_2[:min_len]
        else:
            foo = 1
            for i in range(len_diff):
                ev_1[0] - ev_2[0]
                ev_1[0] - ev_2[len_diff]
                ev_1[len_diff] - ev_2[0]
                ev_2 = ev_2[:min_len]

    return np.array([0]), np.array([0])



if __name__ == '__main__':
    path_root = load_path_lab_data()
    path_kinematics = Path(path_root) / "kinematics"
    path_pressure = Path(path_root) / "pressure"

    df_kinematics = pd.read_excel(path_kinematics / "events_kinematics_ms.xlsx", index_col=0)
    df_kinematics["trial_name"].map(lambda x: x.find(""))
    df_pressure = pd.read_excel(path_pressure / "events_pressure_ms.xlsx", index_col=0)

    df_kinematics["running_speed_kmh"] = df_kinematics["trial_name"].map(parse_speed)
    df_pressure["running_speed_kmh"] = df_pressure["trial"].map(parse_speed)

    keys = ["participant_id", "running_speed_kmh"]
    # consistency check:
    s1 = set(df_pressure[keys].itertuples(index=False, name=None))
    s2 = set(df_kinematics[keys].itertuples(index=False, name=None))
    print(s1 - s2, s2 - s1)

    event_classes = ["Right.IC", "Right.TO", "Left.IC", "Left.TO"]

    df_out = pd.DataFrame(columns=["participant_id",
                           "running_speed_kmh",
                           "side",
                           "event_class",
                           "count",
                           "t_kinematics_ms",
                           "t_pressure_ms"])
    for (p_id, speed) in s1:
        for event_class in event_classes:
            # if p_id != "ITW009":
            #     continue
            # if speed > 12.1:
            #     continue
            # if speed < 11.9:
            #     continue
            mask = (df_pressure["participant_id"] == p_id) & (df_pressure["running_speed_kmh"] == speed)
            ev_pressure = np.array(json.loads(df_pressure.loc[mask, event_class].item()))
            ev_kinematics = np.array(json.loads(df_kinematics.loc[mask, event_class].item()))

            ev_side = event_class.split(".")[0]
            ev_class = event_class.split(".")[1]
            # if ev_side != "Left":
            #     continue

            ev_kinematics, ev_pressure = harmonize_event_arrays(ev_kinematics, ev_pressure)

            # ev_kinematics = ev_kinematics[(1000 < ev_kinematics) & (ev_kinematics < 14000)]
            # ev_pressure = ev_pressure[(1000 < ev_pressure) & (ev_pressure < 14000)]
            # if len(ev_kinematics) != len(ev_pressure):
            #     warnings.warn(f"{len(ev_kinematics)}, {len(ev_pressure)} in {p_id}, {speed}, {event_class}")
            #     continue

            diffs = ev_kinematics - ev_pressure
            np.max(diffs)
            np.min(diffs)

            for i, (val_kin, val_press) in enumerate(zip(ev_kinematics, ev_pressure)):
                row = {"participant_id": [p_id],
                       "running_speed_kmh": [speed],
                       "side": [ev_side],
                       "event_class": [ev_class],
                       "count": [i],
                       "t_kinematics_ms": [val_kin],
                       "t_pressure_ms":[val_press]
                       }
                df_out = pd.concat([df_out, pd.DataFrame(row)], axis=0)
    df_out.to_excel(path_root / "events_comparison.xlsx", index=False)
