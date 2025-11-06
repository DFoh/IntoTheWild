from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from cut_mark_determination.common import load_numbers_dataframe_from_excel, make_empty_final_frame_dataframe, \
    save_final_frames_dataframe_to_excel
from cut_mark_determination.start_number_detection import get_heat_trials
from util import start_numbers_heat_3


def post_process_number_detection():
    df_numbers = load_numbers_dataframe_from_excel()
    detected_numbers = df_numbers["text"].unique()
    for n in detected_numbers:
        if n not in start_numbers_heat_3:
            print(f"Detected number not in expected list: {n}")
            continue
        df_n = df_numbers[df_numbers["text"] == n]
        # Take sparse samples and populate full frame range with zeros
        last_frame = df_n["frame_number"].max()
        # sum the confidences for each frame (in case of multiple camera detections)
        df_n["cam_id"] = df_n["cam_id"].astype(str)
        df_n["frame_number"] = df_n["frame_number"].astype(int)

        df_n = df_n.groupby("frame_number").agg({
            "conf": "sum",
            "cam_id": lambda x: ','.join(x)
        }).reset_index()

        conf_array = np.zeros(last_frame + 1000)  # some extra buffer for later filtering
        conf_array[df_n["frame_number"].values] = df_n["conf"].values

        # floating average filter
        window_size = 50
        conf_array_filt = np.convolve(conf_array, np.ones(window_size) / window_size, mode='same')
        # floating accumulated confidence ove 25 frames
        conf_array_accu = np.convolve(conf_array, np.ones(25), mode='same')
        # normalize
        conf_array_accu = conf_array_accu / 25
        plt.figure(figsize=(12, 8))
        plt.plot(conf_array)
        plt.plot(conf_array_filt)
        plt.plot(conf_array_accu)

        # make some threshold lines
        for thresh in range(5, 51, 5):
            plt.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5)

        # sns.lineplot(data=df_n, x="frame_number", y="conf", hue="cam_id")
        plt.title(f"Confidence over frames for detected number {n}")
        plt.show()


def get_final_frame(conf_array_accu, peak_threshold=0.8) -> int | None:
    maximum = np.max(conf_array_accu)
    if maximum < peak_threshold:
        return None
    lower_thresh = 0.05
    i = np.argmax(conf_array_accu)
    # get the first zero
    first_zero_after_peak = np.where(conf_array_accu[i:] < lower_thresh)[0][0] + i
    return first_zero_after_peak


def get_final_frames(conf_array_accu, peak_threshold=0.8) -> List | None:
    sample_rate = 85
    min_lap_time_s = 50
    min_lap_samples = sample_rate * min_lap_time_s
    peaks, heights = find_peaks(conf_array_accu, height=peak_threshold, distance=min_lap_samples)
    if len(peaks) == 0:
        return None
    final_frames = []
    for peak in peaks:
        lower_thresh = 0.05
        # get the first zero after the peak
        ff = np.where(conf_array_accu[peak:] < lower_thresh)[0][0] + peak
        final_frames.append(ff)
    return final_frames


def post_process_number_detection_gem(heat: int, trial_name: str) -> pd.DataFrame | None:
    df_numbers = load_numbers_dataframe_from_excel(heat, trial_name)
    if df_numbers.empty:
        return None
    detected_numbers = df_numbers["text"].unique()

    df_final_frames = make_empty_final_frame_dataframe()
    for n in detected_numbers:
        if n not in start_numbers_heat_3:
            print(f"Detected number not in expected list: {n}")
            continue
        df_n = df_numbers[df_numbers["text"] == n]
        # Take sparse samples and populate full frame range with zeros
        last_frame = df_n["frame_number"].max()
        # sum the confidences for each frame (in case of multiple camera detections)
        df_n["cam_id"] = df_n["cam_id"].astype(str)
        df_n["frame_number"] = df_n["frame_number"].astype(int)

        df_n = df_n.groupby("frame_number").agg({
            "conf": "sum",
            "cam_id": lambda x: ','.join(x)
        }).reset_index()

        conf_array = np.zeros(last_frame + 1000)  # some extra buffer for later filtering
        # --- Start der neuen Logik zur Überbetonung hoher Werte ---
        # 1. Parameter für die Hervorhebung
        p = 2.0  # Quadratische Überbetonung. Experimentieren Sie mit p > 1

        # Optional: Nur auf Werte > 0 anwenden, um Fehler bei conf=0 zu vermeiden (bei Power-Funktionen)
        # In Ihrem Fall sind Konfidenzen >= 0, also ist das i.d.R. unkritisch für p=2.

        conf_array[df_n["frame_number"].values] = df_n["conf"].values

        # 2. Transformation: Hohe Werte anheben
        conf_array_trans = conf_array ** p

        # 3. Mittelwerte berechnen (nur über die Frames, die nicht Null sind, um leere Stellen nicht zu verzerren)
        # Da Sie ein Array mit Nullen auffüllen, ist es besser, den Mean nur über die Originalwerte zu berechnen!
        # Am einfachsten: Berechnen Sie den Skalierungsfaktor basierend auf den Original-Konfidenzen aus df_n!
        original_confidences = df_n["conf"].values

        if len(original_confidences) > 0:
            mean_original = np.mean(original_confidences)
            mean_transformed = np.mean(original_confidences ** p)

            # Skalierungsfaktor, um den Mittelwert wiederherzustellen
            scaling_factor = mean_original / mean_transformed

            # 4. Neuskalierung des gesamten conf_array
            conf_array_accentuated = conf_array_trans * scaling_factor
        else:
            # Falls keine Daten vorhanden sind
            conf_array_accentuated = conf_array

        # Verwenden Sie nun 'conf_array_accentuated' für die Faltung
        conf_to_convolve = conf_array_accentuated
        # --- Ende der neuen Logik ---

        # Ersetzen Sie nun alle Vorkommen von 'conf_array' in den Faltungsoperationen durch 'conf_to_convolve'
        # floating average filter
        window_size = 25
        conf_array_filt = np.convolve(conf_to_convolve, np.ones(window_size) / window_size, mode='same')
        # floating accumulated confidence ove 25 frames
        conf_array_accu = np.convolve(conf_to_convolve, np.ones(window_size), mode='same')
        # normalize
        conf_array_accu = conf_array_accu / window_size
        # ... und auch für den Plot
        plt.figure(figsize=(12, 8))
        plt.plot(conf_array, label="Original")
        plt.plot(conf_array_accu, label="Accumulated (Accentuated)")
        plt.legend()
        max_red = max(conf_array_accu)
        plt.title(f"Confidence over frames for detected number {n}. Max Akkumuliert (Akzentuiert): {max_red:.2f}")
        thresh = 0.6
        final_frames = get_final_frames(conf_array_accu, peak_threshold=thresh)
        if final_frames is not None:
            for final_frame in final_frames:
                final_frame -= int(window_size / 2)  # adjust for filter delay
                plt.axvline(x=final_frame, color='red', linestyle='--', label='Final Frame')
        plt.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5, label='Peak Threshold')
        df_final_frames_local = pd.DataFrame({
            "heat": heat,
            "number": n,
            "trial_name": trial_name,
            "final_frame": final_frames if final_frames is not None else [None]
        })
        df_final_frames = pd.concat([df_final_frames, df_final_frames_local], ignore_index=True)
        # plt.show()
    return df_final_frames


def main(heat: int):
    trial_names = get_heat_trials(heat)
    df_final_frames_all = make_empty_final_frame_dataframe()
    for trial_name in trial_names:
        df_final_frames_trial = post_process_number_detection_gem(heat, trial_name)
        if df_final_frames_trial is not None:
            save_final_frames_dataframe_to_excel(df_final_frames_trial, heat, trial_name)


if __name__ == '__main__':
    main(heat=3)
