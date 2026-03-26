import warnings

import numpy as np
from scipy.signal import find_peaks


def get_running_events_maiwald(heel_marker_vertical_traj: np.ndarray,
                               toe_marker_vertical_traj: np.ndarray,
                               sample_rate: float,
                               ) -> dict:
    """
    Identify running events (initial contact and toe-off) based on vertical trajectory of heel and toe markers in
    accordance to Maiwald et al. 2009 "Detecting foot-to-ground contact from kinematic data in running"
    Footwear Sci., DOI: 10.1080/19424280903133938

    :param heel_marker_vertical_traj: Vertical trajectory of the heel marker (1D array, in meters)
    :param toe_marker_vertical_traj: Vertical trajectory of the toe marker (1D array, in meters)
    :param sample_rate: Sample rate of the data (in Hz)
    :return: Dictionary with detected events, e.g. {"IC": [list of initial contact frames], "TO": [list of toe off frames]}
    """

    # calc derivatives of the marker trajectories
    heel_vert_vel = np.gradient(heel_marker_vertical_traj) * sample_rate  # Vertical velocity of the heel
    heel_vert_acc = np.gradient(heel_vert_vel) * sample_rate  # Vertical acceleration of the heel
    toe_vert_vel = np.gradient(toe_marker_vertical_traj) * sample_rate
    toe_vert_acc = np.gradient(toe_vert_vel) * sample_rate

    #
    # Get the swing events (peaks in the vertical trajectory of the heel marker) as a robust marker to segment the signal into steps
    #
    # for more robust peak detection, we can set an expected minimum step rate to define "distance" between peaks.
    min_step_rate = 120  # steps per minute (even slow running shouldn't be lower than this)
    min_step_interval = (60 / min_step_rate) * sample_rate  # minimum interval in frames
    min_height = 0.2  # minimum height of the heel marker to be considered a swing event
    heel_pks, _ = find_peaks(heel_marker_vertical_traj,
                             height=min_height,
                             distance=min_step_interval)

    # see if the first swing event (heel_pks) has sufficient frames before start of the file to capture the preceding initial contact event
    avg_swing_evt_frames = np.diff(heel_pks).mean()
    search_window_ratio = 0.66  # adjust from experience
    if heel_pks[0] > search_window_ratio * avg_swing_evt_frames:
        heel_pks = np.insert(heel_pks, 0,
                             0)  # additional start frame for the minimum search up to the first swing event

    # same for the last event
    if heel_pks[-1] < len(heel_marker_vertical_traj) - search_window_ratio * avg_swing_evt_frames:
        heel_pks = np.append(heel_pks, len(heel_marker_vertical_traj) - 1)

    ics = []
    tos = []

    # print(f"heel_pks type: {type(heel_pks)}, shape: {heel_pks.shape}, values: {heel_pks}")
    for pk, next_pk in zip(heel_pks, heel_pks[1:]):
        # "Prior to analyzing the temporal aspect of vertical acceleration of either marker,
        # the time frames of the minimum vertical position of HEEL and MTH5 were compared.
        # In order to accommodate the algorithm for possible midfoot and forefoot strikers,
        # the earlier event and thus the target marker was chosen as an indicator of the
        # type of foot strike and approximate time of touch down (TDapprox)."
        heel_min = np.argmin(heel_marker_vertical_traj[pk:next_pk])
        toe_min = np.argmin(toe_marker_vertical_traj[pk:next_pk])
        if heel_min < toe_min:
            td_approx = pk + int(heel_min)
            acc = heel_vert_acc
        else:
            td_approx = pk + int(toe_min)
            acc = toe_vert_acc

        # "Subsequently, a sufficiently narrow time interval around this minimal position was defined
        # [TDapprox - 50 ms, TDapprox + 100 ms]. This ensured that FCA was able to deal with various
        # types of foot strike and only detected those acceleration peaks that were associated with
        # touch down rather than those associated with take off or any vertical braking that occurs
        # during foot swing and pre touch down phases. To estimate touch down, FCA uses a characteristic
        # maximum in the vertical acceleration curve of the target marker in the given approximation interval."
        window_start = int(max(0, td_approx - 0.05 * sample_rate))
        window_end = int(min(len(heel_vert_acc), td_approx + 0.1 * sample_rate))

        acc_window = acc[window_start:window_end]
        acc_pks, _ = find_peaks(acc_window)
        # if no peaks are found, something went wrong, so raise a warning. Use the maximum of the window as fallback for the initial contact event
        if len(acc_pks) == 0:
            warnings.warn(f"No acceleration peaks found, using maximum as fallback")
            acc_pk = np.argmax(acc_window)
        else:
            # If there are multiple peaks, we take the first one.
            # Base on observations, this was the more robust choice compared to taking the highest peak.
            # However, this was not tested against ground truth data, so it might be worth testing this in the future.
            # TODO: When testing against ground truth data, also try taking the highest peak instead of the first one to see which one matches better.
            acc_pk = acc_pks[0]
        ic = int(window_start + acc_pk)

        # TAKE OFF
        # "For take off, a local maximum in the vertical acceleration of TIP is detected and compared to the
        # minimal vertical position of TIP. A logical operation selects the event that occurs earlier in time,
        # which is then used to estimate take off. The two events usually occur almost simultaneously.
        # For some running styles, e.g. with only limited vertical ground clearance of the foot after take off,
        # we observed a major gap between those events. Choosing the earlier event, which was the minimal vertical
        # position in most cases, yielded a better match to the force platform event, the gold standard.
        # Similar to the detection of touch down, additional constraints were necessary in order to select the
        # correct acceleration peak. This was accomplished by constraining the search for peaks to a suitable
        # time window for ground contact times observed during running at 3.5 m/s [TDapprox + 100 ms, TDapprox + 400 ms]."
        to_window_start = ic + int(0.1 * sample_rate)
        # well, it can't be after the next swing event, so we need to check that as well
        to_window_end = min(next_pk, ic + int(0.5 * sample_rate))

        # make sure we get at least 100ms window to search for the to event
        if (to_window_end - to_window_start) < 0.1 * sample_rate:
            # if not, skip the take off detection for this step and raise a warning
            warnings.warn(
                f"Take off search window is too short, skipping take off detection for this step")
            continue

        # for debugging:
        frames_to = np.arange(to_window_start, to_window_end)
        acc_to_plot = toe_vert_acc[frames_to]
        to_acc_pks, _ = find_peaks(acc_to_plot)
        if len(to_acc_pks) == 0:
            warnings.warn(
                f"No acceleration peaks found in take off window, using maximum as fallback")
            to_acc_pk = np.argmax(acc_to_plot)
        else:
            i_to_acc_pk = np.argmax(acc_to_plot[to_acc_pks])
            to_acc_pk = to_acc_pks[i_to_acc_pk]
        to = int(frames_to[to_acc_pk])
        # all done. write to the lists
        ics.append(ic)
        tos.append(to)

    assert len(ics) == len(tos)
    return {"IC": ics, "TO": tos}


def get_valid_frame_range(data, side: str):
    # ISSUE: the data will have NaN values at the start and end because the skeleton is not solved in the first and last frames.
    # SOLUTION: check the data for the first and last valid frame and only analyze this window.
    # also: save the first frame to later offset the detected events to the actual frame number in the original file.
    # This is important for later comparison with the force platform data.

    heel_pos_data = data[f"{side}_Heel_Pos"][0][0]
    toe_pos_data = data[f"{side}_Toe_Pos"][0][0]
    mask = ~np.isnan(heel_pos_data[:, 0]) & ~np.isnan(toe_pos_data[:, 0])
    valid_indices = np.where(mask)[0]
    first_valid_frame = int(valid_indices[0])
    last_valid_frame = int(valid_indices[-1])

    return first_valid_frame, last_valid_frame


def check_valid_frame_range(valid_start, valid_end, sample_rate) -> bool:
    # Check if the valid frame range is sufficient for analysis (1.5 seconds)
    if (valid_end - valid_start) < (1.5 * sample_rate):
        return False
    return True


def get_running_events(data):
    events = {}
    sample_rate = data['FRAME_RATE'][0][0][0][0]  # Assuming frame rate is stored in this field

    for side in ["Left", "Right"]:
        valid_start, valid_end = get_valid_frame_range(data, side)
        if not check_valid_frame_range(valid_start, valid_end, sample_rate):
            warnings.warn(f"Valid frame range for {side} side is too short for reliable event detection. Skipping this side.")
            continue

        heel_vert_pos = data[f"{side}_Heel_Pos"][0][0][valid_start:valid_end, 2]
        toe_vert_pos = data[f"{side}_Toe_Pos"][0][0][valid_start:valid_end,2]
        events[side] = get_running_events_maiwald(heel_vert_pos, toe_vert_pos, sample_rate)
        # correct the detected event frames to the actual frame numbers in the original file by adding the valid_start offset
        events[side]["IC"] = [frame + valid_start for frame in events[side]["IC"]]
        events[side]["TO"] = [frame + valid_start for frame in events[side]["TO"]]
    return events

