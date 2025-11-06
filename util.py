from pathlib import Path

start_numbers_heat_1 = [100, 166, 177, 185, 186, 213, 214, 218, 222, 224, 225, 231, 245, 311, 313, 318, 363]  # M1
start_numbers_heat_2 = [15, 22, 182, 183, 187, 200, 211, 230, 258, 274, 277, 306, 315]  # F1
start_numbers_heat_3 = [107, 170, 215, 221, 223, 227, 251, 280, 308, 309, 310, 360]  # M2
start_numbers_heat_4 = [98, 219, 246, 247, 248, 252, 283, 288, 289, 295, 362]  # MIXED

start_numbers = {
    "heat_1": start_numbers_heat_1,
    "heat_2": start_numbers_heat_2,
    "heat_3": start_numbers_heat_3,
    "heat_4": start_numbers_heat_4,
}

path_video_data_root = Path("E:\ITW_Backup\HHTGP_HHTGP")
path_heat_1 = path_video_data_root.joinpath("Running - Markerless")
path_heat_2 = path_video_data_root.joinpath("Running - Markerless_3")
path_heat_3 = path_video_data_root.joinpath("Running - Markerless_4")
path_heat_4 = path_video_data_root.joinpath("Running - Markerless_5")

paths_videos = {
    "heat_1": path_heat_1,
    "heat_2": path_heat_2,
    "heat_3": path_heat_3,
    "heat_4": path_heat_4,
}
