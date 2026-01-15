from pathlib import Path
from shutil import move


def rename_heat_3():
    path_files_heat_3 = "E:\ITW_Backup\HHTGP_HHTGP\Running - Markerless_4_cut"
    path_heat = Path(path_files_heat_3)
    trial_count = 1
    for trial_number in range(1, 17, 1):
        files = list(path_heat.glob(f"Running trial Markerless {trial_number}_cut_*.avi"))
        cuts = len(files) // 14
        for cut in range(cuts):
            cut_files = list(
                path_heat.glob(f"Running trial Markerless {trial_number}_cut_{str(cut + 1).zfill(2)}_*.avi"))
            new_trial_number = trial_count
            for cut_file in cut_files:
                parts = cut_file.stem.split("_")
                old_trial_number = parts[0].split(" ")[-1]
                new_trial_name = parts[0].replace(old_trial_number, str(new_trial_number))
                # parts[2] is the trial number
                parts[2] = str(new_trial_number)
                # remove the "cut_xx" part
                new_file_name = "_".join([new_trial_name] + parts[3:]) + cut_file.suffix
                new_file_path = cut_file.parent / new_file_name
                print(f"Renaming {cut_file.name} to {new_file_name}")
                move(cut_file, new_file_path)
            trial_count += 1


def rename_heat_3_again():
    path_files_heat_3 = "E:\ITW_Backup\HHTGP_HHTGP\Running - Markerless_4_cut"
    files = list(Path(path_files_heat_3).glob("Running trial Markerless*.avi"))
    for trial_number in range(40, 4, -1):
        # increase trial number from current files with trial_number 5 to 40 by 1
        print(trial_number)
        trial_files = [f for f in files if "Markerless " + str(trial_number) in f.name]
        assert len(trial_files) == 14, f"Expected 14 files for trial {trial_number}, found {len(trial_files)}"
        for file in trial_files:
            # Markerless {trial_number} -> Markerless {trial_number + 1}
            f_stem = file.stem.replace(f"Markerless {trial_number}", f"Markerless {trial_number + 1}")
            new_file_name = f_stem + file.suffix
            new_file_path = file.with_name(new_file_name)
            print(f"Renaming {file.name} to {new_file_name}")
            move(file, new_file_path)


def rename_heat_1_and_2():
    path_files_heat_1 = "E:\ITW_Backup\HHTGP_HHTGP\Running - Markerless_cut"
    path_files_heat_2 = "E:\ITW_Backup\HHTGP_HHTGP\Running - Markerless_3_cut"
    # "Running trial Markerless 4_cut_01_Miqus_1_25770.avi" -> "Running trial Markerless 1_Miqus_1_25770.avi" etc.
    for heat, path_files_heat in enumerate([path_files_heat_1, path_files_heat_2], start=1):
        path_heat = Path(path_files_heat)
        for file in path_heat.glob("Running trial Markerless *_cut_*.avi"):
            parts = file.stem.split("_")
            trial_number = int(parts[2])
            new_trial_name = f"Running trial Markerless {trial_number}"
            new_file_name = "_".join([new_trial_name] + parts[3:]) + file.suffix
            new_file_path = file.parent / new_file_name
            print(f"Renaming {file.name} to {new_file_name}")
            move(file, new_file_path)


def rename_heat_4():
    path_files_heat_4 = "E:\ITW_Backup\HHTGP_HHTGP\Running - Markerless_5_cut"
    path_heat = Path(path_files_heat_4)
    trial_count = 1
    for trial_number in range(1, 10, 1):
        # skip trial number 3
        if trial_number == 3:
            continue
        files = list(path_heat.glob(f"Running trial Markerless {trial_number}_cut_*.avi"))
        cuts = len(files) // 14
        for cut in range(cuts):
            cut_files = list(
                path_heat.glob(f"Running trial Markerless {trial_number}_cut_{str(cut + 1).zfill(2)}_*.avi"))
            new_trial_number = trial_count
            for cut_file in cut_files:
                parts = cut_file.stem.split("_")
                old_trial_number = parts[0].split(" ")[-1]
                new_trial_name = parts[0].replace(old_trial_number, str(new_trial_number))
                # parts[2] is the trial number
                parts[2] = str(new_trial_number)
                # remove the "cut_xx" part
                new_file_name = "_".join([new_trial_name] + parts[3:]) + cut_file.suffix
                new_file_path = cut_file.parent / new_file_name
                print(f"Renaming {cut_file.name} to {new_file_name}")
                move(cut_file, new_file_path)
            trial_count += 1


if __name__ == '__main__':
    rename_heat_4()
