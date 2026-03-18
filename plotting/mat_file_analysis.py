import matplotlib.pyplot as plt
import matplotlib
import numpy as np

backend = 'Qt5Agg'

matplotlib.use(backend)

from pathlib import Path
from scipy.io import loadmat

PATH_ROOT = r"C:\Users\dominik.fohrmann\OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University\Dokumente\Projects\IntoTheWild\data\TrackGrandPrix"

if __name__ == '__main__':
    path_mat_root = Path(PATH_ROOT) / "kinematics" / "mat"
    path_plot = Path(PATH_ROOT) / "kinematics" / "plots"
    path_plot.mkdir(exist_ok=True)
    heat_directories = [d for d in path_mat_root.iterdir() if d.is_dir()]
    heat_directories.sort()
    heats = [d.name for d in heat_directories]
    print(heats)
    for heat in heats:
        path_mat = path_mat_root / heat
        bib_numbers = [d.name for d in path_mat.iterdir() if d.is_dir()]
        bib_numbers.sort()
        print(bib_numbers)
        for bib_number in bib_numbers:
            mat_files = list((path_mat / bib_number).glob("*filt.mat"))
            size = [2560.5, 1282.5]
            fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
            for m, mat_file in enumerate(mat_files):
                #
                data = loadmat(mat_file)
                #
                pelvis_com_pos = data['Pelvis_COM_Position'][0][0]
                left_heel_pos = data['Left_Heel_Pos'][0][0]
                right_heel_pos = data['Right_Heel_Pos'][0][0]
                left_toe_pos = data['Left_Toe_Pos'][0][0]
                right_toe_pos = data['Right_Toe_Pos'][0][0]
                ax.plot(pelvis_com_pos[:, 0], pelvis_com_pos[:, 2], label=mat_file.stem)
                color = ax.get_lines()[-1].get_color()
                ax.plot(left_heel_pos[:, 0], left_heel_pos[:, 2], label=mat_file.stem, color=color, linestyle="dashed")
                ax.plot(left_toe_pos[:, 0], left_toe_pos[:, 2], label=mat_file.stem, color=color, linestyle="dotted")
                if m > 2:
                    break

                pelvis_com_pos_ap = pelvis_com_pos[:, 0]
                frame_rate_hz = 85  # Hz
                pelvis_com_velocity_ap = np.gradient(pelvis_com_pos_ap) * frame_rate_hz
                avg_running_speed_kmh = np.nanmean(pelvis_com_velocity_ap) * 3.6
                avg_running_speed_kmh = np.nanmedian(pelvis_com_velocity_ap) * 3.6

            ax.set_ylim(-0.1, 1.1)
            title = f"Heat {heat}, Bib {bib_number} - laps 1-{m + 1}"
            ax.set_title(title)
            # plt.legend()
            path_plot_out = path_plot / "pelvis_heel_plots"
            path_plot_out.mkdir(exist_ok=True)
            # plt.savefig(path_plot_out / f"{heat}_{bib_number}_pelvis_heel.png", bbox_inches="tight")
            plt.show()
            plt.close()
