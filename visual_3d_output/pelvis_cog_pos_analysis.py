from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt


def get_pelvis_cog_positions_from_mat_file(path_mat_file: Path) -> dict[str, list[float]]:
    mat_data = loadmat(path_mat_file)
    pelvis_cog_data = mat_data.get('Pelvis_COG_Pos', None)
    if pelvis_cog_data is None:
        raise ValueError(f"'pelvis_cog_positions' not found in {path_mat_file}")
    # handle annoying visual3d format where data is nested in extra arrays
    while pelvis_cog_data.size == 1:
        pelvis_cog_data = pelvis_cog_data[0]

    pelvis_cog_dict = {
        'x': pelvis_cog_data[:, 0].tolist(),
        'y': pelvis_cog_data[:, 1].tolist(),
        'z': pelvis_cog_data[:, 2].tolist()
    }
    return pelvis_cog_dict




if __name__ == '__main__':
    path_root = Path("E:\QTM_Projects\IntoTheWild\Data\Heat 1_Heat 1\Running - Markerless\TheiaFormatData_mat")
    trial_folders = list(path_root.glob("Running trial Markerless *"))
    for trial_folder in trial_folders:
        mat_files = list(trial_folder.glob("*.mat"))
        for mat_file in mat_files:
            print(f"Processing {mat_file}")
            pelvis_cog_positions = get_pelvis_cog_positions_from_mat_file(mat_file)
            plt.figure(figsize=(10, 6))
            # plot capture area
            plt.plot([7, 7, -7, -7, 7], [3, -3, -3, 3, 3], 'k--', label='Capture Area')
            # plot x, y plane of pelvis COG positions
            plt.plot(pelvis_cog_positions['x'], pelvis_cog_positions['y'], marker='o', markersize=2, linestyle='-')
            plt.title(f'Pelvis COG Positions (X-Y) for {mat_file.stem}')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.axis('equal')
            plt.grid(True)
            title = f'Pelvis COG Positions (X-Y) for {mat_file.stem}'
            plt.show()
            # break after first file for demo purposes
            # break
        break
