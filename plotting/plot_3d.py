import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from classes.camera import CameraCalibrationInfo, get_camera_information_from_cal_file

# TODO: DELETE THIS IMPORT
import matplotlib.pyplot as plt

ARROW_LEN = 1000  # length of orientation arrows in same units as positions


def add_global_origin(fig: go.Figure):
    # Add world axes for reference
    fig.add_trace(go.Scatter3d(x=[0, ARROW_LEN], y=[0, 0], z=[0, 0],
                               mode='lines',
                               line=dict(color='red', width=10),
                               name='World X',
                               hoverinfo='skip',
                               showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, ARROW_LEN], z=[0, 0],
                               mode='lines',
                               line=dict(color='green', width=10),
                               name='World Y',
                               hoverinfo='skip',
                               showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, ARROW_LEN],
                               mode='lines',
                               line=dict(color='blue', width=10),
                               name='World Z',
                               hoverinfo='skip',
                               showlegend=False))


def add_floor_grid(fig: go.Figure, span_x=16000, span_y=6000, step=1000):
    # floor grid (Z=0)
    xs = np.arange(-span_x / 2, span_x / 2 + step, step)
    ys = np.arange(-span_y / 2, span_y / 2 + step, step)

    grid_line_col = "white"
    grid_line_width = 2

    for x in xs:  # lines parallel Y
        fig.add_trace(go.Scatter3d(
            x=[x, x], y=[-span_y / 2, span_y / 2], z=[0, 0],
            mode='lines',
            line=dict(color=grid_line_col, width=grid_line_width),
            hoverinfo="skip",
            showlegend=False))
    for y in ys:  # lines parallel X
        fig.add_trace(go.Scatter3d(
            x=[-span_x / 2, span_x / 2], y=[y, y], z=[0, 0],
            mode='lines',
            line=dict(color=grid_line_col, width=grid_line_width),
            hoverinfo="skip",
            showlegend=False))


def style_plot(fig: go.Figure):
    # remove axes titles, labels, and ticks, and set black background with no planes
    bg_col = "black"
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, ticks='', showbackground=False, showgrid=False,
                       zeroline=False, color='white', showspikes=False),
            yaxis=dict(showticklabels=False, ticks='', showbackground=False, showgrid=False,
                       zeroline=False, color='white', showspikes=False),
            zaxis=dict(showticklabels=False, ticks='', showbackground=False, showgrid=False,
                       zeroline=False, color='white', showspikes=False),
            aspectmode='data',
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            bgcolor=bg_col
        ),
        paper_bgcolor=bg_col,
        plot_bgcolor=bg_col
    )


def plot_3d_space() -> go.Figure:
    # Create the 3D space we will work in
    fig = go.Figure()
    add_global_origin(fig)
    add_floor_grid(fig)
    style_plot(fig)
    # set a fixed view position and angle
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=0, y=-1.5, z=1.5),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
    )

    return fig


def add_camera(fig: go.Figure, camera: CameraCalibrationInfo):
    pos = camera.transform.translation
    rot = camera.transform.rotation

    camera_cone_points = 800 * np.array([  # 800 is the scale factor for the pyramid size
        [0, 0, 0],
        [-1.6, 1, -2],
        [1.6, 1, -2],
        [1.6, -1, -2],
        [-1.6, -1, -2],
    ])

    camera_cone_points /= 2

    camera_cone_points_world = (rot.T @ camera_cone_points.T).T + pos
    x, y, z = camera_cone_points_world.T

    # Plotly (add camera‐axis lines the same way)
    axes_world = rot  # rows already are X,Y,Z in world
    for vec, col in zip(axes_world, ('red', 'green', 'blue')):
        fig.add_trace(go.Scatter3d(
            x=[pos[0], pos[0] + ARROW_LEN / 2 * vec[0]],
            y=[pos[1], pos[1] + ARROW_LEN / 2 * vec[1]],
            z=[pos[2], pos[2] + ARROW_LEN / 2 * vec[2]],
            mode='lines',
            hoverinfo='skip',
            line=dict(color=col, width=3),
            showlegend=False))

    # Add the camera cones to the figure
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=[0, 0, 0, 0, 1, 3],
        j=[1, 2, 3, 4, 2, 4],
        k=[2, 3, 4, 1, 3, 4],
        opacity=0.5,
        color='rgba(255, 255, 255, 0.5)',  # semi-transparent white
        name=f'Cam {camera.serial}',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers', marker=dict(size=5, color='black'),
        hoverinfo='name', name=f'Position {camera.serial}', showlegend=False
    ))


def add_camera_calibration(fig: go.Figure, path_calibration_file: Path):
    """
    Add camera calibration information to the 3D plot from a calibration file.
    :param fig: The Plotly figure to add the cameras to.
    :param path_calibration_file: Path to the camera calibration file.
    """
    list_cam_infos = get_camera_information_from_cal_file(path_calibration_file)
    for camera in list_cam_infos:
        add_camera(fig, camera)


def add_camera_ray(fig: go.Figure,
                   start_point: np.ndarray,
                   end_point: np.ndarray,
                   ray_color: str = 'cyan',
                   ray_title: str = 'Camera Ray',
                   show_ray_title: bool = False):
    fig.add_trace(go.Scatter3d(
        x=[start_point[0], end_point[0]],
        y=[start_point[1], end_point[1]],
        z=[start_point[2], end_point[2]],
        mode='lines',
        line=dict(color=ray_color, width=3),
        name=ray_title if show_ray_title else '',
        hoverinfo='name' if show_ray_title else 'skip',
        showlegend=False
    ))


def add_camera_rays_from_json(fig: go.Figure,
                              cam_infos: list[CameraCalibrationInfo],
                              points: dict[str, tuple[int, int]],
                              ray_len=ARROW_LEN):
    # map cam serial/id to CameraCalibrationInfo
    cam_map = {str(cam.serial): cam for cam in cam_infos}
    rays = []
    line_colors = ['cyan', 'magenta', 'yellow', 'orange', 'lime', 'pink', 'lightblue', 'lightgreen']


    for cam_id, cam_points in points.items():
        if cam_points is None:
            continue
        if len(cam_points) > 3:
            continue
        line_color = line_colors[hash(cam_id) % len(line_colors)]
        for cam_point in cam_points:
            cam = cam_map.get(cam_id)
            px = cam_point["uv"][0]
            py = cam_point["uv"][1]
            if cam is None:
                print(f"Warning: no calibration for camera {cam_id}")
                continue
            # compute ray
            start, end = cam.pixel_to_camera_ray(x_pixel=px, y_pixel=py, ray_length=ray_len)
            # direction vector
            v = end - start
            # rays.append((start, v))
            rays.append((start, v, cam_id))

            # add to plot
            print(f"Adding ray: cam {cam_id}, pixel ({px}, {py}), start {start}, end {end}")
            # add consistent color per camera

            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                mode='lines', line=dict(color=line_color, width=3), name=f'Ray {cam_id}', hoverinfo='skip'
            ))
    print(f"Added total of {len(rays)} rays from {len(points)} cameras.")
    clusters = triangulate_clusters(rays, k=3, fig=fig)
    # for c in clusters:
    #     print(f"Adding cluster point at {c['point']} supported by cams {c['cams']}")
    #     plot_point_in_3d(fig, c["point"], name=f'P ({len(c["cams"])} cams)')
    # if rays:
    #     optimal = compute_optimal_point(rays)
    #     print(f"Adding optimal point computed at: {optimal}")
    #     fig.add_trace(go.Scatter3d(
    #         x=[optimal[0]], y=[optimal[1]], z=[optimal[2]],
    #         mode='markers', marker=dict(size=6, color='yellow'), name='Optimal Point'
    #     ))


def compute_distances(rays: list[tuple[np.ndarray, np.ndarray]], point: np.ndarray) -> np.ndarray:
    """
    Compute the distances from a point to a set of rays.
    :param rays: List of tuples (point_on_line, direction_unit_vector).
    :param point: The point to compute distances to.
    :return: List of distances from the point to each ray.
    """
    distances = []
    for p, v in rays:
        v = v / np.linalg.norm(v)  # ensure direction is a unit vector
        d = np.linalg.norm(point - p - np.dot(point - p, v) * v)
        distances.append(d)
    distances = np.array(distances)
    distances.sort()
    return distances


def solve_optimal_point(rays: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Solve for the optimal point that minimizes the distance to a set of rays.
    :param rays: List of tuples (point_on_line, direction_unit_vector).
    :return: The optimal point in 3D space.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, v in rays:
        v = v / np.linalg.norm(v)  # ensure direction is a unit vector
        P = np.eye(3) - np.outer(v, v)
        A += P
        b += P.dot(p)
    # Solve for x: A x = b
    solution = np.linalg.solve(A, b)
    return solution


# Compute the closest point to a set of 3D lines (rays)
def compute_optimal_point(rays: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    solution = solve_optimal_point(rays)
    # compute the distance to each ray
    distances = compute_distances(rays, solution)
    # detect outliers:
    if len(distances) > 2:
        median_dist = np.median(distances)
        outliers = distances[distances > 2 * median_dist]
        if len(outliers) > 0:
            print(f"Warning: {len(outliers)} outliers detected with distances: {outliers}")
    else:
        print("Warning: Not enough rays to compute optimal point reliably.")
    # remove outliers from the rays
    rays_updated = [r for r, d in zip(rays, distances) if d <= 2 * np.median(distances)]
    # solve again without outliers
    solution_updated = solve_optimal_point(rays_updated)
    # get the new distances
    updated_distances = compute_distances(rays_updated, solution_updated)
    return solution_updated


def load_jsonl_line(path_jsonl, n):
    with open(path_jsonl) as f:
        for i, line in enumerate(f):
            if i == n:
                return json.loads(line)
    return None  # out of range


def main(dir_cal_file: Path, dir_json: Path):
    # load calibration
    if not dir_cal_file.exists():
        raise FileNotFoundError(f"Calibration file missing: {dir_cal_file}")
    cam_infos = get_camera_information_from_cal_file(dir_cal_file)

    # load selections
    if not dir_json.exists():
        raise FileNotFoundError(f"Selections JSON missing: {dir_json}")

    line = load_jsonl_line(dir_json, 1190)
    points = line['points']

    # build plot
    fig = plot_3d_space()
    add_camera_calibration(fig, dir_cal_file)

    add_camera_rays_from_json(fig, cam_infos, points, ray_len=15000)
    fig.show()
    print("done")


def plot_point_in_3d(fig: go.Figure, point: np.ndarray, name: str = 'Point', color: str = 'yellow'):
    """
    Plot a single point in 3D space.
    :param fig: The Plotly figure to add the point to.
    :param point: The 3D point to plot.
    :param name: The name of the point for the legend.
    :param color: The color of the point.
    """
    fig.add_trace(go.Scatter3d(
        x=[point[0]], y=[point[1]], z=[point[2]],
        mode='markers',
        marker=dict(size=6, color=color),
        name=name,
        hoverinfo='name'
    ))


# pp1=np.array([0,0,0])
# vv1=np.array([1,0,0])
# pp2=np.array([0,0,0])
# vv2=np.array([0,1,0])
# closest_point_between_lines(pp1, vv1, pp2, vv2)

def closest_point_between_lines(p1, v1, p2, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    w0 = p1 - p2
    a = v1 @ v1
    b = v1 @ v2
    c = v2 @ v2
    d = v1 @ w0
    e = v2 @ w0
    D = a * c - b * b + 1e-9
    s = (b * e - c * d) / D
    t = (a * e - b * d) / D
    c1 = p1 + s * v1
    c2 = p2 + t * v2
    mid = 0.5 * (c1 + c2)
    err = np.linalg.norm(c1 - c2)
    return mid, err


def kmeans(P, k=3, iters=20):
    C = P[np.random.choice(len(P), k, replace=False)]
    for _ in range(iters):
        D = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2)
        a = D.argmin(1)
        newC = np.array([P[a == i].mean(0) if np.any(a == i) else C[i] for i in range(k)])
        if np.allclose(C, newC): break
        C = newC
    return C, a


def triangulate_clusters(rays, k=3, pair_err_thresh=300.0, max_ray_length=10000.0, fig=None):
    # rays: [(p, v, cam_id), ...]
    cand = []
    idx = []
    for i, (p1, v1, c1) in enumerate(rays):
        for j, (p2, v2, c2) in enumerate(rays):
            if j <= i or c1 == c2: continue  # skip same ray (or smaller index) or same camera
            mid, err = closest_point_between_lines(p1, v1, p2, v2)
            if err > pair_err_thresh:
                continue
            l1 = np.linalg.norm(mid - p1)
            l2 = np.linalg.norm(mid - p2)
            if l1 > max_ray_length or l2 > max_ray_length:
                continue
            cand.append(mid)
            idx.append((i, j))
    if not cand: return []
    P = np.vstack(cand)
    C, a = kmeans(P, k=min(k, len(P)))

    # TODO: DELETE!! EXPERIMENTAL
    # print random sample points after candidates
    # ax = plt.figure().add_subplot(projection='3d')

    # randP = P[np.random.choice(len(P), min(80, len(P)), replace=False)]
    for p in range(len(P)):
        point = P[p]
        # ax.scatter(point[0], point[1], point[2], color='cyan', s=20)
        plot_point_in_3d(fig, point, name='Candidate Point', color='cyan')
    # print all cluster centers
    for c in range(len(C)):
        cc = C[c]
        plot_point_in_3d(fig, cc, name=f'Cluster Center {c}', color='red')
        # ax.scatter(cc[0], cc[1], cc[2], color='red', s=100)
    plt.show()
    # END TODO

    # refine per cluster with LS on assigned rays
    out = []
    for ci in range(len(C)):
        pairs = [idx[m] for m in range(len(a)) if a[m] == ci]
        used = set()
        R = []
        for i, j in pairs:
            for rix in (i, j):
                if rix not in used:
                    R.append(rays[rix]);
                    used.add(rix)
        if len(R) >= 2:
            p = solve_optimal_point([(p, v) for (p, v, _) in R])
            out.append({"point": p,
                        "cams": sorted({c for *_, c in R}),
                        "n_rays": len(R)})
    # sort by support
    out.sort(key=lambda d: (d["n_rays"], len(d["cams"])), reverse=True)
    return out[:k]


def reconstruct(cam_rays: list, min_ray_count: int, min_ray_len_m: float, max_ray_len_m: float, max_residual: float):
    pass


if __name__ == '__main__':
    # Use your provided file path
    # path_cal_file = Path("../data/20250710_105205.qca.txt")
    path_cal_file = Path("../data/cal.txt")
    # path_json = Path("../selected_points.json")

    path_json = Path("../output_tracks/merged_tracking_output.jsonl")
    main(path_cal_file, path_json)

    # if path_cal_file.exists():
    #     list_cam_infos = get_camera_information_from_cal_file(path_cal_file)
    # else:
    #     raise FileNotFoundError(f"Calibration file not found: {path_cal_file}")
    # fig = plot_3d_space()
    # add_camera_calibration(fig, path_cal_file)
    #
    # cam = list_cam_infos[5]
    # cam1 = list_cam_infos[6]
    # start, end = cam.pixel_to_camera_ray(x_pixel=900, y_pixel=500, ray_length=10000)
    # add_camera_ray(fig, start_point=start, end_point=end)
    # start, end = cam1.pixel_to_camera_ray(x_pixel=900, y_pixel=500, ray_length=10000)
    # add_camera_ray(fig, start_point=start, end_point=end)
    # # add_camera_ray(fig, start_point=start, end_point=end, ray_color='red', ray_title='Cam1 Ray', show_ray_title=True)
    # #
    # # # Define the corners and center to draw rays for
    # # points_to_draw = [(x, y) for x in range(0, 5, 1) for y in range(0, 5, 1)]
    # #
    # # for coords in points_to_draw:
    # #     # Use the new method on the camera object
    # #     start, end = cam.pixel_to_camera_ray(x_pixel=coords[0], y_pixel=coords[1], ray_length=10000)
    # #     # Add the returned ray to the plot
    # #     add_camera_ray(fig, start_point=start, end_point=end, ray_color='cyan', ray_title=f'Ray {coords}', show_ray_title=True)
    # #
    # fig.show()
