from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from classes.camera import CameraCalibrationInfo, get_camera_information_from_cal_file

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


if __name__ == '__main__':
    # Use your provided file path
    # path_cal_file = Path("../data/20250710_105205.qca.txt")
    path_cal_file = Path("../data/cal.txt")
    if path_cal_file.exists():
        list_cam_infos = get_camera_information_from_cal_file(path_cal_file)
    else:
        raise FileNotFoundError(f"Calibration file not found: {path_cal_file}")
    fig = plot_3d_space()
    add_camera_calibration(fig, path_cal_file)

    cam = list_cam_infos[5]
    # Define the corners and center to draw rays for
    points_to_draw = [(x, y) for x in range(0, 5, 1) for y in range(0, 5, 1)]

    for coords in points_to_draw:
        # Use the new method on the camera object
        start, end = cam.pixel_to_camera_ray(x_pixel=coords[0], y_pixel=coords[1], ray_length=10000)
        # Add the returned ray to the plot
        add_camera_ray(fig, start_point=start, end_point=end, ray_color='cyan', ray_title=f'Ray {coords}', show_ray_title=True)

    fig.show()
