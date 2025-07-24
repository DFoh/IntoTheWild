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
                   camera: CameraCalibrationInfo,
                   pixel_x: int,
                   pixel_y: int,
                   ray_length: int = 20000,
                   ray_color: str = 'cyan',
                   ray_title: str = 'Camera Ray',
                   show_ray_title: bool = False):
    """
    Add a camera ray to the 3D plot based on the camera's calibration information and pixel coordinates.
    :param fig: 3D scene figure to which the camera ray will be added.
    :param camera: CameraInformation object containing camera calibration data.
    :param pixel_x: Object location in the camera image along the x-axis (horizontal).
    :param pixel_y: Object location in the camera image along the y-axis (vertical).
    :param ray_length: Length of the ray in the 3D space.
    :param ray_color: Color of the ray in the plot.
    """
    # 1. Get camera parameters
    pos = camera.transform.translation
    rot = camera.transform.rotation
    intr = camera.intrinsics

    # 2. Un-project the 2D pixel to a 3D direction vector in camera coordinates
    pixel_u = pixel_x / camera.fov.right * (intr.sensor_max_u - intr.sensor_min_u) + intr.sensor_min_u
    pixel_v = pixel_y / camera.fov.bottom * (intr.sensor_max_v - intr.sensor_min_v) + intr.sensor_min_v
    d_cam = np.array([
        (pixel_u - intr.sensor_min_u) / (intr.sensor_max_u - intr.sensor_min_u) * 2 - 1.0,  # x
        (pixel_v - intr.sensor_min_v) / (intr.sensor_max_v - intr.sensor_min_v) * 2 - 1.0,  # y
        -1.0
    ])

    # 3. Normalize the direction vector to make it a unit vector
    d_cam_normalized = d_cam / np.linalg.norm(d_cam)

    # 4. Transform the direction vector from camera to world coordinates
    # The 'rot' matrix transforms from world-to-camera, so we use its transpose
    # to rotate the camera's direction vector into the world frame.
    d_world = rot.T @ d_cam_normalized

    # 5. Calculate the start and end points of the ray in world coordinates
    start_point = pos
    end_point = pos + d_world * ray_length

    # 6. Add the ray as a line trace to the figure
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

    cam = list_cam_infos[1]
    pixel_x = int(3 / 4 * 1920)
    pixel_y = int(3 / 4 * 1080)
    add_camera_ray(camera=cam, fig=fig, pixel_x=960, pixel_y=540, ray_title="Center Ray", show_ray_title=True)
    add_camera_ray(camera=cam, fig=fig, pixel_x=0, pixel_y=0, ray_title="Top Left Ray", show_ray_title=True)
    add_camera_ray(camera=cam, fig=fig, pixel_x=1920, pixel_y=1080, ray_title="Top Right Ray", show_ray_title=True)
    add_camera_ray(camera=cam, fig=fig, pixel_x=960, pixel_y=0, ray_title="Top Center Ray", show_ray_title=True)
    fig.show()
