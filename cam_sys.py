from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objects as go

from classes.camera import CameraCalibrationInfo, get_camera_information_from_cal_file

ARROW_LEN = 1000  # length of orientation arrows in same units as positions


def plot_with_plotly(list_camera_infos: List[CameraCalibrationInfo], frustum_scale=100.0):
    """
    Creates an interactive 3D plot of camera positions and orientations,
    correctly applying the viewrotation roll.
    """
    fig = go.Figure()

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

    for i, cam_info in enumerate(list_camera_infos):
        pos = cam_info.transform.translation
        rot = cam_info.transform.rotation

        pyramid_points_local = frustum_scale * np.array([
            [0, 0, 0],
            [-1.6, 1, -2],
            [1.6, 1, -2],
            [1.6, -1, -2],
            [-1.6, -1, -2],
        ])

        pyramid_points_local /= 2

        pyramid_points_world = (rot.T @ pyramid_points_local.T).T + pos
        x, y, z = pyramid_points_world.T

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

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0, 0, 0, 0, 1, 3],
            j=[1, 2, 3, 4, 2, 4],
            k=[2, 3, 4, 1, 3, 4],
            opacity=0.5,
            name=f'Cam {cam_info.serial}',
            hoverinfo='name'
        ))

        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers', marker=dict(size=5, color='black'),
            hoverinfo='name', name=f'Position {cam_info.serial}', showlegend=False
        ))

    fig.update_layout(
        title_text='NO TTITLE',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        legend_title_text='Cameras',
        legend=dict(traceorder='grouped')
    )
    style_plot(fig)
    fig.show()


def style_plot(fig,
               span_x=16000,
               span_y=6000,
               step=1000):
    # remove axes titles, labels, and ticks
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, ticks=''),
            yaxis=dict(showticklabels=False, ticks=''),
            zaxis=dict(showticklabels=False, ticks=''),
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
        )
    )

    # black bg, no planes
    bg_col = "black"
    fig.update_layout(
        paper_bgcolor=bg_col,
        plot_bgcolor=bg_col,
        scene=dict(
            bgcolor=bg_col,
            xaxis=dict(showbackground=False, showgrid=False,
                       zeroline=False, color='white', showspikes=False),
            yaxis=dict(showbackground=False, showgrid=False,
                       zeroline=False, color='white', showspikes=False),
            zaxis=dict(showbackground=False, showgrid=False,
                       zeroline=False, color='white', showspikes=False)
        )
    )

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

    return fig


if __name__ == '__main__':
    # Use your provided file path
    path_cal_file = Path("data/20250710_105205.qca.txt")
    # path_cal_file = Path("data/cal.txt")
    list_cam_infos = get_camera_information_from_cal_file(path_cal_file)

    # Adjust scale as needed for your scene's dimensions
    plot_with_plotly(list_cam_infos, frustum_scale=800)
