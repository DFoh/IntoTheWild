import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_camera_position(cam: ET.Element) -> np.ndarray:
    """Extract camera information from an XML element."""
    transform = cam.find('transform')
    x = float(transform.get('x'))
    y = float(transform.get('y'))
    z = float(transform.get('z'))

    return np.array([x, y, z])


def get_camera_rotation(cam: ET.Element) -> np.ndarray:
    """Extract camera rotation from an XML element."""
    transform = cam.find('transform')
    r11 = float(transform.get('r11'))
    r12 = float(transform.get('r12'))
    r13 = float(transform.get('r13'))
    r21 = float(transform.get('r21'))
    r22 = float(transform.get('r22'))
    r23 = float(transform.get('r23'))
    r31 = float(transform.get('r31'))
    r32 = float(transform.get('r32'))
    r33 = float(transform.get('r33'))
    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])
    return rotation_matrix


def plot_camera(camera: ET.Element, ax: plt.Axes = None):
    pos = get_camera_position(camera)
    rot = get_camera_rotation(camera)
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    else:
        fig = ax.figure
    # Plot camera position
    # Plot camera orientation
    # Create a local coordinate system for the camera using quiver
    # create the unit vectors for the camera's local coordinate system
    ex = np.array([1000, 0, 0, 1000])  # X-axis
    ey = np.array([0, 1000, 0, 1000])  # Y-axis
    ez = np.array([0, 0, 1000, 1000])  # Z-axis
    # rotate the unit vectors by the camera's rotation matrix
    trans_4_by_4 = np.eye(4)
    trans_4_by_4[:3, :3] = rot
    trans_4_by_4[:3, 3] = pos
    ex_trans = trans_4_by_4 @ ex
    ey_trans = trans_4_by_4 @ ey
    ez_trans = trans_4_by_4 @ ez
    # Plot the camera position
    ax.scatter(pos[0], pos[1], pos[2], color='black', s=50, label='Camera Position')
    # Plot the camera's local coordinate system using quiver
    ax.quiver(pos[0], pos[1], pos[2], ex_trans[0], ex_trans[1], ex_trans[2], color="red")
    ax.quiver(pos[0], pos[1], pos[2], ey_trans[0], ey_trans[1], ey_trans[2], color="green")
    ax.quiver(pos[0], pos[1], pos[2], ez_trans[0], ez_trans[1], ez_trans[2], color="blue")


if __name__ == '__main__':
    path_calibration_file = Path("data/cal.txt")
    tree = ET.parse(path_calibration_file)
    root = tree.getroot()
    cameras_tag = root.find('cameras')
    cameras = cameras_tag.findall('camera')
    # generate a 3d plot of the cameras
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    # plot gcs origin using quiver in mm
    ax.quiver(0, 0, 0, 1000, 0, 0, color='red', label='X-axis')
    ax.quiver(0, 0, 0, 0, 1000, 0, color='green', label='Y-axis')
    ax.quiver(0, 0, 0, 0, 0, 1000, color='blue', label='Z-axis')
    for camera in cameras:
        plot_camera(camera, ax=ax)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions and Orientations')
    plt.show()
