import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import numpy as np


class CameraCalibrationInfo:
    def __init__(self,
                 model: str,
                 serial: str,
                 ):
        self.model = model
        self.serial = serial
        self.view_rotation = 0  # Initialize the new attribute
        self._transform = None
        self._intrinsics = None
        self._fov = None

    def __repr__(self):
        return f"CameraCalibrationInfo(model={self.model}, serial={self.serial})"

    @property
    def transform(self) -> 'Transformation':
        if self._transform is None:
            raise ValueError("Transform has not been set.")
        return self._transform

    @transform.setter
    def transform(self, value: 'Transformation'):
        if not isinstance(value, Transformation):
            raise TypeError("Transform must be an instance of Transformation.")
        self._transform = value

    @property
    def intrinsics(self) -> 'IntrinsicParameters':
        if self._intrinsics is None:
            raise ValueError("Intrinsics have not been set.")
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, value: 'IntrinsicParameters'):
        if not isinstance(value, IntrinsicParameters):
            raise TypeError("Intrinsics must be an instance of IntrinsicParameters.")
        self._intrinsics = value

    @property
    def fov(self) -> 'FOV':
        if self._fov is None:
            raise ValueError("FOV has not been set.")
        return self._fov

    @fov.setter
    def fov(self, value: 'FOV'):
        if not isinstance(value, FOV):
            raise TypeError("FOV must be an instance of FOV.")
        self._fov = value

    @staticmethod
    def from_xml_element(cam_xml: ET.Element):
        model = cam_xml.get('model')
        serial = cam_xml.get('serial')
        if model is None or serial is None:
            raise ValueError("Camera model and serial must be specified in the XML element.")
        cam_info = CameraCalibrationInfo(model, serial)

        cam_info.transform = Transformation.from_xml_element(cam_xml.find('transform'))
        cam_info.intrinsics = IntrinsicParameters.from_xml_element(cam_xml.find('intrinsic'))
        cam_info.fov = FOV.from_xml_element(cam_xml.find('fov_video_max'))
        return cam_info


class FOV:
    def __init__(self, top: int, bottom: int, left: int, right: int):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __repr__(self):
        return f"FOV(top={self.top}, bottom={self.bottom}, left={self.left}, right={self.right})"

    @staticmethod
    def from_xml_element(fov_video_max: ET.Element) -> 'FOV':
        left = int(fov_video_max.get('left'))
        right = int(fov_video_max.get('right'))
        bottom = int(fov_video_max.get('bottom'))
        top = int(fov_video_max.get('top'))
        return FOV(top, bottom, left, right)


class Transformation:
    def __init__(self, translation: np.ndarray, rotation: np.ndarray):
        self.translation = translation
        self.rotation = rotation

    def __repr__(self):
        return f"Transformation(translation={self.translation}, rotation={self.rotation})"

    @staticmethod
    def from_xml_element(transform: ET.Element) -> 'Transformation':
        pos = np.array([float(transform.get('x')), float(transform.get('y')), float(transform.get('z'))])
        rot = np.array([
            [float(transform.get('r11')), float(transform.get('r12')), float(transform.get('r13'))],
            [float(transform.get('r21')), float(transform.get('r22')), float(transform.get('r23'))],
            [float(transform.get('r31')), float(transform.get('r32')), float(transform.get('r33'))],
        ])
        return Transformation(pos, rot)


class IntrinsicParameters:
    def __init__(self,
                 focallength: float,
                 sensor_min_u: float,
                 sensor_max_u: float,
                 sensor_min_v: float,
                 sensor_max_v: float,
                 focal_length_u: float,
                 focal_length_v: float,
                 center_point_u: float,
                 center_point_v: float,
                 skew: float,
                 radial_distortion1: float,
                 radial_distortion2: float,
                 radial_distortion3: float,
                 tangential_distortion1: float,
                 tangential_distortion2: float):
        self.focallength = focallength
        self.sensor_min_u = sensor_min_u
        self.sensor_max_u = sensor_max_u
        self.sensor_min_v = sensor_min_v
        self.sensor_max_v = sensor_max_v
        self.focal_length_u = focal_length_u
        self.focal_length_v = focal_length_v
        self.center_point_u = center_point_u
        self.center_point_v = center_point_v
        self.skew = skew
        self.radial_distortion1 = radial_distortion1
        self.radial_distortion2 = radial_distortion2
        self.radial_distortion3 = radial_distortion3
        self.tangential_distortion1 = tangential_distortion1
        self.tangential_distortion2 = tangential_distortion2

    @staticmethod
    def from_xml_element(intrinsics: ET.Element) -> 'IntrinsicParameters':
        focallength = float(intrinsics.get('focallength'))
        sensor_min_u = float(intrinsics.get('sensorMinU'))
        sensor_max_u = float(intrinsics.get('sensorMaxU'))
        sensor_min_v = float(intrinsics.get('sensorMinV'))
        sensor_max_v = float(intrinsics.get('sensorMaxV'))
        focal_length_u = float(intrinsics.get('focalLengthU'))
        focal_length_v = float(intrinsics.get('focalLengthV'))
        center_point_u = float(intrinsics.get('centerPointU'))
        center_point_v = float(intrinsics.get('centerPointV'))
        skew = float(intrinsics.get('skew', 0.0))
        radial_distortion1 = float(intrinsics.get('radialDistortion1', 0.0))
        radial_distortion2 = float(intrinsics.get('radialDistortion2', 0.0))
        radial_distortion3 = float(intrinsics.get('radialDistortion3', 0.0))
        tangential_distortion1 = float(intrinsics.get('tangentialDistortion1', 0.0))
        tangential_distortion2 = float(intrinsics.get('tangentialDistortion2', 0.0))

        return IntrinsicParameters(
            focallength, sensor_min_u, sensor_max_u,
            sensor_min_v, sensor_max_v, focal_length_u,
            focal_length_v, center_point_u, center_point_v,
            skew, radial_distortion1, radial_distortion2,
            radial_distortion3, tangential_distortion1,
            tangential_distortion2
        )


def get_camera_information_from_cal_file(cal_file: Path) -> List[CameraCalibrationInfo]:
    list_cameras = []
    tree = ET.parse(cal_file)
    root = tree.getroot()
    cams = root.findall(".//camera")
    for cam in cams:
        cam_info = CameraCalibrationInfo.from_xml_element(cam_xml=cam)
        list_cameras.append(cam_info)
    return list_cameras
