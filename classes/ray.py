import numpy as np


class Ray3D:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        if origin.shape != direction.shape:
            raise ValueError("Origin and direction must have the same shape.")
        if origin.ndim != 1 or direction.ndim != 1:
            raise ValueError("Origin and direction must be 1-dimensional arrays.")
        if origin.shape[0] != 3 or direction.shape[0] != 3:
            raise ValueError("Origin and direction must be 3-dimensional vectors.")
        self.origin = origin  # Origin of the ray (a point in space)
        self.direction = direction  # Direction of the ray (a vector)

    def point_at_parameter(self, t: float) -> np.ndarray:
        """Calculate a point along the ray at parameter t."""
        return self.origin + t * self.direction

    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate the distance from the ray to a point."""
        # Vector from ray origin to point
        origin_to_point = point - self.origin
        # Project this vector onto the ray direction
        projection_length = np.dot(origin_to_point, self.direction)
        projection = projection_length * self.direction
        # Calculate the perpendicular vector from the ray to the point
        perpendicular_vector = origin_to_point - projection
        return np.linalg.norm(perpendicular_vector).astype('float')
