"""Pure camera-frame math and RGB image post-processing."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _unit_vector(values: Sequence[float], *, label: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain three finite numbers")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError(f"{label} must be non-zero")
    return vector / norm


def quaternion_rotation_matrix(
    quaternion_xyzw: Sequence[float],
) -> NDArray[np.float64]:
    """Return a 3x3 rotation matrix for a PyBullet-style XYZW quaternion."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must contain four finite numbers")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0:
        raise ValueError("quaternion_xyzw must be non-zero")
    x, y, z, w = quaternion / norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def camera_view_geometry(
    position_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
    forward_xyz: Sequence[float],
    up_xyz: Sequence[float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Transform local optical directions into PyBullet eye/target/up vectors."""
    eye = np.asarray(position_xyz, dtype=np.float64)
    if eye.shape != (3,) or not np.all(np.isfinite(eye)):
        raise ValueError("position_xyz must contain three finite numbers")

    local_forward = _unit_vector(forward_xyz, label="forward_xyz")
    local_up = _unit_vector(up_xyz, label="up_xyz")
    if not np.isclose(float(np.dot(local_forward, local_up)), 0.0, atol=1e-6):
        raise ValueError("forward_xyz and up_xyz must be orthogonal")

    rotation = quaternion_rotation_matrix(quaternion_xyzw)
    world_forward = rotation @ local_forward
    world_up = rotation @ local_up
    target = eye + world_forward

    return (
        tuple(float(value) for value in eye),
        tuple(float(value) for value in target),
        tuple(float(value) for value in world_up),
    )


def rotate_rgb_clockwise(
    image: NDArray[np.generic],
    degrees: int = 90,
) -> NDArray[np.generic]:
    """Rotate an HWC RGB image clockwise by a multiple of 90 degrees."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have HWC shape with exactly three RGB channels")
    if isinstance(degrees, bool) or not isinstance(degrees, int):
        raise ValueError("degrees must be an integer multiple of 90")
    normalized = degrees % 360
    if normalized % 90 != 0:
        raise ValueError("degrees must be an integer multiple of 90")
    return np.ascontiguousarray(np.rot90(image, k=-(normalized // 90)))
