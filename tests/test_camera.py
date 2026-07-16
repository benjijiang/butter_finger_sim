"""Validate camera-frame math and PyBullet RGB capture without PyBullet."""
from __future__ import annotations

import numpy as np
import pytest

from butter_finger.backends.pybullet_arm import PyBulletArm
from butter_finger.camera import (
    camera_view_geometry,
    quaternion_rotation_matrix,
    rotate_rgb_clockwise,
)
from butter_finger.config import load_camera_config


class FakeCameraPyBullet:
    ER_TINY_RENDERER = 7

    def __init__(self) -> None:
        self.link_names = (
            b"turret_link",
            b"upper_arm_link",
            b"forearm_link",
            b"wrist_link",
            b"camera_link",
        )
        self.link_state_call: tuple | None = None
        self.view_call: dict | None = None
        self.projection_call: dict | None = None
        self.image_call: dict | None = None

    def getNumJoints(self, body_id, physicsClientId):
        return len(self.link_names)

    def getJointInfo(self, body_id, index, physicsClientId):
        info = [None] * 13
        info[12] = self.link_names[index]
        return tuple(info)

    def getLinkState(self, *args, **kwargs):
        self.link_state_call = (args, kwargs)
        return (None, None, None, None, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))

    def computeViewMatrix(self, **kwargs):
        self.view_call = kwargs
        return [1.0] * 16

    def computeProjectionMatrixFOV(self, **kwargs):
        self.projection_call = kwargs
        return [2.0] * 16

    def getCameraImage(self, **kwargs):
        self.image_call = kwargs
        height = kwargs["height"]
        width = kwargs["width"]
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = (10, 20, 30)
        rgba[:, :, 3] = 255
        return (width, height, rgba, None, None)


def make_fake_camera_arm() -> tuple[PyBulletArm, FakeCameraPyBullet]:
    arm = object.__new__(PyBulletArm)
    pb = FakeCameraPyBullet()
    arm._pb = pb
    arm._client = 7
    arm._robot_id = 11
    arm._camera_link_index = 4
    arm._camera_config = load_camera_config()
    return arm, pb


def test_identity_camera_view_uses_positive_y_and_negative_x_up() -> None:
    eye, target, up = camera_view_geometry(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
    )
    assert eye == pytest.approx((1.0, 2.0, 3.0))
    assert target == pytest.approx((1.0, 3.0, 3.0))
    assert up == pytest.approx((-1.0, 0.0, 0.0))


def test_quaternion_rotation_matrix_is_orthonormal() -> None:
    matrix = quaternion_rotation_matrix((0.0, 0.0, 2**-0.5, 2**-0.5))
    assert matrix @ matrix.T == pytest.approx(np.identity(3))
    assert np.linalg.det(matrix) == pytest.approx(1.0)
    assert matrix @ np.array((0.0, 1.0, 0.0)) == pytest.approx((-1.0, 0.0, 0.0))


def test_rotate_rgb_clockwise_maps_corners_and_swaps_dimensions() -> None:
    image = np.array(
        [
            [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
            [[4, 0, 0], [5, 0, 0], [6, 0, 0]],
        ],
        dtype=np.uint8,
    )
    rotated = rotate_rgb_clockwise(image)
    assert rotated.shape == (3, 2, 3)
    assert rotated[:, :, 0].tolist() == [[4, 1], [5, 2], [6, 3]]
    assert rotated.flags.c_contiguous


def test_rejects_non_orthogonal_camera_directions() -> None:
    with pytest.raises(ValueError, match="must be orthogonal"):
        camera_view_geometry(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )


def test_discovers_fixed_camera_link_by_child_link_name() -> None:
    arm, _ = make_fake_camera_arm()
    assert arm._discover_link_index("camera_link") == 4


def test_capture_rgb_uses_native_projection_and_rotated_output() -> None:
    arm, pb = make_fake_camera_arm()

    rgb = arm.capture_rgb()

    assert rgb.shape == (640, 480, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb[0, 0] == (10, 20, 30))
    assert pb.link_state_call == (
        (11, 4),
        {"computeForwardKinematics": 1, "physicsClientId": 7},
    )
    assert pb.view_call == {
        "cameraEyePosition": pytest.approx((1.0, 2.0, 3.0)),
        "cameraTargetPosition": pytest.approx((1.0, 3.0, 3.0)),
        "cameraUpVector": pytest.approx((-1.0, 0.0, 0.0)),
    }
    assert pb.projection_call == {
        "fov": 120.0,
        "aspect": pytest.approx(4.0 / 3.0),
        "nearVal": 0.02,
        "farVal": 2.0,
    }
    assert pb.image_call["width"] == 640
    assert pb.image_call["height"] == 480
    assert pb.image_call["renderer"] == pb.ER_TINY_RENDERER
    assert pb.image_call["physicsClientId"] == 7


def test_capture_rgb_rejects_unexpected_buffer_size() -> None:
    arm, pb = make_fake_camera_arm()

    def short_image(**kwargs):
        return (kwargs["width"], kwargs["height"], [0, 0, 0, 255], None, None)

    pb.getCameraImage = short_image
    with pytest.raises(RuntimeError, match="unexpected RGBA buffer size"):
        arm.capture_rgb()
