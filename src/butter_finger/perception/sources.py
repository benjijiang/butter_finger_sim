"""Image sources: where camera frames come from.

An :class:`ImageSource` yields HxWx3 uint8 RGB frames. The detector and
tracker don't care whether the pixels come from a real camera or the
simulated wrist camera.

- ``SimCameraSource`` renders from the arm's ``camera_link`` by delegating to
  ``PyBulletArm.capture_rgb()`` (the simulator already owns camera rendering).
- ``WebcamSource`` reads a local camera via OpenCV. On the Linux sim machine
  this uses the V4L2 backend (``/dev/video*``) transparently.

OpenCV is imported lazily, so importing this module does not require it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from butter_finger.camera import rotate_rgb_clockwise


class ImageSource(ABC):
    """Yields camera frames as HxWx3 uint8 RGB arrays."""

    @abstractmethod
    def read(self) -> Any | None:
        """Return the next frame (RGB), or None if a frame is unavailable."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the source. Safe to call repeatedly."""

    def __enter__(self) -> "ImageSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SimCameraSource(ImageSource):
    """Frames rendered from the arm's wrist camera_link via the backend.

    Delegates to ``PyBulletArm.capture_rgb()`` so camera intrinsics, the
    optical frame, and the 90-degree output rotation all come from the
    simulator's own ``config/camera.yaml`` — this feature adds no second
    camera model.
    """

    def __init__(self, arm: Any) -> None:
        if not hasattr(arm, "capture_rgb"):
            raise TypeError(
                "SimCameraSource requires a backend with capture_rgb() "
                "(PyBulletArm); got " + type(arm).__name__
            )
        self._arm = arm

    def read(self) -> Any | None:
        return self._arm.capture_rgb()


class WebcamSource(ImageSource):
    """Reads frames from a local camera using ``cv2.VideoCapture``.

    OpenCV delivers BGR frames; this converts them to RGB so detectors and the
    sim camera agree on channel order. On Linux, ``cv2.VideoCapture(index)``
    opens ``/dev/video<index>`` through V4L2.

    ``rotate_clockwise_deg`` applies the same quarter-turn correction the
    simulated camera applies in ``PyBulletArm.capture_rgb()``. The physical
    camera is mounted rotated (the top of its native image points along
    wrist-local -X), so a caller feeding frames to an upright-face detector
    must pass ``CameraConfig.rotate_clockwise_deg`` here — otherwise the
    detector sees sideways faces and the image axes do not match the sim's.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        rotate_clockwise_deg: int = 0,
    ) -> None:
        if isinstance(rotate_clockwise_deg, bool) or not isinstance(
            rotate_clockwise_deg, int
        ):
            raise ValueError("rotate_clockwise_deg must be an integer multiple of 90")
        if rotate_clockwise_deg % 90 != 0:
            raise ValueError("rotate_clockwise_deg must be an integer multiple of 90")

        cv2 = _import_cv2()
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open camera at index {camera_index}. On Linux, check "
                f"that /dev/video{camera_index} exists and your user is in the "
                "'video' group (or run with appropriate permissions)."
            )
        if width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Keep the driver queue shallow: a control loop needs the newest frame,
        # not a backlog. Ignored by backends that do not support the property.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cv2 = cv2
        self._capture = capture
        self._rotate_clockwise_deg = rotate_clockwise_deg % 360

    def read(self) -> Any | None:
        ok, frame_bgr = self._capture.read()
        if not ok or frame_bgr is None:
            return None
        frame = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        if self._rotate_clockwise_deg:
            frame = rotate_rgb_clockwise(frame, self._rotate_clockwise_deg)
        return frame

    def close(self) -> None:
        if getattr(self, "_capture", None) is not None:
            self._capture.release()
            self._capture = None


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without cv2
        raise RuntimeError(
            "OpenCV (cv2) is not installed. Install the simulation extras on "
            "the sim machine inside .venv:\n"
            "    python -m pip install -r requirements-sim.txt"
        ) from exc
    return cv2
