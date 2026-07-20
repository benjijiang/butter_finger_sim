"""Face detection: a small abstraction plus two implementations.

- ``HaarFaceDetector`` uses OpenCV's bundled Haar cascade (no download, no
  network). OpenCV is imported lazily so this module compiles and imports on
  any machine; only constructing the detector needs ``cv2``.
- ``ScriptedFaceDetector`` returns caller-supplied detections and needs no
  dependencies, so the control loop can be exercised in tests and in a
  no-camera demo.

Images are HxWx3 uint8 arrays (rows, cols, RGB) — the same format
``PyBulletArm.capture_rgb()`` and a webcam both provide. Detectors do not
care which one the pixels came from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    """One detected face bounding box, in pixel coordinates.

    ``cx``/``cy`` are the box center; ``w``/``h`` its size; ``image_width``/
    ``image_height`` the size of the frame it was found in, so normalized
    errors can be computed without carrying the image around.
    """

    cx: float
    cy: float
    w: float
    h: float
    image_width: int
    image_height: int

    @classmethod
    def from_xywh(
        cls, x: float, y: float, w: float, h: float, image_width: int, image_height: int
    ) -> "Detection":
        """Build from a top-left-corner box (the OpenCV convention)."""
        return cls(
            cx=x + w / 2.0,
            cy=y + h / 2.0,
            w=w,
            h=h,
            image_width=image_width,
            image_height=image_height,
        )

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def width_fraction(self) -> float:
        """Box width as a fraction of image width (apparent-size proxy)."""
        return self.w / self.image_width


def largest(detections: Iterable[Detection]) -> Detection | None:
    """Return the biggest detection (closest/most prominent face), or None."""
    best: Detection | None = None
    for det in detections:
        if best is None or det.area > best.area:
            best = det
    return best


class FaceDetector(ABC):
    """Detects the most prominent face in an HxWx3 uint8 image."""

    @abstractmethod
    def detect(self, image: Any) -> Detection | None:
        """Return the largest detected face, or None if no face is found."""
        raise NotImplementedError


class HaarFaceDetector(FaceDetector):
    """Frontal-face detector using OpenCV's bundled Haar cascade.

    The cascade XML ships with the ``opencv-python`` wheel
    (``cv2.data.haarcascades``); nothing is downloaded at runtime.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
        cascade_path: str | None = None,
    ) -> None:
        cv2 = _import_cv2()
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier.empty():
            raise RuntimeError(
                f"Could not load Haar cascade from {cascade_path!r}. Check the "
                "opencv-python installation."
            )
        self._cv2 = cv2
        self._classifier = classifier
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

    def detect(self, image: Any) -> Detection | None:
        cv2 = self._cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]
        boxes = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )
        detections = (
            Detection.from_xywh(float(x), float(y), float(w), float(h), width, height)
            for (x, y, w, h) in boxes
        )
        return largest(detections)


class ScriptedFaceDetector(FaceDetector):
    """Returns a preset sequence of detections, ignoring the image.

    Used by tests and the no-camera demo. Each ``detect`` call advances to the
    next scripted detection; once the script is exhausted it keeps returning
    the final value (or None if the script ended with None). A ``None`` entry
    simulates a frame where no face was found.
    """

    def __init__(self, script: Iterable[Detection | None]) -> None:
        self._script: list[Detection | None] = list(script)
        self._iter: Iterator[Detection | None] = iter(self._script)
        self._last: Detection | None = None

    def detect(self, image: Any = None) -> Detection | None:
        try:
            self._last = next(self._iter)
        except StopIteration:
            pass
        return self._last


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without cv2
        raise RuntimeError(
            "OpenCV (cv2) is not installed. Install the simulation extras on "
            "the sim machine inside .venv:\n"
            "    python -m pip install -r requirements-sim.txt\n"
            "OpenCV is only needed for face detection and the webcam source."
        ) from exc
    return cv2
