"""Face detection: a small abstraction plus two implementations.

- ``HaarFaceDetector`` uses an OpenCV Haar cascade without downloading
  anything at runtime. It supports both pip-installed OpenCV and the
  Raspberry Pi OS / Debian system OpenCV package.
- ``ScriptedFaceDetector`` returns caller-supplied detections and needs no
  dependencies, so the control loop can be exercised in tests and in a
  no-camera demo.

Images are HxWx3 uint8 arrays (rows, cols, RGB) — the same format
``PyBulletArm.capture_rgb()`` and the project's webcam source provide.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
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
        cls,
        x: float,
        y: float,
        w: float,
        h: float,
        image_width: int,
        image_height: int,
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
        """Box width as a fraction of image width."""
        return self.w / self.image_width


def largest(detections: Iterable[Detection]) -> Detection | None:
    """Return the biggest detection, or None if there are no detections."""
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
    """Frontal-face detector using an OpenCV Haar cascade.

    Supports:

    - pip-installed ``opencv-python`` through ``cv2.data.haarcascades``
    - Raspberry Pi OS / Debian OpenCV through common system paths
    - a custom path supplied through ``cascade_path``
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
            filename = "haarcascade_frontalface_default.xml"
            candidates: list[Path] = []

            # pip-installed opencv-python
            cv2_data = getattr(cv2, "data", None)
            pip_cascade_directory = getattr(cv2_data, "haarcascades", None)

            if pip_cascade_directory:
                candidates.append(
                    Path(pip_cascade_directory) / filename
                )

            # Raspberry Pi OS / Debian system package locations
            candidates.extend(
                [
                    Path("/usr/share/opencv4/haarcascades") / filename,
                    Path("/usr/local/share/opencv4/haarcascades") / filename,
                    Path("/usr/share/opencv/haarcascades") / filename,
                ]
            )

            found_path = next(
                (path for path in candidates if path.is_file()),
                None,
            )

            if found_path is None:
                checked_paths = "\n".join(
                    f"  - {path}" for path in candidates
                )

                raise RuntimeError(
                    f"Could not find {filename}.\n"
                    f"Checked these locations:\n{checked_paths}\n\n"
                    "On Raspberry Pi OS / Debian, install the cascade with:\n"
                    "    sudo apt update\n"
                    "    sudo apt install -y opencv-data"
                )

            cascade_path = str(found_path)

        classifier = cv2.CascadeClassifier(cascade_path)

        if classifier.empty():
            raise RuntimeError(
                f"Could not load Haar cascade from {cascade_path!r}."
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
            Detection.from_xywh(
                float(x),
                float(y),
                float(w),
                float(h),
                width,
                height,
            )
            for x, y, w, h in boxes
        )

        return largest(detections)


class ScriptedFaceDetector(FaceDetector):
    """Returns a preset sequence of detections, ignoring the image.

    Each ``detect`` call advances to the next scripted detection. Once the
    script is exhausted, it keeps returning the final value. A ``None`` entry
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
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV (cv2) is not installed.\n"
            "On Raspberry Pi OS, install it with:\n"
            "    sudo apt update\n"
            "    sudo apt install -y python3-opencv opencv-data"
        ) from exc

    return cv2
