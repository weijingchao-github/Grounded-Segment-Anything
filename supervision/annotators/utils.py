from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position


class ColorLookup(Enum):
    """
    Enumeration class to define strategies for mapping colors to annotations.

    This enum supports three different lookup strategies:
        - `INDEX`: Colors are determined by the index of the detection within the scene.
        - `CLASS`: Colors are determined by the class label of the detected object.
        - `TRACK`: Colors are determined by the tracking identifier of the object.
    """

    INDEX = "index"
    CLASS = "class"
    TRACK = "track"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def resolve_color_idx(
    detections: Detections,
    detection_idx: int,
    color_lookup: Union[ColorLookup, np.ndarray] = ColorLookup.CLASS,
) -> int:
    if detection_idx >= len(detections):
        raise ValueError(
            f"Detection index {detection_idx} "
            f"is out of bounds for detections of length {len(detections)}"
        )

    if isinstance(color_lookup, np.ndarray):
        if len(color_lookup) != len(detections):
            raise ValueError(
                f"Length of color lookup {len(color_lookup)} "
                f"does not match length of detections {len(detections)}"
            )
        return color_lookup[detection_idx]
    elif color_lookup == ColorLookup.INDEX:
        return detection_idx
    elif color_lookup == ColorLookup.CLASS:
        if detections.class_id is None:
            raise ValueError(
                "Could not resolve color by class because "
                "Detections do not have class_id. If using an annotator, "
                "try setting color_lookup to sv.ColorLookup.INDEX or "
                "sv.ColorLookup.TRACK."
            )
        return detections.class_id[detection_idx]
    elif color_lookup == ColorLookup.TRACK:
        if detections.tracker_id is None:
            raise ValueError(
                "Could not resolve color by track because "
                "Detections do not have tracker_id. Did you call "
                "tracker.update_with_detections(...) before annotating?"
            )
        return detections.tracker_id[detection_idx]


def resolve_text_background_xyxy(
    center_coordinates: Tuple[int, int],
    text_wh: Tuple[int, int],
    position: Position,
) -> Tuple[int, int, int, int]:
    center_x, center_y = center_coordinates
    text_w, text_h = text_wh

    if position == Position.TOP_LEFT:
        return center_x, center_y - text_h, center_x + text_w, center_y
    elif position == Position.TOP_RIGHT:
        return center_x - text_w, center_y - text_h, center_x, center_y
    elif position == Position.TOP_CENTER:
        return (
            center_x - text_w // 2,
            center_y - text_h,
            center_x + text_w // 2,
            center_y,
        )
    elif (
        position == Position.CENTER
        or position == Position.CENTER_OF_MASS
        or position == Position.DISTANT_TO_BOUNDARY
    ):
        return (
            center_x - text_w // 2,
            center_y - text_h // 2,
            center_x + text_w // 2,
            center_y + text_h // 2,
        )
    elif position == Position.BOTTOM_LEFT:
        return center_x, center_y, center_x + text_w, center_y + text_h
    elif position == Position.BOTTOM_RIGHT:
        return center_x - text_w, center_y, center_x, center_y + text_h
    elif position == Position.BOTTOM_CENTER:
        return (
            center_x - text_w // 2,
            center_y,
            center_x + text_w // 2,
            center_y + text_h,
        )
    elif position == Position.CENTER_LEFT:
        return (
            center_x - text_w,
            center_y - text_h // 2,
            center_x,
            center_y + text_h // 2,
        )
    elif position == Position.CENTER_RIGHT:
        return (
            center_x,
            center_y - text_h // 2,
            center_x + text_w,
            center_y + text_h // 2,
        )


def get_color_by_index(color: Union[Color, ColorPalette], idx: int) -> Color:
    if isinstance(color, ColorPalette):
        return color.by_idx(idx)
    return color


def resolve_color(
    color: Union[Color, ColorPalette],
    detections: Detections,
    detection_idx: int,
    color_lookup: Union[ColorLookup, np.ndarray] = ColorLookup.CLASS,
) -> Color:
    idx = resolve_color_idx(
        detections=detections,
        detection_idx=detection_idx,
        color_lookup=color_lookup,
    )
    return get_color_by_index(color=color, idx=idx)


class Trace:
    def __init__(
        self,
        max_size: Optional[int] = None,
        start_frame_id: int = 0,
        anchor: Position = Position.CENTER,
    ) -> None:
        self.current_frame_id = start_frame_id
        self.max_size = max_size
        self.anchor = anchor

        self.frame_id = np.array([], dtype=int)
        self.xy = np.empty((0, 2), dtype=np.float32)
        self.tracker_id = np.array([], dtype=int)

    def put(self, detections: Detections) -> None:
        frame_id = np.full(len(detections), self.current_frame_id, dtype=int)
        self.frame_id = np.concatenate([self.frame_id, frame_id])
        self.xy = np.concatenate(
            [self.xy, detections.get_anchors_coordinates(self.anchor)]
        )
        self.tracker_id = np.concatenate([self.tracker_id, detections.tracker_id])

        unique_frame_id = np.unique(self.frame_id)

        if 0 < self.max_size < len(unique_frame_id):
            max_allowed_frame_id = self.current_frame_id - self.max_size + 1
            filtering_mask = self.frame_id >= max_allowed_frame_id
            self.frame_id = self.frame_id[filtering_mask]
            self.xy = self.xy[filtering_mask]
            self.tracker_id = self.tracker_id[filtering_mask]

        self.current_frame_id += 1

    def get(self, tracker_id: int) -> np.ndarray:
        return self.xy[self.tracker_id == tracker_id]
