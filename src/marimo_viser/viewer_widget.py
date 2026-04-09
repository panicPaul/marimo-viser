"""Native image-based 3D viewer widget for marimo notebooks."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anywidget
import cv2
import numpy as np
import torch
import traitlets
from jaxtyping import Float, UInt8
from marimo._plugins.ui._core.ui_element import UIElement
from marimo._plugins.ui._impl.comm import MarimoComm
from marimo._plugins.ui._impl.from_anywidget import (
    AnyWidgetState,
    ModelIdRef,
    get_anywidget_model_id,
    get_anywidget_state,
)
from marimo._plugins.ui._impl.from_anywidget import (
    anywidget as BaseMarimoAnyWidget,
)
from marimo._runtime.virtual_file import VirtualFile
from marimo._utils.code import hash_code
from PIL import Image

CameraConvention = Literal["opencv", "opengl", "blender", "colmap"]
_ASSET_DIR = Path(__file__).with_name("assets")


_CONVENTION_TO_INTERNAL_ROTATION: dict[CameraConvention, np.ndarray] = {
    "opencv": np.diag([1.0, -1.0, 1.0]),
    "opengl": np.diag([1.0, 1.0, -1.0]),
    "blender": np.diag([1.0, 1.0, -1.0]),
    "colmap": np.diag([1.0, -1.0, 1.0]),
}


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Return a normalized copy of the input vector."""
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _camera_basis_from_cam_to_world(
    cam_to_world: Float[np.ndarray, "4 4"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract position, forward, and up vectors from a camera transform."""
    matrix = np.asarray(cam_to_world, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"cam_to_world must have shape (4, 4), got {matrix.shape}."
        )
    position = matrix[:3, 3]
    rotation = matrix[:3, :3]
    forward = _normalize(rotation[:, 2])
    up = _normalize(rotation[:, 1])
    return position, forward, up


def _look_at_cam_to_world(
    position: np.ndarray,
    look_at: np.ndarray,
    up_direction: np.ndarray,
) -> Float[np.ndarray, "4 4"]:
    """Construct a cam-to-world transform from look-at parameters."""
    forward = _normalize(look_at - position)
    right = np.cross(forward, up_direction)
    if np.linalg.norm(right) <= 1e-8:
        fallback_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
    right = _normalize(right)
    up = _normalize(np.cross(right, forward))

    cam_to_world = np.eye(4, dtype=np.float64)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = forward
    cam_to_world[:3, 3] = position
    return cam_to_world


def _convention_transform_matrix(
    camera_convention: CameraConvention,
) -> Float[np.ndarray, "4 4"]:
    """Return the camera-basis transform into the viewer's internal basis."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _CONVENTION_TO_INTERNAL_ROTATION[camera_convention]
    return transform


def _convert_cam_to_world_between_conventions(
    cam_to_world: Float[np.ndarray, "4 4"],
    *,
    source_convention: CameraConvention,
    target_convention: CameraConvention,
) -> Float[np.ndarray, "4 4"]:
    """Convert a camera transform between supported camera conventions."""
    matrix = np.asarray(cam_to_world, dtype=np.float64)
    internal = matrix @ _convention_transform_matrix(source_convention)
    target = internal @ _convention_transform_matrix(target_convention)
    return target


@dataclass(frozen=True)
class CameraState:
    """Serializable native viewer camera state."""

    fov_degrees: float
    width: int
    height: int
    cam_to_world: Float[np.ndarray, "4 4"]
    camera_convention: CameraConvention = "opencv"

    def __post_init__(self) -> None:
        """Validate dimensions and camera matrix shape."""
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}.")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}.")
        if self.fov_degrees <= 0.0:
            raise ValueError(
                f"fov_degrees must be positive, got {self.fov_degrees}."
            )
        if self.fov_degrees >= 180.0:
            raise ValueError(
                "fov_degrees must be less than 180 degrees, "
                f"got {self.fov_degrees}."
            )
        if self.camera_convention not in {
            "opencv",
            "opengl",
            "blender",
            "colmap",
        }:
            raise ValueError(
                "camera_convention must be one of "
                "'opencv', 'opengl', 'blender', or 'colmap', "
                f"got {self.camera_convention!r}."
            )
        matrix = np.asarray(self.cam_to_world, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"cam_to_world must have shape (4, 4), got {matrix.shape}."
            )
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0])):
            raise ValueError("cam_to_world bottom row must be [0, 0, 0, 1].")
        object.__setattr__(self, "cam_to_world", matrix.copy())

    @classmethod
    def default(
        cls,
        *,
        width: int = 800,
        height: int = 600,
        fov_degrees: float = 60.0,
        camera_convention: CameraConvention = "opencv",
    ) -> CameraState:
        """Create a default forward-facing camera state.

        The field of view is expressed in degrees.
        """
        position = np.array([0.0, 0.0, 3.0], dtype=np.float64)
        look_at = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return cls(
            fov_degrees=fov_degrees,
            width=width,
            height=height,
            cam_to_world=_look_at_cam_to_world(position, look_at, up_direction),
            camera_convention=camera_convention,
        )

    @property
    def position(self) -> np.ndarray:
        """Return the camera position."""
        return self.cam_to_world[:3, 3].copy()

    @property
    def forward(self) -> np.ndarray:
        """Return the normalized camera forward vector."""
        return _normalize(self.cam_to_world[:3, 2])

    @property
    def up_direction(self) -> np.ndarray:
        """Return the normalized camera up vector."""
        return _normalize(self.cam_to_world[:3, 1])

    def with_size(self, width: int, height: int) -> CameraState:
        """Return a copy with updated output dimensions."""
        return CameraState(
            fov_degrees=self.fov_degrees,
            width=width,
            height=height,
            cam_to_world=self.cam_to_world,
            camera_convention=self.camera_convention,
        )

    def to_json(self) -> str:
        """Serialize the camera state into a stable JSON string.

        The field of view is expressed in degrees.
        """
        return json.dumps(
            {
                "fov_degrees": self.fov_degrees,
                "width": self.width,
                "height": self.height,
                "cam_to_world": self.cam_to_world.tolist(),
                "camera_convention": self.camera_convention,
            }
        )

    @classmethod
    def from_json(cls, value: str) -> CameraState:
        """Deserialize a camera state from JSON."""
        payload = json.loads(value)
        return cls(
            fov_degrees=float(payload["fov_degrees"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            cam_to_world=np.asarray(payload["cam_to_world"], dtype=np.float64),
            camera_convention=payload.get("camera_convention", "opencv"),
        )


@dataclass(frozen=True)
class ViewerClick:
    """Serializable click event captured from the native viewer."""

    x: int
    y: int
    width: int
    height: int
    camera_state: CameraState

    def __post_init__(self) -> None:
        """Validate click coordinates and viewport dimensions."""
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}.")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}.")
        if not 0 <= self.x < self.width:
            raise ValueError(f"x must be in [0, {self.width}), got {self.x}.")
        if not 0 <= self.y < self.height:
            raise ValueError(f"y must be in [0, {self.height}), got {self.y}.")

    def to_json(self) -> str:
        """Serialize the click into a stable JSON string."""
        return json.dumps(
            {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
                "camera_state": json.loads(self.camera_state.to_json()),
            }
        )

    @classmethod
    def from_json(cls, value: str) -> ViewerClick:
        """Deserialize a click from JSON."""
        payload = json.loads(value)
        return cls(
            x=int(payload["x"]),
            y=int(payload["y"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            camera_state=CameraState.from_json(
                json.dumps(payload["camera_state"])
            ),
        )


def _normalize_frame(
    frame: np.ndarray | torch.Tensor,
) -> UInt8[np.ndarray, "height width 3"]:
    """Convert an RGB numpy or torch frame to contiguous uint8."""
    array: np.ndarray
    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"Expected frame shape (height, width, 3), got {array.shape}."
        )

    if np.issubdtype(array.dtype, np.floating):
        working = np.asarray(array, dtype=np.float32)
        if (
            float(np.nanmax(working)) <= 1.0
            and float(np.nanmin(working)) >= 0.0
        ):
            working = working * 255.0
        array = np.clip(working, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(array)


class _WidgetValueProxy(MutableMapping[str, object]):
    """Live mapping view over synced anywidget traits."""

    def __init__(self, widget: _NativeViewerAnyWidget) -> None:
        self._widget = widget

    def _state(self) -> dict[str, object]:
        return self._widget.get_state()

    def __getitem__(self, key: str) -> object:
        return self._state()[key]

    def __setitem__(self, key: str, value: object) -> None:
        if not self._widget.has_trait(key):
            raise KeyError(key)
        setattr(self._widget, key, value)

    def __delitem__(self, key: str) -> None:
        raise TypeError(
            "Deleting widget traits through .value is not supported."
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._state())

    def __len__(self) -> int:
        return len(self._state())


class _StableMarimoAnyWidget(BaseMarimoAnyWidget):
    """Wrap an anywidget with a stable data URL instead of a temp file."""

    def __init__(self, widget: anywidget.AnyWidget) -> None:
        self.widget = widget
        self._initialized = False

        js = getattr(widget, "_esm", "")
        js_filename = "native_viewer_widget.js"
        if isinstance(js, Path):
            js_filename = js.name
            js = js.read_text(encoding="utf-8")
        if not isinstance(js, str):
            raise TypeError(
                "_StableMarimoAnyWidget expects widget._esm to be a string or Path."
            )

        js_hash = hash_code(js)
        _ = widget.comm
        model_id = get_anywidget_model_id(widget)
        js_url = (
            VirtualFile(
                filename=js_filename,
                buffer=js.encode("utf-8"),
                as_data_url=True,
            ).url
            if js
            else ""
        )

        UIElement.__init__(
            self,
            component_name="marimo-anywidget",
            initial_value=ModelIdRef(model_id=model_id),
            label=None,
            args={
                "js-url": js_url,
                "js-hash": js_hash,
                "model-id": model_id,
            },
            on_change=None,
        )

    def _initialize(
        self,
        initialization_args: Any,
    ) -> None:
        super()._initialize(initialization_args)
        comm = self.widget.comm
        if isinstance(comm, MarimoComm):
            comm.ui_element_id = self._id

    def _convert_value(
        self, value: ModelIdRef | AnyWidgetState
    ) -> AnyWidgetState:
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value)}")

        model_id = value.get("model_id")
        if model_id and len(value) == 1:
            return {}

        self.widget.set_state(value)
        return value

    @property
    def value(self) -> AnyWidgetState:
        """Return the synced anywidget state."""
        return get_anywidget_state(self.widget)

    @value.setter
    def value(self, value: AnyWidgetState) -> None:
        del value
        raise RuntimeError("Setting the value of a UIElement is not allowed.")

    def __setattr__(self, name: str, value: Any) -> None:
        if self._initialized:
            if hasattr(self.widget, name):
                setattr(self.widget, name, value)
                return
            super().__setattr__(name, value)
            return
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name in ("widget", "_initialized"):
            try:
                return self.__getattribute__(name)
            except AttributeError:
                return None
        return getattr(self.widget, name)

    def __getitem__(self, key: Any) -> Any:
        return self.widget[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.widget


class _LatestOnlyRenderer:
    """Background latest-only renderer for camera-driven frames."""

    def __init__(
        self,
        render_fn: Callable[[CameraState], np.ndarray | torch.Tensor],
        publish_frame: Callable[
            [int, CameraState, np.ndarray, float, bool], None
        ],
        publish_error: Callable[[int, str], None],
        set_rendering: Callable[[bool], None],
    ) -> None:
        self._render_fn = render_fn
        self._publish_frame = publish_frame
        self._publish_error = publish_error
        self._set_rendering = set_rendering
        self._condition = threading.Condition()
        self._latest_revision = -1
        self._pending_revision = -1
        self._pending_state: CameraState | None = None
        self._pending_interaction_active = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def request(
        self, revision: int, camera_state: CameraState, interaction_active: bool
    ) -> None:
        """Request a render for the most recent camera state."""
        with self._condition:
            self._latest_revision = revision
            self._pending_revision = revision
            self._pending_state = camera_state
            self._pending_interaction_active = interaction_active
            self._set_rendering(True)
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending_state is None:
                    self._condition.wait()
                revision = self._pending_revision
                camera_state = self._pending_state
                interaction_active = self._pending_interaction_active
                self._pending_state = None

            assert camera_state is not None
            try:
                render_started_at = time.perf_counter()
                rendered_frame = self._render_fn(camera_state)
                render_time_ms = (
                    time.perf_counter() - render_started_at
                ) * 1000.0
                frame = _normalize_frame(rendered_frame)
            except Exception as exception:
                message = "".join(
                    traceback.format_exception(
                        exception.__class__,
                        exception,
                        exception.__traceback__,
                    )
                ).rstrip()
                with self._condition:
                    is_latest = revision == self._latest_revision
                if is_latest:
                    self._publish_error(revision, message)
                    self._set_rendering(False)
                continue

            with self._condition:
                is_latest = revision == self._latest_revision

            if not is_latest:
                continue

            try:
                self._publish_frame(
                    revision,
                    camera_state,
                    frame,
                    render_time_ms,
                    interaction_active,
                )
            except Exception as exception:
                message = "".join(
                    traceback.format_exception(
                        exception.__class__,
                        exception,
                        exception.__traceback__,
                    )
                ).rstrip()
                self._publish_error(revision, message)
            finally:
                self._set_rendering(False)


class _NativeViewerAnyWidget(anywidget.AnyWidget):
    """Internal anywidget for the native camera-controlled viewer."""

    _css = _ASSET_DIR / "native_viewer.css"

    camera_state_json = traitlets.Unicode("").tag(sync=True)
    aspect_ratio = traitlets.Float(16.3 / 9.0).tag(sync=True)
    _camera_revision = traitlets.Int(0).tag(sync=True)
    interaction_active = traitlets.Bool(False).tag(sync=True)
    last_click_json = traitlets.Unicode("").tag(sync=True)
    is_rendering = traitlets.Bool(False).tag(sync=True)
    error_text = traitlets.Unicode("").tag(sync=True)
    controls_hint = traitlets.Unicode(
        "Orbit: drag | Pan: right-drag | Move: WASDQE | Zoom: wheel"
    ).tag(sync=True)

    _esm = _ASSET_DIR / "native_viewer.js"

    def __init__(
        self,
        *,
        camera_state: CameraState,
        aspect_ratio: float,
    ) -> None:
        super().__init__(
            camera_state_json=camera_state.to_json(),
            aspect_ratio=aspect_ratio,
        )


class NativeViewerWidget(_StableMarimoAnyWidget):
    """Marimo-reactive native viewer widget."""

    def __init__(
        self,
        anywidget_instance: _NativeViewerAnyWidget,
        render_fn: Callable[[CameraState], np.ndarray | torch.Tensor],
        interactive_quality: int,
    ) -> None:
        super().__init__(anywidget_instance)
        self._latest_frame_array: np.ndarray | None = None
        self._interactive_quality = interactive_quality
        try:
            self._main_loop: asyncio.AbstractEventLoop | None = (
                asyncio.get_running_loop()
            )
        except RuntimeError:
            self._main_loop = None
        self._renderer = _LatestOnlyRenderer(
            render_fn=render_fn,
            publish_frame=self._publish_frame,
            publish_error=self._publish_error,
            set_rendering=self._set_rendering,
        )
        self.widget.observe(
            self._on_camera_revision_change, names=["_camera_revision"]
        )
        self.rerender()

    def anywidget(self) -> _NativeViewerAnyWidget:
        """Return the underlying raw anywidget instance."""
        return self.widget

    @property
    def value(self) -> MutableMapping[str, object]:
        """Return a live mapping of synced widget traits."""
        return _WidgetValueProxy(self.widget)

    @property
    def camera_state(self) -> CameraState:
        """Return the current synced camera state."""
        return CameraState.from_json(self.widget.camera_state_json)

    @property
    def last_click(self) -> ViewerClick | None:
        """Return the last primary-button click, if any."""
        value = self.widget.last_click_json
        if not value:
            return None
        return ViewerClick.from_json(value)

    def get_camera_state(self) -> CameraState:
        """Return the current synced camera state."""
        return self.camera_state

    def get_last_click(self) -> ViewerClick | None:
        """Return the last primary-button click, if any."""
        return self.last_click

    def get_snapshot(self) -> Image.Image:
        """Return the latest rendered frame as a PIL image."""
        if self._latest_frame_array is None:
            raise RuntimeError("No rendered frame is available yet.")
        return Image.fromarray(self._latest_frame_array.copy(), mode="RGB")

    def _run_on_main_loop(self, callback: Callable[[], None]) -> None:
        """Run a callback on the main asyncio loop when available."""
        if self._main_loop is not None and self._main_loop.is_running():
            self._main_loop.call_soon_threadsafe(callback)
            return
        callback()

    def set_camera_state(self, camera_state: CameraState) -> None:
        """Apply a camera state and request a fresh render."""
        self.widget.camera_state_json = camera_state.to_json()
        self.widget.error_text = ""
        self.widget._camera_revision += 1

    def rerender(self) -> None:
        """Request a fresh render without changing the camera pose."""
        self.widget.error_text = ""
        self.widget._camera_revision += 1

    def _on_camera_revision_change(self, change: dict[str, object]) -> None:
        del change
        self._renderer.request(
            self.widget._camera_revision,
            CameraState.from_json(self.widget.camera_state_json),
            self.widget.interaction_active,
        )

    def _publish_frame(
        self,
        revision: int,
        camera_state: CameraState,
        frame: np.ndarray,
        render_time_ms: float,
        interaction_active: bool,
    ) -> None:
        jpeg_quality = self._interactive_quality if interaction_active else 95
        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not success:
            raise RuntimeError("Failed to encode rendered frame as JPEG.")
        encoded_bytes = encoded.tobytes()
        frame_width = int(frame.shape[1])
        frame_height = int(frame.shape[0])
        next_camera_state_json = None
        if (
            camera_state.width != frame_width
            or camera_state.height != frame_height
        ):
            next_camera_state_json = camera_state.with_size(
                frame_width, frame_height
            ).to_json()

        def _apply_frame_update() -> None:
            self._latest_frame_array = frame.copy()
            self.widget.error_text = ""
            self.widget.send(
                {
                    "type": "frame",
                    "mime_type": "image/jpeg",
                    "width": frame_width,
                    "height": frame_height,
                    "revision": revision,
                    "render_time_ms": render_time_ms,
                    "interaction_active": interaction_active,
                },
                buffers=[encoded_bytes],
            )
            self.widget.send_state("error_text")
            if next_camera_state_json is not None:
                self.widget.camera_state_json = next_camera_state_json
                self.widget.send_state("camera_state_json")

        self._run_on_main_loop(_apply_frame_update)

    def _publish_error(self, revision: int, message: str) -> None:
        del revision

        def _apply_error_update() -> None:
            self.widget.error_text = message
            self.widget.send_state("error_text")

        self._run_on_main_loop(_apply_error_update)

    def _set_rendering(self, value: bool) -> None:
        def _apply_rendering_update() -> None:
            self.widget.is_rendering = value
            self.widget.send_state("is_rendering")

        self._run_on_main_loop(_apply_rendering_update)


def native_viewer(
    render_fn: Callable[[CameraState], np.ndarray | torch.Tensor],
    *,
    fov_degrees: float = 60.0,
    aspect_ratio: float = 16.3 / 9.0,
    interactive_quality: int = 50,
    initial_view: CameraState | None = None,
) -> NativeViewerWidget:
    """Create a native image-based 3D viewer for marimo notebooks.

    The field of view is expressed in degrees. The render size comes from the
    measured notebook layout. `aspect_ratio` controls the initial widget height
    before the first render and resize measurement. `initial_view` sets the
    initial camera pose, convention, and nominal viewport size before the
    widget measures the live layout.

    The returned widget exposes:

    - `camera_state` / `get_camera_state()` for the current synced view
    - `last_click` / `get_last_click()` for the last primary-button click
    - `get_snapshot()` for the latest rendered frame as a PIL image
    - `.value[...]` for direct access to synced anywidget traits

    Clicks are only registered for primary-button press/release interactions
    that stay below the drag threshold; orbiting and panning do not register as
    clicks.
    """
    if aspect_ratio <= 0.0:
        raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio}.")
    if not 1 <= interactive_quality <= 100:
        raise ValueError(
            "interactive_quality must be in [1, 100], "
            f"got {interactive_quality}."
        )
    resolved_camera_state = initial_view or CameraState.default(
        fov_degrees=fov_degrees
    )
    anywidget_instance = _NativeViewerAnyWidget(
        camera_state=resolved_camera_state,
        aspect_ratio=aspect_ratio,
    )
    return NativeViewerWidget(
        anywidget_instance,
        render_fn=render_fn,
        interactive_quality=interactive_quality,
    )
