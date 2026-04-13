"""Desktop offline viewer using pyglet for low-overhead OpenGL display."""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyglet
import pyglet.shapes
import pyglet.text
import pyglet.window

from marimo_3dv.viewer.widget import (
    CameraState,
    ViewerClick,
    ViewerState,
    _look_at_cam_to_world,
    _normalize,
    _normalize_frame,
)

_ORBIT_SENSITIVITY = 0.008
_MOVE_SPEED = 0.05
_SCROLL_ZOOM_SENSITIVITY = 0.0015
_MIN_FOV = 5.0
_MAX_FOV = 170.0
_MIN_ORBIT_DISTANCE = 0.05
_MAX_ORBIT_DISTANCE = 1e5
_CLICK_THRESHOLD_PIXELS = 4.0


@dataclass
class _InputState:
    """Mutable per-frame input state."""

    mode: str | None = None
    keys_held: set = field(default_factory=set)
    drag_start: tuple[int, int] | None = None
    drag_exceeded_click_threshold: bool = False


class DesktopViewer:
    """Blocking desktop viewer window backed by pyglet."""

    def __init__(
        self,
        render_fn: Callable[[CameraState], Any],
        *,
        state: ViewerState | None = None,
        width: int = 1280,
        height: int = 720,
        title: str = "marimo-3dv desktop viewer",
    ) -> None:
        self._render_fn = render_fn
        self._state = state or ViewerState()
        self._logical_window_size = (width, height)

        camera = self._state.camera_state
        if camera.width != width or camera.height != height:
            self._state.camera_state = camera.with_size(width, height)
        self._sync_camera_tracking(self._state.camera_state)

        self._latest_frame: np.ndarray | None = None
        self._render_error: Exception | None = None
        self._render_error_traceback: str | None = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._input = _InputState()

        # Track render timing for stats.
        self._last_render_ms: float = 0.0
        self._last_viewer_fps: float = 0.0
        self._draw_frame_times: list[float] = []
        self._last_render_fps: float = 0.0
        self._render_frame_times: list[float] = []

        self._state._reset_camera_callback = self._on_camera_set

        self._window = pyglet.window.Window(
            width=width, height=height, caption=title, resizable=True
        )
        self._stats_shadow = pyglet.shapes.Rectangle(
            x=18,
            y=height - 134,
            width=276,
            height=126,
            color=(15, 23, 42),
        )
        self._stats_shadow.opacity = 26
        self._stats_background = pyglet.shapes.Rectangle(
            x=16,
            y=height - 136,
            width=276,
            height=126,
            color=(248, 251, 255),
        )
        self._stats_background.opacity = 236
        self._stats_border = pyglet.shapes.Rectangle(
            x=16,
            y=height - 136,
            width=276,
            height=1,
            color=(222, 231, 240),
        )
        self._stats_title = pyglet.text.Label(
            "Stats",
            font_name="monospace",
            font_size=12,
            x=28,
            y=height - 24,
            color=(71, 85, 105, 255),
            anchor_x="left",
            anchor_y="top",
        )
        self._stats_label = pyglet.text.Label(
            "",
            font_name="monospace",
            font_size=14,
            x=28,
            y=height - 42,
            color=(15, 23, 42, 255),
            multiline=True,
            width=238,
            anchor_x="left",
            anchor_y="top",
        )
        self._register_handlers()

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def _draw_frame(self, frame: np.ndarray) -> None:
        """Upload frame as ImageData and blit scaled to window."""
        height, width = frame.shape[:2]
        flipped = np.ascontiguousarray(frame[::-1])
        image_data = pyglet.image.ImageData(
            width, height, "RGB", flipped.tobytes()
        )
        texture = image_data.get_texture()
        win_w, win_h = self._get_framebuffer_size()
        sprite = pyglet.sprite.Sprite(texture, x=0, y=0)
        sprite.scale_x = win_w / sprite.width
        sprite.scale_y = win_h / sprite.height
        sprite.draw()

    # ------------------------------------------------------------------
    # Camera math
    # ------------------------------------------------------------------

    def _get_window_size(self) -> tuple[int, int]:
        """Return the tracked live window size used for interaction coordinates."""
        return self._logical_window_size

    def _get_framebuffer_size(self) -> tuple[int, int]:
        """Return the drawable framebuffer size, falling back to window size."""
        get_framebuffer_size = getattr(
            self._window, "get_framebuffer_size", None
        )
        if get_framebuffer_size is None:
            return self._window.get_size()
        return get_framebuffer_size()

    def _sync_camera_size_from_framebuffer(self) -> None:
        """Match render resolution to the drawable framebuffer size."""
        framebuffer_width, framebuffer_height = self._get_framebuffer_size()
        target_width = framebuffer_width
        target_height = framebuffer_height
        max_side = self._state.internal_render_max_side
        if max_side is not None:
            larger_axis = max(framebuffer_width, framebuffer_height)
            if larger_axis > max_side:
                downscale = max_side / larger_axis
                target_width = max(1, round(framebuffer_width * downscale))
                target_height = max(1, round(framebuffer_height * downscale))
        camera = self._state.camera_state
        if camera.width != target_width or camera.height != target_height:
            self._state.camera_state = camera.with_size(
                target_width,
                target_height,
            )

    def _camera_axes(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (position, right, up, forward) from current camera."""
        c2w = self._state.camera_state.cam_to_world
        position = c2w[:3, 3].copy()
        right = c2w[:3, 0].copy()
        up = c2w[:3, 1].copy()
        forward = c2w[:3, 2].copy()
        return position, right, up, forward

    def _viewer_frame_rotation(self) -> np.ndarray:
        """Return the viewer-frame rotation derived from state rotation sliders."""
        return _rotation_matrix_xyz(
            self._state.viewer_rotation_x_degrees,
            self._state.viewer_rotation_y_degrees,
            self._state.viewer_rotation_z_degrees,
        )

    def _viewer_up_vector(self) -> np.ndarray:
        """Return the desktop viewer up vector matching the JS controller."""
        return _normalize(
            self._viewer_frame_rotation() @ np.array([0.0, -1.0, 0.0])
        )

    def _sync_camera_tracking(self, camera_state: CameraState) -> None:
        """Update tracked position/target/orbit distance from a camera state."""
        self._position = camera_state.cam_to_world[:3, 3].copy()
        forward = camera_state.cam_to_world[:3, 2].copy()
        fallback_distance = max(
            _MIN_ORBIT_DISTANCE, float(np.linalg.norm(self._position))
        )
        if not hasattr(self, "_orbit_distance"):
            self._orbit_distance = max(3.0, fallback_distance)
        self._target = self._position + forward * self._orbit_distance
        self._orbit_distance = max(
            _MIN_ORBIT_DISTANCE,
            min(
                _MAX_ORBIT_DISTANCE,
                float(np.linalg.norm(self._target - self._position)),
            ),
        )

    def _set_camera_pose(
        self, position: np.ndarray, target: np.ndarray
    ) -> None:
        """Rebuild cam_to_world from tracked position/target and viewer up."""
        cam_to_world = _look_at_cam_to_world(
            position,
            target,
            self._viewer_up_vector(),
        )
        cam = self._state.camera_state
        next_camera = CameraState(
            fov_degrees=cam.fov_degrees,
            width=cam.width,
            height=cam.height,
            cam_to_world=cam_to_world,
            camera_convention=cam.camera_convention,
        )
        self._position = position.copy()
        self._target = target.copy()
        self._orbit_distance = max(
            _MIN_ORBIT_DISTANCE,
            min(
                _MAX_ORBIT_DISTANCE,
                float(np.linalg.norm(self._target - self._position)),
            ),
        )
        self._state.camera_state = next_camera

    def _apply_orbit(self, dx: int, dy: int) -> None:
        """Orbit around the explicit tracked target using JS-equivalent math."""
        viewer_up = self._viewer_up_vector()
        offset = self._position - self._target
        radius = max(_MIN_ORBIT_DISTANCE, float(np.linalg.norm(offset)))
        yaw_rotation = _rot_axis(viewer_up, -dx * _ORBIT_SENSITIVITY)
        yawed_offset = yaw_rotation @ offset
        yawed_forward = _normalize(-yawed_offset)
        pitch_axis = np.cross(yawed_forward, viewer_up)
        if np.linalg.norm(pitch_axis) <= 1e-8:
            return
        pitch_axis = _normalize(pitch_axis)
        pitch_rotation = _rot_axis(
            pitch_axis,
            -dy * _ORBIT_SENSITIVITY,
        )
        orbited_offset = pitch_rotation @ yawed_offset
        new_position = self._target + orbited_offset
        self._set_camera_pose(new_position, self._target)

    def _apply_pan(self, dx: int, dy: int) -> None:
        """Pan in the image plane using the JS orbit-distance/FOV scaling."""
        _position, right, up, _forward = self._camera_axes()
        cam = self._state.camera_state
        _window_width, window_height = self._get_window_size()
        fov_radians = np.deg2rad(cam.fov_degrees)
        pan_scale = (
            max(_MIN_ORBIT_DISTANCE, self._orbit_distance)
            * np.tan(fov_radians / 2.0)
            / max(1, window_height)
            * 2.0
        )
        delta = -dx * pan_scale * right + dy * pan_scale * up
        self._set_camera_pose(self._position + delta, self._target + delta)

    def _apply_dolly(self, scroll_y: float) -> None:
        """Zoom by scaling the orbit distance exponentially around the target."""
        zoom_factor = float(np.exp(-scroll_y * _SCROLL_ZOOM_SENSITIVITY))
        offset = self._position - self._target
        direction = _normalize(offset)
        self._orbit_distance = max(
            _MIN_ORBIT_DISTANCE,
            min(_MAX_ORBIT_DISTANCE, self._orbit_distance * zoom_factor),
        )
        self._set_camera_pose(
            self._target + direction * self._orbit_distance,
            self._target,
        )

    def _apply_move(self, dt: float) -> None:
        """Apply WASD/QE keyboard movement each tick."""
        keys = self._input.keys_held
        if not keys:
            return
        position, right, up, forward = self._camera_axes()
        speed = _MOVE_SPEED * dt * 60.0
        delta = np.zeros(3)
        if pyglet.window.key.W in keys:
            delta += forward * speed
        if pyglet.window.key.S in keys:
            delta -= forward * speed
        if pyglet.window.key.A in keys:
            delta -= right * speed
        if pyglet.window.key.D in keys:
            delta += right * speed
        if pyglet.window.key.Q in keys:
            delta -= up * speed
        if pyglet.window.key.E in keys:
            delta += up * speed
        if np.linalg.norm(delta) > 0:
            self._set_camera_pose(position + delta, self._target + delta)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        @self._window.event
        def on_draw() -> None:
            now = time.perf_counter()
            self._draw_frame_times.append(now)
            cutoff = now - 1.0
            self._draw_frame_times = [
                timestamp
                for timestamp in self._draw_frame_times
                if timestamp > cutoff
            ]
            self._last_viewer_fps = float(len(self._draw_frame_times))

            self._window.clear()
            with self._frame_lock:
                frame = self._latest_frame
            if frame is not None:
                self._draw_frame(frame)
            # Stats overlay.
            cam = self._state.camera_state
            if self._state.show_stats:
                logical_width, logical_height = self._get_window_size()
                render_width, render_height = cam.width, cam.height
                self._stats_shadow.draw()
                self._stats_background.draw()
                self._stats_border.draw()
                self._stats_title.draw()
                self._stats_label.text = (
                    f"Viewer {self._last_viewer_fps:.0f}fps\n"
                    f"Render {self._last_render_ms:.0f}ms {self._last_render_fps:.0f}fps\n"
                    f"Window {logical_width}x{logical_height}\n"
                    f"Render {render_width}x{render_height}"
                )
                self._stats_label.draw()

        @self._window.event
        def on_resize(width: int, height: int) -> None:
            self._logical_window_size = (max(1, width), max(1, height))
            self._sync_camera_size_from_framebuffer()
            self._stats_shadow.y = height - 134
            self._stats_background.y = height - 136
            self._stats_border.y = height - 136
            self._stats_title.y = height - 24
            self._stats_label.y = height - 42

        @self._window.event
        def on_mouse_press(x: int, y: int, button: int, modifiers: int) -> None:
            if button == pyglet.window.mouse.LEFT:
                self._input.mode = "orbit"
                self._input.drag_start = (x, y)
                self._input.drag_exceeded_click_threshold = False
            elif button == pyglet.window.mouse.RIGHT:
                self._input.mode = "pan"
                self._input.drag_start = (x, y)
                self._input.drag_exceeded_click_threshold = False

        @self._window.event
        def on_mouse_release(
            x: int, y: int, button: int, modifiers: int
        ) -> None:
            if button == pyglet.window.mouse.LEFT:
                should_emit_click = (
                    self._input.drag_start is not None
                    and not self._input.drag_exceeded_click_threshold
                )
                if should_emit_click:
                    win_w, win_h = self._get_window_size()
                    framebuffer_w, framebuffer_h = self._get_framebuffer_size()
                    scale_x = framebuffer_w / max(1, win_w)
                    scale_y = framebuffer_h / max(1, win_h)
                    self._state.last_click = ViewerClick(
                        x=round(x * scale_x),
                        y=round((win_h - 1 - y) * scale_y),
                        width=framebuffer_w,
                        height=framebuffer_h,
                        camera_state=self._state.camera_state,
                    )
            if button in {
                pyglet.window.mouse.LEFT,
                pyglet.window.mouse.RIGHT,
            }:
                self._input.mode = None
                self._input.drag_start = None
                self._input.drag_exceeded_click_threshold = False

        @self._window.event
        def on_mouse_drag(
            x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
        ) -> None:
            if self._input.drag_start is not None:
                drag_distance = float(
                    np.hypot(
                        x - self._input.drag_start[0],
                        y - self._input.drag_start[1],
                    )
                )
                if drag_distance > _CLICK_THRESHOLD_PIXELS:
                    self._input.drag_exceeded_click_threshold = True
            if self._input.mode == "orbit":
                self._apply_orbit(dx, -dy)
            elif self._input.mode == "pan":
                self._apply_pan(dx, dy)

        @self._window.event
        def on_mouse_scroll(
            x: int, y: int, scroll_x: float, scroll_y: float
        ) -> None:
            self._apply_dolly(scroll_y)

        @self._window.event
        def on_key_press(symbol: int, modifiers: int) -> None:
            self._input.keys_held.add(symbol)
            if symbol == pyglet.window.key.R:
                self._state.reset_camera()
            elif symbol == pyglet.window.key.ESCAPE:
                self._running = False
                self._window.close()

        @self._window.event
        def on_key_release(symbol: int, modifiers: int) -> None:
            self._input.keys_held.discard(symbol)

    # ------------------------------------------------------------------
    # Viewer state callbacks
    # ------------------------------------------------------------------

    def _on_camera_set(self, camera_state: CameraState) -> None:
        self._state.camera_state = camera_state
        self._sync_camera_tracking(camera_state)

    # ------------------------------------------------------------------
    # Render + tick loops
    # ------------------------------------------------------------------

    def _render_once(self, camera_state: CameraState) -> np.ndarray:
        """Render and normalize a single frame, surfacing backend failures."""
        raw = self._render_fn(camera_state)
        return _normalize_frame(raw)

    def _render_loop(self) -> None:
        """Background thread: render frames as fast as possible."""
        while self._running:
            try:
                start = time.perf_counter()
                camera_state = self._state.camera_state
                frame = self._render_once(camera_state)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                now = time.perf_counter()
                with self._frame_lock:
                    self._latest_frame = frame
                    self._last_render_ms = elapsed_ms
                    self._render_frame_times.append(now)
                    cutoff = now - 1.0
                    self._render_frame_times = [
                        timestamp
                        for timestamp in self._render_frame_times
                        if timestamp > cutoff
                    ]
                    self._last_render_fps = float(len(self._render_frame_times))
                self._window.invalid = True
            except Exception as exception:
                self._render_error = exception
                self._render_error_traceback = "".join(
                    traceback.format_exception(
                        type(exception),
                        exception,
                        exception.__traceback__,
                    )
                )
                self._running = False
                pyglet.clock.schedule_once(lambda _dt: pyglet.app.exit(), 0.0)
                return

    def _tick(self, dt: float) -> None:
        """Main-thread tick: apply held-key movement."""
        self._apply_move(dt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the render loop and show the window. Blocks until closed."""
        initial_width, initial_height = self._window.get_size()
        self._logical_window_size = (
            max(1, initial_width),
            max(1, initial_height),
        )
        self._sync_camera_size_from_framebuffer()
        initial_camera_state = self._state.camera_state
        initial_frame = self._render_once(initial_camera_state)
        with self._frame_lock:
            self._latest_frame = initial_frame
            self._last_render_ms = 0.0

        self._running = True
        render_thread = threading.Thread(target=self._render_loop, daemon=True)
        render_thread.start()
        pyglet.clock.schedule(self._tick)
        pyglet.app.run()
        self._running = False
        render_thread.join(timeout=2.0)
        if self._render_error is not None:
            raise RuntimeError(
                "Desktop viewer render loop failed.\n"
                f"{self._render_error_traceback}"
            ) from self._render_error

    def get_camera_state(self) -> CameraState:
        """Return the current desktop camera state."""
        return self._state.camera_state

    def get_last_click(self) -> ViewerClick | None:
        """Return the last click captured by the desktop backend."""
        return self._state.last_click

    def get_snapshot(self) -> np.ndarray:
        """Return the latest rendered desktop frame."""
        if self._latest_frame is None:
            raise RuntimeError("No rendered frame is available yet.")
        with self._frame_lock:
            assert self._latest_frame is not None
            return self._latest_frame.copy()


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for rotating `angle` radians around `axis`."""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _rotation_matrix_xyz(
    x_degrees: float,
    y_degrees: float,
    z_degrees: float,
) -> np.ndarray:
    """Return the XYZ Euler rotation matrix used by the JS viewer."""
    x_radians, y_radians, z_radians = np.radians(
        [x_degrees, y_degrees, z_degrees]
    )
    cx, cy, cz = np.cos([x_radians, y_radians, z_radians])
    sx, sy, sz = np.sin([x_radians, y_radians, z_radians])
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rotation_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rotation_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rotation_z @ rotation_y @ rotation_x


def desktop_viewer(
    render_fn: Callable[[CameraState], Any],
    *,
    state: ViewerState | None = None,
    width: int = 1280,
    height: int = 720,
    title: str = "marimo-3dv desktop viewer",
) -> DesktopViewer:
    """Create and return a ``DesktopViewer`` instance."""
    return DesktopViewer(
        render_fn,
        state=state,
        width=width,
        height=height,
        title=title,
    )
