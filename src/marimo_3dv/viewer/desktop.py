"""Desktop offline viewer using pyglet for low-overhead OpenGL display."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyglet
import pyglet.text
import pyglet.window

from marimo_3dv.viewer.widget import (
    CameraState,
    NativeViewerState,
    ViewerClick,
    _normalize_frame,
)

_ORBIT_SENSITIVITY = 0.005
_PAN_SENSITIVITY = 0.002
_MOVE_SPEED = 0.05
_SCROLL_ZOOM_SENSITIVITY = 0.1
_MIN_FOV = 5.0
_MAX_FOV = 170.0


@dataclass
class _InputState:
    """Mutable per-frame input state."""

    orbit_dragging: bool = False
    pan_dragging: bool = False
    keys_held: set = field(default_factory=set)


class DesktopViewer:
    """Blocking desktop viewer window backed by pyglet."""

    def __init__(
        self,
        render_fn: Callable[[CameraState], Any],
        *,
        state: NativeViewerState | None = None,
        width: int = 1280,
        height: int = 720,
        title: str = "marimo-3dv desktop viewer",
        target_fps: float = 60.0,
    ) -> None:
        self._render_fn = render_fn
        self._state = state or NativeViewerState()
        self._target_fps = target_fps

        camera = self._state.camera_state
        if camera.width != width or camera.height != height:
            self._state.camera_state = camera.with_size(width, height)

        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._input = _InputState()

        # Track render timing for stats.
        self._last_render_ms: float = 0.0
        self._last_fps: float = 0.0
        self._frame_times: list[float] = []

        self._state._reset_camera_callback = self._on_camera_set

        self._window = pyglet.window.Window(
            width=width, height=height, caption=title, resizable=True
        )
        self._stats_label = pyglet.text.Label(
            "",
            font_name="monospace",
            font_size=11,
            x=8,
            y=8,
            color=(255, 255, 80, 220),
            multiline=True,
            width=400,
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
        win_w, win_h = self._window.get_size()
        sprite = pyglet.sprite.Sprite(texture, x=0, y=0)
        sprite.scale_x = win_w / sprite.width
        sprite.scale_y = win_h / sprite.height
        sprite.draw()

    # ------------------------------------------------------------------
    # Camera math
    # ------------------------------------------------------------------

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

    def _set_camera_pose(
        self, position: np.ndarray, forward: np.ndarray
    ) -> None:
        """Rebuild cam_to_world from position and forward, keeping world up.

        Uses -Z as world up to match the opencv convention where camera Y
        points down and scenes are normalized with +Z up.
        """
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = position

        cam = self._state.camera_state
        self._state.camera_state = CameraState(
            fov_degrees=cam.fov_degrees,
            width=cam.width,
            height=cam.height,
            cam_to_world=c2w,
            camera_convention=cam.camera_convention,
        )

    def _apply_orbit(self, dx: int, dy: int) -> None:
        """Orbit around the look-at point (pivot at fixed radius from camera)."""
        position, right, _up, forward = self._camera_axes()

        # Pivot is at some distance in front of the camera.
        pivot_distance = float(np.linalg.norm(position))
        if pivot_distance < 0.1:
            pivot_distance = 1.0
        pivot = position + forward * pivot_distance

        # Yaw around world Z (scene up in opencv/+Z-up convention).
        yaw = -dx * _ORBIT_SENSITIVITY
        rot_y = _rot_axis(np.array([0.0, 0.0, 1.0]), yaw)

        # Pitch around camera right.
        pitch = dy * _ORBIT_SENSITIVITY
        rot_p = _rot_axis(right, pitch)

        rot = rot_p @ rot_y
        new_pos = pivot + rot @ (position - pivot)
        new_forward = pivot - new_pos
        norm = np.linalg.norm(new_forward)
        if norm < 1e-6:
            return
        new_forward /= norm
        self._set_camera_pose(new_pos, new_forward)

    def _apply_pan(self, dx: int, dy: int) -> None:
        """Pan the camera (translate in the image plane)."""
        position, right, up, forward = self._camera_axes()
        cam = self._state.camera_state
        speed = _PAN_SENSITIVITY * cam.fov_degrees
        delta = -dx * speed * right + dy * speed * up
        self._set_camera_pose(position + delta, forward)

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
            self._set_camera_pose(position + delta, forward)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        @self._window.event
        def on_draw() -> None:
            self._window.clear()
            with self._frame_lock:
                frame = self._latest_frame
            if frame is not None:
                self._draw_frame(frame)
            # Stats overlay.
            cam = self._state.camera_state
            self._stats_label.text = (
                f"render: {self._last_render_ms:.1f} ms  "
                f"fps: {self._last_fps:.1f}\n"
                f"fov: {cam.fov_degrees:.1f}°  "
                f"{cam.width}x{cam.height}\n"
                f"pos: {cam.cam_to_world[:3, 3]}"
            )
            self._stats_label.draw()

        @self._window.event
        def on_resize(width: int, height: int) -> None:
            camera = self._state.camera_state
            self._state.camera_state = camera.with_size(width, height)

        @self._window.event
        def on_mouse_press(x: int, y: int, button: int, modifiers: int) -> None:
            if button == pyglet.window.mouse.LEFT:
                self._input.orbit_dragging = True
            elif button == pyglet.window.mouse.MIDDLE:
                self._input.pan_dragging = True
            elif button == pyglet.window.mouse.RIGHT:
                win_w, win_h = self._window.get_size()
                click_y = (
                    win_h - 1 - y
                )  # convert pyglet bottom-left to top-left
                self._state.last_click = ViewerClick(
                    x=x,
                    y=click_y,
                    width=win_w,
                    height=win_h,
                    camera_state=self._state.camera_state,
                )

        @self._window.event
        def on_mouse_release(
            x: int, y: int, button: int, modifiers: int
        ) -> None:
            if button == pyglet.window.mouse.LEFT:
                self._input.orbit_dragging = False
            elif button == pyglet.window.mouse.MIDDLE:
                self._input.pan_dragging = False

        @self._window.event
        def on_mouse_drag(
            x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
        ) -> None:
            if self._input.orbit_dragging:
                self._apply_orbit(dx, dy)
            elif self._input.pan_dragging:
                self._apply_pan(dx, dy)

        @self._window.event
        def on_mouse_scroll(
            x: int, y: int, scroll_x: float, scroll_y: float
        ) -> None:
            # Dolly: move camera forward/back along its view axis.
            position, _right, _up, forward = self._camera_axes()
            speed = (
                scroll_y
                * _SCROLL_ZOOM_SENSITIVITY
                * float(np.linalg.norm(position))
            )
            self._set_camera_pose(position + forward * speed, forward)

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

    # ------------------------------------------------------------------
    # Render + tick loops
    # ------------------------------------------------------------------

    def _render_loop(self) -> None:
        """Background thread: render frames as fast as possible."""
        while self._running:
            start = time.perf_counter()
            camera_state = self._state.camera_state
            raw = self._render_fn(camera_state)
            frame = _normalize_frame(raw)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            with self._frame_lock:
                self._latest_frame = frame
                self._last_render_ms = elapsed_ms
            self._window.invalid = True

    def _tick(self, dt: float) -> None:
        """Main-thread tick: apply held-key movement and update FPS."""
        self._apply_move(dt)
        now = time.perf_counter()
        self._frame_times.append(now)
        cutoff = now - 1.0
        self._frame_times = [t for t in self._frame_times if t > cutoff]
        self._last_fps = float(len(self._frame_times))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the render loop and show the window. Blocks until closed."""
        self._running = True
        render_thread = threading.Thread(target=self._render_loop, daemon=True)
        render_thread.start()
        pyglet.clock.schedule_interval(self._tick, 1.0 / self._target_fps)
        pyglet.app.run()
        self._running = False
        render_thread.join(timeout=2.0)


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


def desktop_viewer(
    render_fn: Callable[[CameraState], Any],
    *,
    state: NativeViewerState | None = None,
    width: int = 1280,
    height: int = 720,
    title: str = "marimo-3dv desktop viewer",
    target_fps: float = 60.0,
) -> DesktopViewer:
    """Create and return a ``DesktopViewer`` instance."""
    return DesktopViewer(
        render_fn,
        state=state,
        width=width,
        height=height,
        title=title,
        target_fps=target_fps,
    )
