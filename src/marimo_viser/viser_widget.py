"""Embed a viser ViserServer inline in a marimo notebook via anywidget.

Two modes
---------

1. Viser-only (no render_fn) — push geometry imperatively:

    server, widget = viser_marimo()
    slider = server.gui.add_slider("scale", min=0.1, max=5.0, initial_value=1.0)
    widget  # renders iframe + read/write camera controls

    # Re-runnable cell — scene calls are idempotent:
    server.scene.add_point_cloud("/pts", points=pts * slider.value)

2. Nerfview render-fn mode — nerfview drives a per-client render loop:

    def render_fn(
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> UInt8[np.ndarray, "height width 3"]:
        width  = render_tab_state.viewer_width
        height = render_tab_state.viewer_height
        c2w    = camera_state.c2w               # (3, 4) float64 cam-to-world
        K      = camera_state.get_K([width, height])  # (3, 3) float64
        return my_renderer(c2w, K, width, height)

    server, viewer, widget = viser_marimo(render_fn=render_fn)
    widget

    # Can still add viser scene objects alongside the rendered background:
    server.scene.add_frame("/origin")

Idempotency rules
-----------------
server.scene.*  Keyed by path string -> safe to re-run freely.
server.gui.*    Keyed by UUID -> creates duplicates on re-run.
                Put GUI setup in the same cell as viser_marimo().

Camera controls
---------------
The embedded widget exposes explicit camera controls instead of a snapshot
abstraction:
  1. "Read Camera" captures the current viewer camera into widget state.
  2. "Write Camera" applies the stored widget camera state back to the viewer.

You can also access the same typed state directly from Python with
`widget.get_camera_state()` and `widget.set_camera_state(...)`.

Graduating to a standalone script
----------------------------------
Viser-only:
    Replace:  server, widget = viser_marimo() / widget
    With:     server = viser.ViserServer(port=8080)
              server.sleep(float("inf"))

Nerfview:
    Replace:  server, viewer, widget = viser_marimo(render_fn=fn) / widget
    With:     server = viser.ViserServer(port=8080)
              viewer = nerfview.Viewer(server=server, render_fn=fn, mode="rendering")
              while True: time.sleep(1e-3)
"""

import socket
import json
import textwrap
import threading
import traceback
from collections.abc import Iterator, MutableMapping
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import anywidget
import nerfview
import numpy as np
import traitlets
import viser
from marimo._plugins.ui._impl.from_anywidget import anywidget as MarimoAnyWidget
from jaxtyping import Float, UInt8
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Port helper
# ---------------------------------------------------------------------------


def _find_free_port(start: int = 8080, attempts: int = 64) -> int:
    """Return the first free TCP port in [start, start + attempts).

    Args:
        start: First port to try. Defaults to 8080.
        attempts: How many consecutive ports to scan. Defaults to 64.

    Returns:
        An available port number.

    Raises:
        RuntimeError: If no free port is found in the scanned range.
    """
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"Could not find a free port in range {start}–{start + attempts}."
    )


def _format_render_exception(exception: BaseException) -> str:
    """Return a concise render error summary followed by the full traceback."""
    frames = traceback.extract_tb(exception.__traceback__)
    location = None
    for frame in reversed(frames):
        filename = Path(frame.filename)
        if "site-packages/nerfview" in frame.filename:
            continue
        if filename.name == "viser_widget.py":
            continue
        location = frame
        break

    lines = [f"render_fn failed: {exception.__class__.__name__}: {exception}"]
    if location is not None:
        location_text = (
            f"{location.filename}:{location.lineno} in {location.name}"
        )
        lines.append(f"Location: {location_text}")
        if location.line:
            lines.append(f"Code: {location.line.strip()}")

    lines.extend(
        [
            "",
            "Traceback:",
            "".join(
                traceback.format_exception(
                    exception.__class__,
                    exception,
                    exception.__traceback__,
                )
            ).rstrip(),
        ]
    )
    return "\n".join(lines)


def _render_error_image(
    message: str,
    width: int,
    height: int,
) -> UInt8[np.ndarray, "height width 3"]:
    """Render an exception summary into an RGB fallback image."""
    image = Image.new("RGB", (width, height), color=(40, 6, 12))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=24)
    body_font = ImageFont.load_default(size=18)

    padding = 20
    title_height = 30
    line_height = 22
    draw.rectangle(
        [(0, 0), (width - 1, height - 1)],
        outline=(241, 174, 181),
        width=2,
    )
    draw.text(
        (padding, padding),
        "render_fn failed",
        fill=(255, 220, 220),
        font=title_font,
    )

    max_chars = max(16, (width - 2 * padding) // 10)
    wrapped_lines: list[str] = []
    for raw_line in message.splitlines():
        segments = textwrap.wrap(
            raw_line,
            width=max_chars,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        wrapped_lines.extend(segments or [""])

    y = padding + title_height + 8
    max_lines = max(1, (height - y - padding) // line_height)
    for line in wrapped_lines[:max_lines]:
        draw.text((padding, y), line, fill=(255, 240, 240), font=body_font)
        y += line_height

    if len(wrapped_lines) > max_lines:
        draw.text(
            (padding, height - padding - line_height),
            "...",
            fill=(255, 240, 240),
            font=body_font,
        )

    return np.asarray(image, dtype=np.uint8)


def _markdown_error_block(message: str) -> str:
    """Format a render error for display in viser's markdown GUI."""
    return (
        "### render_fn failed\n\n"
        "```text\n"
        f"{message}\n"
        "```"
    )


class _WidgetValueProxy(MutableMapping[str, object]):
    """Live mapping view over synced anywidget traits."""

    def __init__(self, widget: "_ViserAnyWidget") -> None:
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
        raise TypeError("Deleting widget traits through .value is not supported.")

    def __iter__(self) -> Iterator[str]:
        return iter(self._state())

    def __len__(self) -> int:
        return len(self._state())


# ---------------------------------------------------------------------------
# Camera state type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ViserCameraState:
    """Serializable camera state for reading, writing, and syncing viewers.

    Note:
        This is distinct from nerfview.CameraState, which nerfview constructs
        internally and passes to your render_fn. That type carries .c2w and
        .get_K(); this one carries the raw viser camera fields.
    """

    position: Float[np.ndarray, "3"]
    wxyz: Float[np.ndarray, "4"]
    look_at: Float[np.ndarray, "3"]
    up_direction: Float[np.ndarray, "3"]
    fov: float

    @classmethod
    def from_camera(cls, camera: object) -> "ViserCameraState":
        """Build a typed state snapshot from a viser camera-like object."""
        up_direction = (
            camera.up_direction if hasattr(camera, "up_direction") else camera.up
        )
        return cls(
            position=np.asarray(camera.position, dtype=np.float64).copy(),
            wxyz=np.asarray(camera.wxyz, dtype=np.float64).copy(),
            look_at=np.asarray(camera.look_at, dtype=np.float64).copy(),
            up_direction=np.asarray(up_direction, dtype=np.float64).copy(),
            fov=float(camera.fov),
        )

    def apply_to_camera(self, camera: object) -> None:
        """Write this state onto a viser camera-like object."""
        camera.position = self.position.copy()
        if hasattr(camera, "wxyz"):
            camera.wxyz = self.wxyz.copy()
        camera.look_at = self.look_at.copy()
        if hasattr(camera, "up_direction"):
            camera.up_direction = self.up_direction.copy()
        else:
            camera.up = self.up_direction.copy()
        camera.fov = self.fov

    def to_json(self) -> str:
        """Serialize the camera state into a stable JSON string."""
        return json.dumps(
            {
                "position": self.position.tolist(),
                "wxyz": self.wxyz.tolist(),
                "look_at": self.look_at.tolist(),
                "up_direction": self.up_direction.tolist(),
                "fov": self.fov,
            }
        )

    @classmethod
    def from_json(cls, value: str) -> "ViserCameraState":
        """Deserialize a camera state from JSON."""
        payload = json.loads(value)
        return cls(
            position=np.asarray(payload["position"], dtype=np.float64),
            wxyz=np.asarray(payload["wxyz"], dtype=np.float64),
            look_at=np.asarray(payload["look_at"], dtype=np.float64),
            up_direction=np.asarray(
                payload["up_direction"], dtype=np.float64
            ),
            fov=float(payload["fov"]),
        )


# ---------------------------------------------------------------------------
# anywidget (camera controls + iframe)
# ---------------------------------------------------------------------------


class _ViserAnyWidget(anywidget.AnyWidget):
    """Internal anywidget rendering camera controls above a viser iframe.

    Not intended for direct instantiation — use viser_marimo() instead.

    Attributes:
        port: The port that the viser HTTP/WebSocket server is listening on.
        height: Height of the embedded iframe in pixels.
        _read_camera_counter: Incremented by the JS read button to trigger a
            camera read in Python.
        _write_camera_counter: Incremented by the JS write button to trigger a
            camera write in Python.
    """

    port = traitlets.Int(8080).tag(sync=True)
    height = traitlets.Int(600).tag(sync=True)
    _read_camera_counter = traitlets.Int(0).tag(sync=True)
    _write_camera_counter = traitlets.Int(0).tag(sync=True)
    camera_state_json = traitlets.Unicode("").tag(sync=True)

    _esm = textwrap.dedent("""\
        function render({ model, el }) {

            // --- Camera controls -----------------------------------------
            const controls = document.createElement("div");
            Object.assign(controls.style, {
                display: "flex",
                gap: "8px",
                marginBottom: "6px",
            });

            function makeButton(label, title) {
                const button = document.createElement("button");
                button.textContent = label;
                button.title = title;
                Object.assign(button.style, {
                    padding:      "4px 12px",
                    fontSize:     "12px",
                    cursor:       "pointer",
                    borderRadius: "6px",
                    border:       "1px solid #ccc",
                    background:   "#f5f5f5",
                });
                return button;
            }

            const readButton = makeButton(
                "Save Camera State",
                "Capture the current viewer camera state into the widget",
            );
            readButton.addEventListener("click", () => {
                model.set(
                    "_read_camera_counter",
                    model.get("_read_camera_counter") + 1,
                );
                model.save_changes();
            });

            const writeButton = makeButton(
                "Restore Saved Camera State",
                "Apply the stored camera state back to the viewer",
            );
            writeButton.addEventListener("click", () => {
                model.set(
                    "_write_camera_counter",
                    model.get("_write_camera_counter") + 1,
                );
                model.save_changes();
            });

            controls.appendChild(readButton);
            controls.appendChild(writeButton);
            el.appendChild(controls);

            // --- iframe ---------------------------------------------------
            function makeIframe() {
                const iframe = document.createElement("iframe");
                iframe.src = `http://localhost:${model.get("port")}`;
                iframe.style.cssText = [
                    "width: 100%",
                    `height: ${model.get("height")}px`,
                    "border: none",
                    "border-radius: 8px",
                    "display: block",
                ].join("; ");
                iframe.allow = "fullscreen";
                return iframe;
            }

            let iframe = makeIframe();
            el.appendChild(iframe);

            const onPortOrHeightChange = () => {
                const next = makeIframe();
                el.replaceChild(next, iframe);
                iframe = next;
            };
            model.on("change:port",   onPortOrHeightChange);
            model.on("change:height", onPortOrHeightChange);

            return () => {
                model.off("change:port",   onPortOrHeightChange);
                model.off("change:height", onPortOrHeightChange);
                iframe.src = "about:blank";
            };
        }

        const widget = { render };

        export { render };
        export default widget;
    """)


# ---------------------------------------------------------------------------
# Public wrapper returned by viser_marimo()
# ---------------------------------------------------------------------------


class ViserMarimoWidget(MarimoAnyWidget):
    """Marimo-reactive widget returned by viser_marimo().

    This wraps the underlying anywidget in a marimo UIElement so the returned
    object is directly displayable and reactive in notebook cells while still
    carrying the `viser` server, optional `nerfview` viewer, and camera-state
    helper methods.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        anywidget_instance: _ViserAnyWidget,
        viewer: object | None = None,
    ) -> None:
        super().__init__(anywidget_instance)
        self.server = server
        self.viewer = viewer
        self._applying_camera_state_json = False
        anywidget_instance.observe(
            self._on_read_camera_counter_change,
            names=["_read_camera_counter"],
        )
        anywidget_instance.observe(
            self._on_write_camera_counter_change,
            names=["_write_camera_counter"],
        )
        anywidget_instance.observe(
            self._on_camera_state_json_change,
            names=["camera_state_json"],
        )

    def anywidget(self) -> _ViserAnyWidget:
        """Return the underlying raw anywidget instance."""
        return self.widget

    @property
    def value(self) -> MutableMapping[str, object]:
        """Live mapping of synced widget traits.

        Reads reflect the current widget state, and item assignment writes
        back to the underlying anywidget trait, e.g.
        `widget.value["camera_state_json"] = ...`.
        """
        return _WidgetValueProxy(self.widget)

    def _get_client_handle(self, client_id: int | None = None) -> object | None:
        """Return the requested client handle, or the first connected client."""
        clients = self.server.get_clients()
        if client_id is not None:
            if client_id not in clients:
                raise KeyError(f"No connected viser client with id {client_id}.")
            return clients[client_id]
        if not clients:
            return None
        return next(iter(clients.values()))

    def get_camera_state(
        self,
        client_id: int | None = None,
    ) -> ViserCameraState:
        """Read the current camera state from a client or the saved reset view.

        Args:
            client_id: Optional viser client id. If omitted, the first connected
                client is used. If no client is connected, the saved reset view
                (`server.initial_camera`) is returned.

        Returns:
            A typed camera state suitable for storing or passing to
            `set_camera_state`.
        """
        client = self._get_client_handle(client_id)
        if client is not None:
            return ViserCameraState.from_camera(client.camera)
        return ViserCameraState.from_camera(self.server.initial_camera)

    @property
    def camera_state(self) -> ViserCameraState | None:
        """Return the last widget-synced camera state, if available."""
        value = self.widget.camera_state_json
        if not value:
            return None
        return ViserCameraState.from_json(value)

    def set_camera_state(
        self,
        camera_state: ViserCameraState,
        *,
        client_id: int | None = None,
        update_reset_view: bool = True,
        sync_gui: bool = True,
    ) -> None:
        """Apply a camera state to the live viewer and optionally reset view.

        Args:
            camera_state: Typed camera state returned by `get_camera_state`.
            client_id: Optional viser client id. If omitted, the state is
                applied to all connected clients.
            update_reset_view: Whether to also persist the state into
                `server.initial_camera` so new clients and viser's built-in
                "Reset View" use it.
            sync_gui: Whether to also synchronize matching nerfview GUI state,
                such as the FOV slider, when a viewer is present.
        """
        if update_reset_view:
            camera_state.apply_to_camera(self.server.initial_camera)

        clients = self.server.get_clients()
        if client_id is not None:
            if client_id not in clients:
                raise KeyError(f"No connected viser client with id {client_id}.")
            target_clients = [clients[client_id]]
        else:
            target_clients = list(clients.values())

        fov_degrees = float(np.rad2deg(camera_state.fov))
        with self.server.atomic():
            if sync_gui and self.viewer is not None:
                fov_slider = getattr(self.viewer, "_rendering_tab_handles", {}).get(
                    "fov_degrees_slider"
                )
                if fov_slider is not None:
                    fov_slider.value = fov_degrees
            for client in target_clients:
                camera_state.apply_to_camera(client.camera)

        self._applying_camera_state_json = True
        try:
            self.widget.camera_state_json = camera_state.to_json()
        finally:
            self._applying_camera_state_json = False
        if self.viewer is not None:
            self.viewer.rerender(None)

    # ------------------------------------------------------------------
    # Camera control plumbing
    # ------------------------------------------------------------------

    def _on_read_camera_counter_change(self, change: dict) -> None:
        """Traitlets observer — offloads camera reads to a daemon thread."""
        del change
        threading.Thread(target=self._execute_read_camera, daemon=True).start()

    def _on_write_camera_counter_change(self, change: dict) -> None:
        """Traitlets observer — offloads camera writes to a daemon thread."""
        del change
        threading.Thread(target=self._execute_write_camera, daemon=True).start()

    def _execute_read_camera(self) -> None:
        """Capture the current camera and sync it into the widget state."""
        try:
            self._applying_camera_state_json = True
            try:
                self.widget.camera_state_json = self.get_camera_state().to_json()
            finally:
                self._applying_camera_state_json = False
        except Exception:
            traceback.print_exc()

    def _execute_write_camera(self) -> None:
        """Apply the synced widget camera state back to the viewer."""
        try:
            camera_state = self.camera_state
            if camera_state is None:
                return
            self.set_camera_state(camera_state)
        except Exception:
            traceback.print_exc()

    def _on_camera_state_json_change(self, change: dict) -> None:
        """Apply programmatic camera-state writes back to the live viewer."""
        del change
        if self._applying_camera_state_json:
            return
        threading.Thread(target=self._execute_write_camera, daemon=True).start()


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------


def viser_marimo(
    *,
    render_fn: Callable | None = None,
    port: int | None = None,
    height: int = 600,
    host: str = "0.0.0.0",
    mode: str = "rendering",
) -> (
    tuple[viser.ViserServer, ViserMarimoWidget]
    | tuple[viser.ViserServer, object, ViserMarimoWidget]
):
    """Start a ViserServer and return an embeddable widget for marimo.

    Args:
        render_fn: Optional render function for nerfview mode. If None,
            returns (server, widget) and you drive the scene imperatively.
            If provided, nerfview is imported, a Viewer is created, and
            (server, viewer, widget) is returned. The signature must be:

                def render_fn(
                    camera_state: nerfview.CameraState,
                    render_tab_state: nerfview.RenderTabState,
                ) -> UInt8[np.ndarray, "height width 3"]:
                    ...

            The function is called in a background thread on every camera
            move. It may also return a (image, depth) tuple where depth is
            Float32[np.ndarray, "height width"].

        port: Port to listen on. If None, a free port is auto-selected
            starting from 8080.
        height: Height of the embedded iframe in pixels. Defaults to 600.
        host: Address to bind the viser server on. Defaults to "0.0.0.0".
        mode: Passed to nerfview.Viewer when render_fn is set. Either
            "rendering" (default) or "training".

    Returns:
        In viser-only mode (render_fn is None):
            A (server, widget) tuple.
        In nerfview mode (render_fn provided):
            A (server, viewer, widget) tuple.

    Raises:
        ImportError: If render_fn is provided but nerfview is not installed.
        RuntimeError: If no free port can be found.

    Examples:
        Viser-only::

            server, widget = viser_marimo()
            slider = server.gui.add_slider(
                "scale", min=0.1, max=5.0, initial_value=1.0
            )

            camera_state = widget.get_camera_state()
            widget.set_camera_state(camera_state)

            widget  # last expression in cell — renders iframe + camera controls

            # In a separate re-runnable cell:
            server.scene.add_point_cloud("/pts", points=pts * slider.value)

        Nerfview::

            def render_fn(camera_state, render_tab_state):
                width  = render_tab_state.viewer_width
                height = render_tab_state.viewer_height
                c2w    = camera_state.c2w
                K      = camera_state.get_K([width, height])
                return my_nerf(c2w, K, width, height)

            server, viewer, widget = viser_marimo(render_fn=render_fn)
            widget

            # Can still add scene objects alongside the rendered background:
            server.scene.add_camera_frustum("/cam", fov=0.9, aspect=width/height)
    """
    resolved_port = port if port is not None else _find_free_port()
    server = viser.ViserServer(host=host, port=resolved_port)
    anywidget_instance = _ViserAnyWidget(port=resolved_port, height=height)
    wrapped_widget = ViserMarimoWidget(server, anywidget_instance, viewer=None)

    viewer = None
    if render_fn is not None:
        error_markdown_handle: viser.GuiMarkdownHandle | None = None
        error_text_handle = None

        def _show_gui_error(message: str) -> None:
            nonlocal error_markdown_handle, error_text_handle
            with server.atomic():
                if error_markdown_handle is None:
                    error_markdown_handle = server.gui.add_markdown(
                        _markdown_error_block(message),
                        order=-10.0,
                    )
                else:
                    error_markdown_handle.content = _markdown_error_block(message)

                if error_text_handle is None:
                    error_text_handle = server.gui.add_text(
                        "Render Traceback",
                        message,
                        multiline=True,
                        disabled=True,
                        order=-9.0,
                    )
                else:
                    error_text_handle.value = message

        def _clear_gui_error() -> None:
            nonlocal error_markdown_handle, error_text_handle
            with server.atomic():
                if error_markdown_handle is not None:
                    error_markdown_handle.remove()
                    error_markdown_handle = None
                if error_text_handle is not None:
                    error_text_handle.remove()
                    error_text_handle = None

        def _get_fallback_render(
            render_state: object,
        ) -> UInt8[np.ndarray, "height width 3"]:
            width = 1
            height_px = 1
            if hasattr(render_state, "viewer_width") and hasattr(
                render_state, "viewer_height"
            ):
                width = max(1, int(render_state.viewer_width))
                height_px = max(1, int(render_state.viewer_height))
            elif (
                isinstance(render_state, tuple)
                and len(render_state) == 2
                and all(isinstance(value, int) for value in render_state)
            ):
                width = max(1, int(render_state[0]))
                height_px = max(1, int(render_state[1]))
            return np.zeros((height_px, width, 3), dtype=np.uint8)

        def _report_render_error(
            exception: BaseException,
            render_state: object,
        ) -> object:
            message = _format_render_exception(exception)
            traceback.print_exception(
                exception.__class__,
                exception,
                exception.__traceback__,
            )
            _show_gui_error(message)
            fallback = _get_fallback_render(render_state)
            return _render_error_image(
                message,
                width=fallback.shape[1],
                height=fallback.shape[0],
            )

        def _safe_render_fn(
            camera_state: object, render_state: object
        ) -> object:
            try:
                rendered = render_fn(camera_state, render_state)
            except TypeError as first_error:
                # Support older nerfview render_fns that still expect (camera_state, img_wh).
                if hasattr(render_state, "viewer_width") and hasattr(
                    render_state, "viewer_height"
                ):
                    img_wh = (
                        int(render_state.viewer_width),
                        int(render_state.viewer_height),
                    )
                    try:
                        rendered = render_fn(camera_state, img_wh)
                    except Exception as second_error:
                        return _report_render_error(second_error, render_state)
                else:
                    return _report_render_error(first_error, render_state)
            except Exception as error:
                return _report_render_error(error, render_state)

            _clear_gui_error()
            return rendered

        viewer = nerfview.Viewer(
            server=server,
            render_fn=_safe_render_fn,
            mode=mode,
        )
        wrapped_widget.viewer = viewer

    if viewer is None:
        return server, wrapped_widget
    return server, viewer, wrapped_widget
