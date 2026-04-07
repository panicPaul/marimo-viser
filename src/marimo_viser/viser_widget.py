"""Embed a viser ViserServer inline in a marimo notebook via anywidget.

Two modes
---------

1. Viser-only (no render_fn) — push geometry imperatively:

    server, widget = viser_marimo()
    slider = server.gui.add_slider("scale", min=0.1, max=5.0, initial_value=1.0)
    widget  # renders iframe + snapshot button

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

The snapshot button
-------------------
Pressing the snapshot button:
  1. Saves the current camera into server.initial_camera, persisting the
     viewpoint for new clients and viser's built-in "Reset View" button.
  2. Fires any on_snapshot callbacks with a ViserCameraState named-tuple,
     which you can use to cleanly rebuild duplicate GUI handles.

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
import textwrap
import threading
import traceback
from collections import namedtuple
from typing import Callable

import anywidget
import numpy as np
import traitlets
import viser
from jaxtyping import UInt8

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


# ---------------------------------------------------------------------------
# Camera snapshot type
# ---------------------------------------------------------------------------

ViserCameraState = namedtuple(
    "ViserCameraState",
    ["position", "wxyz", "look_at", "up_direction", "fov"],
)
"""Snapshot of the viser camera captured when the snapshot button is pressed.

Fields:
    position:     (3,) float64 camera position in world coordinates.
    wxyz:         (4,) float64 orientation quaternion, OpenCV convention.
    look_at:      (3,) float64 point the camera looks at.
    up_direction: (3,) float64 camera up vector.
    fov:          float vertical field of view in radians.

Note:
    This is distinct from nerfview.CameraState, which nerfview constructs
    internally and passes to your render_fn. That type carries .c2w and
    .get_K(); this one carries the raw viser camera fields.
"""


# ---------------------------------------------------------------------------
# anywidget (snapshot button + iframe)
# ---------------------------------------------------------------------------


class _ViserAnyWidget(anywidget.AnyWidget):
    """Internal anywidget rendering a snapshot button above a viser iframe.

    Not intended for direct instantiation — use viser_marimo() instead.

    Attributes:
        port: The port that the viser HTTP/WebSocket server is listening on.
        height: Height of the embedded iframe in pixels.
        _snapshot_counter: Incremented by the JS button on each click; observed
            by Python to trigger snapshot logic without polling.
    """

    port = traitlets.Int(8080).tag(sync=True)
    height = traitlets.Int(600).tag(sync=True)
    _snapshot_counter = traitlets.Int(0).tag(sync=True)

    _esm = textwrap.dedent("""\
        function render({ model, el }) {

            // --- Snapshot button ------------------------------------------
            const button = document.createElement("button");
            button.textContent = "📷 Snapshot";
            button.title = "Save camera position; fire on_snapshot callbacks";
            Object.assign(button.style, {
                display:      "block",
                marginBottom: "6px",
                padding:      "4px 12px",
                fontSize:     "12px",
                cursor:       "pointer",
                borderRadius: "6px",
                border:       "1px solid #ccc",
                background:   "#f5f5f5",
            });
            button.addEventListener("click", () => {
                model.set("_snapshot_counter",
                          model.get("_snapshot_counter") + 1);
                model.save_changes();
                button.textContent = "✓ Saved";
                setTimeout(() => { button.textContent = "📷 Snapshot"; }, 1200);
            });
            el.appendChild(button);

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


class ViserMarimoWidget:
    """Container returned by viser_marimo().

    Holds the ViserServer, the optional nerfview Viewer, and the anywidget,
    and owns the snapshot button plumbing.

    Display in marimo by writing the variable name as the last expression in a
    cell — _repr_mimebundle_ is implemented so the iframe renders directly
    without needing to access .widget.

    Attributes:
        server: The underlying ViserServer. Use for all scene and GUI calls.
        viewer: The nerfview Viewer, or None in viser-only mode.
        widget: The raw anywidget instance. Only needed if you want to pass it
            to mo.ui.anywidget() for marimo reactive bindings.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        anywidget_instance: _ViserAnyWidget,
        viewer: object | None = None,
    ) -> None:
        self.server = server
        self.viewer = viewer
        self.widget = anywidget_instance
        self._snapshot_callbacks: list[Callable[[ViserCameraState], None]] = []
        self._callbacks_lock = threading.Lock()
        anywidget_instance.observe(
            self._on_snapshot_counter_change,
            names=["_snapshot_counter"],
        )

    # ------------------------------------------------------------------
    # marimo display protocol
    # ------------------------------------------------------------------

    def _repr_mimebundle_(self, **kwargs) -> dict:
        return self.widget._repr_mimebundle_(**kwargs)

    # ------------------------------------------------------------------
    # Snapshot plumbing
    # ------------------------------------------------------------------

    def _on_snapshot_counter_change(self, change: dict) -> None:
        """Traitlets observer — offloads snapshot work to a daemon thread."""
        threading.Thread(target=self._execute_snapshot, daemon=True).start()

    def _execute_snapshot(self) -> None:
        """Read the current camera and fire all registered callbacks.

        Reads camera state from the first connected client, persists it into
        server.initial_camera (so viser's Reset View and new clients start
        there), then calls each registered on_snapshot callback in order.
        """
        clients = self.server.get_clients()
        if not clients:
            return

        camera = next(iter(clients.values())).camera
        camera_state = ViserCameraState(
            position=camera.position.copy(),
            wxyz=camera.wxyz.copy(),
            look_at=camera.look_at.copy(),
            up_direction=camera.up_direction.copy(),
            fov=camera.fov,
        )

        initial_camera = self.server.initial_camera
        initial_camera.position = camera_state.position
        initial_camera.look_at = camera_state.look_at
        initial_camera.up = camera_state.up_direction
        initial_camera.fov = camera_state.fov

        with self._callbacks_lock:
            callbacks = list(self._snapshot_callbacks)

        for callback in callbacks:
            try:
                callback(camera_state)
            except Exception:
                traceback.print_exc()

    def on_snapshot(
        self,
        fn: Callable[[ViserCameraState], None],
    ) -> Callable[[ViserCameraState], None]:
        """Register a callback to run when the snapshot button is pressed.

        The callback receives a ViserCameraState named-tuple containing the
        camera position, orientation, look-at point, up direction, and field
        of view at the moment of the snapshot.

        The primary use is to cleanly rebuild GUI handles that have accumulated
        duplicates across cell re-runs:

            @widget.on_snapshot
            def _(camera_state):
                nonlocal slider
                saved_value = slider.value
                slider.remove()
                slider = server.gui.add_slider(
                    "scale", min=0.1, max=5.0, initial_value=saved_value
                )

        Multiple callbacks can be registered and run in registration order.
        Can be used as a decorator or called directly as widget.on_snapshot(fn).

        Args:
            fn: Callable accepting a ViserCameraState, returning None.

        Returns:
            fn unchanged, so the method works as a decorator.
        """
        with self._callbacks_lock:
            self._snapshot_callbacks.append(fn)
        return fn


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
            slider = server.gui.add_slider("scale", min=0.1, max=5.0, initial_value=1.0)

            @widget.on_snapshot
            def _(camera_state):
                nonlocal slider
                saved_value = slider.value
                slider.remove()
                slider = server.gui.add_slider(
                    "scale", min=0.1, max=5.0, initial_value=saved_value
                )

            widget  # last expression in cell — renders iframe + snapshot button

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

    viewer = None
    if render_fn is not None:
        try:
            import nerfview
        except ImportError as error:
            raise ImportError(
                "render_fn requires nerfview. Install it with:\n"
                "  pip install nerfview"
            ) from error
        viewer = nerfview.Viewer(
            server=server,
            render_fn=render_fn,
            mode=mode,
        )

    wrapped_widget = ViserMarimoWidget(
        server, anywidget_instance, viewer=viewer
    )

    if viewer is None:
        return server, wrapped_widget
    return server, viewer, wrapped_widget
