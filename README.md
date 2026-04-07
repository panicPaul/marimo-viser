# marimo-viser

`marimo-viser` is a small wrapper that makes it easier to embed a
[`viser`](https://github.com/nerfstudio-project/viser) server inside a
[`marimo`](https://marimo.io) notebook.

It focuses on the notebook-specific friction points:

- inline display of a live `viser` viewer
- a simple handoff between marimo cells and `viser` state
- optional `nerfview` render-loop integration
- a snapshot button for persisting camera state across cell reruns
- safer handling of `render_fn` failures in notebook workflows

## What It Returns

The main entry point is `viser_marimo`:

```python
from marimo_viser import viser_marimo
```

It supports two modes.

### 1. Viser-only mode

If you do not pass a `render_fn`, you get:

```python
server, widget = viser_marimo()
```

Use this when you want to build the scene imperatively with normal
`server.scene.*` and `server.gui.*` calls.

### 2. Nerfview render mode

If you pass a `render_fn`, you get:

```python
server, viewer, widget = viser_marimo(render_fn=render_fn)
```

Use this when you want `nerfview` to drive a per-client render loop and feed
images into the viewer background.

## Installation

This project currently targets Python 3.14+.

With `uv`:

```bash
uv sync
```

Or install directly into an environment:

```bash
uv pip install -e .
```

## Basic Usage

### Viser-only mode

```python
import marimo as mo
import numpy as np

from marimo_viser import viser_marimo

server, widget = viser_marimo(height=500)

slider = server.gui.add_slider(
    "scale", min=0.1, max=5.0, step=0.1, initial_value=1.0
)

points = np.random.randn(1_000, 3)
server.scene.add_point_cloud("/points", points=points * slider.value)

widget
```

### Nerfview mode

```python
import nerfview
import numpy as np
from jaxtyping import UInt8

from marimo_viser import viser_marimo


def render_fn(
    camera_state: nerfview.CameraState,
    render_tab_state: nerfview.RenderTabState,
) -> UInt8[np.ndarray, "height width 3"]:
    width = render_tab_state.viewer_width
    height = render_tab_state.viewer_height
    c2w = camera_state.c2w
    K = camera_state.get_K([width, height])

    camera_dirs = np.einsum(
        "ij,hwj->hwi",
        np.linalg.inv(K),
        np.pad(
            np.stack(
                np.meshgrid(np.arange(width), np.arange(height), indexing="xy"),
                -1,
            )
            + 0.5,
            ((0, 0), (0, 0), (0, 1)),
            constant_values=1.0,
        ),
    )
    dirs = np.einsum("ij,hwj->hwi", c2w[:3, :3], camera_dirs)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return ((dirs + 1.0) / 2.0 * 255.0).astype(np.uint8)


server, viewer, widget = viser_marimo(render_fn=render_fn, height=600)
widget
```

See [example.py](/home/schlack/Repositories/marimo-viser/notebooks/example.py)
for a notebook example.

## Snapshot Button

The embedded widget includes a snapshot button above the viewer.

When pressed, it:

- copies the current camera into `server.initial_camera`
- preserves the viewpoint for new clients and viser's "Reset View"
- triggers registered `on_snapshot` callbacks

Example:

```python
server, widget = viser_marimo()

slider = server.gui.add_slider("scale", min=0.1, max=5.0, initial_value=1.0)


@widget.on_snapshot
def _(_camera_state):
    global slider
    current = slider.value
    slider.remove()
    slider = server.gui.add_slider(
        "scale", min=0.1, max=5.0, initial_value=current
    )
```

## Reruns and Notebook Semantics

This wrapper is designed for marimo's rerunnable-cell workflow.

- `server.scene.*` calls are path-keyed and are generally safe to rerun.
- `server.gui.*` calls create new handles; if you rerun those cells repeatedly,
  you may want to rebuild them via `on_snapshot`.
- `widget` should typically be the last expression in the cell so marimo renders
  the embedded viewer directly.

## Render Error Handling

`nerfview` normally hard-exits the process when a background `render_fn` raises.
`marimo-viser` intercepts those failures first so the notebook kernel stays
alive.

When `render_fn` fails:

- the original traceback is still printed server-side
- the viewer shows an error frame with the exception and traceback
- the `viser` GUI shows a markdown error panel
- the full traceback is also exposed in a disabled multiline text field so it is
  easy to copy out of the UI

When the next render succeeds, the extra GUI error elements are removed again.

## API

```python
viser_marimo(
    *,
    render_fn: Callable | None = None,
    port: int | None = None,
    height: int = 600,
    host: str = "0.0.0.0",
    mode: str = "rendering",
)
```

Arguments:

- `render_fn`: optional `nerfview` render function
- `port`: explicit port, otherwise auto-select starting at `8080`
- `height`: iframe height in pixels
- `host`: bind address for the `viser` server
- `mode`: `nerfview` mode, usually `"rendering"` or `"training"`

Returns:

- `(server, widget)` in viser-only mode
- `(server, viewer, widget)` in render mode

## Development

Run the example notebook:

```bash
marimo run notebooks/example.py
```

Check the module:

```bash
python -m compileall src/marimo_viser/viser_widget.py
```
