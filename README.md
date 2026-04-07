# marimo-viser

`marimo-viser` embeds a live [`viser`](https://github.com/nerfstudio-project/viser)
viewer inside a [`marimo`](https://marimo.io) notebook.

It gives you:

- a marimo-reactive widget by default
- optional `nerfview` render-loop integration
- explicit save/restore camera state controls
- safer `render_fn` error handling in notebooks

## Installation

Targets Python 3.11+.

```bash
uv pip install -e .
```

## Usage

Viser-only mode:

```python
from marimo_viser import viser_marimo

server, widget = viser_marimo()
widget
```

Nerfview mode:

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


server, viewer, widget = viser_marimo(render_fn=render_fn)
widget
```

The returned `widget` is already reactive in marimo, so downstream cells can
depend on `widget.value`.

## Camera State

The widget includes:

- `Save Camera State`
- `Restore Saved Camera State`

The saved state is exposed through `widget.value["camera_state_json"]`, so
viewer-to-viewer sync can be as simple as:

```python
widget_b.value["camera_state_json"] = widget_a.value["camera_state_json"]
```

For convenience, the same state is also available through typed helpers:

```python
state = widget.get_camera_state()
widget.set_camera_state(state)
```

The typed representation is `ViserCameraState` with:

- `position`
- `wxyz`
- `look_at`
- `up_direction`
- `fov`

## Render Errors

If `render_fn` raises, the kernel stays alive.

- the traceback is printed server-side
- the viewer shows an error image
- the `viser` GUI shows a copyable traceback panel

See [example.py](/home/schlack/Repositories/marimo-viser/notebooks/example.py)
for a notebook example.
