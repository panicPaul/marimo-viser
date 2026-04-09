# marimo-viser

`marimo-viser` embeds a live [`viser`](https://github.com/nerfstudio-project/viser)
viewer inside a [`marimo`](https://marimo.io) notebook.

It gives you:

- a marimo-reactive widget by default
- optional `nerfview` render-loop integration
- a native image-based viewer for custom renderers
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

Native viewer mode:

```python
import torch

from marimo_viser import CameraState, native_viewer


def render_fn(camera_state: CameraState) -> torch.Tensor:
    device = torch.device("cuda")
    width = camera_state.width
    height = camera_state.height
    cam_to_world = torch.as_tensor(
        camera_state.cam_to_world,
        device=device,
        dtype=torch.float32,
    )
    focal_length = 0.5 * height / torch.tan(
        torch.deg2rad(torch.tensor(camera_state.fov_degrees, device=device))
        / 2.0
    )
    intrinsics = torch.eye(3, device=device, dtype=torch.float32)
    intrinsics[0, 0] = focal_length
    intrinsics[1, 1] = focal_length
    intrinsics[0, 2] = width / 2.0
    intrinsics[1, 2] = height / 2.0

    pixel_x, pixel_y = torch.meshgrid(
        torch.arange(width, device=device, dtype=torch.float32),
        torch.arange(height, device=device, dtype=torch.float32),
        indexing="xy",
    )
    pixel_centers = torch.stack((pixel_x, pixel_y), dim=-1) + 0.5
    homogeneous_pixels = torch.nn.functional.pad(pixel_centers, (0, 1), value=1.0)
    camera_dirs = torch.einsum(
        "ij,hwj->hwi",
        torch.linalg.inv(intrinsics),
        homogeneous_pixels,
    )
    world_dirs = torch.einsum("ij,hwj->hwi", cam_to_world[:3, :3], camera_dirs)
    world_dirs = world_dirs / torch.linalg.norm(world_dirs, dim=-1, keepdim=True)
    return (((world_dirs + 1.0) / 2.0) * 255.0).to(torch.uint8)


viewer = native_viewer(render_fn, aspect_ratio=16.3 / 9)
viewer
```

The native viewer callback receives a typed `CameraState` with:

- `fov_degrees`
- `width`
- `height`
- `cam_to_world`
- `camera_convention`, currently `Literal["opencv", "opengl", "blender", "colmap"]`

`width` and `height` are measured from the rendered marimo widget size, so you
do not pass them into `native_viewer()`. Use `aspect_ratio=` to control the
initial layout, which defaults to `16.3 / 9`.

The native viewer currently defaults to OpenCV convention, exposed as
`camera_convention="opencv"`. The widget converts between `opencv`, `opengl`,
`blender`, and `colmap` conventions at the viewer boundary so the Python
callback sees a `cam_to_world` matrix consistent with the declared convention.

The widget also exposes the last primary-button click through
`viewer.last_click` / `viewer.get_last_click()`. Dragging or panning does not
register as a click.

Controls:

- left drag to orbit
- right drag to pan
- wheel to zoom
- `WASD` to move
- `Q` / `E` to move down / up

## Pydantic GUI

You can also generate marimo controls from a small Pydantic model:

```python
from pydantic import BaseModel, Field

from marimo_viser import form_gui


class RenderSettings(BaseModel):
    enabled: bool = True
    steps: int = Field(default=32, ge=1, le=128)
    opacity: float = Field(default=0.5, ge=0.0, le=1.0)
    title: str = "viewer"
```

The generated UI is submit-gated and includes both structured controls and a
JSON editor tab:

```python
form = form_gui(RenderSettings)
form
```

Then in a downstream cell:

```python
submitted = form.value
```

`submitted` is either `None` before the first valid submit or a typed
`RenderSettings` instance afterwards.

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
