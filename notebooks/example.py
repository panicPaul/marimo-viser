import marimo

__generated_with = "0.22.4"
app = marimo.App(width="columns")

with app.setup:
    import anywidget
    import jaxtyping
    import marimo as mo
    import matplotlib
    import nerfview
    import numpy as np
    import ruff
    import splines
    import torch
    import traitlets
    import viser
    from jaxtyping import UInt8
    from marimo_viser.viser_widget import viser_marimo


@app.function
def render_fn(
    camera_state: nerfview.CameraState,
    render_tab_state: nerfview.RenderTabState,
) -> UInt8[np.ndarray, "H W 3"]:
    # Get camera parameters.
    width = render_tab_state.viewer_width
    height = render_tab_state.viewer_height
    c2w = camera_state.c2w
    K = camera_state.get_K([width, height])

    # Render a dummy image as a function of camera direction.
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

    img = ((dirs + 1.0) / 2.0 * 255.0).astype(np.uint8)
    # raise ValueError("test")
    return img


@app.cell
def _():
    server, viewer, widget = viser_marimo(render_fn=render_fn)
    widget
    return


@app.cell(column=1)
def _():
    return


if __name__ == "__main__":
    app.run()
