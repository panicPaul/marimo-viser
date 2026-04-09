"""Example marimo notebook for the native viewer widget."""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="columns")

with app.setup:
    import base64
    import io

    import marimo as mo
    import numpy as np
    from PIL import Image
    from rich import print

    from marimo_viser import CameraState, native_viewer


@app.cell
def _():
    viewer = native_viewer(render_fn)
    viewer
    return (viewer,)


@app.cell
def _(viewer):
    viewer.value["error_text"]
    return


@app.cell
def _(viewer):
    viewer.value["frame_url"]
    return


@app.cell
def _(viewer):
    mo.md(f"""
    Current camera JSON:

    ```json
    {viewer.camera_state.to_json()}
    ```
    """)
    return


@app.function
def render_fn(camera_state: CameraState) -> np.ndarray:
    """Render a simple ray-direction visualization."""
    width = camera_state.width
    height = camera_state.height
    cam_to_world = camera_state.cam_to_world
    fov_radians = np.deg2rad(camera_state.fov_degrees)

    x_coords = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y_coords = np.linspace(1.0, -1.0, height, dtype=np.float32)
    pixel_grid_x, pixel_grid_y = np.meshgrid(x_coords, y_coords, indexing="xy")
    aspect_ratio = width / height
    tan_half_fov = np.tan(fov_radians / 2.0)

    camera_dirs = np.stack(
        [
            pixel_grid_x * tan_half_fov * aspect_ratio,
            pixel_grid_y * tan_half_fov,
            np.ones_like(pixel_grid_x),
        ],
        axis=-1,
    )
    camera_dirs /= np.linalg.norm(camera_dirs, axis=-1, keepdims=True)
    world_dirs = np.einsum("ij,hwj->hwi", cam_to_world[:3, :3], camera_dirs)

    rgb = (world_dirs + 1.0) / 2.0
    return (rgb * 255.0).astype(np.uint8)


@app.cell
def _(viewer):
    print(viewer.camera_state)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
