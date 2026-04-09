"""Example marimo notebook for the native viewer widget."""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import torch
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
    mo.md(f"""
    Current camera JSON:

    ```json
    {viewer.camera_state.to_json()}
    ```
    """)
    return


@app.function
def render_fn(camera_state: CameraState) -> torch.Tensor:
    """Render a simple ray-direction visualization."""
    device = torch.device("cuda")
    width = camera_state.width
    height = camera_state.height
    cam_to_world = torch.as_tensor(
        camera_state.cam_to_world,
        device=device,
        dtype=torch.float32,
    )
    focal_length = (
        0.5
        * height
        / torch.tan(
            torch.deg2rad(torch.tensor(camera_state.fov_degrees, device=device))
            / 2.0
        )
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
    homogeneous_pixels = torch.nn.functional.pad(
        pixel_centers,
        (0, 1),
        value=1.0,
    )
    camera_dirs = torch.einsum(
        "ij,hwj->hwi",
        torch.linalg.inv(intrinsics),
        homogeneous_pixels,
    )
    world_dirs = torch.einsum("ij,hwj->hwi", cam_to_world[:3, :3], camera_dirs)
    world_dirs = world_dirs / torch.linalg.norm(
        world_dirs, dim=-1, keepdim=True
    )
    return (((world_dirs + 1.0) / 2.0) * 255.0).to(torch.uint8)


@app.cell
def _(viewer):
    print(viewer.camera_state)
    return


if __name__ == "__main__":
    app.run()
