# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.0",
#     "numpy==2.4.4",
#     "marimo-3dv",
#     "gsplat==1.5.3",
#     "jaxtyping==0.3.9",
#     "torch==2.11.0",
#     "plyfile",
#     "pydantic",
# ]
#
# [tool.uv.sources]
# marimo-3dv = { path = "..", editable = true }
# ///

"""Interactive 3D Gaussian splatting viewer using gsplat and marimo-3dv."""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")

with app.setup:
    from dataclasses import dataclass
    from math import isqrt
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch
    from gsplat import rasterization
    from jaxtyping import Float
    from plyfile import PlyData
    from pydantic import BaseModel, Field
    from torch import Tensor

    from marimo_3dv import (
        CameraState,
        GuiPipeline,
        NativeViewerState,
        RenderResult,
        desktop_viewer,
        filter_opacity_op,
        filter_size_op,
        form_gui,
        max_sh_degree_op,
        native_viewer,
        show_distribution_op,
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    # 3DGS Viewer
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Viewer
    """)
    return


@app.cell
def _(load_form):
    load_form
    return


@app.cell
def _():
    viewer_state = NativeViewerState()
    return (viewer_state,)


@app.cell
def _(viewer_state):
    (
        viewer_state.set_show_origin(False)
        .set_show_stats(True)
        .set_show_horizon(False)
        .set_show_axes(True)
    )
    return


@app.cell
def _():
    gui_pipeline = (
        GuiPipeline(allow_prepared_copy=True)
        .pipe(max_sh_degree_op())
        .pipe(filter_opacity_op())
        .pipe(filter_size_op())
        .pipe(show_distribution_op())
    )
    return (gui_pipeline,)


@app.cell
def _(gui_pipeline, scene, viewer_state):
    pipeline_result = gui_pipeline.build(scene, viewer_state)
    return (pipeline_result,)


@app.cell
def _(pipeline_result):
    pipeline_config_gui = form_gui(
        pipeline_result.config_model,
        value=pipeline_result.default_config,
        live_update=True,
    )
    pipeline_config_gui
    return (pipeline_config_gui,)


@app.cell
def _(pipeline_config_gui, pipeline_result, viewer_state):
    render_fn = pipeline_result.bind(
        pipeline_config_gui.value or pipeline_result.default_config,
        backend_fn=rasterize_scene,
    )
    if mo.running_in_notebook():
        viewer = native_viewer(
            lambda cam: render_fn(cam).image,
            state=viewer_state,
            camera_convention="opencv",
            interactive_quality=50,
            interactive_max_side=1980,
        )
    else:
        viewer = desktop_viewer(
            lambda cam: render_fn(cam).image,
            state=viewer_state,
        )
        viewer.run()
    return render_fn, viewer


@app.cell
def _(viewer):
    viewer
    return


@app.cell
def _(scene):
    scene
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Viewer Debug
    """)
    return


@app.cell
def _(render_fn, scene, viewer):
    debug_comparison = None
    if scene is not None:
        try:
            camera_state = viewer.get_camera_state()
            pipeline_image = render_fn(camera_state).image
            snapshot_image = np.asarray(viewer.get_snapshot())
            debug_comparison = {
                "same_shape": pipeline_image.shape == snapshot_image.shape,
                "identical": np.array_equal(pipeline_image, snapshot_image),
            }
        except RuntimeError as error:
            debug_comparison = {"snapshot_error": str(error)}
    debug_comparison
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Scene Definition
    """)
    return


@app.class_definition
@dataclass
class SplatScene:
    """Minimal 3DGS scene loaded from a PLY file."""

    center_positions: Float[Tensor, "num_splats 3"]
    log_half_extents: Float[Tensor, "num_splats 3"]
    quaternion_orientation: Float[Tensor, "num_splats 4"]
    spherical_harmonics: Float[Tensor, "num_splats num_bases 3"]
    opacity_logits: Float[Tensor, "num_splats 1"]
    sh_degree: int


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Rasterization
    """)
    return


@app.function
def rasterize_scene(
    camera: CameraState, scene: SplatScene | None
) -> RenderResult:
    """Render a SplatScene from the given camera using gsplat rasterization."""
    if scene is None:
        return RenderResult(
            image=np.full((camera.height, camera.width, 3), 245, dtype=np.uint8)
        )

    gsplat_camera = camera.with_convention("opencv")
    device = scene.center_positions.device
    w2c = np.linalg.inv(gsplat_camera.cam_to_world)
    viewmats = torch.from_numpy(w2c).to(device=device, dtype=torch.float32)[
        None
    ]

    half_fov_rad = np.radians(gsplat_camera.fov_degrees / 2)
    focal = (gsplat_camera.height / 2) / np.tan(half_fov_rad)
    K = torch.tensor(
        [
            [focal, 0.0, gsplat_camera.width / 2],
            [0.0, focal, gsplat_camera.height / 2],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )[None]

    render_colors, _render_alphas, meta = rasterization(
        means=scene.center_positions,
        quats=scene.quaternion_orientation,
        scales=torch.exp(scene.log_half_extents),
        opacities=torch.sigmoid(scene.opacity_logits.squeeze(-1)),
        colors=scene.spherical_harmonics,
        viewmats=viewmats,
        Ks=K,
        width=gsplat_camera.width,
        height=gsplat_camera.height,
        sh_degree=scene.sh_degree,
    )
    image = render_colors[0].clamp(0.0, 1.0).cpu().numpy()
    image_uint8 = (image * 255).astype(np.uint8)

    metadata: dict = {}
    if "means2d" in meta:
        metadata["projected_means"] = meta["means2d"][0]

    return RenderResult(image=image_uint8, metadata=metadata)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## GUI Definition
    """)
    return


@app.cell
def _():
    class LoadConfig(BaseModel):
        """Configuration for loading a PLY file."""

        ply_path: Path = Field(
            default=Path.cwd() / "point_cloud.ply",
            description="Path to a 3DGS-style `.ply` file.",
        )

    load_form = form_gui(
        LoadConfig, value=LoadConfig(), submit_label="Load File"
    )
    return (load_form,)


@app.cell
def _(load_form):
    if mo.running_in_notebook():
        load_config = load_form.value
        scene = (
            load_splat_scene(load_config.ply_path)
            if load_config is not None and load_config.ply_path.exists()
            else None
        )
    else:
        import subprocess

        result = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--title=Open PLY file",
                "--file-filter=*.ply",
            ],
            capture_output=True,
            text=True,
        )
        ply_path = Path(result.stdout.strip())
        scene = load_splat_scene(ply_path) if ply_path.exists() else None
    return (scene,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## PLY Loading
    """)
    return


@app.function
def infer_sh_degree(num_bases: int) -> int:
    """Infer the SH degree from the number of basis functions."""
    degree = isqrt(num_bases) - 1
    if (degree + 1) ** 2 != num_bases:
        raise ValueError(f"Invalid SH basis count: {num_bases}")
    if not 0 <= degree <= 4:
        raise ValueError(f"Only SH degrees 0-4 are supported, got {degree}")
    return degree


@app.function
def get_gsplat_device() -> torch.device:
    """Return the device used for gsplat rasterization."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "gsplat viewer requires CUDA, but CUDA is unavailable."
        )
    return torch.device("cuda")


@app.function
def load_splat_scene(path: Path) -> SplatScene:
    """Load a 3DGS-style `.ply` file into a SplatScene."""
    device = get_gsplat_device()
    ply_data = PlyData.read(path)
    vertices = ply_data["vertex"]
    property_names = list(vertices.data.dtype.names)

    centers = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]],
        axis=1,
    ).astype(np.float32)

    dc_coefficients = np.stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]],
        axis=1,
    ).astype(np.float32)

    rest_feature_names = sorted(
        [name for name in property_names if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    num_rest_coefficients = len(rest_feature_names)
    if num_rest_coefficients % 3 != 0:
        raise ValueError(
            "Expected the number of `f_rest_*` attributes to be divisible by 3."
        )

    num_bases = 1 + num_rest_coefficients // 3
    sh_degree = infer_sh_degree(num_bases)
    sh_coefficients = np.zeros(
        (centers.shape[0], num_bases, 3), dtype=np.float32
    )
    sh_coefficients[:, 0, :] = dc_coefficients
    if rest_feature_names:
        rest_coefficients = np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in rest_feature_names
            ],
            axis=1,
        )
        rest_coefficients = rest_coefficients.reshape(
            centers.shape[0], 3, num_bases - 1
        )
        sh_coefficients[:, 1:num_bases, :] = np.transpose(
            rest_coefficients, (0, 2, 1)
        )

    scale_feature_names = sorted(
        [name for name in property_names if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    rotation_feature_names = sorted(
        [name for name in property_names if name.startswith("rot")],
        key=lambda name: int(name.split("_")[-1]),
    )
    log_scales = (
        np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in scale_feature_names
            ],
            axis=1,
        )
        if scale_feature_names
        else np.full((centers.shape[0], 3), np.log(0.01), dtype=np.float32)
    )
    rotations = (
        np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in rotation_feature_names
            ],
            axis=1,
        )
        if rotation_feature_names
        else np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (centers.shape[0], 1),
        )
    )
    opacity_logits = np.asarray(vertices["opacity"], dtype=np.float32)[:, None]

    return SplatScene(
        center_positions=torch.from_numpy(centers).to(device=device),
        log_half_extents=torch.from_numpy(log_scales).to(device=device),
        quaternion_orientation=torch.from_numpy(rotations).to(device=device),
        spherical_harmonics=torch.from_numpy(sh_coefficients).to(device=device),
        opacity_logits=torch.from_numpy(opacity_logits).to(device=device),
        sh_degree=sh_degree,
    )


if __name__ == "__main__":
    app.run()
