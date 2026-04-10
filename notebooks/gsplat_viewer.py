# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.0",
#     "numpy==2.4.4",
#     "marimo-viser",
#     "gsplat==1.5.3",
#     "jaxtyping==0.3.9",
#     "torch==2.11.0",
# ]
#
# [tool.uv.sources]
# marimo-viser = { path = "..", editable = true }
# ///

"""Interactive 3D Gaussian splatting viewer using gsplat and marimo-viser."""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")

with app.setup:
    from dataclasses import dataclass

    import marimo as mo
    import numpy as np
    import torch
    from gsplat import rasterization
    from jaxtyping import Float

    from marimo_viser import (
        CameraState,
        NativeViewerState,
        form_gui,
        native_viewer,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Viewer Controls
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Scene Processing
    """)
    return


@app.cell
def _():
    # pipeline stuff like filter by opacity, change opacity etc
    # should be modelled after pandas pipe principle
    # i.e. scene = ply_load(path)
    # viewer_scene = pipeline(scene).pipe(filter_opacity, options).pipe(change_size, options) etc.

    # refine later and think how that could integrate well with the overall config gui
    # actually we should probably link them through the viewer class s.t. we can use an offline viewer that doesn't have to go through the same jpeg / browser chain.

    # TODO: note to claude lets skip this for now
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## GUI
    """)
    return


@app.cell
def _():
    # use form gui for viewer settings (i.e. fov for now)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Gaussian Splatting Scene
    """)
    return


@app.cell
def _(Tensor):
    @dataclass
    class SplatScene:
        """Simple 3DGS Scene."""

        center_positions: Float[Tensor, "num_splats 3"]
        log_half_extents: Float[Tensor, "num_splats 3"]
        quaternion_orientation: Float[Tensor, "num_splats 4"]
        spherical_harmonics: Float[Tensor, "num_splats 15 3"]
        opacity_logits: Float[Tensor, "num_splats 1"]

    return (SplatScene,)


@app.cell
def _(SplatScene):
    def rasterize_scene(
        scene: SplatScene, camera: CameraState
    ) -> Float[np.ndarray, "height width 3"]:
        pass  # use gsplat rasterization here

    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""

    """)
    return


if __name__ == "__main__":
    app.run()
