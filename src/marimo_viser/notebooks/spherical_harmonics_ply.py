"""PLY-backed spherical harmonics inspection notebook."""

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns")

with app.setup:
    from dataclasses import replace
    from math import isqrt
    from pathlib import Path
    from typing import Any

    import marimo as mo
    import nerfview
    import numpy as np
    import torch
    from jaxtyping import Float
    from plyfile import PlyData
    from pydantic import BaseModel, Field
    from torch import Tensor

    from marimo_viser import form_gui, viser_marimo
    from marimo_viser.notebooks.spherical_harmonics import render_spheres, sh_to_rgb


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Spherical Harmonics from PLY

    This notebook loads a 3DGS-style `.ply`, shows the Gaussian centers as a
    point cloud, and lets you click a point to inspect that point's spherical
    harmonics in the SH viewer.
    """)
    return


@app.class_definition
class LoadConfig(BaseModel):
    """PLY loader configuration."""

    ply_path: Path = Field(
        default=Path("point_cloud.ply"),
        description="Path to a 3DGS-style PLY file.",
    )


@app.class_definition
class ViewerConfig(BaseModel):
    """Live viewer settings for point selection and SH display."""

    max_sh_degree: int = Field(
        default=3,
        ge=0,
        le=4,
        description="Maximum SH degree to display for the selected point.",
    )
    point_size: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Multiplier for the scale-derived point size.",
    )


@app.function
def infer_sh_degree(num_bases: int) -> int:
    """Infer SH degree from the number of SH bases."""
    degree = isqrt(num_bases) - 1
    if (degree + 1) ** 2 != num_bases:
        raise ValueError(f"Invalid SH basis count: {num_bases}")
    if not 0 <= degree <= 4:
        raise ValueError(f"Only SH degrees 0-4 are supported, got {degree}")
    return degree


@app.function
def load_ply_point_cloud(path: Path, sh_to_rgb: Any) -> dict[str, Any]:
    """Load Gaussian centers and SH coefficients from a 3DGS-style PLY file."""
    ply_data = PlyData.read(path)
    vertex_data = ply_data.elements[0]

    xyz_positions = np.stack(
        (
            np.asarray(vertex_data["x"], dtype=np.float32),
            np.asarray(vertex_data["y"], dtype=np.float32),
            np.asarray(vertex_data["z"], dtype=np.float32),
        ),
        axis=1,
    )

    dc_coefficients = np.stack(
        (
            np.asarray(vertex_data["f_dc_0"], dtype=np.float32),
            np.asarray(vertex_data["f_dc_1"], dtype=np.float32),
            np.asarray(vertex_data["f_dc_2"], dtype=np.float32),
        ),
        axis=1,
    )

    rest_feature_names = sorted(
        [
            property_.name
            for property_ in vertex_data.properties
            if property_.name.startswith("f_rest_")
        ],
        key=lambda name: int(name.split("_")[-1]),
    )

    num_rest_coefficients = len(rest_feature_names)
    if num_rest_coefficients % 3 != 0:
        raise ValueError(
            "Expected the number of f_rest_* attributes to be divisible by 3."
        )

    num_bases = 1 + num_rest_coefficients // 3
    degree_to_use = infer_sh_degree(num_bases)

    full_coefficients = np.zeros(
        (xyz_positions.shape[0], 25, 3),
        dtype=np.float32,
    )
    full_coefficients[:, 0, :] = dc_coefficients

    if rest_feature_names:
        rest_coefficients = np.stack(
            [
                np.asarray(vertex_data[name], dtype=np.float32)
                for name in rest_feature_names
            ],
            axis=1,
        )
        rest_coefficients = rest_coefficients.reshape(
            xyz_positions.shape[0],
            3,
            num_bases - 1,
        )
        rest_coefficients = np.transpose(rest_coefficients, (0, 2, 1))
        full_coefficients[:, 1:num_bases, :] = rest_coefficients

    scale_feature_names = sorted(
        [
            property_.name
            for property_ in vertex_data.properties
            if property_.name.startswith("scale_")
        ],
        key=lambda name: int(name.split("_")[-1]),
    )
    if scale_feature_names:
        log_scales = np.stack(
            [
                np.asarray(vertex_data[name], dtype=np.float32)
                for name in scale_feature_names
            ],
            axis=1,
        )
        gaussian_scales = np.exp(log_scales)
        default_point_size = float(np.exp(np.median(log_scales)) / 3.0)
    else:
        gaussian_scales = None
        default_point_size = 0.01

    point_colors = sh_to_rgb(
        torch.as_tensor(full_coefficients[:, 0, :], dtype=torch.float32)
    ).cpu().numpy()

    return {
        "xyz_positions": xyz_positions,
        "sh_coefficients": full_coefficients,
        "degree_to_use": degree_to_use,
        "point_colors": point_colors,
        "gaussian_scales": gaussian_scales,
        "default_point_size": default_point_size,
    }


@app.function
def point_cloud_extent(
    xyz_positions: Float[np.ndarray, "num_points 3"],
) -> float:
    """Compute a stable scene extent for camera placement and marker sizing."""
    if xyz_positions.shape[0] == 0:
        return 1.0
    bbox_min = xyz_positions.min(axis=0)
    bbox_max = xyz_positions.max(axis=0)
    return float(max(np.linalg.norm(bbox_max - bbox_min), 1.0))


@app.function
def point_cloud_center(
    xyz_positions: Float[np.ndarray, "num_points 3"],
) -> np.ndarray:
    """Compute the mean point-cloud center."""
    if xyz_positions.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    return xyz_positions.mean(axis=0).astype(np.float32)


@app.function
def point_cloud_point_size(scene_extent: float) -> float:
    """Choose a visible point size from the scene extent."""
    return max(scene_extent * 0.003, 0.002)


@app.function
def selected_point_colors(
    point_colors: Float[np.ndarray, "num_points 3"],
    selected_index: int | None,
) -> np.ndarray:
    """Create point-cloud colors with the selected point highlighted."""
    highlighted_colors = np.array(point_colors, copy=True)
    if (
        selected_index is not None
        and 0 <= selected_index < highlighted_colors.shape[0]
    ):
        highlighted_colors[selected_index] = np.array(
            [255, 255, 0], dtype=np.uint8
        )
    return highlighted_colors


@app.function
def selected_point_index_from_ray(
    xyz_positions: Float[np.ndarray, "num_points 3"],
    ray_origin: Float[np.ndarray, "3"],
    ray_direction: Float[np.ndarray, "3"],
) -> int | None:
    """Pick the point whose center lies closest to the click ray."""
    if xyz_positions.shape[0] == 0:
        return None

    normalized_direction = ray_direction / max(
        float(np.linalg.norm(ray_direction)),
        1e-6,
    )
    offsets = xyz_positions - ray_origin[None, :]
    distances_along_ray = offsets @ normalized_direction
    valid_mask = distances_along_ray > 0.0
    if not np.any(valid_mask):
        return None

    closest_points = (
        ray_origin[None, :]
        + distances_along_ray[:, None] * normalized_direction[None, :]
    )
    squared_distances = np.sum((xyz_positions - closest_points) ** 2, axis=1)
    squared_distances[~valid_mask] = np.inf
    return int(np.argmin(squared_distances))


@app.function
def point_view_camera_state(
    xyz_positions: Float[np.ndarray, "num_points 3"],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a sensible point-cloud camera pose."""
    scene_center = point_cloud_center(xyz_positions)
    scene_extent = point_cloud_extent(xyz_positions)
    camera_position = scene_center + np.array(
        [0.0, 0.0, 2.5 * scene_extent],
        dtype=np.float32,
    )
    return scene_center, camera_position


@app.function
def selection_markdown(
    selected_index: int | None,
    xyz_positions: Float[np.ndarray, "num_points 3"] | None,
    degree_to_use: int | None,
) -> str:
    """Format selection status for the viewers."""
    if xyz_positions is None or selected_index is None:
        return "**Loaded point cloud:** none  \n**Selected point:** none"

    selected_position = xyz_positions[selected_index]
    return (
        f"**Selected point:** `{selected_index}`  \n"
        f"**Position:** `[{selected_position[0]:.3f}, {selected_position[1]:.3f}, {selected_position[2]:.3f}]`  \n"
        f"**SH degree:** `{degree_to_use}`"
    )


@app.function
def orbit_camera_position(
    position: Float[np.ndarray, "3"],
    orbit_radius: float,
) -> np.ndarray:
    """Project a numpy camera offset back onto the fixed orbit sphere."""
    direction = np.asarray(position, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 1.0, 0.6], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
    return direction / max(norm, 1e-6) * orbit_radius


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Viewer Settings

    Live settings for point display and SH truncation.
    """)
    return


@app.cell
def _(viewer_form):
    viewer_config = viewer_form.value
    return (viewer_config,)


@app.cell
def _(
    loaded_point_cloud,
    point_server,
    point_status,
    selection_state,
    sh_status,
    sh_view_state,
    sh_viewer,
    viewer_config,
):
    if loaded_point_cloud is not None:
        selection_state["xyz_positions"] = loaded_point_cloud["xyz_positions"]
        selection_state["sh_coefficients"] = loaded_point_cloud[
            "sh_coefficients"
        ]
        selection_state["degree_to_use"] = loaded_point_cloud["degree_to_use"]
        selection_state["selected_index"] = 0
        selection_state["point_colors"] = loaded_point_cloud["point_colors"]
        selection_state["scene_center"] = point_cloud_center(
            loaded_point_cloud["xyz_positions"]
        )
        selection_state["base_point_size"] = loaded_point_cloud[
            "default_point_size"
        ]
        selection_state["point_size_scale"] = (
            viewer_config.point_size if viewer_config is not None else 1.0
        )

        sh_view_state["coefficients"] = torch.as_tensor(
            loaded_point_cloud["sh_coefficients"][0],
            dtype=torch.float32,
        )
        sh_view_state["degrees_to_use"] = min(
            loaded_point_cloud["degree_to_use"],
            viewer_config.max_sh_degree if viewer_config is not None else 4,
        )

        highlighted_colors = selected_point_colors(
            loaded_point_cloud["point_colors"],
            selection_state["selected_index"],
        )
        point_server.scene.add_point_cloud(
            "/ply_points",
            points=loaded_point_cloud["xyz_positions"],
            colors=highlighted_colors,
            point_size=selection_state["base_point_size"]
            * selection_state["point_size_scale"],
            point_shape="circle",
        )
        point_server.scene.add_point_cloud(
            "/selected_point",
            points=loaded_point_cloud["xyz_positions"][[0]],
            colors=np.array([[255, 255, 0]], dtype=np.uint8),
            point_size=selection_state["base_point_size"]
            * selection_state["point_size_scale"]
            * 4.0,
            point_shape="sparkle",
        )

        status_text = selection_markdown(
            selection_state["selected_index"],
            selection_state["xyz_positions"],
            selection_state["degree_to_use"],
        )
        point_status.content = status_text
        sh_status.content = status_text
        sh_viewer.rerender(None)
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## PLY Loader

    Load a 3DGS-style PLY file. Submission is explicit so large loads only
    happen when requested.
    """)
    return


@app.cell
def _():
    load_form = form_gui(LoadConfig, value=LoadConfig())
    load_form
    return (load_form,)


@app.cell
def _(load_form):
    load_config = load_form.value
    return (load_config,)


@app.cell
def _():
    viewer_form = form_gui(
        ViewerConfig,
        value=ViewerConfig(),
        # live_update=True,
    )
    return (viewer_form,)


@app.cell
def _(viewer_form):
    viewer_form
    return


@app.cell
def _(point_widget):
    point_widget
    return


@app.cell
def _(sh_widget):
    sh_widget
    return


@app.cell
def _(load_config):
    if load_config is None:
        loaded_point_cloud = None
    else:
        loaded_point_cloud = load_ply_point_cloud(
            load_config.ply_path, sh_to_rgb
        )
    return (loaded_point_cloud,)


@app.cell
def _():
    selection_state = {
        "xyz_positions": None,
        "sh_coefficients": None,
        "degree_to_use": 0,
        "selected_index": None,
        "point_colors": None,
        "base_point_size": 0.01,
        "point_size_scale": 1.0,
        "scene_center": np.zeros(3, dtype=np.float32),
    }
    return (selection_state,)


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ## Render Functions
    """)
    return


@app.cell
def _():
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    ## Point Cloud Viewer

    Enable picking to select the Gaussian center closest to the click ray.
    """)
    return


@app.cell
def _():
    point_server, point_widget = viser_marimo(height=680)
    pick_point_toggle = point_server.gui.add_checkbox(
        "Pick point",
        initial_value=False,
    )
    point_status = point_server.gui.add_markdown(
        "**Loaded point cloud:** none  \n**Selected point:** none"
    )
    return pick_point_toggle, point_server, point_status, point_widget


@app.cell
def _(loaded_point_cloud, point_server):
    if loaded_point_cloud is not None:
        look_at, camera_position = point_view_camera_state(
            loaded_point_cloud["xyz_positions"]
        )
        point_server.initial_camera.look_at = look_at.astype(np.float64)
        point_server.initial_camera.position = camera_position.astype(
            np.float64
        )
        for client in point_server.get_clients().values():
            client.camera.look_at = look_at.astype(np.float64)
            client.camera.position = camera_position.astype(np.float64)
    return


@app.cell
def _(
    pick_point_toggle,
    point_server,
    point_status,
    selection_state,
    sh_status,
    sh_view_state,
    sh_viewer,
    viewer_config,
):
    def handle_point_pick(event: object) -> None:
        """Select the point cloud center nearest to the clicked scene ray."""
        if selection_state["xyz_positions"] is None:
            return
        if event.ray_origin is None or event.ray_direction is None:
            return

        selected_index = selected_point_index_from_ray(
            selection_state["xyz_positions"],
            np.asarray(event.ray_origin, dtype=np.float32),
            np.asarray(event.ray_direction, dtype=np.float32),
        )
        if selected_index is None:
            return

        selection_state["selected_index"] = selected_index
        sh_view_state["coefficients"] = torch.as_tensor(
            selection_state["sh_coefficients"][selected_index],
            dtype=torch.float32,
        )
        sh_view_state["degrees_to_use"] = selection_state["degree_to_use"]
        if viewer_config is not None:
            sh_view_state["degrees_to_use"] = min(
                sh_view_state["degrees_to_use"],
                viewer_config.max_sh_degree,
            )

        point_server.scene.add_point_cloud(
            "/ply_points",
            points=selection_state["xyz_positions"],
            colors=selected_point_colors(
                selection_state["point_colors"],
                selected_index,
            ),
            point_size=selection_state["base_point_size"]
            * selection_state["point_size_scale"],
            point_shape="circle",
        )
        point_server.scene.add_point_cloud(
            "/selected_point",
            points=selection_state["xyz_positions"][[selected_index]],
            colors=np.array([[255, 255, 0]], dtype=np.uint8),
            point_size=selection_state["base_point_size"]
            * selection_state["point_size_scale"]
            * 4.0,
            point_shape="sparkle",
        )

        status_text = selection_markdown(
            selection_state["selected_index"],
            selection_state["xyz_positions"],
            selection_state["degree_to_use"],
        )
        point_status.content = status_text
        sh_status.content = status_text
        sh_viewer.rerender(None)
        pick_point_toggle.value = False

    @pick_point_toggle.on_update
    def _(_event: object) -> None:
        """Enable scene picking only while point-pick mode is active."""
        if pick_point_toggle.value:
            point_server.scene.on_pointer_event("click")(handle_point_pick)
        elif point_server.scene._scene_pointer_cb is not None:
            point_server.scene.remove_pointer_callback()

    return


@app.cell
def _(
    point_server,
    point_status,
    selection_state,
    sh_status,
    sh_view_state,
    sh_viewer,
    viewer_config,
):
    if (
        viewer_config is not None
        and selection_state["xyz_positions"] is not None
    ):
        selection_state["point_size_scale"] = viewer_config.point_size
        sh_view_state["degrees_to_use"] = min(
            selection_state["degree_to_use"],
            viewer_config.max_sh_degree,
        )

        point_server.scene.add_point_cloud(
            "/ply_points",
            points=selection_state["xyz_positions"],
            colors=selected_point_colors(
                selection_state["point_colors"],
                selection_state["selected_index"],
            ),
            point_size=selection_state["base_point_size"]
            * selection_state["point_size_scale"],
            point_shape="circle",
        )
        if selection_state["selected_index"] is not None:
            point_server.scene.add_point_cloud(
                "/selected_point",
                points=selection_state["xyz_positions"][
                    [selection_state["selected_index"]]
                ],
                colors=np.array([[255, 255, 0]], dtype=np.uint8),
                point_size=selection_state["base_point_size"]
                * selection_state["point_size_scale"]
                * 4.0,
                point_shape="sparkle",
            )
        status_text = selection_markdown(
            selection_state["selected_index"],
            selection_state["xyz_positions"],
            sh_view_state["degrees_to_use"],
        )
        point_status.content = status_text
        sh_status.content = status_text
        sh_viewer.rerender(None)
    return


@app.cell(column=4, hide_code=True)
def _():
    mo.md(r"""
    ## SH Viewer

    The selected point's SH coefficients are visualized with the same SH sphere
    renderer as the main SH notebook.
    """)
    return


@app.cell
def _():
    sh_view_state = {
        "coefficients": torch.zeros((25, 3), dtype=torch.float32),
        "degrees_to_use": 0,
        "orbit_radius": 6.0,
        "sphere_radius": 0.9,
    }
    return (sh_view_state,)


@app.cell
def _(sh_view_state):
    def render_fn(
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> np.ndarray:
        """Render the SH split-screen sphere view for the selected point."""
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        camera_to_world = torch.as_tensor(
            camera_state.c2w,
            dtype=torch.float32,
            device=device,
        )
        intrinsics = torch.as_tensor(
            camera_state.get_K([width, height]),
            dtype=torch.float32,
            device=device,
        )
        image = render_spheres(
            camera_to_world,
            intrinsics,
            sh_view_state["coefficients"].to(
                device=device, dtype=torch.float32
            ),
            int(sh_view_state["degrees_to_use"]),
            float(sh_view_state["orbit_radius"]),
            float(sh_view_state["sphere_radius"]),
            width,
            height,
        )
        return image.detach().cpu().numpy()

    return (render_fn,)


@app.cell
def _(render_fn):
    sh_server, sh_viewer, sh_widget = viser_marimo(
        render_fn=render_fn,
        height=680,
    )
    sh_status = sh_server.gui.add_markdown(
        "**Selected point:** none  \n**SH degree:** `0`"
    )
    return sh_server, sh_status, sh_viewer, sh_widget


@app.cell
def _(sh_server, sh_view_state, sh_viewer, sh_widget):
    @sh_server.on_client_connect
    def _(client: object) -> None:
        """Keep the SH viewer camera on a fixed orbit while preserving zoom as FOV."""
        syncing_camera = False
        client.camera.look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        client.camera.position = np.array(
            [0.0, 0.0, sh_view_state["orbit_radius"]],
            dtype=np.float32,
        )

        def _snap_camera() -> None:
            """Reset translation to the canonical orbit and convert radius changes into FOV."""
            nonlocal syncing_camera
            if syncing_camera:
                return
            look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            relative_position = np.asarray(
                client.camera.position, dtype=np.float32
            ) - np.asarray(client.camera.look_at, dtype=np.float32)
            distance = float(np.linalg.norm(relative_position))
            safe_distance = max(distance, 1e-6)
            zoomed_fov = 2.0 * np.arctan(
                np.tan(float(client.camera.fov) * 0.5)
                * safe_distance
                / float(sh_view_state["orbit_radius"])
            )
            zoomed_fov = float(
                np.clip(zoomed_fov, np.deg2rad(5.0), np.deg2rad(175.0))
            )
            client_id = getattr(
                client, "client_id", getattr(client, "id", None)
            )
            camera_state = sh_widget.get_camera_state(client_id=client_id)
            syncing_camera = True
            try:
                sh_widget.set_camera_state(
                    replace(
                        camera_state,
                        position=orbit_camera_position(
                            relative_position,
                            float(sh_view_state["orbit_radius"]),
                        ).astype(np.float64),
                        look_at=look_at.astype(np.float64),
                        fov=zoomed_fov,
                    ),
                    client_id=client_id,
                    update_reset_view=True,
                    sync_gui=True,
                )
            finally:
                syncing_camera = False

        _snap_camera()

        @client.camera.on_update
        def _(_camera: object) -> None:
            """Cancel translation changes and rerender after orbit updates."""
            _snap_camera()
            sh_viewer.rerender(None)

        sh_viewer.rerender(None)

    return


if __name__ == "__main__":
    app.run()
