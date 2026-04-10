"""Public package exports for marimo-3dv."""

from marimo_3dv.gui.pydantic import form_gui, json_gui
from marimo_3dv.ops.gs import (
    FilterOpacityConfig,
    FilterSizeConfig,
    MaxShDegreeConfig,
    ShowDistributionConfig,
    SplatRenderData,
    filter_opacity_op,
    filter_size_op,
    max_sh_degree_op,
    show_distribution_op,
)
from marimo_3dv.ops.normalization import (
    apply_rotation_to_quaternions,
    apply_rotation_to_sh_coefficients,
    apply_scale_to_log_scales,
    apply_to_cameras,
    apply_to_points,
    compose_transforms,
    pca_transform_from_points,
    similarity_from_cameras,
)
from marimo_3dv.ops.overlay import PaintRayConfig, paint_ray_op
from marimo_3dv.ops.setup import camera_similarity_op, pca_alignment_op
from marimo_3dv.pipeline.context import ViewerContext
from marimo_3dv.pipeline.gui import (
    GuiOp,
    GuiPipeline,
    GuiPipelineResult,
    RenderResult,
)
from marimo_3dv.pipeline.setup import SetupPipeline
from marimo_3dv.viewer import (
    CameraState,
    MarimoViewer,
    Viewer,
    ViewerClick,
    ViewerState,
)

__all__ = [
    "CameraState",
    "FilterOpacityConfig",
    "FilterSizeConfig",
    "GuiOp",
    "GuiPipeline",
    "GuiPipelineResult",
    "MarimoViewer",
    "MaxShDegreeConfig",
    "PaintRayConfig",
    "RenderResult",
    "SetupPipeline",
    "ShowDistributionConfig",
    "SplatRenderData",
    "Viewer",
    "ViewerClick",
    "ViewerContext",
    "ViewerState",
    "apply_rotation_to_quaternions",
    "apply_rotation_to_sh_coefficients",
    "apply_scale_to_log_scales",
    "apply_to_cameras",
    "apply_to_points",
    "camera_similarity_op",
    "compose_transforms",
    "filter_opacity_op",
    "filter_size_op",
    "form_gui",
    "json_gui",
    "max_sh_degree_op",
    "paint_ray_op",
    "pca_alignment_op",
    "pca_transform_from_points",
    "show_distribution_op",
    "similarity_from_cameras",
]
