"""Reusable viewer default controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, create_model

from marimo_3dv.gui.pydantic import form_gui
from marimo_3dv.viewer.widget import ViewerState

if TYPE_CHECKING:
    from marimo_3dv.pipeline.gui import ViewerPipelineResult

PipelineConfigT = TypeVar("PipelineConfigT", bound=BaseModel)


class ViewerOriginConfig(BaseModel):
    """Viewer origin marker position."""

    x: float = Field(default=0.0)
    y: float = Field(default=0.0)
    z: float = Field(default=0.0)


class ViewerRotationConfig(BaseModel):
    """Viewer-frame rotation in degrees."""

    x_degrees: float = Field(default=0.0, ge=-180.0, le=180.0)
    y_degrees: float = Field(default=0.0, ge=-180.0, le=180.0)
    z_degrees: float = Field(default=0.0, ge=-180.0, le=180.0)


class ViewerControlsConfig(BaseModel):
    """Reusable viewer controls schema."""

    fov_degrees: float = Field(default=60.0, gt=0.0, lt=180.0)
    show_axes: bool = Field(default=True)
    show_horizon: bool = Field(default=False)
    show_origin: bool = Field(default=False)
    show_stats: bool = Field(default=False)
    interactive_quality: int = Field(default=50, ge=1, le=100)
    settled_quality: Literal["jpeg_95", "jpeg_100", "png"] = Field(
        default="jpeg_100"
    )
    interactive_max_side: int = Field(default=1980, ge=1)
    internal_render_max_side: int = Field(default=3840, ge=1)
    rotation: ViewerRotationConfig = Field(default_factory=ViewerRotationConfig)
    origin: ViewerOriginConfig = Field(default_factory=ViewerOriginConfig)


@dataclass(frozen=True)
class ViewerControlsHandle:
    """Notebook-ready viewer controls block."""

    config_model: type[ViewerControlsConfig]
    default_config: ViewerControlsConfig
    gui: Any


@dataclass(frozen=True)
class CombinedViewerPipelineControlsHandle(Generic[PipelineConfigT]):
    """Notebook-ready combined viewer and pipeline controls block."""

    config_model: type[BaseModel]
    default_config: BaseModel
    gui: Any
    pipeline_config_model: type[PipelineConfigT]
    pipeline_default_config: PipelineConfigT


def viewer_controls_config(
    viewer_state: ViewerState,
) -> ViewerControlsConfig:
    """Return viewer controls config populated from a ViewerState."""
    return ViewerControlsConfig(
        fov_degrees=viewer_state.camera_state.fov_degrees,
        show_axes=viewer_state.show_axes,
        show_horizon=viewer_state.show_horizon,
        show_origin=viewer_state.show_origin,
        show_stats=viewer_state.show_stats,
        interactive_quality=viewer_state.interactive_quality,
        settled_quality=viewer_state.settled_quality,
        interactive_max_side=viewer_state.interactive_max_side or 1980,
        internal_render_max_side=viewer_state.internal_render_max_side or 3840,
        rotation=ViewerRotationConfig(
            x_degrees=viewer_state.viewer_rotation_x_degrees,
            y_degrees=viewer_state.viewer_rotation_y_degrees,
            z_degrees=viewer_state.viewer_rotation_z_degrees,
        ),
        origin=ViewerOriginConfig(
            x=viewer_state.origin[0],
            y=viewer_state.origin[1],
            z=viewer_state.origin[2],
        ),
    )


def apply_viewer_config(
    viewer_state: ViewerState,
    config: ViewerControlsConfig,
) -> ViewerState:
    """Apply reusable viewer controls config onto a ViewerState."""
    viewer_state.set_fov_degrees(config.fov_degrees, push_to_viewer=False)
    viewer_state.interactive_quality = config.interactive_quality
    viewer_state.settled_quality = config.settled_quality
    viewer_state.interactive_max_side = config.interactive_max_side
    viewer_state.internal_render_max_side = config.internal_render_max_side
    return (
        viewer_state.set_show_axes(config.show_axes)
        .set_show_horizon(config.show_horizon)
        .set_show_origin(config.show_origin)
        .set_show_stats(config.show_stats)
        .set_viewer_rotation(
            config.rotation.x_degrees,
            config.rotation.y_degrees,
            config.rotation.z_degrees,
        )
        .set_origin(
            config.origin.x,
            config.origin.y,
            config.origin.z,
        )
    )


def viewer_controls_gui(
    viewer_state: ViewerState,
    *,
    label: str = "",
    default_config: ViewerControlsConfig | None = None,
) -> ViewerControlsHandle:
    """Build a live notebook GUI for the default viewer controls."""
    resolved_default_config = default_config or viewer_controls_config(
        viewer_state
    )
    return ViewerControlsHandle(
        config_model=ViewerControlsConfig,
        default_config=resolved_default_config,
        gui=form_gui(
            ViewerControlsConfig,
            value=resolved_default_config,
            label=label,
            live_update=True,
        ),
    )


def _combined_viewer_pipeline_model(
    pipeline_config_model: type[PipelineConfigT],
) -> type[BaseModel]:
    return create_model(
        "ViewerPipelineControlsConfig",
        viewer=(
            ViewerControlsConfig,
            Field(default_factory=ViewerControlsConfig),
        ),
        pipeline=(
            pipeline_config_model,
            Field(default_factory=pipeline_config_model),
        ),
    )


def viewer_pipeline_controls_gui(
    viewer_state: ViewerState,
    pipeline_result: ViewerPipelineResult[Any, Any, Any],
    *,
    label: str = "",
    viewer_default_config: ViewerControlsConfig | None = None,
) -> CombinedViewerPipelineControlsHandle[Any]:
    """Build one live config tree containing viewer and pipeline controls."""
    pipeline_config_model = pipeline_result.config_model
    combined_model = _combined_viewer_pipeline_model(pipeline_config_model)
    resolved_viewer_default = viewer_default_config or viewer_controls_config(
        viewer_state
    )
    default_config = combined_model(
        viewer=resolved_viewer_default,
        pipeline=pipeline_result.default_config,
    )
    return CombinedViewerPipelineControlsHandle(
        config_model=combined_model,
        default_config=default_config,
        gui=form_gui(
            combined_model,
            value=default_config,
            label=label,
            live_update=True,
        ),
        pipeline_config_model=pipeline_config_model,
        pipeline_default_config=pipeline_result.default_config,
    )


def apply_viewer_pipeline_config(
    viewer_state: ViewerState,
    config: BaseModel,
) -> BaseModel:
    """Apply combined viewer config and return the pipeline config subtree."""
    apply_viewer_config(viewer_state, config.viewer)
    pipeline_config = config.pipeline
    if not isinstance(pipeline_config, BaseModel):
        raise TypeError(
            "Expected combined config to expose a BaseModel pipeline."
        )
    return pipeline_config


__all__ = [
    "CombinedViewerPipelineControlsHandle",
    "ViewerControlsConfig",
    "ViewerControlsHandle",
    "ViewerOriginConfig",
    "ViewerRotationConfig",
    "apply_viewer_config",
    "apply_viewer_pipeline_config",
    "viewer_controls_config",
    "viewer_controls_gui",
    "viewer_pipeline_controls_gui",
]
