"""Declarative GS render-view nodes and post-render effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from pydantic import BaseModel, Field

from marimo_3dv.pipeline.bundle import ViewerBackendBundle, backend_bundle
from marimo_3dv.pipeline.context import ViewerContext
from marimo_3dv.pipeline.gui import (
    AbstractRenderView,
    EffectNode,
    PipelineGroup,
    RenderNode,
    RenderResult,
    effect_node,
    render_node,
)


@runtime_checkable
class SplatRenderData(Protocol):
    """Protocol for Gaussian splat scenes compatible with GS nodes."""

    @property
    def opacity_logits(self) -> torch.Tensor:
        """(N, 1) raw opacity logits before sigmoid."""
        ...

    @property
    def center_positions(self) -> torch.Tensor:
        """(N, 3) splat centers in world space."""
        ...

    @property
    def log_half_extents(self) -> torch.Tensor:
        """(N, 3) log-scale half-extents."""
        ...

    @property
    def quaternion_orientation(self) -> torch.Tensor:
        """(N, 4) splat orientations."""
        ...

    @property
    def spherical_harmonics(self) -> torch.Tensor:
        """(N, num_bases, 3) SH coefficients."""
        ...

    @property
    def sh_degree(self) -> int:
        """Maximum SH degree present in the data."""
        ...


@dataclass(frozen=True)
class GsRenderView(AbstractRenderView[SplatRenderData | None]):
    """Immutable symbolic GS render view."""

    backend_key: str = "gsplat"
    keep_mask: torch.Tensor | None = None
    max_sh_degree: int | None = None

    def with_mask(self, keep_mask: torch.Tensor) -> GsRenderView:
        """Return a view with an additional symbolic keep-mask applied."""
        next_mask = keep_mask
        if self.keep_mask is not None:
            next_mask = self.keep_mask & keep_mask
        return GsRenderView(
            source_scene=self.source_scene,
            backend_key=self.backend_key,
            capabilities=self.capabilities,
            extensions=self.extensions,
            keep_mask=next_mask,
            max_sh_degree=self.max_sh_degree,
        )

    def with_max_sh_degree(self, max_sh_degree: int) -> GsRenderView:
        """Return a view with a capped active SH degree."""
        active_degree = max_sh_degree
        if self.max_sh_degree is not None:
            active_degree = min(self.max_sh_degree, max_sh_degree)
        return GsRenderView(
            source_scene=self.source_scene,
            backend_key=self.backend_key,
            capabilities=self.capabilities,
            extensions=self.extensions,
            keep_mask=self.keep_mask,
            max_sh_degree=active_degree,
        )


@dataclass(frozen=True)
class CompiledGsRenderView:
    """Materialized GS render view consumed by the rasterizer backend."""

    center_positions: torch.Tensor | None
    log_half_extents: torch.Tensor
    quaternion_orientation: torch.Tensor | None
    spherical_harmonics: torch.Tensor
    opacity_logits: torch.Tensor
    sh_degree: int
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in self.extra_fields:
            return self.extra_fields[name]
        raise AttributeError(name)


def gs_render_view(scene: SplatRenderData | None) -> GsRenderView:
    """Create the initial immutable GS render view from a source scene."""
    return GsRenderView(source_scene=scene)


def _masked_value(value: Any, keep_mask: torch.Tensor | None) -> Any:
    if keep_mask is None:
        return value
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim == 0 or value.shape[0] != keep_mask.shape[0]:
        return value
    return value[keep_mask]


def compile_gs_render_view(view: GsRenderView) -> CompiledGsRenderView | None:
    """Compile a symbolic GS render view into concrete tensors."""
    scene = view.source_scene
    if scene is None:
        return None
    keep_mask = view.keep_mask

    center_positions = _masked_value(
        getattr(scene, "center_positions", None), keep_mask
    )
    log_half_extents = _masked_value(scene.log_half_extents, keep_mask)
    quaternion_orientation = _masked_value(
        getattr(scene, "quaternion_orientation", None), keep_mask
    )
    spherical_harmonics = _masked_value(scene.spherical_harmonics, keep_mask)
    opacity_logits = _masked_value(scene.opacity_logits, keep_mask)
    sh_degree = int(scene.sh_degree)

    if view.max_sh_degree is not None:
        sh_degree = min(sh_degree, view.max_sh_degree)
        num_bases = (sh_degree + 1) ** 2
        spherical_harmonics = spherical_harmonics[:, :num_bases, :]

    known_names = {
        "center_positions",
        "log_half_extents",
        "quaternion_orientation",
        "spherical_harmonics",
        "opacity_logits",
        "sh_degree",
    }
    extra_fields = {
        name: _masked_value(value, keep_mask)
        for name, value in vars(scene).items()
        if name not in known_names
    }
    return CompiledGsRenderView(
        center_positions=center_positions,
        log_half_extents=log_half_extents,
        quaternion_orientation=quaternion_orientation,
        spherical_harmonics=spherical_harmonics,
        opacity_logits=opacity_logits,
        sh_degree=sh_degree,
        extra_fields=extra_fields,
    )


class MaxShDegreeConfig(BaseModel):
    """Configuration for the max_sh_degree node."""

    max_sh_degree: int = Field(
        default=3,
        ge=0,
        le=4,
        description="Maximum SH degree to use during rendering (0 = diffuse).",
    )


def _max_sh_degree_apply(
    render_view: GsRenderView,
    config: MaxShDegreeConfig,
    context: ViewerContext,
) -> GsRenderView:
    del context
    if render_view.source_scene is None:
        return render_view
    return render_view.with_max_sh_degree(config.max_sh_degree)


def max_sh_degree_op(default_degree: int = 3) -> RenderNode[GsRenderView]:
    """Return a render-view node that caps the active SH degree."""
    return render_node(
        name="max_sh_degree",
        config_model=MaxShDegreeConfig,
        default_config=MaxShDegreeConfig(max_sh_degree=default_degree),
        apply=_max_sh_degree_apply,
    )


class FilterOpacityConfig(BaseModel):
    """Configuration for the filter_opacity node."""

    opacity_threshold: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Splats with opacity below this threshold are removed.",
    )


def _filter_opacity_apply(
    render_view: GsRenderView,
    config: FilterOpacityConfig,
    context: ViewerContext,
) -> GsRenderView:
    del context
    if render_view.source_scene is None:
        return render_view
    logits = render_view.source_scene.opacity_logits.squeeze(-1)
    keep_mask = torch.sigmoid(logits) >= config.opacity_threshold
    return render_view.with_mask(keep_mask)


def filter_opacity_op(
    default_threshold: float = 0.005,
) -> RenderNode[GsRenderView]:
    """Return a render-view node that filters splats by opacity."""
    return render_node(
        name="filter_opacity",
        config_model=FilterOpacityConfig,
        default_config=FilterOpacityConfig(opacity_threshold=default_threshold),
        apply=_filter_opacity_apply,
    )


class FilterSizeConfig(BaseModel):
    """Configuration for the filter_size node."""

    max_log_extent: float = Field(
        default=3.0,
        description="Splats with any log-half-extent above this are removed.",
    )


def _filter_size_apply(
    render_view: GsRenderView,
    config: FilterSizeConfig,
    context: ViewerContext,
) -> GsRenderView:
    del context
    if render_view.source_scene is None:
        return render_view
    max_log_extents = render_view.source_scene.log_half_extents.amax(dim=-1)
    keep_mask = max_log_extents <= config.max_log_extent
    return render_view.with_mask(keep_mask)


def filter_size_op(
    default_max_log_extent: float = 3.0,
) -> RenderNode[GsRenderView]:
    """Return a render-view node that filters out oversized splats."""
    return render_node(
        name="filter_size",
        config_model=FilterSizeConfig,
        default_config=FilterSizeConfig(max_log_extent=default_max_log_extent),
        apply=_filter_size_apply,
    )


class ShowDistributionConfig(BaseModel):
    """Configuration for the distribution overlay effect."""

    show_distribution: bool = Field(
        default=False,
        description="Overlay a projected splat count heatmap on the image.",
    )
    distribution_alpha: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Alpha blend weight for the distribution overlay.",
    )


def _show_distribution_apply(
    result: RenderResult,
    config: ShowDistributionConfig,
    context: ViewerContext,
    runtime_state: None,
) -> RenderResult:
    del context, runtime_state
    if not config.show_distribution:
        return result

    projected_means = result.metadata.get("projected_means")
    if projected_means is None:
        return result

    image = result.image.copy()
    height, width = image.shape[:2]
    if isinstance(projected_means, torch.Tensor):
        means_np = projected_means.detach().cpu().numpy()
    else:
        means_np = np.asarray(projected_means)

    xs = np.clip(means_np[:, 0].astype(np.int32), 0, width - 1)
    ys = np.clip(means_np[:, 1].astype(np.int32), 0, height - 1)

    heatmap = np.zeros((height, width), dtype=np.float32)
    np.add.at(heatmap, (ys, xs), 1.0)
    max_count = float(heatmap.max())
    if max_count > 0.0:
        heatmap /= max_count

    import cv2

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)
    heatmap_rgb = heatmap_color[:, :, ::-1]

    alpha = config.distribution_alpha
    blended = (
        image.astype(np.float32) * (1.0 - alpha)
        + heatmap_rgb.astype(np.float32) * alpha
    )
    image = np.clip(blended, 0, 255).astype(np.uint8)
    return RenderResult(image=image, metadata=result.metadata)


def show_distribution_op() -> EffectNode[CompiledGsRenderView, None]:
    """Return a post-render effect that overlays projected splat density."""
    return effect_node(
        name="show_distribution",
        config_model=ShowDistributionConfig,
        default_config=ShowDistributionConfig(),
        apply=_show_distribution_apply,
    )


def gs_backend_bundle() -> ViewerBackendBundle[
    SplatRenderData | None, GsRenderView, CompiledGsRenderView | None
]:
    """Return a lightweight GS backend bundle with optional default groups."""
    return backend_bundle(
        name="gsplat",
        render_view_factory=gs_render_view,
        compile_view=compile_gs_render_view,
        default_render_items=(
            PipelineGroup("shading", max_sh_degree_op()),
            PipelineGroup(
                "filtering",
                filter_opacity_op(),
                filter_size_op(),
            ),
        ),
        default_effect_items=(
            PipelineGroup("diagnostics", show_distribution_op()),
        ),
    )


__all__ = [
    "CompiledGsRenderView",
    "FilterOpacityConfig",
    "FilterSizeConfig",
    "GsRenderView",
    "MaxShDegreeConfig",
    "ShowDistributionConfig",
    "SplatRenderData",
    "compile_gs_render_view",
    "filter_opacity_op",
    "filter_size_op",
    "gs_backend_bundle",
    "gs_render_view",
    "max_sh_degree_op",
    "show_distribution_op",
]
