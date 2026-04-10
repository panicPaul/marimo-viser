"""Built-in GUI pipeline ops for 3DGS / 2DGS rendering.

These ops require render data that exposes splat attributes (SH coefficients,
opacity logits, log scales). They hook into the ``prepare_render`` stage,
modifying inputs before the backend renders, and the ``post_render_metadata``
stage for per-splat diagnostic overlays.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from pydantic import BaseModel, Field

from marimo_3dv.pipeline.context import ViewerContext
from marimo_3dv.pipeline.gui import GuiOp, RenderResult


@runtime_checkable
class SplatRenderData(Protocol):
    """Protocol for Gaussian splat render data compatible with GS ops.

    Implementations must expose these attributes as tensors on the same device.
    All attributes are expected to be float tensors unless noted.
    """

    @property
    def opacity_logits(self) -> torch.Tensor:
        """(N, 1) raw opacity logits before sigmoid."""
        ...

    @property
    def log_half_extents(self) -> torch.Tensor:
        """(N, 3) log-scale half-extents."""
        ...

    @property
    def spherical_harmonics(self) -> torch.Tensor:
        """(N, num_bases, 3) SH coefficients."""
        ...

    @property
    def sh_degree(self) -> int:
        """Maximum SH degree present in the data."""
        ...


# ---------------------------------------------------------------------------
# max_sh_degree
# ---------------------------------------------------------------------------


class MaxShDegreeConfig(BaseModel):
    """Configuration for the max_sh_degree op."""

    max_sh_degree: int = Field(
        default=3,
        ge=0,
        le=4,
        description="Maximum SH degree to use during rendering (0 = diffuse).",
    )


def _max_sh_degree_hook(
    render_data: SplatRenderData,
    config: MaxShDegreeConfig,
    context: ViewerContext,
    runtime_state: None,
) -> SplatRenderData:
    """prepare_render: limit the active SH degree passed to the backend."""

    class _Capped:
        def __init__(self, inner: SplatRenderData, degree: int) -> None:
            self._inner = inner
            self._degree = degree

        @property
        def opacity_logits(self) -> torch.Tensor:
            return self._inner.opacity_logits

        @property
        def log_half_extents(self) -> torch.Tensor:
            return self._inner.log_half_extents

        @property
        def spherical_harmonics(self) -> torch.Tensor:
            num_bases = (self._degree + 1) ** 2
            return self._inner.spherical_harmonics[:, :num_bases, :]

        @property
        def sh_degree(self) -> int:
            return min(self._degree, self._inner.sh_degree)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    active_degree = min(config.max_sh_degree, render_data.sh_degree)
    return _Capped(render_data, active_degree)  # type: ignore[return-value]


def max_sh_degree_op(default_degree: int = 3) -> GuiOp[SplatRenderData]:
    """Return a prepare_render op that caps the active SH degree.

    Args:
        default_degree: Default SH degree shown in the GUI (0-4).

    Returns:
        A ``GuiOp`` configured for the ``prepare_render`` stage.
    """
    return GuiOp(
        name="max_sh_degree",
        config_model=MaxShDegreeConfig,
        default_config=MaxShDegreeConfig(max_sh_degree=default_degree),
        stage="prepare_render",
        hook=_max_sh_degree_hook,
    )


# ---------------------------------------------------------------------------
# filter_opacity
# ---------------------------------------------------------------------------


class FilterOpacityConfig(BaseModel):
    """Configuration for the filter_opacity op."""

    opacity_threshold: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Splats with opacity below this threshold are removed.",
    )


def _filter_opacity_hook(
    render_data: SplatRenderData,
    config: FilterOpacityConfig,
    context: ViewerContext,
    runtime_state: None,
) -> SplatRenderData:
    """prepare_render: remove low-opacity splats before the backend renders."""
    opacities = torch.sigmoid(render_data.opacity_logits.squeeze(-1))
    mask = opacities >= config.opacity_threshold

    class _Filtered:
        def __init__(self, inner: SplatRenderData, keep: torch.Tensor) -> None:
            self._inner = inner
            self._keep = keep

        @property
        def opacity_logits(self) -> torch.Tensor:
            return self._inner.opacity_logits[self._keep]

        @property
        def log_half_extents(self) -> torch.Tensor:
            return self._inner.log_half_extents[self._keep]

        @property
        def spherical_harmonics(self) -> torch.Tensor:
            return self._inner.spherical_harmonics[self._keep]

        @property
        def sh_degree(self) -> int:
            return self._inner.sh_degree

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)[self._keep]

    return _Filtered(render_data, mask)  # type: ignore[return-value]


def filter_opacity_op(
    default_threshold: float = 0.005,
) -> GuiOp[SplatRenderData]:
    """Return a prepare_render op that filters splats below an opacity threshold.

    Args:
        default_threshold: Default minimum opacity (0-1).

    Returns:
        A ``GuiOp`` configured for the ``prepare_render`` stage.
    """
    return GuiOp(
        name="filter_opacity",
        config_model=FilterOpacityConfig,
        default_config=FilterOpacityConfig(opacity_threshold=default_threshold),
        stage="prepare_render",
        hook=_filter_opacity_hook,
    )


# ---------------------------------------------------------------------------
# filter_size
# ---------------------------------------------------------------------------


class FilterSizeConfig(BaseModel):
    """Configuration for the filter_size op."""

    max_log_extent: float = Field(
        default=3.0,
        description="Splats with any log-half-extent above this are removed.",
    )


def _filter_size_hook(
    render_data: SplatRenderData,
    config: FilterSizeConfig,
    context: ViewerContext,
    runtime_state: None,
) -> SplatRenderData:
    """prepare_render: remove oversized splats before the backend renders."""
    max_log_extents = render_data.log_half_extents.amax(dim=-1)
    mask = max_log_extents <= config.max_log_extent

    class _SizeFiltered:
        def __init__(self, inner: SplatRenderData, keep: torch.Tensor) -> None:
            self._inner = inner
            self._keep = keep

        @property
        def opacity_logits(self) -> torch.Tensor:
            return self._inner.opacity_logits[self._keep]

        @property
        def log_half_extents(self) -> torch.Tensor:
            return self._inner.log_half_extents[self._keep]

        @property
        def spherical_harmonics(self) -> torch.Tensor:
            return self._inner.spherical_harmonics[self._keep]

        @property
        def sh_degree(self) -> int:
            return self._inner.sh_degree

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)[self._keep]

    return _SizeFiltered(render_data, mask)  # type: ignore[return-value]


def filter_size_op(
    default_max_log_extent: float = 3.0,
) -> GuiOp[SplatRenderData]:
    """Return a prepare_render op that filters out oversized splats.

    Args:
        default_max_log_extent: Default maximum log-half-extent threshold.

    Returns:
        A ``GuiOp`` configured for the ``prepare_render`` stage.
    """
    return GuiOp(
        name="filter_size",
        config_model=FilterSizeConfig,
        default_config=FilterSizeConfig(max_log_extent=default_max_log_extent),
        stage="prepare_render",
        hook=_filter_size_hook,
    )


# ---------------------------------------------------------------------------
# show_distribution
# ---------------------------------------------------------------------------


class ShowDistributionConfig(BaseModel):
    """Configuration for the show_distribution op."""

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


def _show_distribution_hook(
    result: RenderResult,
    config: ShowDistributionConfig,
    context: ViewerContext,
    runtime_state: None,
) -> RenderResult:
    """post_render_metadata: draw a projected splat count heatmap if enabled."""
    if not config.show_distribution:
        return result

    projected_means = result.metadata.get("projected_means")
    if projected_means is None:
        return result

    image = result.image.copy()
    height, width = image.shape[:2]

    # Build a 2-D histogram of projected splat screen positions.
    if isinstance(projected_means, torch.Tensor):
        means_np = projected_means.detach().cpu().numpy()
    else:
        means_np = np.asarray(projected_means)

    xs = np.clip(means_np[:, 0].astype(np.int32), 0, width - 1)
    ys = np.clip(means_np[:, 1].astype(np.int32), 0, height - 1)

    heatmap = np.zeros((height, width), dtype=np.float32)
    np.add.at(heatmap, (ys, xs), 1.0)

    max_count = float(heatmap.max())
    if max_count > 0:
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


def show_distribution_op() -> GuiOp[SplatRenderData]:
    """Return a post_render_metadata op that overlays a projected splat heatmap.

    Requires the backend to populate ``result.metadata["projected_means"]``
    with an (N, 2) array of screen-space splat positions.

    Returns:
        A ``GuiOp`` configured for the ``post_render_metadata`` stage.
    """
    return GuiOp(
        name="show_distribution",
        config_model=ShowDistributionConfig,
        default_config=ShowDistributionConfig(),
        stage="post_render_metadata",
        hook=_show_distribution_hook,
    )


__all__ = [
    "FilterOpacityConfig",
    "FilterSizeConfig",
    "MaxShDegreeConfig",
    "ShowDistributionConfig",
    "SplatRenderData",
    "filter_opacity_op",
    "filter_size_op",
    "max_sh_degree_op",
    "show_distribution_op",
]
