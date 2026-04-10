"""Tests for built-in 3DGS GUI pipeline ops."""

from dataclasses import dataclass

import numpy as np
import torch

from marimo_3dv.ops.gs import (
    FilterOpacityConfig,
    FilterSizeConfig,
    MaxShDegreeConfig,
    ShowDistributionConfig,
    filter_opacity_op,
    filter_size_op,
    max_sh_degree_op,
    show_distribution_op,
)
from marimo_3dv.pipeline.context import ViewerContext
from marimo_3dv.pipeline.gui import RenderResult
from marimo_3dv.viewer.widget import NativeViewerState

# ---------------------------------------------------------------------------
# Minimal fake SplatRenderData for testing
# ---------------------------------------------------------------------------


@dataclass
class _FakeSplats:
    center_positions: torch.Tensor
    log_half_extents: torch.Tensor
    quaternion_orientation: torch.Tensor
    spherical_harmonics: torch.Tensor
    opacity_logits: torch.Tensor
    sh_degree: int

    @property
    def _keep_mask(self):
        return None


def _make_splats(
    n: int = 10,
    sh_degree: int = 3,
    opacity_logit: float = 0.0,
    log_scale: float = 0.0,
) -> _FakeSplats:
    """Create n splats with uniform opacity and scale."""
    num_bases = (sh_degree + 1) ** 2
    return _FakeSplats(
        center_positions=torch.zeros(n, 3),
        log_half_extents=torch.full((n, 3), log_scale),
        quaternion_orientation=torch.zeros(n, 4),
        spherical_harmonics=torch.zeros(n, num_bases, 3),
        opacity_logits=torch.full((n, 1), opacity_logit),
        sh_degree=sh_degree,
    )


def _context() -> ViewerContext:
    return ViewerContext(viewer_state=NativeViewerState(), last_click=None)


# ---------------------------------------------------------------------------
# max_sh_degree
# ---------------------------------------------------------------------------


def test_max_sh_degree_caps_degree():
    splats = _make_splats(sh_degree=3)
    op = max_sh_degree_op(default_degree=1)
    config = MaxShDegreeConfig(max_sh_degree=1)
    result = op.hook(splats, config, _context(), None)
    assert result.sh_degree == 1
    assert result.spherical_harmonics.shape[1] == 4  # (1+1)^2


def test_max_sh_degree_cannot_exceed_data_degree():
    splats = _make_splats(sh_degree=2)
    op = max_sh_degree_op()
    config = MaxShDegreeConfig(max_sh_degree=3)
    result = op.hook(splats, config, _context(), None)
    assert result.sh_degree == 2


def test_max_sh_degree_zero_gives_diffuse():
    splats = _make_splats(sh_degree=3)
    op = max_sh_degree_op()
    config = MaxShDegreeConfig(max_sh_degree=0)
    result = op.hook(splats, config, _context(), None)
    assert result.sh_degree == 0
    assert result.spherical_harmonics.shape[1] == 1


# ---------------------------------------------------------------------------
# filter_opacity
# ---------------------------------------------------------------------------


def test_filter_opacity_removes_low_opacity_splats():
    # logit=0 → sigmoid≈0.5; logit=-5 → sigmoid≈0.007 (below threshold 0.1)
    low = _make_splats(n=5, opacity_logit=-5.0)
    high = _make_splats(n=5, opacity_logit=2.0)

    # Combine manually
    @dataclass
    class Mixed:
        opacity_logits: torch.Tensor
        log_half_extents: torch.Tensor
        spherical_harmonics: torch.Tensor
        sh_degree: int = 3

    mixed = Mixed(
        opacity_logits=torch.cat([low.opacity_logits, high.opacity_logits]),
        log_half_extents=torch.cat(
            [low.log_half_extents, high.log_half_extents]
        ),
        spherical_harmonics=torch.cat(
            [low.spherical_harmonics, high.spherical_harmonics]
        ),
    )

    op = filter_opacity_op(default_threshold=0.1)
    config = FilterOpacityConfig(opacity_threshold=0.1)
    result = op.hook(mixed, config, _context(), None)

    assert (
        result.opacity_logits.shape[0] == 5
    )  # only high-opacity splats remain


def test_filter_opacity_zero_threshold_keeps_all():
    splats = _make_splats(n=8, opacity_logit=0.0)
    op = filter_opacity_op()
    config = FilterOpacityConfig(opacity_threshold=0.0)
    result = op.hook(splats, config, _context(), None)
    assert result.opacity_logits.shape[0] == 8


# ---------------------------------------------------------------------------
# filter_size
# ---------------------------------------------------------------------------


def test_filter_size_removes_large_splats():
    @dataclass
    class MixedSize:
        opacity_logits: torch.Tensor
        log_half_extents: torch.Tensor
        spherical_harmonics: torch.Tensor
        sh_degree: int = 3

    small_log = torch.full((5, 3), 0.5)
    large_log = torch.full((5, 3), 5.0)
    mixed = MixedSize(
        opacity_logits=torch.zeros(10, 1),
        log_half_extents=torch.cat([small_log, large_log]),
        spherical_harmonics=torch.zeros(10, 16, 3),
    )

    op = filter_size_op()
    config = FilterSizeConfig(max_log_extent=2.0)
    result = op.hook(mixed, config, _context(), None)
    assert result.log_half_extents.shape[0] == 5


# ---------------------------------------------------------------------------
# show_distribution
# ---------------------------------------------------------------------------


def test_show_distribution_no_op_when_disabled():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = RenderResult(
        image=image, metadata={"projected_means": torch.zeros(5, 2)}
    )
    op = show_distribution_op()
    config = ShowDistributionConfig(show_distribution=False)
    out = op.hook(result, config, _context(), None)
    assert out is result


def test_show_distribution_no_op_without_metadata():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = RenderResult(image=image, metadata={})
    op = show_distribution_op()
    config = ShowDistributionConfig(show_distribution=True)
    out = op.hook(result, config, _context(), None)
    assert out is result


def test_show_distribution_blends_image():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    means = torch.tensor([[4.0, 4.0], [8.0, 8.0], [12.0, 12.0]])
    result = RenderResult(image=image, metadata={"projected_means": means})
    op = show_distribution_op()
    config = ShowDistributionConfig(
        show_distribution=True, distribution_alpha=0.5
    )
    out = op.hook(result, config, _context(), None)
    # Output image should differ from all-black input
    assert not np.array_equal(out.image, image)
