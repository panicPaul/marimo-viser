"""Tests for the marimo-3dv public package surface."""

import pytest


def test_core_imports_work():
    from marimo_3dv import (
        CameraState,
        NativeViewerState,
        NativeViewerWidget,
        ViewerClick,
        native_viewer,
    )

    assert CameraState is not None
    assert NativeViewerState is not None
    assert NativeViewerWidget is not None
    assert ViewerClick is not None
    assert native_viewer is not None


def test_pipeline_imports_work():
    from marimo_3dv import (
        GuiOp,
        GuiPipeline,
        GuiPipelineResult,
        RenderResult,
        SetupPipeline,
        ViewerContext,
    )

    assert GuiOp is not None
    assert GuiPipeline is not None
    assert GuiPipelineResult is not None
    assert RenderResult is not None
    assert SetupPipeline is not None
    assert ViewerContext is not None


def test_gs_pipe_imports_work():
    from marimo_3dv import (
        filter_opacity_op,
        filter_size_op,
        max_sh_degree_op,
        paint_ray_op,
        show_distribution_op,
    )

    assert filter_opacity_op is not None
    assert filter_size_op is not None
    assert max_sh_degree_op is not None
    assert paint_ray_op is not None
    assert show_distribution_op is not None


def test_gui_helpers_import():
    from marimo_3dv import form_gui, json_gui

    assert form_gui is not None
    assert json_gui is not None


def test_desktop_viewer_imports():
    from marimo_3dv import DesktopViewer, desktop_viewer

    assert DesktopViewer is not None
    assert desktop_viewer is not None


def test_viser_exports_removed():
    import marimo_3dv

    assert not hasattr(marimo_3dv, "ViserMarimoWidget")
    assert not hasattr(marimo_3dv, "viser_marimo")
    assert not hasattr(marimo_3dv, "ViserCameraState")

    with pytest.raises(ImportError):
        from marimo_3dv import ViserMarimoWidget  # noqa: F401

    with pytest.raises(ImportError):
        from marimo_3dv import viser_marimo  # noqa: F401
