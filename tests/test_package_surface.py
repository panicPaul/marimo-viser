"""Tests for the marimo-3dv public package surface."""

import pytest


def test_core_imports_work():
    from marimo_3dv import (
        CameraState,
        MarimoViewer,
        Viewer,
        ViewerClick,
        ViewerState,
    )

    assert CameraState is not None
    assert MarimoViewer is not None
    assert Viewer is not None
    assert ViewerClick is not None
    assert ViewerState is not None


def test_viewer_uses_marimo_backend_in_notebook_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marimo_3dv.viewer as viewer_module

    sentinel = object()
    monkeypatch.setattr(viewer_module.mo, "running_in_notebook", lambda: True)
    monkeypatch.setattr(
        viewer_module,
        "marimo_viewer",
        lambda render_fn, state=None: sentinel,
    )

    viewer = viewer_module.Viewer(lambda camera: camera)

    assert viewer is sentinel


def test_viewer_runs_desktop_backend_outside_notebook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marimo_3dv.viewer as viewer_module

    class _StubDesktopViewer:
        def __init__(self) -> None:
            self.ran = False

        def run(self) -> None:
            self.ran = True

    stub = _StubDesktopViewer()
    monkeypatch.setattr(viewer_module.mo, "running_in_notebook", lambda: False)
    monkeypatch.setattr(
        viewer_module,
        "desktop_viewer",
        lambda render_fn, state=None, width=1280, height=720, title="": stub,
    )

    viewer = viewer_module.Viewer(lambda camera: camera)

    assert viewer is stub
    assert stub.ran is True


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


def test_backend_specific_viewer_imports_are_internal_only():
    from marimo_3dv.viewer.desktop import DesktopViewer, desktop_viewer

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
