"""Viewer runtimes and shared viewer primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import marimo as mo

from marimo_3dv.viewer.defaults import (
    CombinedViewerPipelineControlsHandle,
    ViewerCameraConfig,
    ViewerControlsConfig,
    ViewerControlsHandle,
    ViewerNavigationConfig,
    ViewerOriginConfig,
    ViewerOverlayConfig,
    ViewerRenderConfig,
    ViewerRotationConfig,
    ViewerTransformConfig,
    apply_viewer_config,
    apply_viewer_pipeline_config,
    viewer_controls_config,
    viewer_controls_gui,
    viewer_pipeline_controls_gui,
)
from marimo_3dv.viewer.desktop import DesktopViewer, desktop_viewer
from marimo_3dv.viewer.widget import (
    CameraState,
    MarimoViewer,
    ViewerClick,
    ViewerState,
    marimo_viewer,
)


def Viewer(
    render_fn: Callable[[CameraState], Any],
    *,
    state: ViewerState | None = None,
    title: str = "marimo-3dv viewer",
) -> MarimoViewer | DesktopViewer:
    """Create the appropriate viewer backend for the current runtime.

    In notebook runtimes this returns a marimo-backed viewer widget. In
    script runtimes it creates and immediately runs the desktop backend.
    """
    viewer_state = state or ViewerState()
    if mo.running_in_notebook():
        return marimo_viewer(render_fn, state=viewer_state)

    window_width = 1280
    window_height = max(1, round(window_width / viewer_state.aspect_ratio))

    viewer = desktop_viewer(
        render_fn,
        state=viewer_state,
        width=window_width,
        height=window_height,
        title=title,
    )
    viewer.run()
    return viewer


__all__ = [
    "CameraState",
    "CombinedViewerPipelineControlsHandle",
    "MarimoViewer",
    "Viewer",
    "ViewerCameraConfig",
    "ViewerClick",
    "ViewerControlsConfig",
    "ViewerControlsHandle",
    "ViewerNavigationConfig",
    "ViewerOriginConfig",
    "ViewerOverlayConfig",
    "ViewerRenderConfig",
    "ViewerRotationConfig",
    "ViewerState",
    "ViewerTransformConfig",
    "apply_viewer_config",
    "apply_viewer_pipeline_config",
    "viewer_controls_config",
    "viewer_controls_gui",
    "viewer_pipeline_controls_gui",
]
