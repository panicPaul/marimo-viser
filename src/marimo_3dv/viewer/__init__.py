"""Viewer runtimes and shared viewer primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import marimo as mo

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
    width: int = 1280,
    height: int = 720,
    title: str = "marimo-3dv viewer",
    target_fps: float = 60.0,
) -> MarimoViewer | DesktopViewer:
    """Create the appropriate viewer backend for the current runtime.

    In notebook runtimes this returns a marimo-backed viewer widget. In
    script runtimes it creates and immediately runs the desktop backend.
    """
    if mo.running_in_notebook():
        return marimo_viewer(render_fn, state=state)

    viewer = desktop_viewer(
        render_fn,
        state=state,
        width=width,
        height=height,
        title=title,
        target_fps=target_fps,
    )
    viewer.run()
    return viewer


__all__ = [
    "CameraState",
    "MarimoViewer",
    "Viewer",
    "ViewerClick",
    "ViewerState",
]
