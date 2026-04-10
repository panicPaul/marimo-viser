from __future__ import annotations

import pytest

from marimo_3dv.viewer.desktop import DesktopViewer


class _StubWindow:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        caption: str,
        resizable: bool,
    ) -> None:
        self._size = (width, height)
        self.invalid = False

    def event(self, func):
        return func

    def get_size(self) -> tuple[int, int]:
        return self._size

    def clear(self) -> None:
        return None

    def close(self) -> None:
        return None


class _StubLabel:
    def __init__(self, *args, **kwargs) -> None:
        self.text = ""

    def draw(self) -> None:
        return None


def test_desktop_viewer_run_raises_render_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marimo_3dv.viewer.desktop as desktop_module

    monkeypatch.setattr(desktop_module.pyglet.window, "Window", _StubWindow)
    monkeypatch.setattr(desktop_module.pyglet.text, "Label", _StubLabel)
    monkeypatch.setattr(
        desktop_module.pyglet.clock,
        "schedule_interval",
        lambda callback, interval: None,
    )
    monkeypatch.setattr(desktop_module.pyglet.app, "run", lambda: None)

    viewer = DesktopViewer(
        lambda camera_state: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        viewer.run()
