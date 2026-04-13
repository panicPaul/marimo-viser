from __future__ import annotations

import numpy as np
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


class _StubTexture:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.blit_calls: list[tuple[int, int, int, int, int]] = []
        self.deleted = False

    def blit_into(self, image_data, x: int, y: int, z: int) -> None:
        self.blit_calls.append((image_data.width, image_data.height, x, y, z))

    def delete(self) -> None:
        self.deleted = True


class _StubSprite:
    def __init__(self, texture: _StubTexture, x: int, y: int) -> None:
        self.image = texture
        self.width = texture.width
        self.height = texture.height
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.draw_calls = 0

    def draw(self) -> None:
        self.draw_calls += 1


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


def test_desktop_viewer_reuses_texture_until_frame_size_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marimo_3dv.viewer.desktop as desktop_module

    created_textures: list[_StubTexture] = []

    monkeypatch.setattr(desktop_module.pyglet.window, "Window", _StubWindow)
    monkeypatch.setattr(desktop_module.pyglet.text, "Label", _StubLabel)
    monkeypatch.setattr(desktop_module.pyglet.shapes, "Rectangle", _StubLabel)
    monkeypatch.setattr(
        desktop_module.pyglet.image.Texture,
        "create",
        lambda width, height, fmt: (
            created_textures.append(_StubTexture(width, height))
            or created_textures[-1]
        ),
    )
    monkeypatch.setattr(desktop_module.pyglet.sprite, "Sprite", _StubSprite)

    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        )
    )

    viewer._draw_frame(np.zeros((20, 10, 3), dtype=np.uint8))
    first_texture = created_textures[-1]
    viewer._draw_frame(np.zeros((20, 10, 3), dtype=np.uint8))
    viewer._draw_frame(np.zeros((10, 5, 3), dtype=np.uint8))

    assert len(created_textures) == 2
    assert first_texture.deleted is True
    assert created_textures[-1].deleted is False


def test_desktop_viewer_keeps_stable_scale_with_reused_sprite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marimo_3dv.viewer.desktop as desktop_module

    created_textures: list[_StubTexture] = []

    monkeypatch.setattr(desktop_module.pyglet.window, "Window", _StubWindow)
    monkeypatch.setattr(desktop_module.pyglet.text, "Label", _StubLabel)
    monkeypatch.setattr(desktop_module.pyglet.shapes, "Rectangle", _StubLabel)
    monkeypatch.setattr(
        desktop_module.pyglet.image.Texture,
        "create",
        lambda width, height, fmt: (
            created_textures.append(_StubTexture(width, height))
            or created_textures[-1]
        ),
    )
    monkeypatch.setattr(desktop_module.pyglet.sprite, "Sprite", _StubSprite)

    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        )
    )
    monkeypatch.setattr(
        viewer,
        "_get_framebuffer_size",
        lambda: (20, 40),
    )

    viewer._draw_frame(np.zeros((20, 10, 3), dtype=np.uint8))
    first_scale = (viewer._frame_sprite.scale_x, viewer._frame_sprite.scale_y)
    viewer._draw_frame(np.zeros((20, 10, 3), dtype=np.uint8))
    second_scale = (viewer._frame_sprite.scale_x, viewer._frame_sprite.scale_y)

    assert first_scale == (2.0, 2.0)
    assert second_scale == (2.0, 2.0)
