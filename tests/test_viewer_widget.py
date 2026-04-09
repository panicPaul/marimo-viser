from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest
import torch
from PIL import Image

from marimo_viser import CameraState, ViewerClick, native_viewer
from marimo_viser.viewer_widget import (
    _convert_cam_to_world_between_conventions,
    _LatestOnlyRenderer,
    _normalize_frame,
)


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 2.0,
    interval: float = 0.01,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("Timed out waiting for condition.")


def test_camera_state_json_round_trip() -> None:
    state = CameraState.default(
        width=320,
        height=240,
        fov_degrees=75.0,
    )

    restored = CameraState.from_json(state.to_json())

    assert restored.width == 320
    assert restored.height == 240
    assert restored.fov_degrees == 75.0
    assert restored.camera_convention == "opencv"
    assert np.allclose(restored.cam_to_world, state.cam_to_world)


@pytest.mark.parametrize(
    "camera_convention",
    ["opencv", "opengl", "blender", "colmap"],
)
def test_camera_state_round_trips_supported_camera_conventions(
    camera_convention: str,
) -> None:
    state = CameraState(
        fov_degrees=60.0,
        width=32,
        height=24,
        cam_to_world=np.eye(4, dtype=np.float64),
        camera_convention=camera_convention,  # type: ignore[arg-type]
    )

    restored = CameraState.from_json(state.to_json())

    assert restored.camera_convention == camera_convention


@pytest.mark.parametrize(
    "camera_convention",
    ["opencv", "opengl", "blender", "colmap"],
)
def test_camera_convention_transform_round_trips(
    camera_convention: str,
) -> None:
    source = CameraState.default(
        width=48,
        height=32,
        camera_convention="opencv",
    ).cam_to_world

    converted = _convert_cam_to_world_between_conventions(
        source,
        source_convention="opencv",
        target_convention=camera_convention,  # type: ignore[arg-type]
    )
    round_tripped = _convert_cam_to_world_between_conventions(
        converted,
        source_convention=camera_convention,  # type: ignore[arg-type]
        target_convention="opencv",
    )

    assert np.allclose(round_tripped, source)


def test_camera_state_rejects_unknown_camera_convention() -> None:
    with pytest.raises(ValueError, match="camera_convention must be one of"):
        CameraState(
            fov_degrees=60.0,
            width=32,
            height=24,
            cam_to_world=np.eye(4, dtype=np.float64),
            camera_convention="unknown",  # type: ignore[arg-type]
        )


def test_viewer_click_json_round_trip() -> None:
    click = ViewerClick(
        x=10,
        y=12,
        width=32,
        height=24,
        camera_state=CameraState.default(width=32, height=24),
    )

    restored = ViewerClick.from_json(click.to_json())

    assert restored.x == 10
    assert restored.y == 12
    assert restored.width == 32
    assert restored.height == 24
    assert restored.camera_state.width == 32
    assert restored.camera_state.height == 24


def test_native_viewer_last_click_reads_synced_widget_state() -> None:
    viewer = native_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8)
    )
    click = ViewerClick(
        x=4,
        y=5,
        width=20,
        height=10,
        camera_state=CameraState.default(width=20, height=10),
    )

    viewer.anywidget().last_click_json = click.to_json()

    assert viewer.last_click is not None
    assert viewer.get_last_click() is not None
    assert viewer.last_click.x == click.x
    assert viewer.last_click.y == click.y
    assert viewer.last_click.width == click.width
    assert viewer.last_click.height == click.height
    assert np.allclose(
        viewer.last_click.camera_state.cam_to_world,
        click.camera_state.cam_to_world,
    )
    assert viewer.get_last_click().x == click.x


def test_native_viewer_get_snapshot_decodes_latest_frame() -> None:
    viewer = native_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8)
    )
    image = Image.new("RGB", (3, 2), color=(12, 34, 56))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    viewer._latest_frame_bytes = buffer.getvalue()

    snapshot = viewer.get_snapshot()

    assert snapshot.size == (3, 2)
    assert snapshot.mode == "RGB"


def test_native_viewer_get_snapshot_requires_available_frame() -> None:
    viewer = native_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8)
    )
    viewer._latest_frame_bytes = None

    with pytest.raises(
        RuntimeError, match="No rendered frame is available yet"
    ):
        viewer.get_snapshot()


def test_normalize_frame_accepts_float_rgb() -> None:
    frame = np.array(
        [[[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]]],
        dtype=np.float32,
    )

    normalized = _normalize_frame(frame)

    assert normalized.dtype == np.uint8
    assert normalized.shape == (1, 2, 3)
    assert normalized.tolist() == [[[0, 127, 255], [255, 63, 0]]]


def test_normalize_frame_accepts_torch_tensor() -> None:
    frame = torch.tensor([[[0.0, 255.0, 32.0]]], dtype=torch.float32)

    normalized = _normalize_frame(frame)

    assert normalized.dtype == np.uint8
    assert normalized.tolist() == [[[0, 255, 32]]]


def test_normalize_frame_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Expected frame shape"):
        _normalize_frame(np.zeros((4, 4), dtype=np.uint8))


def test_latest_only_renderer_drops_stale_results() -> None:
    started_first = threading.Event()
    release_first = threading.Event()
    published: list[tuple[int, int]] = []

    def render_fn(camera_state: CameraState) -> np.ndarray:
        if camera_state.width == 10:
            started_first.set()
            assert release_first.wait(timeout=2.0)
        return np.full(
            (camera_state.height, camera_state.width, 3),
            fill_value=camera_state.width,
            dtype=np.uint8,
        )

    def publish_frame(
        revision: int, camera_state: CameraState, frame: np.ndarray
    ) -> None:
        del frame
        published.append((revision, camera_state.width))

    renderer = _LatestOnlyRenderer(
        render_fn=render_fn,
        publish_frame=publish_frame,
        publish_error=lambda revision, message: None,
        set_rendering=lambda value: None,
    )

    renderer.request(1, CameraState.default(width=10, height=4))
    assert started_first.wait(timeout=2.0)
    renderer.request(2, CameraState.default(width=20, height=4))
    release_first.set()

    _wait_until(lambda: published == [(2, 20)])


def test_native_viewer_set_camera_state_updates_widget_state() -> None:
    viewer = native_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8),
        initial_view=CameraState.default(width=64, height=48),
    )
    updated = CameraState.default(width=32, height=24, fov_degrees=45.0)

    viewer.set_camera_state(updated)

    assert viewer.camera_state.width == 32
    assert viewer.camera_state.height == 24
    assert viewer.camera_state.fov_degrees == 45.0


def test_native_viewer_exposes_configured_aspect_ratio() -> None:
    viewer = native_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8),
        aspect_ratio=2.0,
    )

    assert viewer.anywidget().aspect_ratio == 2.0


def test_native_viewer_render_errors_surface_in_widget_state() -> None:
    viewer = native_viewer(
        lambda state: (_ for _ in ()).throw(RuntimeError("boom")),
        initial_view=CameraState.default(width=40, height=30),
    )

    viewer.rerender()

    _wait_until(lambda: "RuntimeError: boom" in viewer.anywidget().error_text)


def test_native_viewer_rejects_non_positive_aspect_ratio() -> None:
    with pytest.raises(ValueError, match="aspect_ratio must be positive"):
        native_viewer(
            lambda state: np.zeros(
                (state.height, state.width, 3), dtype=np.uint8
            ),
            aspect_ratio=0.0,
        )
