from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import pytest
import torch

from marimo_3dv import (
    CameraState,
    ViewerClick,
    ViewerState,
)
from marimo_3dv.viewer.widget import (
    _convert_cam_to_world_between_conventions,
    _LatestOnlyRenderer,
    _normalize_frame,
    marimo_viewer,
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


def test_default_camera_state_uses_proper_rotation_matrix() -> None:
    rotation = CameraState.default().cam_to_world[:3, :3]

    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert np.allclose(rotation[:, 1], np.array([0.0, 1.0, 0.0]))


def test_viewer_state_defaults_show_axes_and_hides_guides() -> None:
    state = ViewerState()

    assert state.show_axes is True
    assert state.show_horizon is False
    assert state.show_origin is False
    assert state.show_stats is False


def test_viewer_state_overlay_setters_are_fluent() -> None:
    state = ViewerState()

    chained_state = (
        state.set_show_axes(False)
        .set_show_origin(True)
        .set_show_horizon(True)
        .set_show_stats(True)
        .set_origin(1.0, 2.0, 3.0)
    )

    assert chained_state is state
    assert state.show_axes is False
    assert state.show_origin is True
    assert state.show_horizon is True
    assert state.show_stats is True
    assert state.origin == (1.0, 2.0, 3.0)


def test_camera_state_with_convention_round_trips() -> None:
    state = CameraState.default(width=48, height=32, camera_convention="opencv")

    converted = state.with_convention("opengl")
    restored = converted.with_convention("opencv")

    assert converted.camera_convention == "opengl"
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


def test_viewer_last_click_reads_synced_widget_state() -> None:
    viewer = marimo_viewer(
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

    assert viewer.get_last_click() is not None
    assert viewer.get_last_click().x == click.x
    assert viewer.get_last_click().y == click.y
    assert viewer.get_last_click().width == click.width
    assert viewer.get_last_click().height == click.height
    assert np.allclose(
        viewer.get_last_click().camera_state.cam_to_world,
        click.camera_state.cam_to_world,
    )
    assert viewer.get_last_click().x == click.x


def test_viewer_get_snapshot_decodes_latest_frame() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8)
    )
    viewer._latest_frame_array = np.full(
        (2, 3, 3),
        fill_value=np.array([12, 34, 56], dtype=np.uint8),
        dtype=np.uint8,
    )

    snapshot = viewer.get_snapshot()

    assert snapshot.size == (3, 2)
    assert snapshot.mode == "RGB"


def test_viewer_get_snapshot_requires_available_frame() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8)
    )
    viewer._latest_frame_array = None

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
        revision: int,
        camera_state: CameraState,
        frame: np.ndarray,
        render_queue_time_ms: float,
        render_time_ms: float,
        interaction_active: bool,
    ) -> None:
        del frame
        assert render_queue_time_ms >= 0.0
        assert render_time_ms >= 0.0
        assert isinstance(interaction_active, bool)
        published.append((revision, camera_state.width))

    renderer = _LatestOnlyRenderer(
        render_fn=render_fn,
        publish_frame=publish_frame,
        publish_error=lambda revision, message: None,
        set_rendering=lambda value: None,
    )

    renderer.request(1, CameraState.default(width=10, height=4), True)
    assert started_first.wait(timeout=2.0)
    renderer.request(2, CameraState.default(width=20, height=4), False)
    release_first.set()

    _wait_until(lambda: published == [(2, 20)])


def test_marimo_viewer_set_camera_state_updates_widget_state() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8),
        initial_view=CameraState.default(width=64, height=48),
    )
    updated = CameraState.default(width=32, height=24, fov_degrees=45.0)

    viewer.set_camera_state(updated)

    assert viewer.get_camera_state().width == 32
    assert viewer.get_camera_state().height == 24
    assert viewer.get_camera_state().fov_degrees == 45.0


def test_marimo_viewer_reuses_explicit_state_across_reruns() -> None:
    state = ViewerState(camera_state=CameraState.default(width=64, height=48))
    first_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
    )
    updated_camera_state = CameraState.default(
        width=32, height=24, fov_degrees=45.0
    )

    first_viewer.set_camera_state(updated_camera_state)

    second_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
    )

    assert second_viewer.get_camera_state().width == 32
    assert second_viewer.get_camera_state().height == 24
    assert second_viewer.get_camera_state().fov_degrees == 45.0


def test_viewer_state_can_reset_camera_to_initial_value() -> None:
    initial = CameraState.default(width=64, height=48, fov_degrees=60.0)
    state = ViewerState(camera_state=initial)
    viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
    )
    viewer.set_camera_state(
        CameraState.default(width=32, height=24, fov_degrees=45.0)
    )

    state.reset_camera()

    assert viewer.get_camera_state().width == 64
    assert viewer.get_camera_state().height == 48
    assert viewer.get_camera_state().fov_degrees == 60.0
    assert state.camera_state.width == 64
    assert state.camera_state.height == 48
    assert state.camera_state.fov_degrees == 60.0


def test_marimo_viewer_reuses_show_axes_from_explicit_state() -> None:
    state = ViewerState(show_axes=True)

    first_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
    )
    assert first_viewer.anywidget().show_axes is True

    state.show_axes = False
    second_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
    )

    assert second_viewer.anywidget().show_axes is False


def test_marimo_viewer_get_debug_info_reads_synced_metrics() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8)
    )
    widget = viewer.anywidget()
    widget.error_text = "boom"
    widget.latency_ms = 12.5
    widget.latency_sample_ms = 11.0
    widget.render_time_ms = 1.25
    widget.render_queue_time_ms = 3.5
    widget.encode_time_ms = 0.75
    widget.stream_queue_time_ms = 0.5
    widget.stream_send_time_ms = 1.0
    widget.backend_to_browser_time_ms = 3.0
    widget.packet_size_bytes = 12345
    widget.browser_receive_queue_ms = 4.0
    widget.browser_post_receive_ms = 14.0
    widget.browser_decode_time_ms = 2.0
    widget.browser_draw_time_ms = 0.25
    widget.browser_present_wait_ms = 8.5

    assert viewer.get_debug_info() == {
        "error_text": "boom",
        "latency_ms": 12.5,
        "latency_sample_ms": 11.0,
        "render_time_ms": 1.25,
        "render_queue_time_ms": 3.5,
        "encode_time_ms": 0.75,
        "stream_queue_time_ms": 0.5,
        "stream_send_time_ms": 1.0,
        "backend_to_browser_time_ms": 3.0,
        "packet_size_bytes": 12345,
        "browser_receive_queue_ms": 4.0,
        "browser_post_receive_ms": 14.0,
        "browser_decode_time_ms": 2.0,
        "browser_draw_time_ms": 0.25,
        "browser_present_wait_ms": 8.5,
        "accounted_leaf_latency_ms": 21.75,
        "unaccounted_leaf_latency_ms": -9.25,
        "unaccounted_leaf_latency_sample_ms": -10.75,
        "accounted_coarse_latency_ms": 25.0,
        "unaccounted_coarse_latency_ms": -12.5,
        "unaccounted_coarse_latency_sample_ms": -14.0,
    }


def test_marimo_viewer_exposes_configured_aspect_ratio() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8),
        state=ViewerState(aspect_ratio=2.0),
    )

    assert viewer.anywidget().aspect_ratio == 2.0


def test_marimo_viewer_uses_requested_default_camera_convention() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8),
        state=ViewerState(camera_convention="opengl"),
    )

    assert viewer.get_camera_state().camera_convention == "opengl"


def test_marimo_viewer_caps_motion_render_larger_axis_only() -> None:
    rendered_sizes: list[tuple[int, int, bool]] = []
    viewer = marimo_viewer(
        lambda state: (
            rendered_sizes.append((state.width, state.height, True))
            or np.zeros((state.height, state.width, 3), dtype=np.uint8)
        ),
        initial_view=CameraState.default(width=100, height=80),
        state=ViewerState(interactive_max_side=50),
    )
    rendered_sizes.clear()
    viewer.anywidget().interaction_active = True

    viewer.rerender()
    _wait_until(lambda: len(rendered_sizes) >= 1)

    viewer.anywidget().interaction_active = False
    viewer.rerender()
    _wait_until(lambda: len(rendered_sizes) >= 2)

    assert rendered_sizes[0][:2] == (50, 40)
    assert rendered_sizes[1][:2] == (100, 80)
    assert viewer.get_camera_state().width == 100
    assert viewer.get_camera_state().height == 80


def test_marimo_viewer_caps_settled_render_larger_axis_with_internal_limit() -> (
    None
):
    rendered_sizes: list[tuple[int, int]] = []
    viewer = marimo_viewer(
        lambda state: (
            rendered_sizes.append((state.width, state.height))
            or np.zeros((state.height, state.width, 3), dtype=np.uint8)
        ),
        initial_view=CameraState.default(width=160, height=80),
        state=ViewerState(internal_render_max_side=100),
    )

    assert rendered_sizes[0] == (100, 50)
    assert viewer.get_camera_state().width == 160
    assert viewer.get_camera_state().height == 80


def test_marimo_viewer_render_errors_raise_by_default() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        marimo_viewer(
            lambda state: (_ for _ in ()).throw(RuntimeError("boom")),
            initial_view=CameraState.default(width=40, height=30),
        )


def test_marimo_viewer_can_surface_render_errors_in_widget_state() -> None:
    viewer = marimo_viewer(
        lambda state: (_ for _ in ()).throw(RuntimeError("boom")),
        initial_view=CameraState.default(width=40, height=30),
        state=ViewerState(raise_on_error=False),
    )

    viewer.rerender()

    _wait_until(lambda: "RuntimeError: boom" in viewer.anywidget().error_text)


def test_viewer_state_rejects_non_positive_aspect_ratio() -> None:
    with pytest.raises(ValueError, match="aspect_ratio must be positive"):
        ViewerState(aspect_ratio=0.0)


def test_viewer_state_rejects_out_of_range_interactive_quality() -> None:
    with pytest.raises(ValueError, match="interactive_quality must be in"):
        ViewerState(interactive_quality=0)


def test_viewer_state_rejects_non_positive_interactive_max_side() -> None:
    with pytest.raises(
        ValueError, match="interactive_max_side must be None or a positive"
    ):
        ViewerState(interactive_max_side=0)


def test_viewer_state_rejects_non_positive_internal_render_max_side() -> None:
    with pytest.raises(
        ValueError, match="internal_render_max_side must be None or a positive"
    ):
        ViewerState(internal_render_max_side=0)


def test_viewer_state_accepts_none_interactive_max_side() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8),
        state=ViewerState(interactive_max_side=None),
    )

    assert viewer is not None
