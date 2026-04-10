"""Tests for GuiPipeline, GuiOp, RenderResult, and stage ordering."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from marimo_3dv.pipeline.context import ViewerContext
from marimo_3dv.pipeline.gui import GuiOp, GuiPipeline, RenderResult
from marimo_3dv.viewer.widget import NativeViewerState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class EmptyConfig(BaseModel):
    pass


def _make_viewer_state() -> NativeViewerState:
    return NativeViewerState()


def _dummy_image(h: int = 4, w: int = 4) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_op(
    name: str,
    stage: str,
    log: list[str],
    *,
    config_model: type[BaseModel] = EmptyConfig,
    default_config: BaseModel | None = None,
    mutate_render_data: bool = False,
    mutate_result_image: bool = False,
) -> GuiOp:
    def hook(data_or_result, config, context, runtime_state):
        log.append(name)
        if stage == "prepare_render" and mutate_render_data:
            return data_or_result + [name]
        if (
            stage in ("post_render_metadata", "image_overlay")
            and mutate_result_image
        ):
            img = data_or_result.image.copy()
            return RenderResult(image=img, metadata=data_or_result.metadata)
        return data_or_result

    return GuiOp(
        name=name,
        config_model=config_model,
        default_config=default_config or config_model(),
        stage=stage,
        hook=hook,
    )


def _make_backend(render_count: list[int], image: np.ndarray):
    def backend_fn(camera_state, render_data):
        render_count.append(1)
        return RenderResult(image=image.copy(), metadata={})

    return backend_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_pipeline_builds():
    pipeline = GuiPipeline()
    result = pipeline.build(render_data=None, viewer_state=_make_viewer_state())
    assert result is not None


def test_single_op_config_exposed():
    class MyConfig(BaseModel):
        threshold: float = 0.5

    op = GuiOp(
        name="my_op",
        config_model=MyConfig,
        default_config=MyConfig(),
        stage="prepare_render",
        hook=lambda data, cfg, ctx, rs: data,
    )
    result = GuiPipeline().pipe(op).build(None, _make_viewer_state())
    assert "threshold" in result.config_model.model_fields
    assert result.default_config.threshold == 0.5


def test_duplicate_field_name_raises():
    class A(BaseModel):
        value: int = 1

    class B(BaseModel):
        value: int = 2

    op_a = GuiOp(
        name="a",
        config_model=A,
        default_config=A(),
        stage="prepare_render",
        hook=lambda *a: a[0],
    )
    op_b = GuiOp(
        name="b",
        config_model=B,
        default_config=B(),
        stage="image_overlay",
        hook=lambda *a: a[0],
    )

    with pytest.raises(ValueError, match="value"):
        GuiPipeline().pipe(op_a).pipe(op_b).build(None, _make_viewer_state())


def test_stages_run_in_correct_order():
    log: list[str] = []
    ops = [
        _make_op("overlay", "image_overlay", log),
        _make_op("prepare", "prepare_render", log),
        _make_op("post", "post_render_metadata", log),
    ]
    pipeline = GuiPipeline()
    for op in ops:
        pipeline = pipeline.pipe(op)

    result = pipeline.build(render_data=[], viewer_state=_make_viewer_state())

    render_count: list[int] = []
    image = _dummy_image()
    backend = _make_backend(render_count, image)

    config = result.default_config
    render_fn = result.bind(config, backend_fn=backend)

    from marimo_3dv.viewer.widget import CameraState

    cam = CameraState.default()
    render_fn(cam)

    # prepare → backend → post → overlay
    assert log == ["prepare", "post", "overlay"]


def test_exactly_one_backend_render_per_frame():
    render_count: list[int] = []
    pipeline = GuiPipeline()
    result = pipeline.build(render_data=None, viewer_state=_make_viewer_state())

    image = _dummy_image()
    backend = _make_backend(render_count, image)

    from marimo_3dv.viewer.widget import CameraState

    cam = CameraState.default()
    render_fn = result.bind(result.default_config, backend_fn=backend)

    for _ in range(5):
        render_fn(cam)

    assert render_count == [1, 1, 1, 1, 1]


def test_runtime_state_persists_across_renders():
    @dataclass
    class CounterState:
        count: int = 0

    def hook(result, config, context, runtime_state):
        runtime_state.count += 1
        return result

    op = GuiOp(
        name="counter",
        config_model=EmptyConfig,
        default_config=EmptyConfig(),
        stage="image_overlay",
        hook=hook,
        runtime_state_factory=CounterState,
    )
    result = GuiPipeline().pipe(op).build(None, _make_viewer_state())

    image = _dummy_image()
    backend = _make_backend([], image)

    from marimo_3dv.viewer.widget import CameraState

    cam = CameraState.default()
    render_fn = result.bind(result.default_config, backend_fn=backend)

    for _ in range(3):
        render_fn(cam)

    assert result.runtime_state["counter"].count == 3


def test_viewer_context_carries_viewer_state():
    captured: list[ViewerContext] = []

    def hook(result, config, context, runtime_state):
        captured.append(context)
        return result

    op = GuiOp(
        name="ctx_reader",
        config_model=EmptyConfig,
        default_config=EmptyConfig(),
        stage="image_overlay",
        hook=hook,
    )
    viewer_state = _make_viewer_state()
    result = GuiPipeline().pipe(op).build(None, viewer_state)

    image = _dummy_image()
    backend = _make_backend([], image)

    from marimo_3dv.viewer.widget import CameraState

    cam = CameraState.default()
    render_fn = result.bind(result.default_config, backend_fn=backend)
    render_fn(cam)

    assert len(captured) == 1
    assert captured[0].viewer_state is viewer_state


def test_prepare_render_can_modify_render_data():
    log: list[str] = []
    received_data: list[Any] = []

    def prepare_hook(render_data, config, context, runtime_state):
        log.append("prepare")
        return render_data + ["modified"]

    def backend(camera_state, render_data):
        received_data.append(render_data)
        return RenderResult(image=_dummy_image(), metadata={})

    op = GuiOp(
        name="modifier",
        config_model=EmptyConfig,
        default_config=EmptyConfig(),
        stage="prepare_render",
        hook=prepare_hook,
    )
    result = (
        GuiPipeline()
        .pipe(op)
        .build(render_data=[], viewer_state=_make_viewer_state())
    )

    from marimo_3dv.viewer.widget import CameraState

    cam = CameraState.default()
    render_fn = result.bind(result.default_config, backend_fn=backend)
    render_fn(cam)

    assert received_data[0] == ["modified"]
