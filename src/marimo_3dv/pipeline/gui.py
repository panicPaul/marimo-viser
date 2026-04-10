"""Typed GUI pipeline for composing render ops around a single backend render.

Stage model (exactly one backend render per frame):

1. ``prepare_render``   — modify backend inputs before the backend renders.
2. ``backend_render``   — the single actual scene render (owned by the caller).
3. ``post_render_metadata`` — consume backend outputs beyond the final image.
4. ``image_overlay``    — draw on top of the already-rendered image.

Each GuiOp hooks into exactly one stage. All ops compose around the single
backend render; no op may trigger a second backend render.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from pydantic import BaseModel

from marimo_3dv.pipeline.context import ViewerContext
from marimo_3dv.viewer.widget import CameraState, NativeViewerState

RenderDataT = TypeVar("RenderDataT")

PipelineStage = Literal[
    "prepare_render",
    "post_render_metadata",
    "image_overlay",
]


@dataclass
class RenderResult:
    """Structured output from a backend render.

    Attributes:
        image: Final rendered image as uint8 (H, W, 3) numpy array.
        metadata: Backend-specific extras (e.g. gsplat projected means,
            visibility masks, depth buffers).
    """

    image: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuiOp(Generic[RenderDataT]):
    """A single composable unit in a GuiPipeline.

    Each op contributes a config submodel, optional runtime state, and a
    hook that fires at one pipeline stage.

    Attributes:
        name: Unique name identifying this op in the pipeline.
        config_model: Pydantic model class for this op's configuration.
        default_config: Default instance of ``config_model``.
        stage: Which pipeline stage this op hooks into.
        hook: The op's stage callback (see GuiPipeline.build for signatures).
        runtime_state_factory: Optional factory returning fresh mutable state
            for this op. Called once per ``GuiPipeline.build()`` call.
    """

    name: str
    config_model: type[BaseModel]
    default_config: BaseModel
    stage: PipelineStage
    hook: Callable[..., Any]
    runtime_state_factory: Callable[[], Any] | None = None


class _EmptyConfig(BaseModel):
    """Empty pydantic model for ops that require no configuration."""


_EMPTY_DEFAULT = _EmptyConfig()


@dataclass
class _PipelineRuntimeState(Generic[RenderDataT]):
    """Collected runtime state for all ops in a built pipeline."""

    ops: list[GuiOp[RenderDataT]]
    op_configs: dict[str, type[BaseModel]]
    op_runtime_states: dict[str, Any]
    render_data: RenderDataT
    viewer_state: NativeViewerState


@dataclass
class GuiPipelineResult(Generic[RenderDataT]):
    """Result of building a GuiPipeline.

    Exposes a combined Pydantic config model, default config, and a
    ``render_fn`` that executes all pipeline stages around a caller-supplied
    backend render function.

    Attributes:
        config_model: Combined Pydantic model merging all op configs.
        default_config: Combined default config instance.
        runtime_state: Mutable per-op runtime state keyed by op name.
        render_fn: Full pipeline render function
            ``(CameraState, combined_config, ViewerContext,
               backend_fn) -> RenderResult``.
    """

    config_model: type[BaseModel]
    default_config: BaseModel
    runtime_state: dict[str, Any]
    _pipeline_state: _PipelineRuntimeState[RenderDataT]

    def bind(
        self,
        config: BaseModel,
        backend_fn: Callable[[CameraState, RenderDataT], RenderResult],
    ) -> Callable[[CameraState], RenderResult]:
        """Bind a config and backend to produce a single-argument render callable.

        The returned callable accepts only a ``CameraState`` and runs the full
        pipeline: prepare_render → backend_render → post_render_metadata →
        image_overlay.

        Args:
            config: Combined pipeline config (instance of ``config_model``).
            backend_fn: Callable that performs the actual scene render and
                returns a ``RenderResult``.

        Returns:
            A ``(CameraState) -> RenderResult`` callable ready for the viewer.
        """

        def render(camera_state: CameraState) -> RenderResult:
            viewer_context = ViewerContext(
                viewer_state=self._pipeline_state.viewer_state,
                last_click=self._pipeline_state.viewer_state.last_click,
            )
            return _run_pipeline(
                camera_state=camera_state,
                config=config,
                context=viewer_context,
                backend_fn=backend_fn,
                pipeline_state=self._pipeline_state,
            )

        return render


def _run_pipeline(
    *,
    camera_state: CameraState,
    config: BaseModel,
    context: ViewerContext,
    backend_fn: Callable[[CameraState, RenderDataT], RenderResult],
    pipeline_state: _PipelineRuntimeState[RenderDataT],
) -> RenderResult:
    """Execute all pipeline stages for one frame.

    Exactly one call to ``backend_fn`` occurs per invocation.
    """
    render_data = pipeline_state.render_data
    config_dict = config.model_dump()

    # Stage 1: prepare_render — ops modify backend inputs
    for op in pipeline_state.ops:
        if op.stage != "prepare_render":
            continue
        op_config = _extract_op_config(op, config_dict)
        op_runtime = pipeline_state.op_runtime_states.get(op.name)
        render_data = op.hook(render_data, op_config, context, op_runtime)

    # Stage 2: backend_render — exactly once
    result = backend_fn(camera_state, render_data)

    # Stage 3: post_render_metadata — ops consume backend outputs
    for op in pipeline_state.ops:
        if op.stage != "post_render_metadata":
            continue
        op_config = _extract_op_config(op, config_dict)
        op_runtime = pipeline_state.op_runtime_states.get(op.name)
        result = op.hook(result, op_config, context, op_runtime)

    # Stage 4: image_overlay — ops draw on top of the rendered image
    for op in pipeline_state.ops:
        if op.stage != "image_overlay":
            continue
        op_config = _extract_op_config(op, config_dict)
        op_runtime = pipeline_state.op_runtime_states.get(op.name)
        result = op.hook(result, op_config, context, op_runtime)

    return result


def _extract_op_config(
    op: GuiOp[Any], config_dict: dict[str, Any]
) -> BaseModel:
    """Extract this op's config fields from the combined config dict."""
    op_fields = set(op.config_model.model_fields)
    op_data = {k: v for k, v in config_dict.items() if k in op_fields}
    return op.config_model(**op_data)


def _build_combined_config_model(
    ops: list[GuiOp[Any]],
) -> tuple[type[BaseModel], BaseModel]:
    """Merge all op config models into one flat Pydantic model.

    Field names must be unique across all ops.

    Args:
        ops: List of GuiOps in the pipeline.

    Returns:
        ``(CombinedModel, default_instance)`` tuple.

    Raises:
        ValueError: If two ops define a field with the same name.
    """
    from pydantic import create_model

    seen: dict[str, str] = {}
    field_definitions: dict[str, Any] = {}
    defaults: dict[str, Any] = {}

    for op in ops:
        op_defaults = op.default_config.model_dump()
        for field_name, field_info in op.config_model.model_fields.items():
            if field_name in seen:
                raise ValueError(
                    f"Field {field_name!r} is defined by both op "
                    f"{seen[field_name]!r} and op {op.name!r}. "
                    "All op config fields must have unique names."
                )
            seen[field_name] = op.name
            field_definitions[field_name] = (
                field_info.annotation,
                field_info,
            )
            defaults[field_name] = op_defaults.get(field_name)

    combined_model: type[BaseModel] = create_model(
        "PipelineConfig", **field_definitions
    )
    default_instance = combined_model(**defaults)
    return combined_model, default_instance


class GuiPipeline(Generic[RenderDataT]):
    """Typed GUI pipeline that composes render ops around a single backend render.

    Example::

        pipeline = (
            GuiPipeline()
            .pipe(max_sh_degree_op())
            .pipe(filter_opacity_op())
            .pipe(paint_ray_op())
        )
        result = pipeline.build(render_data, viewer_state)
        viewer = native_viewer(
            result.bind(config_gui.value, backend_fn=rasterize),
            state=viewer_state,
        )
    """

    def __init__(self) -> None:
        self._ops: list[GuiOp[RenderDataT]] = []

    def pipe(self, op: GuiOp[RenderDataT]) -> GuiPipeline[RenderDataT]:
        """Append a GuiOp and return a new pipeline.

        Args:
            op: The op to append.

        Returns:
            A new GuiPipeline with the op added.
        """
        next_pipeline: GuiPipeline[RenderDataT] = GuiPipeline()
        next_pipeline._ops = [*self._ops, op]
        return next_pipeline

    def build(
        self,
        render_data: RenderDataT,
        viewer_state: NativeViewerState,
    ) -> GuiPipelineResult[RenderDataT]:
        """Build the pipeline, instantiating runtime state and merging configs.

        Args:
            render_data: The prepared scene render data (output of SetupPipeline).
            viewer_state: Shared viewer state for camera/overlay control.

        Returns:
            A ``GuiPipelineResult`` ready to bind a config and backend render fn.
        """
        combined_model, default_instance = _build_combined_config_model(
            self._ops
        )

        op_runtime_states: dict[str, Any] = {}
        for op in self._ops:
            if op.runtime_state_factory is not None:
                op_runtime_states[op.name] = op.runtime_state_factory()

        pipeline_state = _PipelineRuntimeState(
            ops=self._ops,
            op_configs={op.name: op.config_model for op in self._ops},
            op_runtime_states=op_runtime_states,
            render_data=render_data,
            viewer_state=viewer_state,
        )

        return GuiPipelineResult(
            config_model=combined_model,
            default_config=default_instance,
            runtime_state=op_runtime_states,
            _pipeline_state=pipeline_state,
        )
