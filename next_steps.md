
  # Next Steps

  ## Goal
  Refactor the package into a viewer + pipeline toolkit with:

  - a marimo viewer runtime
  - an offline native desktop viewer runtime
  - a typed setup pipeline
  - a typed GUI/render/tool pipeline
  - a small built-in library of common pipes for 3DGS / 2DGS

  The old `viser_widget` path should be removed, and the package should be renamed now.

  ## Summary
  We want two pipeline layers:

  - `SetupPipeline`: transforms source scene data into reusable render data once.
  - `GuiPipeline`: composes config, runtime tool state, render-stage ops, and overlay ops into one final viewer-compatible renderer.

  The framework remains scene-agnostic. Scene-specific behavior is provided by typed operations. Some ops will be backend-independent, while others depend
  on backend-specific render outputs such as gsplat metadata.

  ## Package Surface
  Rename the package away from `marimo_viser`.

  The new public package should expose only:

  - camera and shared viewer state types
  - marimo viewer entrypoint
  - offline desktop viewer entrypoint
  - pydantic GUI helpers
  - pipeline builders and pipeline result types
  - a curated set of built-in pipes

  Remove entirely:

  - `viser_widget.py`
  - `viser_marimo`
  - `ViserMarimoWidget`

  No backward compatibility layer is required.

  ## Viewer Runtime
  Keep one shared viewer state API across both runtimes.

  Viewer state remains responsible for:

  - camera get/set/reset
  - overlay toggles
  - origin setters
  - viewer rotation setters
  - other renderer-agnostic viewer controls

  Do not put scene-tool-specific mutable state into viewer state.

  Add a native desktop offline viewer that mirrors the current marimo viewer:

  - same `CameraState` render function signature
  - same interaction semantics
  - same overlays and viewer-state methods
  - same error-handling behavior
  - no websocket/browser transfer in the hot path

  Runtime direction:

  - `marimo edit` / notebook mode uses the marimo viewer
  - `marimo run` / script mode uses the desktop viewer
  - introduce the desktop viewer as a separate entrypoint first

  ## Setup Pipeline
  Add a typed `SetupPipeline[InputT, RenderDataT]`.

  Contract:

  - `.pipe(op)` appends one typed setup operation
  - `.run(input_data) -> RenderDataT`

  Operation model:

  - setup ops are plain Python callables
  - each op consumes one typed input and returns the next typed output
  - use `beartype` for runtime compatibility checks
  - fail early on incompatible wiring

  Current scene normalization should move into setup-pipeline ops instead of remaining a top-level public utility surface.

  ## GUI Pipeline
  Add a typed `GuiPipeline[RenderDataT]`.

  Contract:

  - `.pipe(op)` appends one GUI/render/tool operation
  - `.build(render_data, viewer_state) -> GuiPipelineResult`

  `GuiPipelineResult` should expose:

  - `config_model: type[BaseModel]`
  - `default_config: BaseModel`
  - `runtime_state`
  - `render_fn: Callable[[CameraState, BaseModel, ViewerContext], np.ndarray | torch.Tensor]`
  - `bind(config) -> Callable[[CameraState], np.ndarray | torch.Tensor]`

  Each GUI op contributes:

  - `name`
  - a pydantic config submodel
  - default config
  - optional runtime state factory
  - one stage hook

  ## Pipe Interaction Model
  We should define explicit pipeline stages so ops do not accidentally duplicate work or trigger extra renders.

  ### Stage model
  A GUI op may hook into exactly one or more of these stages:

  1. `prepare_render`
  - modifies backend inputs before the backend render
  - examples: `max_sh_degree`, `filter_opacity`, `filter_size`
  - operates on render data / render params
  - must not render the scene itself

  2. `backend_render`
  - the single actual scene render
  - this happens exactly once per frame
  - owned by the backend-specific renderer

  3. `post_render_metadata`
  - consumes backend render outputs beyond the final image
  - examples: gsplat metadata, visibility, projected means, alpha/debug buffers
  - must not trigger a second backend render

  4. `image_overlay`
  - draws on top of the already rendered image
  - examples: `paint_ray`, screen-space distributions, guides, markers
  - should work from existing image plus metadata/runtime state
  - must not re-render the scene

  ### Core rule
  There is exactly one backend render per frame.

  All pipes must compose around that single render pass:
  - pre-render pipes modify inputs
  - backend renders once
  - post-render pipes consume outputs
  - overlay pipes draw on the resulting image

  No pipe should call the backend renderer recursively or independently.

  ### Backend-independent vs backend-dependent ops
  We need two broad classes of GUI ops:

  #### Backend-independent ops
  These work from:
  - `CameraState`
  - render data
  - viewer context
  - final rendered image
  - pipeline runtime state

  Examples:
  - `paint_ray`
  - horizon/origin/axes-style overlays
  - generic world annotations
  - image-space guides

  These should work across backends.

  #### Backend-dependent ops
  These require metadata produced by a specific renderer.

  Examples:
  - on-screen distributions derived from gsplat outputs
  - visibility statistics from gsplat internals
  - splat-projection diagnostics

  These should declare the metadata they require and fail clearly if the backend cannot provide it.

  ## Render Result Contract
  The backend render should return a structured render result, not just an image.

  Proposed shape:

  - `image`
  - `metadata`

  Where:
  - `image` is the final base render
  - `metadata` is a typed or namespaced dict-like payload with backend-specific extras

  This lets backend-dependent ops read gsplat outputs without rerendering.

  Examples of metadata:
  - projected splat statistics
  - visibility masks
  - per-splat screen-space values
  - depth-like buffers if available

  ## Stateful Tools
  The GUI pipeline must support stateful tools such as `paint_ray`.

  State model:
  - mutable tool state lives in pipeline runtime state
  - runtime state is separate from pydantic config
  - runtime state persists across renders for that pipeline instance

  Event model:
  - stateful ops receive a `ViewerContext`
  - ops should not read notebook globals directly

  `ViewerContext` should include:

  - shared viewer state
  - last click data
  - camera access helpers
  - renderer-agnostic viewer metadata

  `paint_ray` behavior:
  - on click, store a world-space ray from the camera origin through the clicked pixel
  - store rays in op runtime state
  - visualize them later from new viewpoints
  - implement as an overlay op, not a separate scene render

  ## Built-In Pipe Surface
  Ship a small core set of built-in pipes in v1.

  ### Setup pipes
  Include:
  - normalization pipes migrated from current scene normalization utilities
  - transform/composition helpers

  ### GUI/render pipes
  Include a focused GS-oriented core:
  - `max_sh_degree`
  - `filter_opacity`
  - `filter_size`
  - `show_distribution`

  These should work for 3DGS / 2DGS when the render-data type and backend outputs support them.

  ## Notebook Authoring
  Notebook authors should be able to define ops inline as plain Python functions.

  Setup op example:

  ```python
  def normalize_scene(scene: SplatScene) -> SplatRenderData: ...

  GUI op example:

  def max_sh_degree_op(render_data: SplatRenderData) -> GuiOp[SplatRenderData]: ...

  Stateful tool example:

  def paint_ray_op(render_data: RenderDataT) -> GuiOp[RenderDataT]: ...

  Do not require decorators or registration systems in v1.
  Provide lightweight helper constructors for common op shapes.

  ## Integration Flow

  ### Marimo notebook

  - render_data = setup_pipeline.run(source_scene)
  - gui_result = gui_pipeline.build(render_data, viewer_state)
  - config_gui = form_gui(gui_result.config_model, value=gui_result.default_config, ...)
  - viewer = native_viewer(gui_result.bind(config_gui.value), state=viewer_state)

  ### Script / marimo run

  - same setup pipeline
  - same GUI pipeline
  - same pydantic config model
  - desktop config UI generated from the same config
  - desktop viewer launched with the same shared viewer state

  ## Tests

  Add tests for:

  - package surface after rename
  - removal of old viser_widget exports
  - SetupPipeline typed composition
  - normalization behavior through setup pipes
  - GuiPipeline config composition
  - deterministic wrapper/stage ordering
  - exactly one backend render per frame
  - backend-independent overlay ops not rerendering
  - backend-dependent ops consuming existing metadata without rerendering
  - runtime state persistence for stateful tools
  - ViewerContext event access
  - paint_ray click-to-world-ray behavior
  - built-in GS pipes
  - offline runtime numpy/torch rendering
  - offline runtime error behavior
  - notebook-style inline op authoring

  ## Defaults and Assumptions

  - the package is renamed now
  - no backward compatibility is required
  - the old viser widget path is removed entirely
  - scene normalization becomes setup-pipeline functionality
  - one shared viewer state API is kept across runtimes
  - scene/tool-specific mutable state belongs to pipeline runtime state
  - stateful tools receive a ViewerContext
  - beartype is used for runtime compatibility checks
  - GUI ops compose around a single backend render per frame
  - backend-dependent ops consume backend metadata instead of triggering rerenders
  - built-in pipes ship as a small curated core set
