"""Spherical harmonics demo notebook."""

import marimo

__generated_with = "0.22.4"
app = marimo.App(width="columns")

with app.setup:
    import torch
    import torch.nn.functional as F
    from jaxtyping import Float
    from rich import print
    from torch import Tensor


@app.function
def spherical_harmonics(
    dirs: Float[Tensor, "*batch 3"],
    coefficients: Float[Tensor, "*batch num_bases 3"],
    degrees_to_use: int,
) -> Float[Tensor, "*batch 3"]:
    """Evaluate spherical harmonics at unit directions using the efficient method.

    "Efficient Spherical Harmonic Evaluation", Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/

    Args:
        dirs: Unit direction vectors. Will be normalized internally.
        coefficients: SH coefficients per basis and color channel.
        degrees_to_use: Maximum SH degree to evaluate (0-4).

    Returns:
        Color values computed from SH evaluation.
    """
    assert 0 <= degrees_to_use <= 4, (
        f"Only degrees 0-4 supported, got {degrees_to_use}"
    )
    num_bases = (degrees_to_use + 1) ** 2
    assert num_bases <= coefficients.shape[-2], (
        f"Need at least {num_bases} SH bases, got {coefficients.shape[-2]}"
    )

    dirs = F.normalize(dirs, p=2, dim=-1)
    x, y, z = dirs.unbind(-1)

    basis_values = torch.zeros(
        (*dirs.shape[:-1], coefficients.shape[-2]),
        dtype=dirs.dtype,
        device=dirs.device,
    )

    # Degree 0
    basis_values[..., 0] = 0.2820947917738781

    if degrees_to_use == 0:
        return (basis_values[..., None] * coefficients).sum(dim=-2)

    # Degree 1
    basis_values[..., 1] = 0.48860251190292 * y
    basis_values[..., 2] = -0.48860251190292 * z
    basis_values[..., 3] = (
        -0.48860251190292 * x
    )  # negated by convention, same as original

    if degrees_to_use == 1:
        return (basis_values[..., None] * coefficients).sum(dim=-2)

    # Degree 2
    z2 = z * z
    cos_2_azimuth = x * x - y * y  # cos(2φ) * sin²θ component
    sin_2_azimuth = 2.0 * x * y  # sin(2φ) * sin²θ component

    basis_values[..., 4] = 0.5462742152960395 * sin_2_azimuth
    basis_values[..., 5] = -1.092548430592079 * z * y
    basis_values[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    basis_values[..., 7] = -1.092548430592079 * z * x
    basis_values[..., 8] = 0.5462742152960395 * cos_2_azimuth

    if degrees_to_use == 2:
        return (basis_values[..., None] * coefficients).sum(dim=-2)

    # Degree 3
    cos_3_azimuth = x * cos_2_azimuth - y * sin_2_azimuth
    sin_3_azimuth = x * sin_2_azimuth + y * cos_2_azimuth

    basis_values[..., 9] = -0.5900435899266435 * sin_3_azimuth
    basis_values[..., 10] = 1.445305721320277 * z * sin_2_azimuth
    basis_values[..., 11] = (-2.285228997322329 * z2 + 0.4570457994644658) * y
    basis_values[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    basis_values[..., 13] = (-2.285228997322329 * z2 + 0.4570457994644658) * x
    basis_values[..., 14] = 1.445305721320277 * z * cos_2_azimuth
    basis_values[..., 15] = -0.5900435899266435 * cos_3_azimuth

    if degrees_to_use == 3:
        return (basis_values[..., None] * coefficients).sum(dim=-2)

    # Degree 4
    cos_4_azimuth = x * cos_3_azimuth - y * sin_3_azimuth
    sin_4_azimuth = x * sin_3_azimuth + y * cos_3_azimuth

    basis_values[..., 16] = 0.6258357354491763 * sin_4_azimuth
    basis_values[..., 17] = -1.770130769779931 * z * sin_3_azimuth
    basis_values[..., 18] = (
        3.31161143515146 * z2 - 0.47308734787878
    ) * sin_2_azimuth
    basis_values[..., 19] = (
        z * (-4.683325804901025 * z2 + 2.007139630671868) * y
    )
    basis_values[..., 20] = 1.984313483298443 * z2 * (
        1.865881662950577 * z2 - 1.119528997770346
    ) - 1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    basis_values[..., 21] = (
        z * (-4.683325804901025 * z2 + 2.007139630671868) * x
    )
    basis_values[..., 22] = (
        3.31161143515146 * z2 - 0.47308734787878
    ) * cos_2_azimuth
    basis_values[..., 23] = -1.770130769779931 * z * cos_3_azimuth
    basis_values[..., 24] = 0.6258357354491763 * cos_4_azimuth

    return (basis_values[..., None] * coefficients).sum(dim=-2)


@app.cell
def _():
    return


@app.cell(column=1)
def _(Config):
    from marimo_viser import form_gui

    form_widget = form_gui(Config)
    form_widget
    return (form_widget,)


@app.cell
def _(form_widget):
    config = form_widget.value
    print(config)
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _(INIT_FNS, SharedInitializationConfig):
    from pathlib import Path
    from typing import Annotated, Literal

    from pydantic import BaseModel, ConfigDict, Field, model_validator

    class _ConfigModel(BaseModel):
        """Base config model with strict field validation."""

        model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    class ExponentialDecaySchedulerConfig(_ConfigModel):
        """Configuration for an exponential decay learning-rate scheduler."""

        final_ratio: float = 0.01
        max_steps: int = 30_000

        def gamma(self) -> float:
            """Return the exponential decay factor per optimization step."""
            return self.final_ratio ** (1.0 / self.max_steps)

        def scale_by_factor(self, scale_factor: float) -> None:
            """Scale the scheduler horizon by the provided factor."""
            self.max_steps = max(1, round(self.max_steps * scale_factor))

    class OptimizationConfig(_ConfigModel):
        """Optimizer and scheduler configuration for vanilla 3DGS."""

        means_lr: Annotated[
            float,
            Field(
                gt=0,
                description="The learning rates for the center positions of the primitives.",
            ),
        ] = 1.6e-4
        log_scales_lr: float = 5e-3
        logit_opacities_lr: float = 5e-2
        unnormalized_quats_lr: float = 1e-3
        sh_0_lr: float = 2.5e-3
        sh_N_lr: float = 2.5e-3 / 20.0
        means_scheduler: ExponentialDecaySchedulerConfig = Field(
            default_factory=ExponentialDecaySchedulerConfig
        )

        def scale_to_max_steps(self, new_max_steps: int) -> None:
            """Scale all schedule horizons to a new optimization horizon."""
            if new_max_steps <= 0:
                raise ValueError("new_max_steps must be positive.")
            scale_factor = new_max_steps / self.means_scheduler.max_steps
            self.means_scheduler.scale_by_factor(scale_factor)

    class DensificationConfig(_ConfigModel):
        """Configuration for vanilla 3DGS densification.

        Args:
            enabled: this is a test for the enabled flag
        """

        enabled: bool = True
        reference_training_steps: int = Field(
            30_000,
            description="NOTE: this will be overwritten by another config",
        )
        prune_opacity_threshold: float = 0.005
        image_plane_gradient_magnitude_threshold: float = 0.0002
        duplicate_max_normalized_scale_3d: float = 0.01
        split_max_normalized_radius_2d: float = 0.05
        prune_max_normalized_scale_3d: float = 0.1
        prune_max_normalized_radius_2d: float = 0.15
        screen_space_refinement_stop_iteration: int = 0
        refinement_start_iteration: int = 500
        refinement_stop_iteration: int = 15_000
        opacity_reset_interval: int = 3_000
        refinement_interval: int = 100
        refinement_pause_after_opacity_reset: int = 0
        use_absolute_image_plane_gradients: bool = False
        use_revised_opacity_after_split: bool = False
        verbose: bool = False

        def scale_to_max_steps(self, new_max_steps: int) -> None:
            """Scale densification cadence values to a new training horizon."""
            if new_max_steps <= 0:
                raise ValueError("new_max_steps must be positive.")
            scale_factor = new_max_steps / self.reference_training_steps
            self.refinement_start_iteration = max(
                1, round(self.refinement_start_iteration * scale_factor)
            )
            self.refinement_stop_iteration = max(
                1, round(self.refinement_stop_iteration * scale_factor)
            )
            self.opacity_reset_interval = max(
                1, round(self.opacity_reset_interval * scale_factor)
            )
            self.refinement_interval = max(
                1, round(self.refinement_interval * scale_factor)
            )
            self.refinement_pause_after_opacity_reset = max(
                0,
                round(self.refinement_pause_after_opacity_reset * scale_factor),
            )
            self.screen_space_refinement_stop_iteration = max(
                0,
                round(
                    self.screen_space_refinement_stop_iteration * scale_factor
                ),
            )
            self.reference_training_steps = new_max_steps

    class RandomInitializationConfig(_ConfigModel):
        """Configuration for random point-based initialization."""

        init_num_points: int = 100_000
        init_extent: float = 3.0
        init_opacity: float = 0.1
        init_scale: float = 1.0

    class PointCloudInitializationConfig(_ConfigModel):
        """Configuration for sparse point-cloud initialization."""

        init_opacity: float = 0.1
        init_scale: float = 1.0

    class CheckpointInitializationConfig(_ConfigModel):
        """Configuration for checkpoint-based initialization."""

        checkpoint_path: Path

    ValidInitializationConfig = (
        RandomInitializationConfig
        | PointCloudInitializationConfig
        | CheckpointInitializationConfig
    )

    class InitializationConfig(_ConfigModel):
        """Selected initialization method and its corresponding config."""

        method: Literal["random", "point_cloud", "checkpoint"] = "point_cloud"
        config: ValidInitializationConfig = Field(
            default_factory=PointCloudInitializationConfig
        )

        @model_validator(mode="after")
        def validate_method_and_config(self) -> "InitializationConfig":
            """Ensure the selected method is registered and matches the config."""
            if self.method not in INIT_FNS:
                raise ValueError(
                    f"Initialization method {self.method!r} is not registered."
                )

            expected_config_types: dict[
                str, type[ValidInitializationConfig]
            ] = {
                "random": RandomInitializationConfig,
                "point_cloud": PointCloudInitializationConfig,
                "checkpoint": CheckpointInitializationConfig,
            }
            expected_config_type = expected_config_types[self.method]
            if not isinstance(self.config, expected_config_type):
                raise ValueError(
                    f"Initialization method {self.method!r} requires "
                    f"{expected_config_type.__name__}, got "
                    f"{type(self.config).__name__}."
                )
            return self

        def to_shared_initialization_config(self) -> SharedInitializationConfig:
            """Convert the example initialization wrapper to the shared init config."""
            return SharedInitializationConfig(
                strategy=self.method,
                **self.config.model_dump(),
            )

    class TrainingConfig(_ConfigModel):
        """Training-loop configuration for the vanilla 3DGS example."""

        result_dir: Path = Path("results/vanilla_3dgs")
        seed: int = 42  # hello from the other side
        device: str = "cuda"
        max_steps: int = 30_000
        batch_size: int = 1
        num_workers: int = 4
        pin_memory: bool = True
        persistent_workers: bool = True
        log_every: int = 100
        eval_every: int = 1_000
        save_at_steps: list[int] = Field(default_factory=list)

        @model_validator(mode="after")
        def validate_training_config(self) -> "TrainingConfig":
            """Validate the training-loop configuration."""
            if self.max_steps <= 0:
                raise ValueError("max_steps must be positive.")
            if self.batch_size <= 0:
                raise ValueError("batch_size must be positive.")
            if self.num_workers < 0:
                raise ValueError("num_workers must be non-negative.")
            if self.log_every <= 0:
                raise ValueError("log_every must be positive.")
            if self.eval_every <= 0:
                raise ValueError("eval_every must be positive.")
            if any(save_step <= 0 for save_step in self.save_at_steps):
                raise ValueError(
                    "save_at_steps must contain only positive steps."
                )
            self.save_at_steps = sorted(set(self.save_at_steps))
            return self

    class Config(_ConfigModel):
        """Top-level configuration for the vanilla 3DGS example.

        Args:
            test_literal: Another test for the helptext generation
        """

        # data: ColmapSourceConfig
        # train_dataset: DatasetConfig = Field(
        #     default_factory=lambda: DatasetConfig(split="train")
        # )
        # val_dataset: DatasetConfig = Field(
        #     default_factory=lambda: DatasetConfig(split="val")
        # )
        # init: InitializationConfig = Field(default_factory=InitializationConfig)
        densification: DensificationConfig = Field(
            default_factory=DensificationConfig
        )
        optimization: OptimizationConfig = Field(
            default_factory=OptimizationConfig
        )
        training: TrainingConfig = Field(default_factory=TrainingConfig)
        test_flag: bool = Field(False, description="this is a test")
        test_literal: Literal["option_a", "option_b"]
        test_bar: Annotated[float, Field(ge=0, le=1, description="blah")]

        @model_validator(mode="after")
        def scale_configs_to_training_horizon(self) -> "Config":
            """Scale step-based configs to the configured training horizon."""
            self.optimization.scale_to_max_steps(self.training.max_steps)
            self.densification.scale_to_max_steps(self.training.max_steps)
            return self

    return (Config,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
