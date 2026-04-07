import marimo

__generated_with = "0.22.4"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo

    import torch
    from jaxtyping import UInt8
    from marimo_viser.viser_widget import viser_marimo
    import torch.nn.functional as F
    from torch import Tensor
    from jaxtyping import Float


@app.function
def spherical_harmonics(
    dirs: Float[Tensor, "*batch 3"],
    coefficients: Float[Tensor, "*batch num_bases 3"],
    degrees_to_use: int,
) -> Float[Tensor, "*batch 3"]:
    """
    Evaluate spherical harmonics at unit directions using the efficient method from:
    "Efficient Spherical Harmonic Evaluation", Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/

    Args:
        dirs: Unit direction vectors. Will be normalized internally.
        coefficients: SH coefficients per basis and color channel.
        degrees_to_use: Maximum SH degree to evaluate (0-4).

    Returns:
        Color values computed from SH evaluation.
    """
    assert 0 <= degrees_to_use <= 4, f"Only degrees 0-4 supported, got {degrees_to_use}"
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
    basis_values[..., 3] = -0.48860251190292 * x  # negated by convention, same as original

    if degrees_to_use == 1:
        return (basis_values[..., None] * coefficients).sum(dim=-2)

    # Degree 2
    z2 = z * z
    cos_2_azimuth = x * x - y * y   # cos(2φ) * sin²θ component
    sin_2_azimuth = 2.0 * x * y     # sin(2φ) * sin²θ component

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

    basis_values[..., 9]  = -0.5900435899266435 * sin_3_azimuth
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
    basis_values[..., 18] = (3.31161143515146 * z2 - 0.47308734787878) * sin_2_azimuth
    basis_values[..., 19] = z * (-4.683325804901025 * z2 + 2.007139630671868) * y
    basis_values[..., 20] = (
        1.984313483298443 * z2 * (1.865881662950577 * z2 - 1.119528997770346)
        - 1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    )
    basis_values[..., 21] = z * (-4.683325804901025 * z2 + 2.007139630671868) * x
    basis_values[..., 22] = (3.31161143515146 * z2 - 0.47308734787878) * cos_2_azimuth
    basis_values[..., 23] = -1.770130769779931 * z * cos_3_azimuth
    basis_values[..., 24] = 0.6258357354491763 * cos_4_azimuth

    return (basis_values[..., None] * coefficients).sum(dim=-2)


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    return


if __name__ == "__main__":
    app.run()
