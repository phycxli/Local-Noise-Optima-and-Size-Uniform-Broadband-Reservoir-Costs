"""Representative pseudospectrum contour for the local reservoir rapidity block."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from liouvillian_geometry import (  # noqa: E402
    LocalBosonicHatanoParams,
    local_bosonic_hatano_drift,
)


FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
PARAMS = LocalBosonicHatanoParams(t_right=1.0, t_left=0.25, gamma=1.2)


def pbc_symbol(k: np.ndarray, p: LocalBosonicHatanoParams = PARAMS) -> np.ndarray:
    return -p.gamma + p.t_right * np.exp(1j * k) + p.t_left * np.exp(-1j * k)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    n_sites = 60
    x = local_bosonic_hatano_drift(n_sites, PARAMS)
    eigvals = np.linalg.eigvals(x)
    gap = -float(np.max(eigvals.real))

    real_values = np.linspace(-2.35, 0.12, 190)
    imag_values = np.linspace(-0.92, 0.92, 150)
    ident = np.eye(n_sites, dtype=complex)
    log_sigma = np.empty((len(imag_values), len(real_values)), dtype=float)

    for i, imag in enumerate(imag_values):
        for j, real in enumerate(real_values):
            z = real + 1j * imag
            sigma_min = np.linalg.svd(z * ident - x, compute_uv=False)[-1]
            log_sigma[i, j] = np.log10(max(sigma_min, 1e-16))

    np.savetxt(
        RES_DIR / "pseudospectrum_contour_grid_N60.csv",
        log_sigma,
        delimiter=",",
        header="rows: imag grid from -0.92 to 0.92; columns: real grid from -2.35 to 0.12; values: log10 sigma_min(zI-X)",
        comments="",
    )

    k = np.linspace(0.0, 2.0 * np.pi, 600, endpoint=True)
    pbc = pbc_symbol(k)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    extent = [real_values[0], real_values[-1], imag_values[0], imag_values[-1]]
    image = ax.imshow(
        log_sigma,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        vmin=-8,
        vmax=0,
    )
    levels = [-8, -6, -4, -2, -1]
    contours = ax.contour(
        real_values,
        imag_values,
        log_sigma,
        levels=levels,
        colors="white",
        linewidths=0.85,
        linestyles="solid",
    )
    ax.clabel(contours, fmt=lambda value: f"1e{int(value)}", fontsize=8)

    ax.plot(eigvals.real, eigvals.imag, "o", ms=3.0, color="black", label="OBC eigenvalues")
    ax.plot(pbc.real, pbc.imag, color="tab:red", lw=1.8, label="PBC symbol")
    ax.axvline(0.0, color="white", lw=1.0, ls="--", alpha=0.9, label="stability boundary")
    ax.axvline(-gap, color="black", lw=1.0, ls=":", label="OBC gap edge")
    ax.plot([-0.5 * gap], [0.0], marker="*", ms=10, color="gold", mec="black", label="-gap/2")

    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    ax.set_title("Pseudospectrum of a gapped local Liouvillian rapidity block")
    ax.legend(frameon=True, loc="upper left", fontsize=8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10 sigma_min(zI-X)")
    fig.savefig(FIG_DIR / "pseudospectrum_contour_N60.png", dpi=220)
    plt.close(fig)

    note = f"""# Pseudospectrum Contour

Representative local-reservoir rapidity block with `N={n_sites}`.

- OBC gap edge: `{-gap:.6g}`;
- PBC rightmost real part: `{-PARAMS.gamma + PARAMS.t_right + PARAMS.t_left:.6g}`;
- chosen point for midgap singular value: `{-0.5 * gap:.6g}`.

The OBC eigenvalues are real and gapped, but the small-singular-value contours
stretch toward the PBC ellipse and the stability boundary.

Generated files:

- `figures/pseudospectrum_contour_N60.png`
- `results/pseudospectrum_contour_grid_N60.csv`
"""
    (RES_DIR / "reports/pseudospectrum_contour.txt").write_text(note, encoding="utf-8")

    print("Wrote figures/pseudospectrum_contour_N60.png")
    print("Wrote results/pseudospectrum_contour_grid_N60.csv")


if __name__ == "__main__":
    main()
