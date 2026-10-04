"""Generate the sign-adaptive noise/resource tradeoff diagnostic."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
ARXIV_FIG_DIR = ROOT / "figures"
sys.path.insert(0, str(ROOT / "scripts"))

from run_sign_adaptive_local_cp import one_row  # noqa: E402
from scan_sign_transition import scalar_drift, signature  # noqa: E402


def main() -> None:
    curves: list[dict[str, float]] = []
    root_rows: list[dict[str, float]] = []
    grid = np.linspace(0.0, 0.25, 251)
    for n in (24, 40, 80, 160):
        x = scalar_drift(n)
        values = [signature(x, float(w))[0] for w in grid]
        for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if fa * fb < 0.0:
                lo, hi = float(a), float(b)
                flo = fa
                for _ in range(70):
                    mid = 0.5 * (lo + hi)
                    fmid = signature(x, mid)[0]
                    if flo * fmid <= 0.0:
                        hi = mid
                    else:
                        lo, flo = mid, fmid
                root = 0.5 * (lo + hi)
                s, transverse, norm_y = signature(x, root)
                root_rows.append(
                    {"N": n, "omega_star": root, "s": s,
                     "transverse_norm": transverse, "response_norm": norm_y}
                )
        for w, s in zip(grid, values):
            curves.append({"N": n, "omega": float(w), "s_unit": float(s)})

    local_rows: list[dict[str, float | str]] = []
    n = 40
    root = next(row["omega_star"] for row in root_rows if row["N"] == n)
    for delta in (-0.020, -0.015, -0.010, -0.0075, -0.005, -0.003,
                  -0.002, -0.001, 0.001, 0.002, 0.003, 0.005, 0.0075,
                  0.010, 0.015, 0.020):
        row = one_row("scalar", n, root + delta)
        row["abs_s_unit"] = abs(float(row["s_unit"]))
        local_rows.append(row)

    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / "rate_noise_tradeoff_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    with (RES_DIR / "rate_noise_tradeoff_roots.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(root_rows[0]))
        writer.writeheader()
        writer.writerows(root_rows)
    with (RES_DIR / "rate_noise_tradeoff_local_N40.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(local_rows[0]))
        writer.writeheader()
        writer.writerows(local_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)
    for n in (24, 40, 80, 160):
        subset = [row for row in curves if row["N"] == n]
        axes[0].plot(
            [row["omega"] for row in subset],
            [row["s_unit"] for row in subset],
            lw=1.0,
            label=rf"$N={n}$",
        )
    axes[0].axhline(0.0, color="0.25", lw=0.7)
    axes[0].set_xlabel(r"frequency $\omega$")
    axes[0].set_ylabel(r"$s(\omega)=u^\dagger M u$")
    axes[0].set_title("(a) Pointwise noise branch", loc="left", fontsize=9)
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    axes[0].grid(alpha=0.18, lw=0.5)

    local_rows.sort(key=lambda row: float(row["abs_s_unit"]))
    axes[1].loglog(
        [float(row["abs_s_unit"]) for row in local_rows],
        [float(row["resource_lower_bound"]) for row in local_rows],
        "o-",
        ms=3.0,
        lw=0.9,
        label=r"universal lower bound $\|PMu\|^2/|s|$",
    )
    axes[1].loglog(
        [float(row["abs_s_unit"]) for row in local_rows],
        [float(row["rate"]) for row in local_rows],
        "s--",
        ms=3.0,
        lw=0.9,
        label="local construction",
    )
    axes[1].set_xlabel(r"$|s(\omega)|$")
    axes[1].set_ylabel(r"required rate")
    axes[1].set_title(r"(b) Resource divergence near $s=0$", loc="left", fontsize=9)
    axes[1].legend(frameon=False, fontsize=5.8)
    axes[1].xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    axes[1].xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    axes[1].xaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    axes[1].grid(alpha=0.18, which="both", lw=0.5)
    for extension in ("pdf", "png"):
        path = FIG_DIR / f"rate_noise_tradeoff.{extension}"
        fig.savefig(path, dpi=260 if extension == "png" else None)
        ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
        (ARXIV_FIG_DIR / path.name).write_bytes(path.read_bytes())
    plt.close(fig)

    print("sign-transition roots")
    for row in root_rows:
        print(row)
    print("max local-minus-bound", max(float(row["noise"]) - float(row["lower_bound"]) for row in local_rows))
    print("max identity error", max(float(row["identity_error"]) for row in local_rows))
    print("max PSD violation", max(max(0.0, -float(row["gain_min"]), -float(row["loss_min"])) for row in local_rows))


if __name__ == "__main__":
    main()
