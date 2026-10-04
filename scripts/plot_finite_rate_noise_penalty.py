"""Plot the exact finite-rate CP noise lower bound near a sign crossing."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
ARXIV_FIG_DIR = ROOT / "figures"
sys.path.insert(0, str(ROOT / "scripts"))

from run_sign_adaptive_local_cp import response_data, scalar_drift  # noqa: E402


N = 40
OMEGA_STAR = 0.1329063645380138
RATE_CAPS = (50.0, 100.0, 200.0, 500.0, 1000.0)
DELTAS = np.linspace(-0.02, 0.02, 161)


def bound_row(delta: float, rate: float) -> dict[str, float]:
    data = response_data(scalar_drift(N), OMEGA_STAR + float(delta), 0, N - 1)
    y, m, _, denominator = data
    unit = y / np.linalg.norm(y)
    signature = float(np.real(np.vdot(unit, m @ unit)))
    transverse = m @ unit - signature * unit
    b2 = float(np.vdot(transverse, transverse).real)
    prefactor = float(np.vdot(y, y).real / denominator)
    excess = 0.0
    if b2 > 1.0e-14:
        excess = prefactor * max(b2 - rate * abs(signature), 0.0) ** 2 / (
            4.0 * rate * b2
        )
    return {
        "N": N,
        "omega": OMEGA_STAR + float(delta),
        "delta_omega": float(delta),
        "rate_cap": rate,
        "s_unit": signature,
        "b_norm": float(np.sqrt(b2)),
        "input_referred_excess_bound": excess,
    }


def main() -> None:
    rows = [
        bound_row(float(delta), rate)
        for rate in RATE_CAPS
        for delta in DELTAS
    ]
    RES_DIR.mkdir(exist_ok=True)
    with (RES_DIR / "finite_rate_noise_penalty.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axis = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    for rate in RATE_CAPS:
        subset = [row for row in rows if row["rate_cap"] == rate]
        values = np.asarray([row["input_referred_excess_bound"] for row in subset])
        values[values <= 0.0] = np.nan
        axis.semilogy(
            [row["delta_omega"] for row in subset],
            values,
            lw=1.0,
            label=rf"$R={rate:g}$",
        )
    axis.axvline(0.0, color="0.25", lw=0.7)
    axis.set_xlabel(r"$\omega-\omega_*$")
    axis.set_ylabel(r"finite-$R$ excess-noise bound")
    axis.set_title("(a) Bounded-rate noise penalty", loc="left", fontsize=9)
    axis.legend(frameon=False, fontsize=6, ncol=2)
    axis.grid(alpha=0.18, which="both", lw=0.5)
    for extension in ("pdf", "png"):
        path = FIG_DIR / f"finite_rate_noise_penalty.{extension}"
        fig.savefig(path, dpi=260 if extension == "png" else None)
        ARXIV_FIG_DIR.mkdir(parents=True, exist_ok=True)
        (ARXIV_FIG_DIR / path.name).write_bytes(path.read_bytes())
    plt.close(fig)
    print(f"wrote {RES_DIR / 'finite_rate_noise_penalty.csv'}")


if __name__ == "__main__":
    main()
