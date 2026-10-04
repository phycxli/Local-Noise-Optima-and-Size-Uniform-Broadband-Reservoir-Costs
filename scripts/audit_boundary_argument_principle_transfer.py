"""Transfer-recursion argument-principle audit without symbolic elimination.

This evaluates the analytic continuation of the crossing numerator directly
from the 4x4 block transfer recurrence. It therefore reaches larger N than
the exact symbolic polynomial audit. The result is numerical evidence for a
complex-frequency strip, not an all-N proof.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
RES = ROOT / "results"
TR = np.array([1.0, 0.8])
TL = np.array([0.25, 0.35])
GAMMA = np.array([1.2, 1.15])
J_MIX = 0.01
KAPPA = 0.2
PORT = np.array([1.0, 0.0], dtype=complex)


def profile(z: complex, n: int, sharp: bool = False) -> np.ndarray:
    """Return the adjugate-profile blocks for the analytic or sharp recurrence."""
    sign = 1.0 if sharp else -1.0
    diag = np.diag(GAMMA + sign * 1j * z).astype(complex)
    off = sign * 1j * J_MIX
    onsite_bulk = diag + np.array([[0.0, off], [off, 0.0]], dtype=complex)
    onsite_edge = onsite_bulk.copy()
    onsite_edge[0, 0] += KAPPA / 2.0
    c = np.diag(1.0 / TR).astype(complex)
    d = np.diag(-TL / TR).astype(complex)
    lower = np.diag(-TL).astype(complex)
    eye = np.eye(2, dtype=complex)
    previous = np.zeros((2, 2), dtype=complex)
    current = eye
    blocks = [current]
    for _ in range(n - 1):
        nxt = c @ (onsite_edge if len(blocks) == 1 else onsite_bulk) @ current + d @ previous
        blocks.append(nxt)
        previous, current = current, nxt
    boundary = lower @ blocks[-2] + onsite_edge @ blocks[-1]
    adj_source = np.array([boundary[1, 1], -boundary[1, 0]], dtype=complex)
    return np.array([block @ adj_source for block in blocks])


def numerator(z: complex, n: int) -> complex:
    analytic = profile(z, n, sharp=False)
    sharp = profile(z, n, sharp=True)
    value = 0.0 + 0.0j
    h = TR + TL
    for cell in range(n):
        for orbital in range(2):
            value += sharp[cell, orbital] * (-2.0 * GAMMA[orbital]) * analytic[cell, orbital]
            if cell + 1 < n:
                value += h[orbital] * (
                    sharp[cell, orbital] * analytic[cell + 1, orbital]
                    + sharp[cell + 1, orbital] * analytic[cell, orbital]
                )
    return value


def contour(left: float, right: float, height: float, samples: int) -> np.ndarray:
    bottom = np.linspace(left - 1j * height, right - 1j * height, samples, endpoint=False)
    side_r = np.linspace(right - 1j * height, right + 1j * height, samples, endpoint=False)
    top = np.linspace(right + 1j * height, left + 1j * height, samples, endpoint=False)
    side_l = np.linspace(left + 1j * height, left - 1j * height, samples, endpoint=False)
    return np.concatenate([bottom, side_r, top, side_l])


def winding(values: np.ndarray) -> tuple[int, float]:
    increments = np.angle(np.roll(values, -1) / values)
    total = float(np.sum(increments) / (2.0 * np.pi))
    return int(np.rint(total)), total


def main() -> None:
    sizes = (24, 32, 40, 48, 56, 64, 80, 100, 120)
    heights = (0.02, 0.05, 0.07)
    samples = 500
    rows: list[dict[str, float]] = []
    for n in sizes:
        for height in heights:
            points = contour(-2.0, 2.0, height, samples)
            values = []
            normalized_values = []
            for z in points:
                analytic = profile(complex(z), n, sharp=False)
                sharp = profile(complex(z), n, sharp=True)
                value = 0.0 + 0.0j
                h = TR + TL
                for cell in range(n):
                    for orbital in range(2):
                        value += sharp[cell, orbital] * (-2.0 * GAMMA[orbital]) * analytic[cell, orbital]
                        if cell + 1 < n:
                            value += h[orbital] * (
                                sharp[cell, orbital] * analytic[cell + 1, orbital]
                                + sharp[cell + 1, orbital] * analytic[cell, orbital]
                            )
                values.append(value)
                profile_scale = np.linalg.norm(analytic) * np.linalg.norm(sharp)
                normalized_values.append(value / profile_scale)
            values = np.asarray(values)
            normalized_values = np.asarray(normalized_values)
            count, count_float = winding(values)
            scale = float(np.max(np.abs(values)))
            row = {
                "N": float(n),
                "height": float(height),
                "samples_per_side": float(samples),
                "winding": float(count),
                "winding_float": count_float,
                "min_abs_boundary": float(np.min(np.abs(values))),
                "max_abs_boundary": scale,
                "min_normalized_boundary": float(np.min(np.abs(values)) / scale),
                "min_profile_normalized_boundary": float(np.min(np.abs(normalized_values))),
                "max_profile_normalized_boundary": float(np.max(np.abs(normalized_values))),
            }
            rows.append(row)
            print(row, flush=True)
    out = RES / "boundary_argument_principle_transfer_audit.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RES / "reports/boundary_argument_principle_transfer_audit.txt").write_text(
        "# Transfer-recursion boundary argument-principle audit\n\n"
        "The crossing numerator is evaluated by the analytic/sharp transfer recurrences. "
        "The contour is the rectangle `[-2,2] x [-h,h]`.\n\n"
        + "\n".join(
            f"- N={int(row['N'])}, h={row['height']}: winding={int(row['winding'])}, "
            f"min profile-normalized boundary={row['min_profile_normalized_boundary']:.3e}"
            for row in rows
        )
        + "\n\nThis is a numerical contour audit. A theorem needs a uniform analytic lower bound on the normalized boundary function.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
