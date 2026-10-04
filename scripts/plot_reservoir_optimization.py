"""Plot the physical device, certified shared-band costs, and design targets."""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from scipy.integrate import simpson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_band_resources import MODELS, normalized_profile, source_matrix

COLORS = {"all": "#146b78", "1": "#b34232", "2": "#645791"}


def read_json(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def style(ax):
    ax.grid(alpha=.16, linewidth=.45)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_axisbelow(True)


def save(fig, name):
    directory = ROOT / "figures"
    directory.mkdir(exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{extension}", dpi=240)
    plt.close(fig)


def group_for(resources, model, width=.05):
    return next(g for g in resources["groups"] if g["model"] == model and g["halfwidth"] == width)


def nearest(group):
    return next(b for b in group["locality_bounds"] if b["radius"] == 1)


def device_figure(rows):
    fig, (device, ax) = plt.subplots(2, 1, figsize=(3.38, 3.30), layout="constrained",
                                    gridspec_kw={"height_ratios": [1, 1.35]})
    device.set(xlim=(-.06, 1.06), ylim=(-.03, 1.04))
    device.axis("off")
    device.set_title("(a) Local bosonic device", loc="left")
    xs, y = [.12, .27, .43, .59, .75, .90], .56
    for left, right in zip(xs[:-1], xs[1:]):
        device.add_patch(FancyArrowPatch((left + .032, y), (right - .032, y),
                         arrowstyle="<->", mutation_scale=7, color=".28", linewidth=.8))
    for x, label in zip(xs, ("1", "2", r"$j$", r"$j+1$", r"$N-1$", r"$N$")):
        device.add_patch(Circle((x, y), .030, facecolor="#f3f6f7", edgecolor=".25", linewidth=.8))
        device.text(x, y - .12, label, ha="center", va="center", fontsize=7.7)
    for start, end in ((-.04, .085), (.935, 1.04)):
        device.annotate("", xy=(end, y), xytext=(start, y),
                        arrowprops={"arrowstyle": "->", "lw": 1.1})
    device.text(.04, .73, "input", ha="center", fontsize=7.5)
    device.text(.98, .73, "output", ha="center", fontsize=7.5)
    device.text(.20, .83, r"coherent $H$", ha="center", fontsize=7.5)
    device.annotate("", xy=(.20, .585), xytext=(.20, .77),
                    arrowprops={"arrowstyle": "-", "lw": .6, "color": ".5"})
    for x in (.43, .59):
        device.annotate("", xy=(x, .60), xytext=(.51, .84),
                        arrowprops={"arrowstyle": "->", "lw": 1, "color": COLORS["all"]})
        device.text((x + .51) / 2 + .014, .72, "+", color=COLORS["all"], ha="center", fontsize=7.8)
    device.text(.51, .97, r"$L_j^g$  gain", ha="center", va="center", color=COLORS["all"])
    for x, symbol in ((.43, "+"), (.59, "-")):
        device.annotate("", xy=(.51, .10), xytext=(x, .39),
                        arrowprops={"arrowstyle": "->", "lw": 1, "color": COLORS["1"]})
        device.text((x + .51) / 2 + .015, .25, symbol, color=COLORS["1"], ha="center")
    device.text(.51, .015, r"$L_j^\ell$  loss", ha="center", va="center", color=COLORS["1"])
    device.annotate("", xy=(.27, .13), xytext=(.27, .38),
                    arrowprops={"arrowstyle": "->", "lw": .8, "color": ".45"})
    device.text(.23, .02, r"$L_2^0$", ha="center", color=".4")
    rows = sorted(rows, key=lambda row: float(row["N"]))
    sizes = [float(row["N"]) for row in rows]
    ax.semilogy(sizes, [float(row["current_noise"]) for row in rows], color=".45", lw=1.1,
                label="original bonds")
    ax.semilogy(sizes, [float(row["fixed_drift_value"]) for row in rows], color=COLORS["all"],
                lw=1.4, label="fixed-drift floor")
    valid = [row for row in rows if float(row["construction_valid"]) > .5]
    ax.semilogy([float(row["N"]) for row in valid], [float(row["noise"]) for row in valid],
                "o", ms=2.7, markevery=3, color=COLORS["1"], label="local construction")
    ax.semilogy(sizes, [float(row["caves_floor"]) for row in rows], ":", color=".20", lw=1,
                label="Caves floor")
    ax.set(xlabel=r"chain length $N$", ylabel="input-referred noise", ylim=(.3, 500))
    ax.set_title(r"(b) Selected frequency $\omega=0$", loc="left")
    ax.legend(frameon=False, loc="center right", fontsize=7.1)
    style(ax)
    save(fig, "fig1_local_device")


def shared_figure(rows, certificate, resources):
    model, n, width, cap = MODELS["uniform"], 64, .05, 8
    omega = np.linspace(-width, width, 1201)
    profiles = np.array([normalized_profile(model, n, w) for w in omega])
    source = np.asarray(source_matrix(model, n), dtype=float)
    floor = np.maximum(0, np.einsum("fi,ij,fj->f", profiles.conj(), source, profiles).real)
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.75), layout="constrained",
                             gridspec_kw={"width_ratios": [1.10, .92, 1.05]})
    ax, plotted = axes[0], []
    for radius in ("all", "2", "1"):
        record = next(r for r in certificate["rows"] if r["N"] == n and
                      r["rate_cap"] == cap and r["radius"] == radius)
        path = ROOT / record["matrix_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["matrix_sha256"]:
            raise ValueError("The spectral reservoir changed after certification.")
        with np.load(path, allow_pickle=False) as saved:
            gain = saved["loss"] + source
            noise = np.einsum("fi,ij,fj->f", profiles.conj(), gain, profiles).real
        mean = float(simpson(noise, x=omega) / (2 * width))
        if not record["reported_lower"] - 1e-9 <= mean <= record["reported_upper"] + 1e-9:
            raise ValueError("Displayed spectrum disagrees with its certified band mean.")
        ax.plot(omega, noise, color=COLORS[radius], lw=1.25,
                label="shared, all" if radius == "all" else rf"shared, $r={radius}$")
        plotted.append({"radius": radius, "matrix_file": record["matrix_file"],
                        "matrix_sha256": record["matrix_sha256"], "spectral_mean": mean,
                        "certified_interval": [record["reported_lower"], record["reported_upper"]],
                        "sample_frequencies": omega[::300].tolist(),
                        "sample_noise": noise[::300].tolist()})
    ax.plot(omega, floor, "--", color=".25", lw=1, label="adaptive floor")
    ax.set(xlabel=r"frequency $\omega$", ylabel="input-referred noise", ylim=(.65, 4.2))
    ax.set_xticks([-.05, 0, .05])
    ax.set_title("(a) Shared noise spectra", loc="left", y=1.22)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2,
              fontsize=6.5, columnspacing=.7, handletextpad=.3, borderaxespad=0)

    ax = axes[1]
    selected = {radius: next(r for r in rows if int(r["N"]) == n and
                float(r["rate_cap"]) == cap and r["radius"] == radius)
                for radius in ("all", "2", "1")}
    adaptive = float(selected["all"]["point_floor_average"])
    ax.scatter([0], [adaptive], color=".25", marker="x", s=24, zorder=3)
    ax.text(0, adaptive + .14, f"{adaptive:.3f}", ha="center", fontsize=7.2)
    for x, radius in enumerate(("all", "2", "1"), start=1):
        low, high = float(selected[radius]["lower"]), float(selected[radius]["upper"])
        mean = (low + high) / 2
        ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], color=COLORS[radius],
                    fmt="o", ms=4.5, capsize=3, lw=1, zorder=3)
        ax.plot([x, x], [adaptive, mean], color=COLORS[radius], alpha=.5, lw=1)
        ax.text(x, mean + .14, f"{mean:.3f}", ha="center", fontsize=7.2)
    ax.axhline(adaptive, color=".4", ls="--", lw=.7)
    ax.set(xticks=range(4), xticklabels=["adaptive", "all", "2", "1"],
           xlabel="reservoir / matrix range", ylabel="band-average noise",
           xlim=(-.55, 3.55), ylim=(.65, 4.2))
    ax.set_title(r"(b) $N=64$, $R=8$", loc="left", y=1.22)
    ax.tick_params(axis="x", labelsize=7)

    ax = axes[2]
    for y, name in ((1, "uniform"), (0, "dimerized")):
        group = group_for(resources, name)
        bound = nearest(group)
        upper, lower = group["nonlocal_construction_upper"], bound["all_N_local_noise_lower_at_R8"]
        ax.plot([upper, lower], [y, y], color=".7", lw=5, solid_capstyle="butt", zorder=1)
        ax.scatter([upper], [y], color=COLORS["all"], marker="v", s=33, zorder=3,
                   label="reference upper" if y else None)
        ax.scatter([lower], [y], color=COLORS["1"], marker="^", s=33, zorder=3,
                   label="nearest-neighbor lower" if y else None)
        ax.text((upper + lower) / 2, y + .20,
                f"gap > {np.floor(bound['all_N_locality_gap_lower_at_R8'] * 1e5) / 1e5:.5f}",
                ha="center", fontsize=7.5)
    ax.set(yticks=[0, 1], yticklabels=["dimerized", "uniform"], xlabel="band-average noise bound",
           xlim=(2.25, 3.65), ylim=(-.75, 1.55))
    ax.set_xticks([2.4, 2.8, 3.2, 3.6])
    ax.set_title(r"(c) Every finite $N\geq64$", loc="left", y=1.22)
    ax.legend(frameon=False, loc="lower left", fontsize=6.7, handletextpad=.3)
    for ax in axes:
        style(ax)
    save(fig, "fig2_shared_band_cost")
    return {"frequency_grid_points": len(omega),
            "pointwise_floor_mean": float(simpson(floor, x=omega) / (2 * width)),
            "shared_spectra": plotted, "spectra_not_claimed_pointwise_ordered": True,
            "all_length_panel_shows_bounds_not_exact_optima": True}


def design_figure(resources, audit):
    fig, axes = plt.subplots(3, 1, figsize=(3.38, 4.60), layout="constrained",
                             gridspec_kw={"height_ratios": [1.12, 1.0, .90]})
    group, caps, ax = group_for(resources, "uniform"), np.linspace(8, 16, 201), axes[0]
    for bound in sorted(group["locality_bounds"], key=lambda b: b["radius"]):
        radius = str(bound["radius"])
        ax.plot(caps, np.maximum(0, bound["alpha"] - caps * bound["beta_with_boundary"]),
                color=COLORS[radius], lw=1.2, label=rf"$r={radius}$ lower")
    upper = group["nonlocal_construction_upper"]
    target = upper + .1
    ax.axhline(upper, color=COLORS["all"], lw=1, label="fixed reference upper")
    ax.axhline(target, color=".35", ls="--", lw=.8)
    threshold = nearest(group)["necessary_R_for_reference_plus_0p1"]
    ax.axvline(threshold, color=".5", ls=":", lw=.8)
    ax.annotate("13.0093", xy=(threshold, target), xytext=(13.55, 3.60), fontsize=7.3, ha="center",
                arrowprops={"arrowstyle": "->", "lw": .7, "color": ".35"})
    ax.text(8.15, target + .06, "reference + 0.1", color=".3", fontsize=6.9)
    ax.set(xlabel=r"complete rate cap $R$", ylabel="noise bound", xlim=(8, 16), ylim=(1.85, 3.9))
    ax.set_title("(a) Necessary rate, uniform chain", loc="left")
    ax.text(8.15, 3.55, r"$r=1$ lower", color=COLORS["1"], fontsize=7.1)
    ax.text(15.85, 2.20, r"$r=2$ lower", color=COLORS["2"], fontsize=7.1, ha="right")
    ax.text(8.15, 2.20, "fixed reference upper", color=COLORS["all"], fontsize=6.9)

    ax = axes[1]
    for name, marker, color in (("uniform", "o", COLORS["1"]), ("dimerized", "s", COLORS["all"])):
        groups = sorted([g for g in resources["groups"] if g["model"] == name],
                        key=lambda g: g["halfwidth"])
        ax.plot([g["halfwidth"] for g in groups],
                [nearest(g)["all_N_locality_gap_lower_at_R8"] for g in groups],
                marker + "-", color=color, lw=1.1, ms=3.5, label=name)
    ax.axvspan(.0495, .0505, color=".5", alpha=.16, linewidth=0)
    ax.set(xlabel=r"band halfwidth $\Omega$", ylabel="gap lower bound", ylim=(0, 1.75))
    ax.set_xticks([.03, .05, .07])
    ax.set_title(r"(b) Bandwidth, every $N\geq64$", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=7, labelspacing=.2)

    ax = axes[2]
    for y, name, target in ((1, "uniform", 2.6), (0, "dimerized", 2.9)):
        lower = nearest(group_for(resources, name))["all_N_local_noise_lower_at_R8"]
        upper = next(c for c in audit["range_two_constructions"] if c["model"] == name)["all_N_noise_upper"]
        if not upper < target < lower:
            raise ValueError("Minimum-range target is not separated by the certified bounds.")
        ax.plot([upper, lower], [y, y], color=".72", lw=2, zorder=1)
        ax.scatter([upper], [y], marker="v", color=COLORS["2"], s=28, zorder=3,
                   label=r"$r=2$ upper" if y else None)
        ax.scatter([lower], [y], marker="^", color=COLORS["1"], s=28, zorder=3,
                   label=r"$r=1$ lower" if y else None)
        ax.plot([target, target], [y - .18, y + .18], ls="--", lw=.8, color=".25")
        ax.text(target + .025, y + .20, rf"$T={target}$", fontsize=7.1, va="bottom")
        ax.text((upper + lower) / 2, y - .13, r"$r_{\min}=2$", fontsize=7.5, ha="center", va="top")
    ax.set(yticks=[0, 1], yticklabels=["dimerized", "uniform"], xlim=(2.4, 3.6), ylim=(-.4, 1.8),
           xlabel="noise bound / target")
    ax.set_xticks([2.4, 2.8, 3.2, 3.6])
    ax.set_title(r"(c) Minimum range, $R=8$", loc="left")
    ax.legend(frameon=False, loc="upper right", ncols=2, fontsize=6.8, handletextpad=.3, columnspacing=.8)
    for ax in axes:
        style(ax)
    save(fig, "fig3_design_targets")


def main():
    with (ROOT / "results/local_noise_optimization_explicit.csv").open(encoding="utf-8") as handle:
        points = [r for r in csv.DictReader(handle) if float(r["power_gain"]) > 1]
    with (ROOT / "results/rate_capped_band_optimization.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    certificate = read_json("rate_capped_band_optimization_certificate.json")
    resources = read_json("size_uniform_resources_certificate.json")
    audit = read_json("size_uniform_resources_audit.json")
    if not (certificate["all_checks_passed"] and resources["all_checks_passed"] and
            audit["all_continuous_nearest_neighbor_gaps_positive"]):
        raise RuntimeError("Figures require the existing passing resource certificates.")
    plt.rcParams.update({"font.size": 8.0, "axes.titlesize": 8.4, "axes.labelsize": 8.1,
                         "legend.fontsize": 7, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    device_figure(points)
    result = shared_figure(rows, certificate, resources)
    design_figure(resources, audit)
    result["figure_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["figure_files"] = [f"figures/{name}.pdf" for name in
                             ("fig1_local_device", "fig2_shared_band_cost", "fig3_design_targets")]
    output = ROOT / "results/reports/physical_narrative_figure_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
